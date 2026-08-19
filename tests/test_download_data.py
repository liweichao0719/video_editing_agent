import http.client
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from evaluation import download_data


class FakeResponse:
    def __init__(self, *reads):
        self.reads = list(reads)

    def __enter__(self):
        return self

    def __exit__(self, _error_type, _error, _traceback):
        return False

    def read(self, _size):
        if not self.reads:
            return b""
        value = self.reads.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class DownloadRetryTests(unittest.TestCase):
    @staticmethod
    def http_error(status, headers=None):
        return urllib.error.HTTPError(
            "https://example.test/video.webm",
            status,
            "download failed",
            headers or {},
            None,
        )

    def test_retries_retryable_http_status_and_honors_retry_after(self):
        with tempfile.TemporaryDirectory(prefix="download-retry-test-") as temp_dir:
            destination = Path(temp_dir) / "video.webm"
            error = self.http_error(429, {"Retry-After": "2.5"})
            with (
                patch(
                    "evaluation.download_data.urllib.request.urlopen",
                    side_effect=[error, FakeResponse(b"video", b"")],
                ) as urlopen,
                patch("evaluation.download_data.time.sleep") as sleep,
            ):
                download_data.download_file(
                    "https://example.test/video.webm", destination, 5
                )

            self.assertEqual(destination.read_bytes(), b"video")
            self.assertFalse(destination.with_suffix(".webm.part").exists())
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once_with(2.5)

    def test_retries_transient_connection_and_stream_errors(self):
        transient_errors = [
            urllib.error.URLError("temporary DNS failure"),
            TimeoutError("timed out"),
            http.client.RemoteDisconnected("closed"),
            http.client.IncompleteRead(b"partial", 10),
            self.http_error(408),
            self.http_error(503),
        ]
        for index, error in enumerate(transient_errors):
            with self.subTest(error=type(error).__name__, status=getattr(error, "code", None)):
                with tempfile.TemporaryDirectory(
                    prefix="download-transient-test-"
                ) as temp_dir:
                    destination = Path(temp_dir) / f"video-{index}.webm"
                    first_response = (
                        FakeResponse(b"partial", error)
                        if isinstance(error, http.client.IncompleteRead)
                        else error
                    )
                    with (
                        patch(
                            "evaluation.download_data.urllib.request.urlopen",
                            side_effect=[first_response, FakeResponse(b"ok", b"")],
                        ) as urlopen,
                        patch("evaluation.download_data.time.sleep") as sleep,
                    ):
                        download_data.download_file(
                            "https://example.test/video.webm", destination, 2
                        )

                    self.assertEqual(destination.read_bytes(), b"ok")
                    self.assertFalse(destination.with_suffix(".webm.part").exists())
                    self.assertEqual(urlopen.call_count, 2)
                    sleep.assert_called_once_with(1.0)

    def test_retries_size_mismatch_as_an_incomplete_transfer(self):
        with tempfile.TemporaryDirectory(prefix="download-size-test-") as temp_dir:
            destination = Path(temp_dir) / "video.webm"
            with (
                patch(
                    "evaluation.download_data.urllib.request.urlopen",
                    side_effect=[
                        FakeResponse(b"x", b""),
                        FakeResponse(b"ok", b""),
                    ],
                ) as urlopen,
                patch("evaluation.download_data.time.sleep") as sleep,
            ):
                download_data.download_file(
                    "https://example.test/video.webm", destination, 2
                )

            self.assertEqual(destination.read_bytes(), b"ok")
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once_with(1.0)

    def test_permanent_http_error_is_not_retried_and_keeps_existing_file(self):
        with tempfile.TemporaryDirectory(prefix="download-permanent-test-") as temp_dir:
            destination = Path(temp_dir) / "video.webm"
            destination.write_bytes(b"existing")
            with (
                patch(
                    "evaluation.download_data.urllib.request.urlopen",
                    side_effect=self.http_error(404),
                ) as urlopen,
                patch("evaluation.download_data.time.sleep") as sleep,
            ):
                with self.assertRaises(urllib.error.HTTPError):
                    download_data.download_file(
                        "https://example.test/video.webm", destination, 2
                    )

            self.assertEqual(destination.read_bytes(), b"existing")
            self.assertFalse(destination.with_suffix(".webm.part").exists())
            self.assertEqual(urlopen.call_count, 1)
            sleep.assert_not_called()

    def test_exhausted_stream_retries_use_backoff_and_remove_partial_file(self):
        with tempfile.TemporaryDirectory(prefix="download-exhausted-test-") as temp_dir:
            destination = Path(temp_dir) / "video.webm"
            destination.write_bytes(b"existing")
            responses = [
                FakeResponse(
                    b"partial",
                    http.client.IncompleteRead(b"partial", 10),
                )
                for _ in range(3)
            ]
            with (
                patch("evaluation.download_data.DOWNLOAD_ATTEMPTS", 3),
                patch(
                    "evaluation.download_data.urllib.request.urlopen",
                    side_effect=responses,
                ) as urlopen,
                patch("evaluation.download_data.time.sleep") as sleep,
            ):
                with self.assertRaises(http.client.IncompleteRead):
                    download_data.download_file(
                        "https://example.test/video.webm", destination, 2
                    )

            self.assertEqual(destination.read_bytes(), b"existing")
            self.assertFalse(destination.with_suffix(".webm.part").exists())
            self.assertEqual(urlopen.call_count, 3)
            self.assertEqual(
                [call.args for call in sleep.call_args_list],
                [(1.0,), (2.0,)],
            )


if __name__ == "__main__":
    unittest.main()
