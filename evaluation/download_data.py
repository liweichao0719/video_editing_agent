#!/usr/bin/env python3
"""Download licensed evaluation media and record source metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.client
import hashlib
from html import unescape
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "evaluation" / "source_catalog.json"
DEFAULT_MEDIA_DIR = ROOT / "evaluation" / "data" / "media"
DEFAULT_METADATA = ROOT / "evaluation" / "data" / "metadata.json"
API_URL = "https://commons.wikimedia.org/w/api.php"
MAX_SOURCE_BYTES = 150 * 1024 * 1024
USER_AGENT = "MultimodalAgentPoC/0.1 (local evaluation dataset builder)"
DOWNLOAD_ATTEMPTS = 5
MAX_RETRY_DELAY_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Wikimedia evaluation data")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_html(value: str | None) -> str:
    if not value:
        return ""
    return unescape(re.sub(r"<[^>]+>", "", value)).strip()


class _DownloadSizeMismatch(RuntimeError):
    pass


def _open_url_once(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=120)


def _retryable_http_status(status: int) -> bool:
    return status in {408, 429} or 500 <= status < 600


def _retry_delay(attempt: int, error: Exception) -> float:
    retry_after = None
    headers = getattr(error, "headers", None)
    if headers is not None:
        retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = 2**attempt
    else:
        delay = 2**attempt
    return min(MAX_RETRY_DELAY_SECONDS, max(1.0, delay))


def _sleep_before_retry(attempt: int, error: Exception) -> None:
    time.sleep(_retry_delay(attempt, error))


def open_url(url: str):
    """Open a small metadata request, retrying only transient connection failures."""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            return _open_url_once(url)
        except urllib.error.HTTPError as exc:
            if not _retryable_http_status(exc.code) or attempt + 1 >= DOWNLOAD_ATTEMPTS:
                raise
            _sleep_before_retry(attempt, exc)
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ) as exc:
            if attempt + 1 >= DOWNLOAD_ATTEMPTS:
                raise
            _sleep_before_retry(attempt, exc)
    raise RuntimeError("下载重试次数已耗尽")


def commons_info(file_title: str) -> dict:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|extmetadata",
            "titles": file_title,
        }
    )
    with open_url(f"{API_URL}?{query}") as response:
        payload = json.load(response)
    page = payload["query"]["pages"][0]
    if page.get("missing"):
        raise RuntimeError(f"Wikimedia 文件不存在：{file_title}")
    info = page["imageinfo"][0]
    metadata = info.get("extmetadata", {})

    def field(name: str) -> str:
        return clean_html(metadata.get(name, {}).get("value"))

    return {
        "canonical_title": page["title"],
        "description_url": info["descriptionurl"],
        "download_url": info["url"],
        "size": int(info["size"]),
        "width": info.get("width"),
        "height": info.get("height"),
        "mime": info.get("mime"),
        "wikimedia_sha1": info.get("sha1"),
        "license": field("LicenseShortName"),
        "license_url": field("LicenseUrl"),
        "artist": field("Artist"),
        "credit": field("Credit"),
        "date_time_original": field("DateTimeOriginal"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_once(
    url: str,
    destination: Path,
    temporary: Path,
    expected_size: int,
) -> None:
    temporary.unlink(missing_ok=True)
    succeeded = False
    try:
        received = 0
        with _open_url_once(url) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > MAX_SOURCE_BYTES:
                    raise RuntimeError(f"下载文件超过限制：{destination.name}")
                output.write(chunk)
        if received != expected_size:
            raise _DownloadSizeMismatch(
                f"文件大小不符：{destination.name}，"
                f"期望 {expected_size}，实际 {received}"
            )
        temporary.replace(destination)
        succeeded = True
    finally:
        if not succeeded:
            temporary.unlink(missing_ok=True)


def download_file(url: str, destination: Path, expected_size: int) -> None:
    if expected_size < 0:
        raise RuntimeError(f"源文件大小无效：{expected_size} bytes")
    if expected_size > MAX_SOURCE_BYTES:
        raise RuntimeError(
            f"源文件超过 {MAX_SOURCE_BYTES // 1024 // 1024}MB：{expected_size} bytes"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            _download_once(url, destination, temporary, expected_size)
            return
        except urllib.error.HTTPError as exc:
            if not _retryable_http_status(exc.code) or attempt + 1 >= DOWNLOAD_ATTEMPTS:
                raise
            _sleep_before_retry(attempt, exc)
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            _DownloadSizeMismatch,
        ) as exc:
            if attempt + 1 >= DOWNLOAD_ATTEMPTS:
                raise
            _sleep_before_retry(attempt, exc)
    raise RuntimeError("下载重试次数已耗尽")


def clip_media(source: Path, destination: Path, start: float, end: float) -> None:
    if start < 0 or end <= start:
        raise RuntimeError(f"无效裁剪时间：{start}-{end}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part.webm")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-c:a",
            "libopus",
            "-b:a",
            "96k",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(destination)


def write_metadata(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "items": items}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        if catalog.get("version") != 1:
            raise RuntimeError("catalog version 必须为 1")
        records = []
        with tempfile.TemporaryDirectory(prefix="evaluation-sources-") as temp_dir:
            source_cache: dict[str, Path] = {}
            info_cache: dict[str, dict] = {}
            for index, item in enumerate(catalog["items"], start=1):
                print(
                    f"[{index}/{len(catalog['items'])}] {item['id']}", file=sys.stderr
                )
                info = info_cache.get(item["file_title"])
                if info is None:
                    info = commons_info(item["file_title"])
                    info_cache[item["file_title"]] = info
                if item.get("local_existing"):
                    destination = ROOT / item.get("local_path", item["filename"])
                    if not destination.is_file():
                        raise RuntimeError(f"本地文件不存在：{destination}")
                else:
                    destination = args.media_dir / item["filename"]
                    if args.force or not destination.is_file():
                        derivative_url = item.get("derivative_url")
                        if derivative_url:
                            source = source_cache.get(derivative_url)
                            if source is None:
                                source = Path(temp_dir) / (
                                    hashlib.sha256(derivative_url.encode()).hexdigest()
                                    + ".webm"
                                )
                                download_file(
                                    derivative_url,
                                    source,
                                    int(item["derivative_size"]),
                                )
                                source_cache[derivative_url] = source
                            clip_media(
                                source,
                                destination,
                                float(item["clip_start"]),
                                float(item["clip_end"]),
                            )
                        else:
                            download_file(
                                info["download_url"], destination, info["size"]
                            )
                record = {
                    **item,
                    **info,
                    "local_path": str(destination.resolve().relative_to(ROOT)),
                    "local_size": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "license_matches_expected": (
                        item["license_expected"].casefold()
                        in info["license"].casefold()
                        or info["license"].casefold()
                        in item["license_expected"].casefold()
                    ),
                }
                records.append(record)
                write_metadata(args.metadata, records)
        print(args.metadata.resolve())
        return 0
    except (
        OSError,
        KeyError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        urllib.error.URLError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
