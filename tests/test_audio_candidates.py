from pathlib import Path
import shutil
import tempfile
import unittest

from audio_candidates import (
    DEFAULT_EVENT_CONFIG,
    DEFAULT_MODEL_PATH,
    detect_candidates,
    extract_audio,
    format_time,
    load_event_labels,
    merge_frames,
    video_duration,
)


ROOT = Path(__file__).resolve().parents[1]
GLASS_FIXTURE = ROOT / "tests" / "fixtures" / "glass_shattering_cc0.mp3"
BLENDER_FIXTURE = ROOT / "samples" / "test_blender_av.webm"


class CandidateUtilitiesTests(unittest.TestCase):
    def test_chinese_event_alias_resolves_to_yamnet_labels(self):
        labels = load_event_labels(DEFAULT_EVENT_CONFIG, "玻璃破碎")
        self.assertIn("Glass", labels)
        self.assertIn("Shatter", labels)

    def test_time_format(self):
        self.assertEqual(format_time(65.25), "01:05.25")

    def test_overlapping_frames_are_merged_and_padded(self):
        frames = [
            {
                "start": 1.0,
                "end": 1.96,
                "score": 0.6,
                "top_label": "Glass",
                "label_scores": {"Glass": 0.6},
            },
            {
                "start": 1.48,
                "end": 2.44,
                "score": 0.8,
                "top_label": "Shatter",
                "label_scores": {"Glass": 0.5, "Shatter": 0.8},
            },
        ]
        candidates = merge_frames(frames, 10.0, padding=0.5, merge_gap=0.24)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["start"], 0.5)
        self.assertEqual(candidates[0]["end"], 2.94)
        self.assertEqual(candidates[0]["top_label"], "Shatter")


@unittest.skipUnless(
    (DEFAULT_MODEL_PATH / "saved_model.pb").is_file()
    and GLASS_FIXTURE.is_file()
    and BLENDER_FIXTURE.is_file()
    and shutil.which("ffmpeg"),
    "YAMNet model or audio fixture is unavailable",
)
class YamnetIntegrationTests(unittest.TestCase):
    def test_detects_cc0_glass_shattering_sample(self):
        duration = video_duration(GLASS_FIXTURE)
        with tempfile.TemporaryDirectory(prefix="yamnet-test-") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            extract_audio(GLASS_FIXTURE, wav_path)
            candidates, scores, _ = detect_candidates(
                wav_path,
                DEFAULT_MODEL_PATH,
                load_event_labels(DEFAULT_EVENT_CONFIG, "玻璃破碎"),
                threshold=0.1,
                duration=duration,
                padding=0.5,
                merge_gap=0.24,
            )
        self.assertTrue(candidates)
        self.assertGreater(scores["Glass"], 0.5)
        self.assertGreater(scores["Shatter"], 0.5)

    def test_does_not_report_blender_as_glass_shattering(self):
        duration = video_duration(BLENDER_FIXTURE)
        with tempfile.TemporaryDirectory(prefix="yamnet-test-") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            extract_audio(BLENDER_FIXTURE, wav_path)
            candidates, scores, _ = detect_candidates(
                wav_path,
                DEFAULT_MODEL_PATH,
                load_event_labels(DEFAULT_EVENT_CONFIG, "玻璃破碎"),
                threshold=0.1,
                duration=duration,
                padding=0.5,
                merge_gap=0.24,
            )
        self.assertFalse(candidates)
        self.assertLess(max(scores.values()), 0.1)


if __name__ == "__main__":
    unittest.main()
