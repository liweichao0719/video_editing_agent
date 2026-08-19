from pathlib import Path
import tempfile
import unittest

from pipeline import (
    as_bool,
    as_confidence,
    cut_clip,
    extract_json,
    make_visual_scan,
    normalize_visual_candidates,
    probe_media,
    technical_review,
)


ROOT = Path(__file__).resolve().parents[1]
BLENDER_FIXTURE = ROOT / "samples" / "test_blender_av.webm"


class ModelResponseTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"confirmed": true}\n```')["confirmed"], True)

    def test_extracts_json_surrounded_by_text(self):
        self.assertEqual(extract_json('结果：{"complete": true}。')["complete"], True)

    def test_normalizes_model_values(self):
        self.assertTrue(as_bool("是"))
        self.assertEqual(as_confidence("87"), 0.87)

    def test_normalizes_visual_candidates_before_padding(self):
        candidates = normalize_visual_candidates(
            {
                "candidates": [
                    {
                        "start": 1,
                        "end": 2,
                        "confidence": 80,
                        "visual_evidence": "出现裂纹",
                    },
                    {
                        "start": 2.5,
                        "end": 3,
                        "confidence": 0.7,
                        "visual_evidence": "碎片飞散",
                    },
                    {"start": 8, "end": 9, "confidence": 0.1},
                ]
            },
            10,
            padding=1,
            max_candidates=3,
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["start"], 0.0)
        self.assertEqual(candidates[0]["end"], 3.0)
        self.assertEqual(candidates[0]["score"], 0.8)
        self.assertEqual(candidates[1]["start"], 1.5)
        self.assertEqual(candidates[1]["end"], 4.0)


@unittest.skipUnless(BLENDER_FIXTURE.is_file(), "video fixture is unavailable")
class ClipIntegrationTests(unittest.TestCase):
    def test_cut_clip_keeps_audio_and_video(self):
        with tempfile.TemporaryDirectory(prefix="clip-test-") as temp_dir:
            clip = Path(temp_dir) / "clip.mp4"
            cut_clip(BLENDER_FIXTURE, clip, 1.0, 3.0)
            media = probe_media(clip)
            review = technical_review(clip, 2.0)
        self.assertTrue(media["has_video"])
        self.assertTrue(media["has_audio"])
        self.assertTrue(review["passed"])

    def test_visual_scan_is_small_silent_video(self):
        with tempfile.TemporaryDirectory(prefix="visual-scan-test-") as temp_dir:
            scan = Path(temp_dir) / "scan.mp4"
            make_visual_scan(BLENDER_FIXTURE, scan, 2.0)
            media = probe_media(scan)
        self.assertTrue(media["has_video"])
        self.assertFalse(media["has_audio"])
        self.assertLess(media["size"], BLENDER_FIXTURE.stat().st_size)


if __name__ == "__main__":
    unittest.main()
