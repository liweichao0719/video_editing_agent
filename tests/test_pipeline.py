from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import (
    as_bool,
    as_confidence,
    cut_clip,
    extract_json,
    fuse_candidates,
    make_visual_scan,
    normalize_visual_candidates,
    probe_media,
    refine_clip_bounds,
    scan_visual_candidates,
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

    def test_fuses_cross_modal_duplicate_and_keeps_uncovered_visual_event(self):
        candidates = fuse_candidates(
            [
                {
                    "start": 10.0,
                    "end": 13.0,
                    "score": 0.7,
                    "origin": "audio",
                    "top_label": "Breaking",
                }
            ],
            [
                {
                    "start": 9.5,
                    "end": 13.5,
                    "score": 0.8,
                    "origin": "visual_fallback",
                    "top_label": "visual_scan",
                    "visual_evidence": "后半段发生碎裂",
                },
                {
                    "start": 2.0,
                    "end": 5.0,
                    "score": 0.9,
                    "origin": "visual_fallback",
                    "top_label": "visual_scan",
                    "visual_evidence": "前半段发生碎裂",
                },
            ],
            max_candidates=3,
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["start"], 2.0)
        self.assertEqual(candidates[1]["origin"], "audio_visual")
        self.assertEqual(candidates[1]["start"], 9.5)
        self.assertEqual(candidates[1]["end"], 13.5)

    def test_refines_model_bounds_with_context(self):
        start, end, used = refine_clip_bounds(
            {"start": 10.0, "end": 15.0},
            {"event_start": 1.0, "event_end": 3.0},
            20.0,
            padding=0.5,
        )
        self.assertTrue(used)
        self.assertEqual((start, end), (10.5, 13.5))

    def test_broad_visual_candidate_does_not_swallow_narrow_audio_event(self):
        candidates = fuse_candidates(
            [
                {
                    "start": 15.0,
                    "end": 18.0,
                    "score": 0.7,
                    "origin": "audio",
                    "top_label": "Breaking",
                }
            ],
            [
                {
                    "start": 3.0,
                    "end": 19.0,
                    "score": 0.9,
                    "origin": "visual_fallback",
                    "top_label": "visual_scan",
                    "visual_evidence": "宽范围内发生状态变化",
                }
            ],
            max_candidates=3,
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {candidate["origin"] for candidate in candidates},
            {"audio", "visual_fallback"},
        )

    def test_keeps_candidate_when_model_bounds_are_invalid(self):
        start, end, used = refine_clip_bounds(
            {"start": 10.0, "end": 15.0},
            {"event_start": 4.0, "event_end": 2.0},
            20.0,
            padding=0.5,
        )
        self.assertFalse(used)
        self.assertEqual((start, end), (10.0, 15.0))

    def test_visual_scan_retries_transient_model_failure(self):
        with patch(
            "pipeline.call_model",
            side_effect=[RuntimeError("timeout"), {"candidates": []}],
        ) as model:
            candidates, details = scan_visual_candidates(
                Path("scan.mp4"),
                "玻璃破碎",
                10.0,
                interval=2.0,
                padding=1.0,
                max_candidates=3,
                attempts=2,
                api_key="test-key",
            )
        self.assertEqual(candidates, [])
        self.assertEqual(details["attempts"], 2)
        self.assertEqual(model.call_count, 2)


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
        self.assertLess(media["duration"], 3.0)
        self.assertLess(media["size"], BLENDER_FIXTURE.stat().st_size)


if __name__ == "__main__":
    unittest.main()
