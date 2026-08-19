import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import evaluate
from evaluate import (
    EVALUATION_VERSION,
    build_evaluation_fingerprint,
    calculate_metrics,
    fingerprint_path,
    full_predictions,
    interval_iou,
    match_intervals,
    merge_resume_fingerprint,
)


class IntervalMetricTests(unittest.TestCase):
    def test_interval_iou(self):
        self.assertAlmostEqual(
            interval_iou({"start": 1.0, "end": 3.0}, {"start": 2.0, "end": 4.0}),
            1 / 3,
        )

    def test_match_reports_complete_coverage(self):
        matches = match_intervals(
            [{"start": 5.0, "end": 6.0}],
            [{"start": 4.5, "end": 6.5}],
        )
        self.assertTrue(matches[0]["complete_coverage"])

    def test_localization_uses_event_core_but_coverage_uses_output_clip(self):
        matches = match_intervals(
            [{"start": 5.0, "end": 6.0}],
            [
                {
                    "start": 5.2,
                    "end": 5.8,
                    "clip_start": 4.5,
                    "clip_end": 6.5,
                }
            ],
        )
        self.assertAlmostEqual(matches[0]["iou"], 0.6)
        self.assertTrue(matches[0]["complete_coverage"])

    def test_matching_maximizes_event_count_before_total_iou(self):
        matches = match_intervals(
            [
                {"start": 0.0, "end": 10.0},
                {"start": 8.0, "end": 12.0},
            ],
            [
                {"start": 0.0, "end": 12.0},
                {"start": 0.0, "end": 7.0},
            ],
        )
        self.assertEqual(matches[0]["prediction"], {"start": 0.0, "end": 7.0})
        self.assertEqual(matches[1]["prediction"], {"start": 0.0, "end": 12.0})

    def test_grazing_overlap_below_threshold_is_not_a_match(self):
        matches = match_intervals(
            [{"start": 0.0, "end": 10.0}],
            [{"start": 9.5, "end": 10.5}],
        )
        self.assertIsNone(matches[0]["prediction"])

    def test_matching_is_stable_when_inputs_are_reordered(self):
        truth = [
            {"start": 0.0, "end": 10.0},
            {"start": 8.0, "end": 12.0},
        ]
        predictions = [
            {"start": 0.0, "end": 12.0},
            {"start": 0.0, "end": 7.0},
        ]
        forward = match_intervals(truth, predictions)
        reversed_order = match_intervals(
            list(reversed(truth)),
            list(reversed(predictions)),
        )
        self.assertEqual(
            sorted(item["iou"] for item in forward),
            sorted(item["iou"] for item in reversed_order),
        )

    def test_matching_handles_many_predictions_without_exponential_search(self):
        truth = [
            {"start": 0.0, "end": 1.0, "id": f"truth-{index}"}
            for index in range(12)
        ]
        predictions = [
            {"start": 0.0, "end": 1.0, "id": f"prediction-{index}"}
            for index in range(30)
        ]
        matches = match_intervals(truth, predictions)
        self.assertEqual(
            sum(item["prediction"] is not None for item in matches),
            len(truth),
        )


class SummaryMetricTests(unittest.TestCase):
    @staticmethod
    def full_args() -> SimpleNamespace:
        return SimpleNamespace(
            output=Path("outputs/evaluation.json"),
            threshold=0.1,
            max_candidates=3,
            min_confidence=0.55,
            retry_padding=1.0,
            event_padding=1.0,
            merge_confirmed_gap=0.5,
            no_visual_fallback=False,
            visual_scan_interval=2.0,
            visual_padding=2.0,
            visual_scan_attempts=2,
            skip_final_review=False,
            model_path=Path("models/yamnet"),
        )

    def test_classification_and_temporal_metrics(self):
        results = [
            {
                "status": "ok",
                "expected": True,
                "predicted": True,
                "events": [{"start": 5.0, "end": 6.0}],
                "predictions": [{"start": 4.5, "end": 6.5}],
            },
            {
                "status": "ok",
                "expected": False,
                "predicted": False,
                "events": [],
                "predictions": [],
            },
        ]
        metrics = calculate_metrics(results)
        self.assertEqual(metrics["classification"]["accuracy"], 1.0)
        self.assertEqual(metrics["classification"]["precision"], 1.0)
        self.assertEqual(metrics["temporal"]["complete_coverage_recall"], 1.0)
        self.assertEqual(metrics["temporal"]["complete_coverage_rate"], 1.0)
        self.assertEqual(
            metrics["temporal"]["complete_coverage_rate_on_matched"], 1.0
        )
        self.assertEqual(metrics["temporal"]["mean_iou"], 0.5)
        self.assertEqual(metrics["temporal"]["event_precision"], 1.0)
        self.assertEqual(metrics["temporal"]["event_recall"], 1.0)
        self.assertEqual(metrics["temporal"]["event_f1"], 1.0)
        self.assertEqual(metrics["temporal"]["match_iou_threshold"], 0.1)

    def test_extra_prediction_reduces_event_precision(self):
        metrics = calculate_metrics(
            [
                {
                    "status": "ok",
                    "expected": True,
                    "predicted": True,
                    "events": [{"start": 5.0, "end": 6.0}],
                    "predictions": [
                        {"start": 4.5, "end": 6.5},
                        {"start": 5.2, "end": 5.8},
                    ],
                }
            ]
        )
        self.assertEqual(metrics["temporal"]["event_precision"], 0.5)
        self.assertEqual(metrics["temporal"]["event_recall"], 1.0)
        self.assertEqual(metrics["temporal"]["unmatched_predictions"], 1)
        self.assertEqual(metrics["temporal"]["true_positive"], 1)
        self.assertEqual(metrics["temporal"]["false_positive"], 1)
        self.assertEqual(metrics["temporal"]["false_negative"], 0)
        self.assertEqual(
            metrics["temporal"]["strict_at_iou_0_5"]["match_iou_threshold"],
            0.5,
        )

    def test_complete_clip_coverage_is_independent_from_core_localization(self):
        metrics = calculate_metrics(
            [
                {
                    "status": "ok",
                    "expected": True,
                    "predicted": True,
                    "events": [{"start": 5.0, "end": 6.0}],
                    "predictions": [
                        {
                            "start": 7.0,
                            "end": 8.0,
                            "clip_start": 4.0,
                            "clip_end": 9.0,
                        }
                    ],
                }
            ]
        )
        temporal = metrics["temporal"]
        self.assertEqual(temporal["event_recall"], 0.0)
        self.assertEqual(temporal["complete_coverage_recall"], 1.0)
        self.assertEqual(temporal["complete_coverage_rate"], 1.0)
        self.assertIsNone(temporal["complete_coverage_rate_on_matched"])

    def test_complete_clip_coverage_is_one_to_one(self):
        metrics = calculate_metrics(
            [
                {
                    "status": "ok",
                    "expected": True,
                    "predicted": True,
                    "events": [
                        {"start": 1.0, "end": 2.0},
                        {"start": 3.0, "end": 4.0},
                    ],
                    "predictions": [
                        {
                            "start": 1.0,
                            "end": 4.0,
                            "clip_start": 0.0,
                            "clip_end": 5.0,
                        }
                    ],
                }
            ]
        )
        self.assertEqual(metrics["temporal"]["complete_coverage_recall"], 0.5)

    def test_full_evaluation_does_not_score_partial_visual_scan(self):
        args = SimpleNamespace(
            output=Path("outputs/evaluation.json"),
            threshold=0.1,
            max_candidates=3,
            min_confidence=0.55,
            retry_padding=1.0,
            event_padding=1.0,
            merge_confirmed_gap=0.5,
            no_visual_fallback=False,
            visual_scan_interval=2.0,
            visual_padding=2.0,
            visual_scan_attempts=2,
            skip_final_review=False,
            model_path=Path("models/yamnet"),
        )
        report = {
            "status": "completed",
            "visual_fallback": {"error": "read timeout"},
            "results": [],
        }
        with patch(
            "evaluate.run_pipeline",
            return_value=(report, Path("pipeline_report.json")),
        ):
            with self.assertRaisesRegex(RuntimeError, "完整流程结果不计分"):
                full_predictions(
                    {
                        "id": "sample",
                        "path": Path("sample.mp4"),
                        "event": "玻璃破碎",
                    },
                    args,
                )

    def test_full_predictions_use_unpadded_event_bounds(self):
        args = SimpleNamespace(
            output=Path("outputs/evaluation.json"),
            threshold=0.1,
            max_candidates=3,
            min_confidence=0.55,
            retry_padding=1.0,
            event_padding=1.0,
            merge_confirmed_gap=0.5,
            no_visual_fallback=False,
            visual_scan_interval=2.0,
            visual_padding=2.0,
            visual_scan_attempts=2,
            skip_final_review=False,
            model_path=Path("models/yamnet"),
        )
        report = {
            "status": "completed",
            "visual_fallback": {"error": None, "used": True},
            "results": [
                {
                    "status": "completed",
                    "event_bounds": {"start": 5.2, "end": 5.8},
                    "clip": {
                        "start": 4.2,
                        "end": 6.8,
                        "path": "event.mp4",
                    },
                }
            ],
        }
        with patch(
            "evaluate.run_pipeline",
            return_value=(report, Path("pipeline_report.json")),
        ):
            predictions, _ = full_predictions(
                {
                    "id": "sample",
                    "path": Path("sample.mp4"),
                    "event": "玻璃破碎",
                },
                args,
            )
        self.assertEqual(
            predictions,
            [
                {
                    "start": 5.2,
                    "end": 5.8,
                    "clip_start": 4.2,
                    "clip_end": 6.8,
                    "path": "event.mp4",
                }
            ],
        )

    def test_full_predictions_reject_missing_or_malformed_event_bounds(self):
        invalid_cases = (
            ("missing", None, False),
            ("not-an-object", "5-6", True),
            ("non-finite", {"start": 5.0, "end": float("inf")}, True),
            ("empty", {"start": 5.0, "end": 5.0}, True),
        )
        for label, event_bounds, include_field in invalid_cases:
            with self.subTest(label=label):
                result = {
                    "status": "completed",
                    "clip": {
                        "start": 4.0,
                        "end": 7.0,
                        "path": "event.mp4",
                    },
                }
                if include_field:
                    result["event_bounds"] = event_bounds
                report = {
                    "status": "completed",
                    "visual_fallback": {"error": None, "used": True},
                    "results": [result],
                }
                with patch(
                    "evaluate.run_pipeline",
                    return_value=(report, Path("pipeline_report.json")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "event_bounds"):
                        full_predictions(
                            {
                                "id": "sample",
                                "path": Path("sample.mp4"),
                                "event": "玻璃破碎",
                            },
                            self.full_args(),
                        )

    def test_full_predictions_reject_missing_or_malformed_clip_bounds(self):
        invalid_cases = (
            ("missing", None, False),
            (
                "negative",
                {"start": -1.0, "end": 6.8, "path": "event.mp4"},
                True,
            ),
            (
                "non-finite",
                {"start": 4.2, "end": float("nan"), "path": "event.mp4"},
                True,
            ),
        )
        for label, clip, include_field in invalid_cases:
            with self.subTest(label=label):
                result = {
                    "status": "completed",
                    "event_bounds": {"start": 5.2, "end": 5.8},
                }
                if include_field:
                    result["clip"] = clip
                report = {
                    "status": "completed",
                    "visual_fallback": {"error": None, "used": True},
                    "results": [result],
                }
                with patch(
                    "evaluate.run_pipeline",
                    return_value=(report, Path("pipeline_report.json")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "clip"):
                        full_predictions(
                            {
                                "id": "sample",
                                "path": Path("sample.mp4"),
                                "event": "玻璃破碎",
                            },
                            self.full_args(),
                        )

    def test_full_evaluation_does_not_score_partial_materialization(self):
        args = SimpleNamespace(
            output=Path("outputs/evaluation.json"),
            threshold=0.1,
            max_candidates=3,
            min_confidence=0.55,
            retry_padding=1.0,
            event_padding=1.0,
            merge_confirmed_gap=0.5,
            no_visual_fallback=False,
            visual_scan_interval=2.0,
            visual_padding=2.0,
            visual_scan_attempts=2,
            skip_final_review=False,
            model_path=Path("models/yamnet"),
        )
        report = {
            "status": "completed_with_errors",
            "visual_fallback": {"error": None, "used": True},
            "results": [
                {
                    "status": "error",
                    "error": {"type": "RuntimeError", "message": "timeout"},
                }
            ],
        }
        with patch(
            "evaluate.run_pipeline",
            return_value=(report, Path("pipeline_report.json")),
        ):
            with self.assertRaisesRegex(RuntimeError, "部分结果不计分"):
                full_predictions(
                    {
                        "id": "sample",
                        "path": Path("sample.mp4"),
                        "event": "玻璃破碎",
                    },
                    args,
                )

    def test_full_predictions_reject_unknown_pipeline_status(self):
        report = {
            "status": "future_failure_mode",
            "visual_fallback": {"error": None, "used": False},
            "results": [],
        }
        with patch(
            "evaluate.run_pipeline",
            return_value=(report, Path("pipeline_report.json")),
        ):
            with self.assertRaisesRegex(RuntimeError, "非成功状态"):
                full_predictions(
                    {
                        "id": "sample",
                        "path": Path("sample.mp4"),
                        "event": "玻璃破碎",
                    },
                    self.full_args(),
                )


class EvaluationFingerprintTests(unittest.TestCase):
    def test_directory_fingerprint_changes_with_nested_model_content(self):
        with tempfile.TemporaryDirectory(prefix="evaluation-fingerprint-") as name:
            model = Path(name) / "model"
            variables = model / "variables"
            variables.mkdir(parents=True)
            weights = variables / "weights.bin"
            weights.write_bytes(b"first")
            before = fingerprint_path(model)
            weights.write_bytes(b"second")
            after = fingerprint_path(model)

        self.assertEqual(before["kind"], "directory")
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_full_fingerprint_includes_versions_media_model_and_ark(self):
        with tempfile.TemporaryDirectory(prefix="evaluation-fingerprint-") as name:
            root = Path(name)
            media = root / "sample.mp4"
            media.write_bytes(b"media-v1")
            media_sha256 = evaluate.sha256_file(media)
            model = root / "model"
            model.mkdir()
            (model / "saved_model.pb").write_bytes(b"model-v1")
            args = SimpleNamespace(stage="full", model_path=model)
            with patch.dict(
                "os.environ",
                {
                    "ARK_MODEL": "test-model",
                    "ARK_BASE_URL": "https://example.invalid/api/",
                },
            ):
                fingerprint = build_evaluation_fingerprint(
                    args,
                    [{"id": "sample", "path": media}],
                    manifest_path=root / "manifest.json",
                    manifest_sha256="manifest-sha",
                    settings={"threshold": 0.1},
                )

        self.assertEqual(fingerprint["evaluation_version"], EVALUATION_VERSION)
        self.assertEqual(fingerprint["pipeline_version"], evaluate.PIPELINE_VERSION)
        self.assertEqual(fingerprint["media"]["sample"]["sha256"], media_sha256)
        self.assertEqual(fingerprint["local_model"]["kind"], "directory")
        self.assertEqual(fingerprint["ark"]["model"], "test-model")
        self.assertEqual(
            fingerprint["ark"]["base_url"],
            "https://example.invalid/api",
        )

    def test_audio_fingerprint_ignores_ark_but_tracks_local_model(self):
        with tempfile.TemporaryDirectory(prefix="evaluation-fingerprint-") as name:
            root = Path(name)
            media = root / "sample.mp4"
            media.write_bytes(b"media")
            model = root / "model"
            model.write_bytes(b"local-model")
            args = SimpleNamespace(stage="audio", model_path=model)
            with patch.dict(
                "os.environ",
                {"ARK_MODEL": "unused-model", "ARK_BASE_URL": "https://unused"},
            ):
                fingerprint = build_evaluation_fingerprint(
                    args,
                    [{"id": "sample", "path": media}],
                    manifest_path=root / "manifest.json",
                    manifest_sha256="manifest-sha",
                    settings={},
                )

        self.assertNotIn("ark", fingerprint)
        self.assertIsNotNone(fingerprint["local_model"]["sha256"])

    def test_resume_fingerprint_rejects_version_or_runtime_changes(self):
        current = {
            "evaluation_version": EVALUATION_VERSION,
            "pipeline_version": evaluate.PIPELINE_VERSION,
            "stage": "full",
            "ark": {"model": "model-a", "base_url": "https://example.invalid"},
            "local_model": {"sha256": "model-sha"},
            "media": {"sample": {"path": "/sample.mp4", "sha256": "media-sha"}},
        }
        for field, changed_value in (
            ("evaluation_version", EVALUATION_VERSION - 1),
            ("pipeline_version", evaluate.PIPELINE_VERSION - 1),
            ("ark", {"model": "model-b", "base_url": "https://example.invalid"}),
            ("local_model", {"sha256": "different-model-sha"}),
        ):
            with self.subTest(field=field):
                previous = json.loads(json.dumps(current))
                previous[field] = changed_value
                with self.assertRaisesRegex(RuntimeError, "运行指纹"):
                    merge_resume_fingerprint(
                        previous,
                        current,
                        previous_result_ids={"sample"},
                    )

    def test_resume_fingerprint_rejects_changed_selected_media(self):
        current = {
            "evaluation_version": EVALUATION_VERSION,
            "media": {"sample": {"path": "/sample.mp4", "sha256": "new"}},
        }
        previous = {
            "evaluation_version": EVALUATION_VERSION,
            "media": {"sample": {"path": "/sample.mp4", "sha256": "old"}},
        }
        with self.assertRaisesRegex(RuntimeError, "媒体内容已变化"):
            merge_resume_fingerprint(
                previous,
                current,
                previous_result_ids={"sample"},
            )

    def test_resume_fingerprint_rejects_changed_unselected_saved_media(self):
        previous = {
            "evaluation_version": EVALUATION_VERSION,
            "media": {
                "selected": {"path": "/selected.mp4", "sha256": "same"},
                "saved": {"path": "/saved.mp4", "sha256": "old"},
            },
        }
        current = {
            "evaluation_version": EVALUATION_VERSION,
            "media": {
                "selected": {"path": "/selected.mp4", "sha256": "same"},
                "saved": {"path": "/saved.mp4", "sha256": "new"},
            },
        }
        with self.assertRaisesRegex(RuntimeError, "saved.*媒体内容已变化"):
            merge_resume_fingerprint(
                previous,
                current,
                previous_result_ids={"selected", "saved"},
            )

    def test_resume_fingerprint_requires_media_hash_for_every_saved_result(self):
        fingerprint = {
            "evaluation_version": EVALUATION_VERSION,
            "media": {},
        }
        with self.assertRaisesRegex(RuntimeError, "缺少媒体内容指纹"):
            merge_resume_fingerprint(
                fingerprint,
                fingerprint,
                previous_result_ids={"saved-sample"},
            )


class ResumeSelectionTests(unittest.TestCase):
    @staticmethod
    def make_args(
        manifest: Path,
        output: Path,
        model: Path,
        *,
        resume: bool,
        sample_ids: list[str],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            manifest=manifest,
            stage="audio",
            output=output,
            split=None,
            sample_id=sample_ids,
            limit=None,
            resume=resume,
            threshold=0.1,
            max_candidates=3,
            min_confidence=0.55,
            retry_padding=1.0,
            event_padding=1.0,
            merge_confirmed_gap=0.5,
            visual_scan_interval=2.0,
            visual_padding=2.0,
            visual_scan_attempts=2,
            no_visual_fallback=False,
            skip_final_review=False,
            model_path=model,
        )

    @staticmethod
    def successful_result(sample: dict, _args: SimpleNamespace) -> dict:
        return {
            "id": sample["id"],
            "status": "ok",
            "expected": sample["expected"],
            "predicted": False,
            "events": sample["events"],
            "predictions": [],
        }

    def test_resume_with_sample_id_preserves_unselected_results(self):
        with tempfile.TemporaryDirectory(prefix="evaluation-resume-") as name:
            root = Path(name)
            for sample_id in ("one", "two"):
                (root / f"{sample_id}.mp4").write_bytes(sample_id.encode("utf-8"))
            model = root / "model"
            model.write_bytes(b"model")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "name": "resume-test",
                        "default_event": "测试事件",
                        "samples": [
                            {"id": "one", "path": "one.mp4", "expected": False},
                            {"id": "two", "path": "two.mp4", "expected": False},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "evaluation.json"
            first_args = self.make_args(
                manifest, output, model, resume=False, sample_ids=[]
            )
            with patch("evaluate.parse_args", return_value=first_args), patch(
                "evaluate.evaluate_sample", side_effect=self.successful_result
            ):
                self.assertEqual(evaluate.main(), 0)

            resume_args = self.make_args(
                manifest, output, model, resume=True, sample_ids=["one"]
            )
            with patch("evaluate.parse_args", return_value=resume_args), patch(
                "evaluate.evaluate_sample"
            ) as evaluator:
                self.assertEqual(evaluate.main(), 0)
                evaluator.assert_not_called()

            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], EVALUATION_VERSION)
        self.assertEqual(payload["fingerprint"]["pipeline_version"], evaluate.PIPELINE_VERSION)
        self.assertEqual(
            [item["id"] for item in payload["results"]],
            ["one", "two"],
        )
        self.assertEqual(set(payload["fingerprint"]["media"]), {"one", "two"})

    def test_resume_rejects_changed_unselected_saved_media(self):
        with tempfile.TemporaryDirectory(prefix="evaluation-resume-") as name:
            root = Path(name)
            one = root / "one.mp4"
            two = root / "two.mp4"
            one.write_bytes(b"one")
            two.write_bytes(b"two")
            model = root / "model"
            model.write_bytes(b"model")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "name": "resume-test",
                        "default_event": "测试事件",
                        "samples": [
                            {"id": "one", "path": "one.mp4", "expected": False},
                            {"id": "two", "path": "two.mp4", "expected": False},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "evaluation.json"
            first_args = self.make_args(
                manifest, output, model, resume=False, sample_ids=[]
            )
            with patch("evaluate.parse_args", return_value=first_args), patch(
                "evaluate.evaluate_sample", side_effect=self.successful_result
            ):
                self.assertEqual(evaluate.main(), 0)

            previous_output = output.read_bytes()
            two.write_bytes(b"two-changed")
            resume_args = self.make_args(
                manifest, output, model, resume=True, sample_ids=["one"]
            )
            with patch("evaluate.parse_args", return_value=resume_args), patch(
                "evaluate.evaluate_sample"
            ) as evaluator:
                self.assertEqual(evaluate.main(), 1)
                evaluator.assert_not_called()

            self.assertEqual(output.read_bytes(), previous_output)


if __name__ == "__main__":
    unittest.main()
