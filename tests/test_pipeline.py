from pathlib import Path
import http.client
import io
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from pipeline import (
    as_bool,
    as_confidence,
    call_model,
    call_model_with_retries,
    cached_confirmation,
    collect_event_proposals,
    confirmation_cache_identity,
    confirm_candidate,
    consolidate_completed_results,
    cut_clip,
    deduplicate_core_events,
    detect_scene_change_hints,
    event_to_source_core,
    extract_json,
    fuse_candidates,
    group_adjacent_confirmed_results,
    make_scene_change_candidates,
    make_visual_scan,
    materialize_event_result,
    load_visual_scan_cache,
    load_confirmation_cache,
    normalize_confirmation_events,
    normalize_visual_candidates,
    padded_bounds_for_core,
    probe_media,
    refine_clip_bounds,
    refine_event_bounds,
    run_pipeline,
    scan_visual_candidates,
    split_confirmation_candidates,
    stable_json_sha256,
    technical_review,
    visual_scan_cache_identity,
    write_visual_scan_cache,
    write_confirmation_cache,
    RetryableModelError,
)


ROOT = Path(__file__).resolve().parents[1]
BLENDER_FIXTURE = ROOT / "samples" / "test_blender_av.webm"


class ModelResponseTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"confirmed": true}\n```')["confirmed"], True)

    def test_extracts_json_surrounded_by_text(self):
        self.assertEqual(extract_json('结果：{"complete": true}。')["complete"], True)

    def test_remote_disconnect_is_reported_as_retryable_model_error(self):
        with tempfile.TemporaryDirectory(prefix="model-error-test-") as temp_dir:
            clip = Path(temp_dir) / "clip.mp4"
            clip.write_bytes(b"video")
            with patch(
                "pipeline.urllib.request.urlopen",
                side_effect=http.client.RemoteDisconnected("closed"),
            ):
                with self.assertRaisesRegex(RetryableModelError, "模型请求失败"):
                    call_model(clip, "prompt", "test-key")

    def test_permanent_http_error_is_not_marked_retryable(self):
        with tempfile.TemporaryDirectory(prefix="model-error-test-") as temp_dir:
            clip = Path(temp_dir) / "clip.mp4"
            clip.write_bytes(b"video")
            error = urllib.error.HTTPError(
                "https://example.invalid",
                401,
                "unauthorized",
                {},
                io.BytesIO(b"unauthorized"),
            )
            with patch("pipeline.urllib.request.urlopen", side_effect=error):
                with self.assertRaises(RuntimeError) as raised:
                    call_model(clip, "prompt", "test-key")
        self.assertNotIsInstance(raised.exception, RetryableModelError)

    def test_invalid_service_json_is_retryable(self):
        with tempfile.TemporaryDirectory(prefix="model-error-test-") as temp_dir:
            clip = Path(temp_dir) / "clip.mp4"
            clip.write_bytes(b"video")
            with patch(
                "pipeline.urllib.request.urlopen",
                return_value=io.BytesIO(b"not-json"),
            ):
                with self.assertRaises(RetryableModelError):
                    call_model(clip, "prompt", "test-key")

    def test_candidate_confirmation_retries_transient_model_error(self):
        response = {
            "confirmed": True,
            "confidence": 0.9,
            "event_start": 0.5,
            "event_end": 1.0,
        }
        with (
            patch(
                "pipeline.call_model",
                side_effect=[RetryableModelError("disconnect"), response],
            ) as model,
            patch("pipeline.time.sleep") as sleep,
            patch("pipeline.random.random", return_value=0.0),
        ):
            confirmation = confirm_candidate(
                Path("clip.mp4"),
                "玻璃破碎",
                {"origin": "audio", "score": 0.8, "start": 10.0, "end": 12.0},
                "test-key",
            )
        self.assertTrue(confirmation["confirmed"])
        self.assertEqual(confirmation["attempts"], 2)
        self.assertEqual(confirmation["occurrence_count"], 1)
        self.assertEqual(model.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_permanent_model_error_is_not_retried(self):
        with patch(
            "pipeline.call_model",
            side_effect=RuntimeError("missing API key"),
        ) as model:
            with self.assertRaisesRegex(RuntimeError, "missing API key"):
                call_model_with_retries(
                    Path("clip.mp4"),
                    "prompt",
                    "",
                    attempts=3,
                )
        self.assertEqual(model.call_count, 1)

    def test_retry_after_controls_model_backoff(self):
        with (
            patch(
                "pipeline.call_model",
                side_effect=[
                    RetryableModelError("rate limited", retry_after=5.0),
                    {"ok": True},
                ],
            ),
            patch("pipeline.time.sleep") as sleep,
            patch("pipeline.random.random", return_value=0.0),
        ):
            result, attempts = call_model_with_retries(
                Path("clip.mp4"), "prompt", "test-key", attempts=2
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 2)
        sleep.assert_called_once_with(5.0)

    def test_normalizes_model_values(self):
        self.assertTrue(as_bool("是"))
        self.assertEqual(as_confidence("87"), 0.87)

    def test_nonfinite_confidence_is_not_accepted(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertEqual(as_confidence(value), 0.0)
                events = normalize_confirmation_events(
                    {
                        "events": [
                            {
                                "event_start": 0.1,
                                "event_end": 0.2,
                                "confidence": value,
                            }
                        ]
                    },
                    1.0,
                )
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["confidence"], 0.0)
                self.assertEqual(
                    normalize_visual_candidates(
                        {
                            "candidates": [
                                {
                                    "start": 0.1,
                                    "end": 0.2,
                                    "confidence": value,
                                }
                            ]
                        },
                        1.0,
                        padding=0.0,
                        max_candidates=1,
                    ),
                    [],
                )

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
        core_start, core_end, core_used = refine_event_bounds(
            {"start": 10.0, "end": 15.0},
            {"event_start": 1.0, "event_end": 3.0},
            20.0,
        )
        self.assertTrue(core_used)
        self.assertEqual((core_start, core_end), (11.0, 13.0))

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

    def test_splits_broad_confirmation_candidate_into_bounded_windows(self):
        candidates = split_confirmation_candidates(
            [
                {
                    "start": 3.0,
                    "end": 19.0,
                    "score": 0.9,
                    "origin": "visual_fallback",
                    "visual_evidence": "宽范围状态变化",
                }
            ]
        )
        self.assertEqual(
            [(item["start"], item["end"]) for item in candidates],
            [(3.0, 11.0), (7.0, 15.0), (11.0, 19.0)],
        )
        self.assertTrue(
            all(item["end"] - item["start"] <= 8.0 for item in candidates)
        )
        self.assertEqual(
            [item["segment_index"] for item in candidates],
            [1, 2, 3],
        )
        self.assertTrue(
            all(item["segment_count"] == 3 for item in candidates)
        )

    def test_does_not_modify_short_confirmation_candidate(self):
        candidate = {
            "start": 3.0,
            "end": 7.0,
            "score": 0.7,
            "origin": "audio",
        }
        self.assertEqual(split_confirmation_candidates([candidate]), [candidate])

    def test_scene_scores_do_not_crowd_out_remote_visual_candidates(self):
        remote = [
            {
                "start": float(index * 10),
                "end": float(index * 10 + 2),
                "score": 0.4 + index * 0.05,
                "origin": "visual_fallback",
            }
            for index in range(3)
        ]
        scene = [
            {
                "start": float(40 + index * 10),
                "end": float(42 + index * 10),
                "score": 0.99 - index * 0.01,
                "origin": "visual_change",
            }
            for index in range(3)
        ]
        candidates = fuse_candidates([], [*remote, *scene], max_candidates=3)
        self.assertEqual(
            sum(item["origin"] == "visual_fallback" for item in candidates),
            3,
        )
        self.assertEqual(
            sum(item["origin"] == "visual_change" for item in candidates),
            2,
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
        with (
            patch(
                "pipeline.call_model",
                side_effect=[RetryableModelError("timeout"), {"candidates": []}],
            ) as model,
            patch("pipeline.time.sleep"),
        ):
            candidates, details = scan_visual_candidates(
                Path("scan.mp4"),
                "玻璃破碎",
                10.0,
                interval=2.0,
                padding=1.0,
                max_candidates=3,
                attempts=2,
                api_key="test-key",
                time_hints=[{"start": 4.0, "end": 6.0}],
                scene_hints=[{"time": 8.25, "score": 0.4}],
            )
        self.assertEqual(candidates, [])
        self.assertEqual(details["attempts"], 2)
        self.assertEqual(model.call_count, 2)
        self.assertIn("4.00-6.00秒", model.call_args.args[1])
        self.assertIn("8.25秒", model.call_args.args[1])

    def test_extracts_strongest_scene_change_hints(self):
        output = """frame:0 pts:100 pts_time:1.25
lavfi.scene_score=0.3
frame:1 pts:200 pts_time:2.5
lavfi.scene_score=0.8
frame:2 pts:300 pts_time:3.75
lavfi.scene_score=0.5
"""
        with patch("pipeline.run_command") as command:
            command.return_value.stdout = output
            hints = detect_scene_change_hints(Path("video.mp4"), max_hints=2)
        self.assertEqual(
            hints,
            [
                {"time": 2.5, "score": 0.8},
                {"time": 3.75, "score": 0.5},
            ],
        )

    def test_scene_change_candidates_only_use_uncovered_strong_changes(self):
        candidates = make_scene_change_candidates(
            [
                {"time": 1.2, "score": 0.8},
                {"time": 7.9, "score": 0.1},
                {"time": 11.785, "score": 0.43},
                {"time": 19.0, "score": 0.32},
            ],
            21.0,
            covered_by=[
                {"start": 0.0, "end": 6.0},
                {"start": 16.3, "end": 18.74},
            ],
            padding=2.0,
            max_candidates=3,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["origin"], "visual_change")
        self.assertEqual(candidates[0]["start"], 11.535)
        self.assertEqual(candidates[0]["end"], 13.785)
        self.assertEqual(candidates[0]["focus_time"], 0.25)

    def test_scene_change_candidates_suppress_dense_duplicates(self):
        candidates = make_scene_change_candidates(
            [
                {"time": 5.0, "score": 0.9},
                {"time": 5.2, "score": 0.8},
                {"time": 11.0, "score": 0.7},
            ],
            15.0,
            covered_by=[],
            padding=2.0,
            max_candidates=3,
        )
        self.assertEqual([item["change_time"] for item in candidates], [5.0, 11.0])

    def test_visual_scan_cache_requires_exact_source_prompt_and_model(self):
        source = Path("/video.mp4")
        with patch.dict(
            "pipeline.os.environ",
            {"ARK_MODEL": "model-a", "ARK_BASE_URL": "https://model.example/v1"},
        ):
            identity = visual_scan_cache_identity(
                source=source,
                source_sha256="a" * 64,
                prompt="prompt-a",
                interval=2.0,
            )
        with tempfile.TemporaryDirectory(prefix="scan-cache-test-") as temp_dir:
            cache = Path(temp_dir) / "video_visual_scan_cache.json"
            report = Path(temp_dir) / "video_pipeline_report.json"
            report.write_text('{"status":"completed"}\n', encoding="utf-8")
            write_visual_scan_cache(
                cache,
                identity=identity,
                raw={"candidates": []},
                scan_media={"duration": 1.0},
            )
            cached = load_visual_scan_cache(
                cache,
                identity=identity,
            )
            changed_source = {**identity, "source_sha256": "b" * 64}
            changed_prompt = {**identity, "prompt_sha256": "c" * 64}
            changed_model = {**identity, "model": "model-b"}
            stale_values = [
                load_visual_scan_cache(cache, identity=value)
                for value in (changed_source, changed_prompt, changed_model)
            ]
            final_report = report.read_text(encoding="utf-8")
        self.assertEqual(cached["raw"], {"candidates": []})
        self.assertEqual(stale_values, [None, None, None])
        self.assertEqual(final_report, '{"status":"completed"}\n')

    def test_confirmation_cache_requires_exact_identity(self):
        source = Path("/video.mp4")
        candidate = {
            "start": 1.0,
            "end": 2.0,
            "score": 0.8,
            "origin": "audio",
        }
        identity = confirmation_cache_identity(
            source=source,
            source_sha256="a" * 64,
            candidate=candidate,
            prompt="prompt-a",
        )
        confirmation = {
            "confirmed": True,
            "events": [{"event_start": 0.2, "event_end": 0.4}],
            "attempts": 1,
        }
        key = stable_json_sha256(identity)
        entries = {key: {"identity": identity, "confirmation": confirmation}}
        with tempfile.TemporaryDirectory(prefix="confirmation-cache-") as temp_dir:
            cache = Path(temp_dir) / "confirmation.json"
            write_confirmation_cache(
                cache,
                source=source,
                source_sha256="a" * 64,
                entries=entries,
            )
            loaded = load_confirmation_cache(cache, source_sha256="a" * 64)
            hit = cached_confirmation(loaded, identity)
            miss = cached_confirmation(
                loaded,
                {**identity, "prompt_sha256": "b" * 64},
            )
            changed_source = load_confirmation_cache(
                cache, source_sha256="c" * 64
            )
        self.assertTrue(hit["cache_hit"])
        self.assertEqual(hit["attempts"], 0)
        self.assertIsNone(miss)
        self.assertEqual(changed_source, {})

    def test_confirmation_cache_keeps_progress_before_later_failure(self):
        candidates = [
            {
                "start": start,
                "end": end,
                "score": 0.8,
                "origin": "audio",
            }
            for start, end in ((1.0, 2.0), (3.0, 4.0))
        ]
        confirmation = {
            "confirmed": True,
            "confidence": 0.9,
            "occurrence_count": 1,
            "events": [
                {
                    "event_start": 0.2,
                    "event_end": 0.4,
                    "confidence": 0.9,
                }
            ],
            "attempts": 1,
        }
        with tempfile.TemporaryDirectory(prefix="confirmation-cache-") as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            cache = root / "confirmation.json"
            with (
                patch("pipeline.make_candidate_clip", return_value=root / "candidate.mp4"),
                patch(
                    "pipeline.confirm_candidate",
                    side_effect=[confirmation, RuntimeError("timeout")],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "timeout"):
                    collect_event_proposals(
                        source,
                        root,
                        candidates,
                        "玻璃破碎",
                        source_duration=5.0,
                        source_sha256="a" * 64,
                        confirmation_cache_path=cache,
                        min_confidence=0.55,
                        api_key="test-key",
                    )
            loaded = load_confirmation_cache(cache, source_sha256="a" * 64)
        self.assertEqual(len(loaded), 1)


class CoreEventTests(unittest.TestCase):
    @staticmethod
    def proposal(proposal_id, candidate_id, start, end, confidence=0.9):
        return {
            "stable_id": proposal_id,
            "candidate_id": candidate_id,
            "confidence": confidence,
            "core": {"start": start, "end": end},
        }

    def test_confirmation_normalizes_multiple_events_and_legacy_single(self):
        events = normalize_confirmation_events(
            {
                "confidence": 0.9,
                "events": [
                    {"event_start": 4.0, "event_end": 4.5, "confidence": 0.8},
                    {"event_start": 1.0, "event_end": 1.4, "confidence": 0.95},
                    {"event_start": 1.0, "event_end": 1.4, "confidence": 0.5},
                ],
            },
            5.0,
        )
        self.assertEqual(
            [(item["event_start"], item["event_end"]) for item in events],
            [(1.0, 1.4), (4.0, 4.5)],
        )
        self.assertEqual(events[0]["confidence"], 0.95)

        legacy = normalize_confirmation_events(
            {"event_start": 0.5, "event_end": 0.8, "confidence": 0.7},
            2.0,
        )
        self.assertEqual(len(legacy), 1)

    def test_confirmation_keeps_only_first_six_valid_events_in_time_order(self):
        events = normalize_confirmation_events(
            {
                "events": [
                    {
                        "event_start": index + 0.1,
                        "event_end": index + 0.2,
                        "confidence": 0.9,
                    }
                    for index in reversed(range(20))
                ]
            },
            20.0,
        )
        self.assertEqual(len(events), 6)
        self.assertEqual(
            [item["event_start"] for item in events],
            [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
        )

    def test_explicit_empty_or_invalid_events_never_fall_back_to_candidate(self):
        self.assertEqual(
            normalize_confirmation_events(
                {
                    "events": [],
                    "event_start": 1.0,
                    "event_end": 2.0,
                    "confidence": 0.9,
                },
                5.0,
            ),
            [],
        )
        self.assertEqual(
            normalize_confirmation_events(
                {
                    "events": [
                        {"event_start": None, "event_end": 2.0},
                        {"event_start": 4.0, "event_end": 3.0},
                    ]
                },
                5.0,
            ),
            [],
        )

    def test_wide_candidate_keeps_multiple_strict_source_cores(self):
        candidate = {"start": 10.0, "end": 20.0}
        cores = [
            event_to_source_core(candidate, event, 30.0)
            for event in (
                {"event_start": 1.0, "event_end": 1.5},
                {"event_start": 4.0, "event_end": 4.5},
                {"event_start": 8.0, "event_end": 8.5},
            )
        ]
        self.assertEqual(
            cores,
            [
                {"start": 11.0, "end": 11.5},
                {"start": 14.0, "end": 14.5},
                {"start": 18.0, "end": 18.5},
            ],
        )

    def test_pipeline_materializes_three_events_from_one_wide_candidate(self):
        detection = {
            "candidates": [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "score": 0.9,
                    "top_label": "Breaking",
                }
            ]
        }
        def confirmation_for_segment(_clip, _event, candidate, _api_key, **_kwargs):
            local_events = (
                ((1.0, 1.5), (4.0, 4.5))
                if candidate["segment_index"] == 1
                else ((2.0, 2.5), (6.0, 6.5))
            )
            events = [
                {
                    "event_start": start,
                    "event_end": end,
                    "confidence": 0.9,
                    "visual_evidence": "破碎",
                    "audio_evidence": "",
                    "reason": "新事件",
                }
                for start, end in local_events
            ]
            return {
                "confirmed": True,
                "confidence": 0.95,
                "occurrence_count": len(events),
                "events": events,
                "attempts": 1,
            }

        def materialize(_source, output, _event, number, cluster, **_kwargs):
            core = cluster["representative"]["core"]
            return {
                "status": "completed",
                "event_bounds": {**core, "source": "model"},
                "clip": {
                    "path": str(output / f"event-{number}.mp4"),
                    "start": core["start"],
                    "end": core["end"],
                },
            }

        with tempfile.TemporaryDirectory(prefix="pipeline-multi-event-") as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            with (
                patch(
                    "pipeline.probe_media",
                    return_value={
                        "duration": 30.0,
                        "size": 6,
                        "has_video": True,
                        "has_audio": True,
                        "streams": [],
                    },
                ),
                patch("pipeline.locate_candidates", return_value=detection),
                patch("pipeline.make_candidate_clip", return_value=root / "candidate.mp4"),
                patch(
                    "pipeline.confirm_candidate",
                    side_effect=confirmation_for_segment,
                ),
                patch("pipeline.materialize_event_result", side_effect=materialize),
            ):
                report, _ = run_pipeline(
                    source,
                    "玻璃破碎",
                    root / "outputs",
                    threshold=0.1,
                    max_candidates=3,
                    min_confidence=0.55,
                    retry_padding=1.0,
                    event_padding=1.0,
                    merge_confirmed_gap=0.5,
                    visual_fallback=False,
                    visual_scan_interval=2.0,
                    visual_padding=2.0,
                    visual_scan_attempts=2,
                    require_visual_scan_success=False,
                    final_review=False,
                    model_path=root / "model",
                    api_key="test-key",
                )

        self.assertEqual(report["deduplication"]["cluster_count"], 3)
        self.assertEqual(len(report["confirmation_candidates"]), 2)
        self.assertEqual(len(report["results"]), 3)
        self.assertEqual(len(report["outputs"]), 3)

    def test_cross_candidate_duplicate_uses_tight_representative_without_union(self):
        tight = self.proposal("tight", "candidate-a", 5.2, 5.8, 0.8)
        broad = self.proposal("broad", "candidate-b", 5.0, 6.0, 0.95)
        clusters, decisions = deduplicate_core_events([broad, tight])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["representative"]["core"], tight["core"])
        self.assertEqual(decisions[0]["status"], "duplicate_suppressed")

    def test_two_close_events_from_same_candidate_are_never_deduplicated(self):
        clusters, _ = deduplicate_core_events(
            [
                self.proposal("first", "same-candidate", 1.0, 2.0),
                self.proposal("second", "same-candidate", 1.2, 2.2),
            ]
        )
        self.assertEqual(len(clusters), 2)

    def test_complete_link_prevents_chain_merge(self):
        clusters, _ = deduplicate_core_events(
            [
                self.proposal("a", "candidate-a", 0.0, 2.0),
                self.proposal("b", "candidate-b", 0.6, 2.6),
                self.proposal("c", "candidate-c", 1.2, 3.2),
            ]
        )
        self.assertEqual(len(clusters), 2)
        self.assertEqual(
            sorted(len(cluster["members"]) for cluster in clusters),
            [1, 2],
        )

    def test_broad_bridge_cannot_join_two_tight_events(self):
        clusters, decisions = deduplicate_core_events(
            [
                self.proposal("left", "candidate-left", 0.0, 1.0),
                self.proposal("right", "candidate-right", 1.1, 2.1),
                self.proposal("bridge", "candidate-bridge", 0.1, 2.0),
            ]
        )
        self.assertEqual(len(clusters), 2)
        self.assertEqual(
            [item["status"] for item in decisions],
            ["ambiguous_bridge_suppressed"],
        )

    def test_core_dedupe_is_stable_when_proposals_are_reordered(self):
        proposals = [
            self.proposal("a", "candidate-a", 0.0, 2.0),
            self.proposal("b", "candidate-b", 0.6, 2.6),
            self.proposal("c", "candidate-c", 1.2, 3.2),
        ]
        forward, _ = deduplicate_core_events(proposals)
        reverse, _ = deduplicate_core_events(list(reversed(proposals)))
        self.assertEqual(
            [item["representative"]["stable_id"] for item in forward],
            [item["representative"]["stable_id"] for item in reverse],
        )

    def test_padding_is_capped_at_neighbor_midpoints(self):
        start, end, limits = padded_bounds_for_core(
            {"start": 3.0, "end": 4.0},
            previous_core={"start": 1.0, "end": 2.0},
            next_core={"start": 5.0, "end": 6.0},
            padding=3.0,
            source_duration=10.0,
        )
        self.assertEqual((start, end), (2.5, 4.5))
        self.assertEqual((limits["min_start"], limits["max_end"]), (2.5, 4.5))

    def test_overlapping_neighbor_cores_remove_padding_on_that_side(self):
        start, end, limits = padded_bounds_for_core(
            {"start": 3.0, "end": 4.0},
            previous_core={"start": 2.0, "end": 3.5},
            next_core={"start": 3.5, "end": 5.0},
            padding=2.0,
            source_duration=10.0,
        )
        self.assertEqual((start, end), (3.0, 4.0))
        self.assertEqual(
            limits["core_overlap_conflicts"],
            ["previous_core_overlap", "next_core_overlap"],
        )

    def test_retry_padding_stays_within_neighbor_limits_and_review_focuses_core(self):
        proposal = {
            "stable_id": "proposal",
            "candidate_id": "candidate",
            "candidate_index": 1,
            "occurrence_index": 1,
            "candidate": {"origin": "visual_fallback"},
            "candidate_confirmation": {"attempts": 1, "occurrence_count": 1},
            "event": {
                "event_start": 1.0,
                "event_end": 2.0,
                "confidence": 0.9,
                "visual_evidence": "破碎",
                "audio_evidence": "",
                "reason": "新事件",
            },
            "core": {"start": 3.0, "end": 4.0},
            "confidence": 0.9,
        }
        cluster = {"representative": proposal, "members": [proposal]}

        def write_clip(_source, destination, _start, _end):
            destination.write_bytes(b"clip")

        reviews = [
            {
                "event_present": True,
                "complete": False,
                "confidence": 0.9,
                "needs_more_before": True,
                "needs_more_after": True,
            },
            {
                "event_present": True,
                "complete": True,
                "confidence": 0.9,
                "needs_more_before": False,
                "needs_more_after": False,
            },
        ]
        with tempfile.TemporaryDirectory(prefix="materialize-test-") as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            with (
                patch("pipeline.cut_clip", side_effect=write_clip),
                patch("pipeline.technical_review", return_value={"passed": True}),
                patch("pipeline.review_clip", side_effect=reviews) as review,
            ):
                result = materialize_event_result(
                    source,
                    root,
                    "玻璃破碎",
                    1,
                    cluster,
                    previous_core={"start": 1.0, "end": 2.0},
                    next_core={"start": 5.0, "end": 6.0},
                    source_duration=10.0,
                    event_padding=0.1,
                    retry_padding=10.0,
                    min_confidence=0.55,
                    final_review=True,
                    api_key="test-key",
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual((result["clip"]["start"], result["clip"]["end"]), (2.5, 4.5))
        self.assertEqual(review.call_count, 2)
        self.assertAlmostEqual(review.call_args_list[0].kwargs["focus_start"], 0.1)
        self.assertAlmostEqual(review.call_args_list[1].kwargs["focus_start"], 0.5)

    def test_groups_only_adjacent_cross_modal_fragments(self):
        visual = {
            "status": "completed",
            "candidate": {"origin": "visual_fallback"},
            "clip": {"start": 9.0, "end": 15.0},
        }
        audio = {
            "status": "completed",
            "candidate": {"origin": "audio"},
            "clip": {"start": 15.3, "end": 18.9},
        }
        groups = group_adjacent_confirmed_results(
            [visual, audio],
            max_gap=0.5,
        )
        self.assertEqual(groups, [[visual, audio]])

        overlapping = {
            "status": "completed",
            "candidate": {"origin": "audio"},
            "clip": {"start": 14.0, "end": 17.0},
        }
        groups = group_adjacent_confirmed_results(
            [visual, overlapping],
            max_gap=0.5,
        )
        self.assertEqual(groups, [[visual], [overlapping]])

    def test_does_not_group_adjacent_fragments_from_same_modality(self):
        first = {
            "status": "completed",
            "candidate": {"origin": "visual_fallback"},
            "clip": {"start": 1.0, "end": 3.0},
        }
        second = {
            "status": "completed",
            "candidate": {"origin": "visual_fallback"},
            "clip": {"start": 3.2, "end": 5.0},
        }
        self.assertEqual(
            group_adjacent_confirmed_results([first, second], max_gap=0.5),
            [[first], [second]],
        )

    def test_does_not_chain_three_adjacent_fragments(self):
        results = [
            {
                "status": "completed",
                "candidate": {"origin": origin},
                "clip": {"start": start, "end": end},
            }
            for origin, start, end in (
                ("visual_fallback", 1.0, 2.0),
                ("audio", 2.2, 3.0),
                ("visual_fallback", 3.2, 4.0),
            )
        ]
        groups = group_adjacent_confirmed_results(results, max_gap=0.5)
        self.assertEqual([len(group) for group in groups], [2, 1])

    def test_does_not_group_unknown_origin_or_invalid_bounds(self):
        unknown = {
            "status": "completed",
            "candidate": {"origin": "unknown"},
            "clip": {"start": 1.0, "end": 2.0},
        }
        audio = {
            "status": "completed",
            "candidate": {"origin": "audio"},
            "clip": {"start": 2.2, "end": 3.0},
        }
        invalid = {
            "status": "completed",
            "candidate": {"origin": "visual_fallback"},
            "clip": {"start": None, "end": 4.0},
        }
        groups = group_adjacent_confirmed_results(
            [unknown, audio, invalid], max_gap=0.5
        )
        self.assertEqual([len(group) for group in groups], [1, 1, 1])


class ConsolidationTests(unittest.TestCase):
    @staticmethod
    def adjacent_results(first_path: Path, second_path: Path) -> list[dict]:
        return [
            {
                "status": "completed",
                "candidate": {"origin": "visual_fallback"},
                "clip": {"path": str(first_path), "start": 1.0, "end": 3.0},
            },
            {
                "status": "completed",
                "candidate": {"origin": "audio"},
                "clip": {"path": str(second_path), "start": 3.2, "end": 5.0},
            },
        ]

    def test_consolidates_adjacent_cross_modal_clips(self):
        with tempfile.TemporaryDirectory(prefix="consolidate-test-") as temp_dir:
            temp = Path(temp_dir)
            first_path = temp / "first.mp4"
            second_path = temp / "second.mp4"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            results = [
                {
                    "status": "completed",
                    "candidate": {"origin": "visual_fallback"},
                    "clip": {"path": str(first_path), "start": 1.0, "end": 3.0},
                },
                {
                    "status": "completed",
                    "candidate": {"origin": "audio"},
                    "clip": {"path": str(second_path), "start": 3.2, "end": 5.0},
                },
            ]

            def write_merged(_source, destination, _start, _end):
                destination.write_bytes(b"merged")

            with (
                patch("pipeline.cut_clip", side_effect=write_merged),
                patch(
                    "pipeline.technical_review",
                    return_value={"passed": True},
                ),
                patch(
                    "pipeline.review_consolidation",
                    return_value={
                        "same_occurrence": True,
                        "occurrence_count": 1,
                        "confidence": 0.9,
                        "reason": "同一次事件",
                    },
                ),
            ):
                merged = consolidate_completed_results(
                    Path("source.mp4"),
                    results,
                    max_gap=0.5,
                    event="玻璃破碎",
                    min_confidence=0.55,
                    api_key="test-key",
                )

            self.assertEqual(first_path.read_bytes(), b"merged")
            self.assertFalse(second_path.exists())

        self.assertEqual(len(merged), 1)
        self.assertEqual(results[0]["clip"]["start"], 1.0)
        self.assertEqual(results[0]["clip"]["end"], 5.0)
        self.assertEqual(results[0]["consolidation"]["component_count"], 2)
        self.assertEqual(results[1]["status"], "consolidated")

    def test_keeps_independent_adjacent_occurrences_separate(self):
        with tempfile.TemporaryDirectory(prefix="consolidate-test-") as temp_dir:
            temp = Path(temp_dir)
            paths = [temp / "first.mp4", temp / "second.mp4"]
            for path in paths:
                path.write_bytes(b"clip")
            results = [
                {
                    "status": "completed",
                    "candidate": {"origin": origin},
                    "clip": {"path": str(path), "start": start, "end": end},
                }
                for origin, path, start, end in (
                    ("visual_fallback", paths[0], 1.0, 3.0),
                    ("audio", paths[1], 3.2, 5.0),
                )
            ]

            def write_merged(_source, destination, _start, _end):
                destination.write_bytes(b"merged")

            with (
                patch("pipeline.cut_clip", side_effect=write_merged),
                patch("pipeline.technical_review", return_value={"passed": True}),
                patch(
                    "pipeline.review_consolidation",
                    return_value={
                        "same_occurrence": False,
                        "occurrence_count": 2,
                        "confidence": 0.95,
                        "reason": "两次独立事件",
                    },
                ),
            ):
                decisions = consolidate_completed_results(
                    Path("source.mp4"),
                    results,
                    max_gap=0.5,
                    event="玻璃破碎",
                    min_confidence=0.55,
                    api_key="test-key",
                )

        self.assertEqual(decisions[0]["status"], "kept_separate")
        self.assertEqual([item["status"] for item in results], ["completed", "completed"])

    def test_consolidation_error_keeps_original_clips(self):
        with tempfile.TemporaryDirectory(prefix="consolidate-test-") as temp_dir:
            temp = Path(temp_dir)
            first_path = temp / "first.mp4"
            second_path = temp / "second.mp4"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            results = self.adjacent_results(first_path, second_path)

            def write_merged(_source, destination, _start, _end):
                destination.write_bytes(b"merged")

            with (
                patch("pipeline.cut_clip", side_effect=write_merged),
                patch("pipeline.technical_review", return_value={"passed": True}),
                patch(
                    "pipeline.review_consolidation",
                    side_effect=ValueError("invalid model JSON"),
                ),
            ):
                decisions = consolidate_completed_results(
                    Path("source.mp4"),
                    results,
                    max_gap=0.5,
                    event="玻璃破碎",
                    min_confidence=0.55,
                    api_key="test-key",
                )

            self.assertEqual(first_path.read_bytes(), b"first")
            self.assertEqual(second_path.read_bytes(), b"second")
            self.assertFalse((temp / "first_consolidating.mp4").exists())

        self.assertEqual(decisions[0]["status"], "kept_separate")
        self.assertEqual(decisions[0]["reason"], "consolidation_error")
        self.assertEqual([item["status"] for item in results], ["completed", "completed"])


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
