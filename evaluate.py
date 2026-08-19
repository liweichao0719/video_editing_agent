#!/usr/bin/env python3
"""Batch evaluation for audio candidates or the full evidence pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from analyze_video import DEFAULT_BASE_URL, DEFAULT_MODEL
from audio_candidates import DEFAULT_MODEL_PATH, locate_candidates
from pipeline import (
    DEFAULT_EVENT_PADDING,
    DEFAULT_MERGE_CONFIRMED_GAP,
    DEFAULT_VISUAL_SCAN_ATTEMPTS,
    DEFAULT_VISUAL_PADDING,
    DEFAULT_VISUAL_SCAN_INTERVAL,
    PIPELINE_VERSION,
    run_pipeline,
)


EVALUATION_VERSION = 4
EVENT_MATCH_IOU = 0.10
STRICT_EVENT_MATCH_IOU = 0.50
SUCCESSFUL_PIPELINE_STATUSES = {"completed", "no_candidate", "no_confirmed_event"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: Path) -> dict:
    """Return a deterministic content fingerprint for a file or directory."""
    resolved = path.resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "kind": "missing",
            "sha256": None,
        }
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "sha256": sha256_file(resolved),
        }
    if not resolved.is_dir():
        return {
            "path": str(resolved),
            "kind": "unsupported",
            "sha256": None,
        }

    digest = hashlib.sha256()
    files = sorted(
        (item for item in resolved.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(resolved).as_posix(),
    )
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return {
        "path": str(resolved),
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def fingerprint_samples(samples: list[dict]) -> dict:
    fingerprints = {}
    path_cache = {}
    for sample in samples:
        path = sample["path"].resolve()
        cache_key = str(path)
        if cache_key not in path_cache:
            path_cache[cache_key] = (
                sha256_file(path) if path.is_file() else None
            )
        fingerprints[sample["id"]] = {
            "path": cache_key,
            "sha256": path_cache[cache_key],
        }
    return fingerprints


def build_evaluation_fingerprint(
    args: argparse.Namespace,
    samples: list[dict],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    settings: dict,
) -> dict:
    fingerprint = {
        "evaluation_version": EVALUATION_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "stage": args.stage,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "settings": settings,
        "local_model": fingerprint_path(args.model_path),
        "media": fingerprint_samples(samples),
    }
    if args.stage == "full":
        fingerprint["ark"] = {
            "model": os.environ.get("ARK_MODEL", DEFAULT_MODEL),
            "base_url": os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        }
    return fingerprint


def merge_resume_fingerprint(
    previous: dict,
    current: dict,
    *,
    previous_result_ids: set[str],
) -> dict:
    if not isinstance(previous, dict):
        raise RuntimeError("已有结果缺少运行指纹，不能安全续跑")
    previous_static = {key: value for key, value in previous.items() if key != "media"}
    current_static = {key: value for key, value in current.items() if key != "media"}
    if previous_static != current_static:
        changed = sorted(
            key
            for key in set(previous_static) | set(current_static)
            if (
                key not in previous_static
                or key not in current_static
                or previous_static[key] != current_static[key]
            )
        )
        raise RuntimeError(
            "已有结果的运行指纹与当前环境不同：" + "、".join(changed)
        )

    previous_media = previous.get("media")
    current_media = current.get("media")
    if not isinstance(previous_media, dict) or not isinstance(current_media, dict):
        raise RuntimeError("已有结果缺少媒体内容指纹，不能安全续跑")
    missing_media = sorted(previous_result_ids - set(previous_media))
    if missing_media:
        raise RuntimeError(
            "已有样本缺少媒体内容指纹：" + "、".join(missing_media)
        )
    for sample_id, media_fingerprint in current_media.items():
        if (
            sample_id in previous_media
            and previous_media[sample_id] != media_fingerprint
        ):
            raise RuntimeError(f"已有样本 {sample_id} 的媒体内容已变化")

    merged = {**current_static}
    merged["media"] = {**previous_media, **current_media}
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate event detection on a manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("audio", "full"), default="audio")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="只评测指定样本；可重复使用",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=3,
        help="每种候选来源最多保留的时间段数量",
    )
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--retry-padding", type=float, default=1.0)
    parser.add_argument("--event-padding", type=float, default=DEFAULT_EVENT_PADDING)
    parser.add_argument(
        "--merge-confirmed-gap",
        type=float,
        default=DEFAULT_MERGE_CONFIRMED_GAP,
    )
    parser.add_argument(
        "--visual-scan-interval", type=float, default=DEFAULT_VISUAL_SCAN_INTERVAL
    )
    parser.add_argument("--visual-padding", type=float, default=DEFAULT_VISUAL_PADDING)
    parser.add_argument(
        "--visual-scan-attempts",
        type=int,
        default=DEFAULT_VISUAL_SCAN_ATTEMPTS,
    )
    parser.add_argument(
        "--no-visual-scan",
        "--no-visual-fallback",
        dest="no_visual_fallback",
        action="store_true",
    )
    parser.add_argument("--skip-final-review", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise RuntimeError("评测清单 version 必须为 1")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("评测清单必须包含非空 samples 数组")
    default_event = data.get("default_event")
    root = (path.parent / data.get("root", ".")).resolve()
    normalized = []
    seen_ids = set()
    for raw in samples:
        sample_id = str(raw.get("id", "")).strip()
        if not sample_id or sample_id in seen_ids:
            raise RuntimeError(f"样本 id 缺失或重复：{sample_id!r}")
        seen_ids.add(sample_id)
        event = str(raw.get("event") or default_event or "").strip()
        if not event:
            raise RuntimeError(f"样本 {sample_id} 缺少 event")
        if not isinstance(raw.get("expected"), bool):
            raise RuntimeError(f"样本 {sample_id} 的 expected 必须是布尔值")
        annotations = raw.get("events", [])
        if not isinstance(annotations, list):
            raise RuntimeError(f"样本 {sample_id} 的 events 必须是数组")
        events = []
        for annotation in annotations:
            start = float(annotation["start"])
            end = float(annotation["end"])
            if start < 0 or end <= start:
                raise RuntimeError(f"样本 {sample_id} 包含无效时间段")
            events.append({"start": start, "end": end})
        if raw["expected"] and not events:
            raise RuntimeError(f"正样本 {sample_id} 必须标注至少一个事件时间段")
        normalized.append(
            {
                **raw,
                "id": sample_id,
                "event": event,
                "path": (root / raw["path"]).resolve(),
                "events": events,
            }
        )
    return data, normalized


def interval_iou(left: dict, right: dict) -> float:
    overlap = max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
    union = max(left["end"], right["end"]) - min(left["start"], right["start"])
    return overlap / union if union > 0 else 0.0


def match_intervals(
    truth: list[dict],
    predictions: list[dict],
    *,
    min_iou: float = EVENT_MATCH_IOU,
) -> list[dict]:
    truth_order = sorted(
        range(len(truth)),
        key=lambda index: (
            truth[index]["start"],
            truth[index]["end"],
            json.dumps(truth[index], ensure_ascii=False, sort_keys=True),
        ),
    )
    prediction_order = sorted(
        range(len(predictions)),
        key=lambda index: (
            predictions[index]["start"],
            predictions[index]["end"],
            json.dumps(predictions[index], ensure_ascii=False, sort_keys=True),
        ),
    )
    ordered_truth = [truth[index] for index in truth_order]
    ordered_predictions = [predictions[index] for index in prediction_order]

    truth_count = len(ordered_truth)
    prediction_count = len(ordered_predictions)
    source = 0
    truth_offset = 1
    prediction_offset = truth_offset + truth_count
    sink = prediction_offset + prediction_count
    graph: list[list[dict]] = [[] for _ in range(sink + 1)]

    def add_edge(left: int, right: int, cost: int) -> dict:
        forward = {
            "to": right,
            "reverse": len(graph[right]),
            "capacity": 1,
            "cost": cost,
        }
        backward = {
            "to": left,
            "reverse": len(graph[left]),
            "capacity": 0,
            "cost": -cost,
        }
        graph[left].append(forward)
        graph[right].append(backward)
        return forward

    for truth_index in range(truth_count):
        add_edge(source, truth_offset + truth_index, 0)
    for prediction_index in range(prediction_count):
        add_edge(prediction_offset + prediction_index, sink, 0)

    match_edges: list[list[tuple[int, dict]]] = [[] for _ in range(truth_count)]
    for truth_index, annotation in enumerate(ordered_truth):
        for prediction_index, prediction in enumerate(ordered_predictions):
            iou = interval_iou(annotation, prediction)
            if iou < min_iou:
                continue
            edge = add_edge(
                truth_offset + truth_index,
                prediction_offset + prediction_index,
                -round(iou * 1_000_000_000),
            )
            match_edges[truth_index].append((prediction_index, edge))

    node_count = len(graph)
    while True:
        distances: list[int | None] = [None] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0
        for _ in range(node_count - 1):
            changed = False
            for left, edges in enumerate(graph):
                if distances[left] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if not edge["capacity"]:
                        continue
                    candidate_distance = distances[left] + edge["cost"]
                    current_distance = distances[edge["to"]]
                    if current_distance is None or candidate_distance < current_distance:
                        distances[edge["to"]] = candidate_distance
                        previous[edge["to"]] = (left, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            left, edge_index = previous[node]
            edge = graph[left][edge_index]
            edge["capacity"] = 0
            graph[node][edge["reverse"]]["capacity"] = 1
            node = left

    assignment = [-1] * truth_count
    for truth_index, edges in enumerate(match_edges):
        for prediction_index, edge in edges:
            if edge["capacity"] == 0:
                assignment[truth_index] = prediction_index
                break
    matches_by_truth_index = {}
    for ordered_truth_index, prediction_index in enumerate(assignment):
        original_truth_index = truth_order[ordered_truth_index]
        annotation = truth[original_truth_index]
        if prediction_index < 0:
            matches_by_truth_index[original_truth_index] = {
                "truth": annotation,
                "prediction": None,
                "iou": 0.0,
            }
            continue
        prediction = ordered_predictions[prediction_index]
        best_iou = interval_iou(annotation, prediction)
        coverage_start = prediction.get("clip_start", prediction["start"])
        coverage_end = prediction.get("clip_end", prediction["end"])
        matches_by_truth_index[original_truth_index] = {
            "truth": annotation,
            "prediction": prediction,
            "iou": best_iou,
            "start_error": abs(prediction["start"] - annotation["start"]),
            "end_error": abs(prediction["end"] - annotation["end"]),
            "complete_coverage": (
                coverage_start <= annotation["start"]
                and coverage_end >= annotation["end"]
            ),
        }
    return [matches_by_truth_index[index] for index in range(len(truth))]


def count_complete_clip_coverage_matches(
    truth: list[dict], predictions: list[dict]
) -> int:
    """Return the maximum one-to-one count of GT events fully covered by clips."""
    ordered_truth = sorted(
        truth,
        key=lambda item: (
            item["start"],
            item["end"],
            json.dumps(item, ensure_ascii=False, sort_keys=True),
        ),
    )
    ordered_predictions = sorted(
        predictions,
        key=lambda item: (
            item.get("clip_start", item["start"]),
            item.get("clip_end", item["end"]),
            json.dumps(item, ensure_ascii=False, sort_keys=True),
        ),
    )
    eligible_predictions = []
    for annotation in ordered_truth:
        eligible_predictions.append(
            [
                prediction_index
                for prediction_index, prediction in enumerate(ordered_predictions)
                if (
                    prediction.get("clip_start", prediction["start"])
                    <= annotation["start"]
                    and prediction.get("clip_end", prediction["end"])
                    >= annotation["end"]
                )
            ]
        )

    matched_truth_by_prediction: list[int | None] = [None] * len(
        ordered_predictions
    )

    def assign(truth_index: int, visited_predictions: set[int]) -> bool:
        for prediction_index in eligible_predictions[truth_index]:
            if prediction_index in visited_predictions:
                continue
            visited_predictions.add(prediction_index)
            previous_truth = matched_truth_by_prediction[prediction_index]
            if previous_truth is None or assign(previous_truth, visited_predictions):
                matched_truth_by_prediction[prediction_index] = truth_index
                return True
        return False

    return sum(
        assign(truth_index, set()) for truth_index in range(len(ordered_truth))
    )


def safe_divide(numerator: int | float, denominator: int | float):
    return round(numerator / denominator, 4) if denominator else None


def mean_or_none(values: list[float]):
    return round(statistics.fmean(values), 4) if values else None


def calculate_metrics(results: list[dict]) -> dict:
    successful = [item for item in results if item["status"] == "ok"]
    tp = sum(item["expected"] and item["predicted"] for item in successful)
    tn = sum(not item["expected"] and not item["predicted"] for item in successful)
    fp = sum(not item["expected"] and item["predicted"] for item in successful)
    fn = sum(item["expected"] and not item["predicted"] for item in successful)
    all_matches = []
    prediction_count = 0
    complete_coverage_matches = 0
    for item in successful:
        prediction_count += len(item["predictions"])
        complete_coverage_matches += count_complete_clip_coverage_matches(
            item["events"], item["predictions"]
        )
        all_matches.extend(
            match_intervals(
                item["events"],
                item["predictions"],
                min_iou=EVENT_MATCH_IOU,
            )
        )
    matched = [item for item in all_matches if item["prediction"] is not None]
    complete = [item for item in matched if item["complete_coverage"]]
    unmatched_predictions = prediction_count - len(matched)
    strict_matches = []
    for item in successful:
        strict_matches.extend(
            match_intervals(
                item["events"],
                item["predictions"],
                min_iou=STRICT_EVENT_MATCH_IOU,
            )
        )
    strict_matched = [
        item for item in strict_matches if item["prediction"] is not None
    ]
    unmatched_truth = len(all_matches) - len(matched)
    strict_unmatched_truth = len(strict_matches) - len(strict_matched)
    complete_coverage_recall = safe_divide(
        complete_coverage_matches, len(all_matches)
    )
    return {
        "samples": {
            "total": len(results),
            "evaluated": len(successful),
            "errors": len(results) - len(successful),
        },
        "classification": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "precision": safe_divide(tp, tp + fp),
            "recall": safe_divide(tp, tp + fn),
            "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
            "accuracy": safe_divide(tp + tn, len(successful)),
        },
        "temporal": {
            "localization_bounds": "event_core_for_full_stage",
            "coverage_bounds": "output_clip_when_available",
            "match_iou_threshold": EVENT_MATCH_IOU,
            "ground_truth_events": len(all_matches),
            "predicted_events": prediction_count,
            "matched_events": len(matched),
            "true_positive": len(matched),
            "false_positive": unmatched_predictions,
            "false_negative": unmatched_truth,
            "unmatched_predictions": unmatched_predictions,
            "unmatched_ground_truth_events": unmatched_truth,
            "event_precision": safe_divide(len(matched), prediction_count),
            "event_recall": safe_divide(len(matched), len(all_matches)),
            "event_f1": safe_divide(
                2 * len(matched), prediction_count + len(all_matches)
            ),
            "mean_iou": mean_or_none([item["iou"] for item in matched]),
            "mean_start_error_seconds": mean_or_none(
                [item["start_error"] for item in matched]
            ),
            "mean_end_error_seconds": mean_or_none(
                [item["end_error"] for item in matched]
            ),
            "complete_coverage_recall": complete_coverage_recall,
            "complete_coverage_rate": complete_coverage_recall,
            "complete_coverage_rate_on_matched": safe_divide(
                len(complete), len(matched)
            ),
            "strict_at_iou_0_5": {
                "match_iou_threshold": STRICT_EVENT_MATCH_IOU,
                "matched_events": len(strict_matched),
                "true_positive": len(strict_matched),
                "false_positive": prediction_count - len(strict_matched),
                "false_negative": strict_unmatched_truth,
                "unmatched_predictions": prediction_count - len(strict_matched),
                "unmatched_ground_truth_events": strict_unmatched_truth,
                "event_precision": safe_divide(
                    len(strict_matched), prediction_count
                ),
                "event_recall": safe_divide(
                    len(strict_matched), len(strict_matches)
                ),
                "event_f1": safe_divide(
                    2 * len(strict_matched), prediction_count + len(strict_matches)
                ),
            },
        },
    }


def write_evaluation(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def audio_predictions(sample: dict, args: argparse.Namespace) -> tuple[list[dict], dict]:
    report = locate_candidates(
        sample["path"],
        sample["event"],
        model_path=args.model_path.resolve(),
        threshold=args.threshold,
    )
    predictions = [
        {
            "start": item["start"],
            "end": item["end"],
            "score": item["score"],
            "label": item["top_label"],
        }
        for item in report["candidates"]
    ]
    details = {
        "candidate_count": len(predictions),
        "max_scores": report["detector"]["max_scores"],
    }
    return predictions, details


def require_valid_full_interval(
    value: object, *, name: str, report_path: Path
) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise RuntimeError(
            f"完整流程 completed 结果缺少合法 {name} 区间；报告：{report_path}"
        )

    def finite_float(candidate: object) -> float | None:
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            return None
        try:
            normalized = float(candidate)
        except (OverflowError, ValueError):
            return None
        return normalized if math.isfinite(normalized) else None

    start = finite_float(value.get("start"))
    end = finite_float(value.get("end"))
    if (
        start is None
        or end is None
        or start < 0
        or end <= start
    ):
        raise RuntimeError(
            f"完整流程 completed 结果的 {name} 不是有限合法区间；报告：{report_path}"
        )
    return start, end


def full_predictions(sample: dict, args: argparse.Namespace) -> tuple[list[dict], dict]:
    sample_output = args.output.resolve().parent / "evaluation_clips" / sample["id"]
    report, report_path = run_pipeline(
        sample["path"],
        sample["event"],
        sample_output,
        threshold=args.threshold,
        max_candidates=args.max_candidates,
        min_confidence=args.min_confidence,
        retry_padding=args.retry_padding,
        event_padding=args.event_padding,
        merge_confirmed_gap=args.merge_confirmed_gap,
        visual_fallback=not args.no_visual_fallback,
        visual_scan_interval=args.visual_scan_interval,
        visual_padding=args.visual_padding,
        visual_scan_attempts=args.visual_scan_attempts,
        require_visual_scan_success=not args.no_visual_fallback,
        final_review=not args.skip_final_review,
        model_path=args.model_path,
        api_key=os.environ.get("ARK_API_KEY", "").strip(),
    )
    pipeline_status = report.get("status")
    if pipeline_status not in SUCCESSFUL_PIPELINE_STATUSES:
        raise RuntimeError(
            f"完整流程返回非成功状态 {pipeline_status!r}，部分结果不计分；"
            f"报告：{report_path}"
        )
    visual_error = report["visual_fallback"].get("error")
    if not args.no_visual_fallback and visual_error:
        raise RuntimeError(
            f"补充视觉扫描失败，完整流程结果不计分：{visual_error}；"
            f"报告：{report_path}"
        )
    processing_errors = [
        item for item in report["results"] if item.get("status") == "error"
    ]
    if processing_errors:
        raise RuntimeError(
            f"完整流程存在候选确认或出片错误，部分结果不计分；报告：{report_path}"
        )
    predictions = []
    for item in report["results"]:
        if item["status"] != "completed":
            continue
        event_start, event_end = require_valid_full_interval(
            item.get("event_bounds"), name="event_bounds", report_path=report_path
        )
        clip = item.get("clip")
        clip_start, clip_end = require_valid_full_interval(
            clip, name="clip", report_path=report_path
        )
        predictions.append(
            {
                "start": event_start,
                "end": event_end,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "path": clip["path"],
            }
        )
    return predictions, {
        "pipeline_status": report["status"],
        "pipeline_report": str(report_path),
        "visual_fallback_used": report["visual_fallback"]["used"],
        "visual_scan_used": report["visual_fallback"]["used"],
    }


def evaluate_sample(sample: dict, args: argparse.Namespace) -> dict:
    started = time.monotonic()
    base = {
        "id": sample["id"],
        "path": str(sample["path"]),
        "event": sample["event"],
        "split": sample.get("split"),
        "kind": sample.get("kind"),
        "expected": sample["expected"],
        "events": sample["events"],
    }
    try:
        if not sample["path"].is_file():
            raise RuntimeError(f"文件不存在：{sample['path']}")
        if args.stage == "audio":
            predictions, details = audio_predictions(sample, args)
        else:
            predictions, details = full_predictions(sample, args)
        return {
            **base,
            "status": "ok",
            "predicted": bool(predictions),
            "predictions": predictions,
            "details": details,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "predicted": False,
            "predictions": [],
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def main() -> int:
    args = parse_args()
    try:
        manifest_path = args.manifest.resolve()
        manifest, manifest_samples = load_manifest(manifest_path)
        samples = manifest_samples
        if args.split:
            samples = [item for item in samples if item.get("split") == args.split]
        if args.sample_id:
            selected_ids = set(args.sample_id)
            samples = [item for item in samples if item["id"] in selected_ids]
        if args.limit is not None:
            if args.limit < 1:
                raise RuntimeError("limit 必须至少为 1")
            samples = samples[: args.limit]
        if not samples:
            raise RuntimeError("筛选后没有可评测样本")
        if args.visual_scan_interval <= 0:
            raise RuntimeError("visual-scan-interval 必须大于 0")
        if args.visual_padding < 0:
            raise RuntimeError("visual-padding 不能小于 0")
        if args.event_padding < 0:
            raise RuntimeError("event-padding 不能小于 0")
        if args.merge_confirmed_gap < 0:
            raise RuntimeError("merge-confirmed-gap 不能小于 0")
        if args.visual_scan_attempts < 1:
            raise RuntimeError("visual-scan-attempts 必须至少为 1")

        output = args.output.resolve()
        manifest_sha256 = sha256_file(manifest_path)
        settings = {
            "threshold": args.threshold,
            "max_candidates": args.max_candidates,
            "min_confidence": args.min_confidence,
            "retry_padding": args.retry_padding,
            "event_padding": args.event_padding,
            "merge_confirmed_gap": args.merge_confirmed_gap,
            "final_review": not args.skip_final_review,
            "visual_fallback": not args.no_visual_fallback,
            "visual_scan_mode": (
                "supplemental" if not args.no_visual_fallback else "disabled"
            ),
            "visual_scan_interval": args.visual_scan_interval,
            "visual_padding": args.visual_padding,
            "visual_scan_attempts": args.visual_scan_attempts,
        }
        selected_ids = {item["id"] for item in samples}
        previous = None
        previous_results = []
        previous_result_ids: set[str] = set()
        if args.resume and output.is_file():
            previous = json.loads(output.read_text(encoding="utf-8"))
            if previous.get("version") != EVALUATION_VERSION:
                raise RuntimeError("已有结果的评测器版本与当前版本不同")
            if previous.get("stage") != args.stage:
                raise RuntimeError("已有结果的 stage 与当前参数不同")
            if previous.get("manifest") != str(manifest_path):
                raise RuntimeError("已有结果来自不同的评测清单")
            if previous.get("manifest_sha256") != manifest_sha256:
                raise RuntimeError("已有结果对应的评测清单内容已变化")
            if previous.get("settings") != settings:
                raise RuntimeError("已有结果的评测参数与当前参数不同")
            previous_results = previous.get("results")
            if not isinstance(previous_results, list):
                raise RuntimeError("已有结果的 results 格式无效")
            manifest_ids = {item["id"] for item in manifest_samples}
            for item in previous_results:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    raise RuntimeError("已有结果包含无效样本")
                sample_id = item["id"]
                if sample_id in previous_result_ids:
                    raise RuntimeError(f"已有结果包含重复样本：{sample_id}")
                if sample_id not in manifest_ids:
                    raise RuntimeError(f"已有结果包含清单外样本：{sample_id}")
                previous_result_ids.add(sample_id)

        fingerprint_ids = selected_ids | previous_result_ids
        fingerprint_manifest_samples = [
            item for item in manifest_samples if item["id"] in fingerprint_ids
        ]
        current_fingerprint = build_evaluation_fingerprint(
            args,
            fingerprint_manifest_samples,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            settings=settings,
        )
        evaluation_fingerprint = current_fingerprint
        results = []
        if previous is not None:
            evaluation_fingerprint = merge_resume_fingerprint(
                previous.get("fingerprint"),
                current_fingerprint,
                previous_result_ids=previous_result_ids,
            )
            results = list(previous_results)

        completed_ids = {
            item["id"]
            for item in results
            if item.get("id") in selected_ids and item.get("status") == "ok"
        }
        result_positions = {item["id"]: index for index, item in enumerate(results)}

        for index, sample in enumerate(samples, start=1):
            if sample["id"] in completed_ids:
                print(f"[{index}/{len(samples)}] 跳过 {sample['id']}", file=sys.stderr)
                continue
            print(f"[{index}/{len(samples)}] 评测 {sample['id']}", file=sys.stderr)
            result = evaluate_sample(sample, args)
            if sample["id"] in result_positions:
                results[result_positions[sample["id"]]] = result
            else:
                result_positions[sample["id"]] = len(results)
                results.append(result)
            payload = {
                "version": EVALUATION_VERSION,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "dataset": manifest.get("name"),
                "stage": args.stage,
                "settings": settings,
                "fingerprint": evaluation_fingerprint,
                "metrics": calculate_metrics(results),
                "results": results,
            }
            write_evaluation(output, payload)

        metrics = calculate_metrics(results)
        write_evaluation(
            output,
            {
                "version": EVALUATION_VERSION,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "dataset": manifest.get("name"),
                "stage": args.stage,
                "settings": settings,
                "fingerprint": evaluation_fingerprint,
                "metrics": metrics,
                "results": results,
            },
        )
        print(json.dumps({"output": str(output), "metrics": metrics}, ensure_ascii=False))
        return 0 if metrics["samples"]["errors"] == 0 else 1
    except (RuntimeError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
