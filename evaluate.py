#!/usr/bin/env python3
"""Batch evaluation for audio candidates or the full evidence pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

from audio_candidates import DEFAULT_MODEL_PATH, locate_candidates
from pipeline import (
    DEFAULT_VISUAL_PADDING,
    DEFAULT_VISUAL_SCAN_INTERVAL,
    run_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate event detection on a manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("audio", "full"), default="audio")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--retry-padding", type=float, default=1.0)
    parser.add_argument(
        "--visual-scan-interval", type=float, default=DEFAULT_VISUAL_SCAN_INTERVAL
    )
    parser.add_argument("--visual-padding", type=float, default=DEFAULT_VISUAL_PADDING)
    parser.add_argument("--no-visual-fallback", action="store_true")
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


def match_intervals(truth: list[dict], predictions: list[dict]) -> list[dict]:
    available = set(range(len(predictions)))
    matches = []
    for annotation in truth:
        if not available:
            matches.append({"truth": annotation, "prediction": None, "iou": 0.0})
            continue
        best_index = max(available, key=lambda index: interval_iou(annotation, predictions[index]))
        best_iou = interval_iou(annotation, predictions[best_index])
        if best_iou <= 0:
            matches.append({"truth": annotation, "prediction": None, "iou": 0.0})
            continue
        prediction = predictions[best_index]
        available.remove(best_index)
        matches.append(
            {
                "truth": annotation,
                "prediction": prediction,
                "iou": best_iou,
                "start_error": abs(prediction["start"] - annotation["start"]),
                "end_error": abs(prediction["end"] - annotation["end"]),
                "complete_coverage": (
                    prediction["start"] <= annotation["start"]
                    and prediction["end"] >= annotation["end"]
                ),
            }
        )
    return matches


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
    for item in successful:
        prediction_count += len(item["predictions"])
        all_matches.extend(match_intervals(item["events"], item["predictions"]))
    matched = [item for item in all_matches if item["prediction"] is not None]
    complete = [item for item in matched if item["complete_coverage"]]
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
            "ground_truth_events": len(all_matches),
            "predicted_events": prediction_count,
            "matched_events": len(matched),
            "event_recall": safe_divide(len(matched), len(all_matches)),
            "mean_iou": mean_or_none([item["iou"] for item in matched]),
            "mean_start_error_seconds": mean_or_none(
                [item["start_error"] for item in matched]
            ),
            "mean_end_error_seconds": mean_or_none(
                [item["end_error"] for item in matched]
            ),
            "complete_coverage_rate": safe_divide(len(complete), len(all_matches)),
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
        visual_fallback=not args.no_visual_fallback,
        visual_scan_interval=args.visual_scan_interval,
        visual_padding=args.visual_padding,
        final_review=not args.skip_final_review,
        model_path=args.model_path,
        api_key=os.environ.get("ARK_API_KEY", "").strip(),
    )
    predictions = [
        {
            "start": item["clip"]["start"],
            "end": item["clip"]["end"],
            "path": item["clip"]["path"],
        }
        for item in report["results"]
        if item["status"] == "completed"
    ]
    return predictions, {
        "pipeline_status": report["status"],
        "pipeline_report": str(report_path),
        "visual_fallback_used": report["visual_fallback"]["used"],
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
        manifest, samples = load_manifest(manifest_path)
        if args.split:
            samples = [item for item in samples if item.get("split") == args.split]
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

        output = args.output.resolve()
        settings = {
            "threshold": args.threshold,
            "max_candidates": args.max_candidates,
            "min_confidence": args.min_confidence,
            "retry_padding": args.retry_padding,
            "final_review": not args.skip_final_review,
            "visual_fallback": not args.no_visual_fallback,
            "visual_scan_interval": args.visual_scan_interval,
            "visual_padding": args.visual_padding,
        }
        results = []
        if args.resume and output.is_file():
            previous = json.loads(output.read_text(encoding="utf-8"))
            if previous.get("stage") != args.stage:
                raise RuntimeError("已有结果的 stage 与当前参数不同")
            if previous.get("manifest") != str(manifest_path):
                raise RuntimeError("已有结果来自不同的评测清单")
            if previous.get("settings") != settings:
                raise RuntimeError("已有结果的评测参数与当前参数不同")
            selected_ids = {item["id"] for item in samples}
            results = [
                item
                for item in previous.get("results", [])
                if item.get("id") in selected_ids
            ]
        completed_ids = {item["id"] for item in results}

        for index, sample in enumerate(samples, start=1):
            if sample["id"] in completed_ids:
                print(f"[{index}/{len(samples)}] 跳过 {sample['id']}", file=sys.stderr)
                continue
            print(f"[{index}/{len(samples)}] 评测 {sample['id']}", file=sys.stderr)
            results.append(evaluate_sample(sample, args))
            payload = {
                "version": 1,
                "manifest": str(manifest_path),
                "dataset": manifest.get("name"),
                "stage": args.stage,
                "settings": settings,
                "metrics": calculate_metrics(results),
                "results": results,
            }
            write_evaluation(output, payload)

        metrics = calculate_metrics(results)
        write_evaluation(
            output,
            {
                "version": 1,
                "manifest": str(manifest_path),
                "dataset": manifest.get("name"),
                "stage": args.stage,
                "settings": settings,
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
