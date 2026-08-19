#!/usr/bin/env python3
"""Run the multimodal video evidence pipeline end to end."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

from analyze_video import DEFAULT_BASE_URL, DEFAULT_MODEL
from audio_candidates import DEFAULT_MODEL_PATH, locate_candidates


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_VISUAL_SCAN_INTERVAL = 2.0
DEFAULT_VISUAL_PADDING = 2.0
DEFAULT_VISUAL_SCAN_ATTEMPTS = 2
VISUAL_SCAN_PLAYBACK_FPS = 4
DEFAULT_EVENT_PADDING = 1.0
MIN_VISUAL_SCAN_CONFIDENCE = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate, confirm, clip, and verify a video event"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=3,
        help="每种候选来源最多保留的时间段数量",
    )
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--retry-padding", type=float, default=1.0)
    parser.add_argument(
        "--event-padding",
        type=float,
        default=DEFAULT_EVENT_PADDING,
        help="模型定位出的事件前后保留的上下文秒数",
    )
    parser.add_argument(
        "--visual-scan-interval",
        type=float,
        default=DEFAULT_VISUAL_SCAN_INTERVAL,
        help="补充视觉抽帧扫描的间隔秒数",
    )
    parser.add_argument(
        "--visual-padding",
        type=float,
        default=DEFAULT_VISUAL_PADDING,
        help="视觉候选前后扩展的秒数",
    )
    parser.add_argument(
        "--visual-scan-attempts",
        type=int,
        default=DEFAULT_VISUAL_SCAN_ATTEMPTS,
        help="视觉扫描请求的最大尝试次数",
    )
    parser.add_argument(
        "--no-visual-scan",
        "--no-visual-fallback",
        dest="no_visual_fallback",
        action="store_true",
        help="关闭补充视觉扫描，仅使用音频候选",
    )
    parser.add_argument("--skip-final-review", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def probe_media(path: Path) -> dict:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ]
    )
    raw = json.loads(result.stdout)
    streams = raw.get("streams", [])
    return {
        "duration": round(float(raw["format"]["duration"]), 3),
        "size": int(raw["format"].get("size", path.stat().st_size)),
        "has_video": any(item.get("codec_type") == "video" for item in streams),
        "has_audio": any(item.get("codec_type") == "audio" for item in streams),
        "streams": streams,
    }


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("模型没有返回 JSON")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise RuntimeError("模型 JSON 顶层必须是对象")
    return value


def model_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    raise RuntimeError("模型返回了无法识别的内容格式")


def call_model(video: Path, prompt: str, api_key: str) -> dict:
    if not api_key:
        raise RuntimeError("缺少环境变量 ARK_API_KEY")
    size = video.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise RuntimeError(f"候选片段超过 25MB：{video}")
    media_type = mimetypes.guess_type(video.name)[0] or "video/mp4"
    data_url = (
        f"data:{media_type};base64,"
        + base64.b64encode(video.read_bytes()).decode("ascii")
    )
    payload = {
        "model": os.environ.get("ARK_MODEL", DEFAULT_MODEL),
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"模型调用失败（HTTP {exc.code}）：{detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"模型请求失败：{exc}") from exc
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("模型响应缺少 choices/message/content") from exc
    return extract_json(model_text(content))


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "1", "是", "有"}
    return bool(value)


def as_confidence(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    return round(max(0.0, min(1.0, number)), 3)


def normalize_visual_candidates(
    raw: dict,
    duration: float,
    *,
    padding: float,
    max_candidates: int,
) -> list[dict]:
    values = raw.get("candidates", [])
    if not isinstance(values, list):
        return []
    candidates = []
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            start = float(value["start"])
            end = float(value["end"])
        except (KeyError, TypeError, ValueError):
            continue
        confidence = as_confidence(value.get("confidence"))
        if confidence < MIN_VISUAL_SCAN_CONFIDENCE:
            continue
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end <= start:
            continue
        candidates.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "score": confidence,
                "origin": "visual_fallback",
                "top_label": "visual_scan",
                "visual_evidence": str(value.get("visual_evidence", "")),
            }
        )

    merged = []
    for candidate in sorted(candidates, key=lambda item: item["start"]):
        if merged and candidate["start"] <= merged[-1]["end"]:
            current = merged[-1]
            current["end"] = max(current["end"], candidate["end"])
            if candidate["score"] > current["score"]:
                current["score"] = candidate["score"]
            evidence = candidate["visual_evidence"].strip()
            if evidence and evidence not in current["visual_evidence"]:
                current["visual_evidence"] = "；".join(
                    item
                    for item in (current["visual_evidence"].strip(), evidence)
                    if item
                )
        else:
            merged.append(candidate)
    selected = sorted(merged, key=lambda item: item["score"], reverse=True)[
        :max_candidates
    ]
    for candidate in selected:
        candidate["start"] = round(max(0.0, candidate["start"] - padding), 3)
        candidate["end"] = round(min(duration, candidate["end"] + padding), 3)
        candidate["time"] = f"{candidate['start']:.2f}-{candidate['end']:.2f}"
    return sorted(selected, key=lambda item: item["start"])


def candidate_iou(left: dict, right: dict) -> float:
    overlap = max(
        0.0,
        min(float(left["end"]), float(right["end"]))
        - max(float(left["start"]), float(right["start"])),
    )
    union = (
        max(float(left["end"]), float(right["end"]))
        - min(float(left["start"]), float(right["start"]))
    )
    return overlap / union if union > 0 else 0.0


def fuse_candidates(
    audio_candidates: list[dict],
    visual_candidates: list[dict],
    *,
    max_candidates: int,
) -> list[dict]:
    """Keep candidates from both modalities and merge clear cross-modal duplicates."""
    selected_audio = sorted(
        audio_candidates, key=lambda item: item["score"], reverse=True
    )[:max_candidates]
    selected_visual = sorted(
        visual_candidates, key=lambda item: item["score"], reverse=True
    )[:max_candidates]
    fused = [{**candidate} for candidate in selected_audio]

    for visual in selected_visual:
        matches = [
            (index, candidate_iou(candidate, visual))
            for index, candidate in enumerate(fused)
            if candidate.get("origin") in {"audio", "audio_visual"}
        ]
        best_index, best_overlap = max(matches, key=lambda item: item[1], default=(-1, 0.0))
        if best_overlap < 0.50:
            fused.append({**visual})
            continue

        current = fused[best_index]
        current["origin"] = "audio_visual"
        current["audio_score"] = current.get("audio_score", current["score"])
        current["visual_score"] = visual["score"]
        current["score"] = max(current["score"], visual["score"])
        current["start"] = round(min(current["start"], visual["start"]), 3)
        current["end"] = round(max(current["end"], visual["end"]), 3)
        current["time"] = f"{current['start']:.2f}-{current['end']:.2f}"
        evidence = str(visual.get("visual_evidence", "")).strip()
        if evidence:
            current["visual_evidence"] = evidence

    return sorted(fused, key=lambda item: item["start"])


def refine_clip_bounds(
    candidate: dict,
    confirmation: dict,
    source_duration: float,
    *,
    padding: float,
) -> tuple[float, float, bool]:
    """Convert model-reported clip-local bounds into padded source bounds."""
    candidate_start = float(candidate["start"])
    candidate_end = float(candidate["end"])
    clip_duration = candidate_end - candidate_start
    try:
        event_start = float(confirmation["event_start"])
        event_end = float(confirmation["event_end"])
    except (KeyError, TypeError, ValueError):
        return candidate_start, candidate_end, False
    if not math.isfinite(event_start) or not math.isfinite(event_end):
        return candidate_start, candidate_end, False
    tolerance = 0.25
    if (
        event_start < -tolerance
        or event_end > clip_duration + tolerance
        or event_end <= event_start
    ):
        return candidate_start, candidate_end, False
    event_start = max(0.0, min(clip_duration, event_start))
    event_end = max(0.0, min(clip_duration, event_end))
    start = max(0.0, candidate_start + event_start - padding)
    end = min(source_duration, candidate_start + event_end + padding)
    if end <= start:
        return candidate_start, candidate_end, False
    return round(start, 3), round(end, 3), True


def make_visual_scan(source: Path, destination: Path, interval: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    timestamp = (
        "drawtext=text='source %{pts\\:hms}':x=10:y=10:fontsize=24:"
        "fontcolor=yellow:box=1:boxcolor=black@0.65"
    )
    filters = (
        "scale=640:-2,"
        f"{timestamp},fps=1/{interval:.3f},"
        f"setpts=N/({VISUAL_SCAN_PLAYBACK_FPS}*TB)"
    )
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            filters,
            "-r",
            str(VISUAL_SCAN_PLAYBACK_FPS),
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def scan_visual_candidates(
    scan: Path,
    event: str,
    source_duration: float,
    *,
    interval: float,
    padding: float,
    max_candidates: int,
    attempts: int,
    api_key: str,
) -> tuple[list[dict], dict]:
    prompt = f"""你是视频事件粗筛器。这个视频是从原视频每隔 {interval:.3f} 秒抽取一帧组成的无声扫描视频。
每帧左上角的 source 时间是原视频时间。请只根据画面寻找可能发生“{event}”的位置。
重点观察事件发生前后的状态变化，不要因为已经破损的物体一直存在就重复报候选。
返回最多 {max_candidates} 个原视频时间段；证据不足时返回空数组。只输出 JSON：
{{"candidates":[{{"start":0.0,"end":1.0,"confidence":0.0,"visual_evidence":"..."}}]}}"""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            raw = call_model(scan, prompt, api_key)
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        raise RuntimeError(
            f"视觉扫描在 {attempts} 次尝试后失败：{last_error}"
        ) from last_error
    candidates = normalize_visual_candidates(
        raw,
        source_duration,
        padding=padding,
        max_candidates=max_candidates,
    )
    return candidates, {
        "raw": raw,
        "candidates": candidates,
        "attempts": attempt,
    }


def confirm_candidate(
    clip: Path, event: str, candidate: dict, api_key: str
) -> dict:
    if candidate.get("origin") == "visual_fallback":
        source_description = (
            f"候选来自稀疏视觉扫描，初筛置信度为 {candidate['score']}，"
            f"初筛证据为：{candidate.get('visual_evidence', '')}。"
        )
    elif candidate.get("origin") == "audio_visual":
        source_description = (
            f"音频和视觉扫描都命中了这个候选。音频得分为 "
            f"{candidate.get('audio_score', candidate['score'])}，视觉初筛得分为 "
            f"{candidate.get('visual_score', candidate['score'])}，视觉证据为："
            f"{candidate.get('visual_evidence', '')}。两种初筛都可能误报。"
        )
    else:
        source_description = (
            f"候选来自音频模型，最高分为 {candidate['score']}，但它可能误报。"
        )
    prompt = f"""你是视频事件取证审核器。请综合画面和声音，判断候选片段中是否发生了“{event}”。
{source_description}
不要仅凭文件名或提示词判断。只有片段里的实际证据足够时才确认。
只输出 JSON：
{{"confirmed":true,"confidence":0.0,"event_start":0.0,"event_end":0.0,"visual_evidence":"...","audio_evidence":"...","reason":"..."}}
event_start 和 event_end 使用该候选片段内的秒数；无法精确定位时可填 null。"""
    raw = call_model(clip, prompt, api_key)
    return {
        "confirmed": as_bool(raw.get("confirmed")),
        "confidence": as_confidence(raw.get("confidence")),
        "event_start": raw.get("event_start"),
        "event_end": raw.get("event_end"),
        "visual_evidence": str(raw.get("visual_evidence", "")),
        "audio_evidence": str(raw.get("audio_evidence", "")),
        "reason": str(raw.get("reason", "")),
        "raw": raw,
    }


def review_clip(clip: Path, event: str, api_key: str) -> dict:
    prompt = f"""你是视频剪辑质检员。检查这个成品是否完整包含“{event}”。
请确认事件确实存在，关键声音没有被截断，并且画面包含足够的事件上下文。
只输出 JSON：
{{"event_present":true,"complete":true,"confidence":0.0,"needs_more_before":false,"needs_more_after":false,"reason":"..."}}"""
    raw = call_model(clip, prompt, api_key)
    return {
        "event_present": as_bool(raw.get("event_present")),
        "complete": as_bool(raw.get("complete")),
        "confidence": as_confidence(raw.get("confidence")),
        "needs_more_before": as_bool(raw.get("needs_more_before")),
        "needs_more_after": as_bool(raw.get("needs_more_after")),
        "reason": str(raw.get("reason", "")),
        "raw": raw,
    }


def cut_clip(source: Path, destination: Path, start: float, end: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def technical_review(clip: Path, expected_duration: float) -> dict:
    media = probe_media(clip)
    duration_error = abs(media["duration"] - expected_duration)
    checks = {
        "has_video": media["has_video"],
        "has_audio": media["has_audio"],
        "duration_ok": duration_error <= 0.75,
        "nonempty": media["size"] > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actual_duration": media["duration"],
        "expected_duration": round(expected_duration, 3),
        "duration_error": round(duration_error, 3),
        "size": media["size"],
        "streams": media["streams"],
    }


def safe_name(value: str) -> str:
    name = "".join(char if char.isalnum() else "_" for char in value.strip())
    return re.sub(r"_+", "_", name).strip("_") or "event"


def make_candidate_clip(
    source: Path, temp_dir: Path, index: int, start: float, end: float
) -> Path:
    path = temp_dir / f"candidate_{index:02d}.mp4"
    cut_clip(source, path, start, end)
    return path


def write_report(output_dir: Path, video: Path, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video.stem}_pipeline_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_pipeline(
    video: Path,
    event: str,
    output_dir: Path,
    *,
    threshold: float,
    max_candidates: int,
    min_confidence: float,
    retry_padding: float,
    event_padding: float,
    visual_fallback: bool,
    visual_scan_interval: float,
    visual_padding: float,
    visual_scan_attempts: int,
    final_review: bool,
    model_path: Path,
    api_key: str,
) -> tuple[dict, Path]:
    import tempfile

    source = video.resolve()
    output_dir = output_dir.resolve()
    source_media = probe_media(source)
    if not source_media["has_video"] or not source_media["has_audio"]:
        raise RuntimeError("输入必须同时包含视频流和音频流")

    detection = locate_candidates(
        source,
        event,
        model_path=model_path.resolve(),
        threshold=threshold,
    )
    audio_candidates = [
        {**item, "origin": "audio"} for item in detection["candidates"]
    ]
    candidates = fuse_candidates(audio_candidates, [], max_candidates=max_candidates)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": 3,
        "source": str(source),
        "event": event,
        "status": "processing",
        "settings": {
            "multimodal_model": os.environ.get("ARK_MODEL", DEFAULT_MODEL),
            "audio_threshold": threshold,
            "min_confirmation_confidence": min_confidence,
            "max_candidates": max_candidates,
            "final_model_review": final_review,
            "retry_padding": retry_padding,
            "event_padding": event_padding,
            "visual_fallback": visual_fallback,
            "visual_scan_mode": "supplemental" if visual_fallback else "disabled",
            "visual_scan_interval": visual_scan_interval,
            "visual_padding": visual_padding,
            "visual_scan_attempts": visual_scan_attempts,
        },
        "source_media": source_media,
        "detection": detection,
        "visual_fallback": {
            "enabled": visual_fallback,
            "mode": "supplemental" if visual_fallback else "disabled",
            "used": False,
            "scan_media": None,
            "raw": None,
            "candidates": [],
            "attempts": None,
            "error": None,
        },
        "results": [],
        "outputs": [],
    }

    with tempfile.TemporaryDirectory(prefix="evidence-pipeline-") as temp_name:
        temp_dir = Path(temp_name)
        if visual_fallback:
            scan_path = temp_dir / "visual_scan.mp4"
            make_visual_scan(source, scan_path, visual_scan_interval)
            try:
                visual_candidates, visual_scan = scan_visual_candidates(
                    scan_path,
                    event,
                    source_media["duration"],
                    interval=visual_scan_interval,
                    padding=visual_padding,
                    max_candidates=max_candidates,
                    attempts=visual_scan_attempts,
                    api_key=api_key,
                )
            except RuntimeError as exc:
                report["visual_fallback"].update(
                    {
                        "used": True,
                        "scan_media": probe_media(scan_path),
                        "error": str(exc),
                    }
                )
                if not candidates:
                    raise
            else:
                report["visual_fallback"] = {
                    "enabled": True,
                    "mode": "supplemental",
                    "used": True,
                    "scan_media": probe_media(scan_path),
                    "raw": visual_scan["raw"],
                    "candidates": visual_candidates,
                    "attempts": visual_scan["attempts"],
                    "error": None,
                }
                candidates = fuse_candidates(
                    audio_candidates,
                    visual_candidates,
                    max_candidates=max_candidates,
                )
        if not candidates:
            report["status"] = "no_candidate"
            report_path = write_report(output_dir, source, report)
            return report, report_path

        for index, candidate in enumerate(candidates, start=1):
            item = {"candidate": candidate, "status": "pending"}
            candidate_clip = make_candidate_clip(
                source, temp_dir, index, candidate["start"], candidate["end"]
            )
            confirmation = confirm_candidate(candidate_clip, event, candidate, api_key)
            item["confirmation"] = confirmation
            if not confirmation["confirmed"] or confirmation["confidence"] < min_confidence:
                item["status"] = "rejected"
                report["results"].append(item)
                continue

            start, end, refined = refine_clip_bounds(
                candidate,
                confirmation,
                source_media["duration"],
                padding=event_padding,
            )
            item["boundary_refinement"] = {
                "used": refined,
                "event_padding": event_padding,
                "start": start,
                "end": end,
            }
            clip_name = f"{source.stem}_{safe_name(event)}_{index:02d}.mp4"
            clip_path = output_dir / clip_name
            cut_clip(source, clip_path, start, end)
            technical = technical_review(clip_path, end - start)
            item["technical_review"] = technical
            if not technical["passed"]:
                clip_path.unlink(missing_ok=True)
                item["status"] = "technical_review_failed"
                report["results"].append(item)
                continue

            reviews = []
            if final_review:
                review = review_clip(clip_path, event, api_key)
                reviews.append(review)
                passed = (
                    review["event_present"]
                    and review["complete"]
                    and review["confidence"] >= min_confidence
                )
                if not passed and review["event_present"]:
                    add_before = review["needs_more_before"] or not (
                        review["needs_more_before"] or review["needs_more_after"]
                    )
                    add_after = review["needs_more_after"] or not (
                        review["needs_more_before"] or review["needs_more_after"]
                    )
                    start = max(0.0, start - (retry_padding if add_before else 0.0))
                    end = min(
                        source_media["duration"],
                        end + (retry_padding if add_after else 0.0),
                    )
                    cut_clip(source, clip_path, start, end)
                    technical = technical_review(clip_path, end - start)
                    item["technical_review"] = technical
                    if technical["passed"]:
                        review = review_clip(clip_path, event, api_key)
                        reviews.append(review)
                        passed = (
                            review["event_present"]
                            and review["complete"]
                            and review["confidence"] >= min_confidence
                        )
                item["model_reviews"] = reviews
                if not passed:
                    clip_path.unlink(missing_ok=True)
                    item["status"] = "model_review_failed"
                    report["results"].append(item)
                    continue

            item["status"] = "completed"
            item["clip"] = {
                "path": str(clip_path),
                "start": round(start, 3),
                "end": round(end, 3),
                "time": f"{start:.2f}-{end:.2f}",
            }
            report["outputs"].append(str(clip_path))
            report["results"].append(item)

    report["status"] = "completed" if report["outputs"] else "no_confirmed_event"
    report_path = write_report(output_dir, source, report)
    return report, report_path


def main() -> int:
    args = parse_args()
    try:
        video = args.video.resolve()
        if not video.is_file():
            raise RuntimeError(f"视频不存在：{video}")
        if not 0.0 <= args.threshold <= 1.0:
            raise RuntimeError("threshold 必须在 0 到 1 之间")
        if not 0.0 <= args.min_confidence <= 1.0:
            raise RuntimeError("min-confidence 必须在 0 到 1 之间")
        if args.max_candidates < 1:
            raise RuntimeError("max-candidates 必须至少为 1")
        if args.visual_scan_interval <= 0:
            raise RuntimeError("visual-scan-interval 必须大于 0")
        if args.visual_padding < 0:
            raise RuntimeError("visual-padding 不能小于 0")
        if args.event_padding < 0:
            raise RuntimeError("event-padding 不能小于 0")
        if args.visual_scan_attempts < 1:
            raise RuntimeError("visual-scan-attempts 必须至少为 1")
        report, report_path = run_pipeline(
            video,
            args.event,
            args.output,
            threshold=args.threshold,
            max_candidates=args.max_candidates,
            min_confidence=args.min_confidence,
            retry_padding=args.retry_padding,
            event_padding=args.event_padding,
            visual_fallback=not args.no_visual_fallback,
            visual_scan_interval=args.visual_scan_interval,
            visual_padding=args.visual_padding,
            visual_scan_attempts=args.visual_scan_attempts,
            final_review=not args.skip_final_review,
            model_path=args.model_path,
            api_key=os.environ.get("ARK_API_KEY", "").strip(),
        )
        print(json.dumps({"status": report["status"], "report": str(report_path), "outputs": report["outputs"]}, ensure_ascii=False))
        return 0 if report["status"] in {"completed", "no_candidate", "no_confirmed_event"} else 1
    except (RuntimeError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
