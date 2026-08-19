#!/usr/bin/env python3
"""Run the multimodal video evidence pipeline end to end."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import json
import math
import mimetypes
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from analyze_video import DEFAULT_BASE_URL, DEFAULT_MODEL
from audio_candidates import DEFAULT_MODEL_PATH, locate_candidates


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_VISUAL_SCAN_INTERVAL = 2.0
DEFAULT_VISUAL_PADDING = 2.0
DEFAULT_VISUAL_SCAN_ATTEMPTS = 2
DEFAULT_MODEL_ATTEMPTS = 2
DEFAULT_MODEL_RETRY_BASE_DELAY = 1.0
VISUAL_SCAN_PLAYBACK_FPS = 4
DEFAULT_EVENT_PADDING = 1.0
DEFAULT_MERGE_CONFIRMED_GAP = 0.5
DEFAULT_SCENE_CHANGE_THRESHOLD = 0.05
DEFAULT_MAX_SCENE_HINTS = 12
MIN_SCENE_CANDIDATE_SCORE = 0.20
SCENE_CANDIDATE_PRE_PADDING = 0.25
MIN_VISUAL_SCAN_CONFIDENCE = 0.30
MAX_SCENE_CANDIDATES_PER_RUN = 2
MAX_CONFIRMATION_CANDIDATE_SECONDS = 8.0
CONFIRMATION_WINDOW_OVERLAP_SECONDS = 1.0
MAX_CONFIRMATION_EVENTS = 6
SCENE_CANDIDATE_NMS_IOU = 0.50
SCENE_CANDIDATE_NMS_OVERLAP = 0.80
EVENT_DEDUPE_IOU = 0.50
EVENT_DEDUPE_OVERLAP = 0.80
VISUAL_SCAN_CACHE_VERSION = 1
CONFIRMATION_CACHE_VERSION = 2
PIPELINE_VERSION = 9


class RetryableModelError(RuntimeError):
    """A transient model-service failure that may succeed after a delay."""

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


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
        "--merge-confirmed-gap",
        type=float,
        default=DEFAULT_MERGE_CONFIRMED_GAP,
        help="兼容旧版参数；v6 已改用事件核心全局去重，不再按时间间隔合并",
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
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("模型返回的 JSON 无法解析") from exc
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


def retry_after_seconds(headers) -> float | None:
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


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
            try:
                result = json.load(response)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RetryableModelError(
                    "模型服务返回了无法解析的响应"
                ) from exc
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
        except (OSError, http.client.HTTPException):
            detail = ""
        message = f"模型调用失败（HTTP {exc.code}）：{detail}"
        if exc.code in {408, 425, 429} or 500 <= exc.code <= 599:
            raise RetryableModelError(
                message,
                retry_after=retry_after_seconds(exc.headers),
            ) from exc
        raise RuntimeError(message) from exc
    except (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
    ) as exc:
        raise RetryableModelError(f"模型请求失败：{exc}") from exc
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RetryableModelError("模型响应缺少 choices/message/content") from exc
    try:
        return extract_json(model_text(content))
    except RuntimeError as exc:
        raise RetryableModelError(f"模型内容解析失败：{exc}") from exc


def call_model_with_retries(
    video: Path,
    prompt: str,
    api_key: str,
    *,
    attempts: int = DEFAULT_MODEL_ATTEMPTS,
) -> tuple[dict, int]:
    if attempts < 1:
        raise ValueError("模型尝试次数必须至少为 1")
    last_error: RetryableModelError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call_model(video, prompt, api_key), attempt
        except RetryableModelError as exc:
            last_error = exc
            if attempt < attempts:
                delay = max(
                    exc.retry_after or 0.0,
                    DEFAULT_MODEL_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                )
                time.sleep(delay + random.random() * 0.25)
    raise RuntimeError(f"模型在 {attempts} 次尝试后失败：{last_error}") from last_error


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
    if not math.isfinite(number):
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


def candidate_overlap_fraction(left: dict, right: dict) -> float:
    overlap = max(
        0.0,
        min(float(left["end"]), float(right["end"]))
        - max(float(left["start"]), float(right["start"])),
    )
    shorter = min(
        float(left["end"]) - float(left["start"]),
        float(right["end"]) - float(right["start"]),
    )
    return overlap / shorter if shorter > 0 else 0.0


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
    remote_visual = [
        item for item in visual_candidates if item.get("origin") != "visual_change"
    ]
    scene_visual = [
        item for item in visual_candidates if item.get("origin") == "visual_change"
    ]
    # Remote model confidence and FFmpeg scene score are not comparable scales.
    # Keep the remote visual budget intact and admit a small, separate scene budget.
    selected_visual = sorted(
        remote_visual, key=lambda item: item["score"], reverse=True
    )[:max_candidates]
    selected_visual.extend(
        sorted(scene_visual, key=lambda item: item["score"], reverse=True)[
            : min(max_candidates, MAX_SCENE_CANDIDATES_PER_RUN)
        ]
    )
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


def split_confirmation_candidates(
    candidates: list[dict],
    *,
    max_duration: float = MAX_CONFIRMATION_CANDIDATE_SECONDS,
    overlap: float = CONFIRMATION_WINDOW_OVERLAP_SECONDS,
) -> list[dict]:
    """Split broad candidates into bounded, overlapping confirmation windows."""
    if max_duration <= 0:
        raise ValueError("max_duration must be positive")
    if overlap < 0 or overlap >= max_duration:
        raise ValueError("overlap must be non-negative and smaller than max_duration")

    segmented = []
    for candidate in candidates:
        start = float(candidate["start"])
        end = float(candidate["end"])
        duration = end - start
        if duration <= max_duration:
            segmented.append({**candidate})
            continue

        segment_count = math.ceil(
            (duration - overlap) / (max_duration - overlap)
        )
        step = (duration - max_duration) / (segment_count - 1)
        absolute_focus = None
        if candidate.get("focus_time") is not None:
            try:
                absolute_focus = start + float(candidate["focus_time"])
            except (TypeError, ValueError):
                absolute_focus = None
        for segment_index in range(segment_count):
            segment_start = start + step * segment_index
            segment_end = segment_start + max_duration
            if segment_index == segment_count - 1:
                segment_end = end
                segment_start = end - max_duration
            item = {
                **candidate,
                "start": round(segment_start, 3),
                "end": round(segment_end, 3),
                "parent_start": round(start, 3),
                "parent_end": round(end, 3),
                "segment_index": segment_index + 1,
                "segment_count": segment_count,
            }
            item["time"] = f"{item['start']:.2f}-{item['end']:.2f}"
            if absolute_focus is not None:
                item["focus_time"] = round(
                    max(0.0, min(max_duration, absolute_focus - item["start"])),
                    3,
                )
            segmented.append(item)
    return segmented


def refine_event_bounds(
    candidate: dict,
    confirmation: dict,
    source_duration: float,
) -> tuple[float, float, bool]:
    """Convert model-reported clip-local event bounds into source time."""
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
    start = max(0.0, candidate_start + event_start)
    end = min(source_duration, candidate_start + event_end)
    if end <= start:
        return candidate_start, candidate_end, False
    return round(start, 3), round(end, 3), True


def refine_clip_bounds(
    candidate: dict,
    confirmation: dict,
    source_duration: float,
    *,
    padding: float,
) -> tuple[float, float, bool]:
    """Convert model event bounds into a padded source clip."""
    event_start, event_end, refined = refine_event_bounds(
        candidate,
        confirmation,
        source_duration,
    )
    if not refined:
        return event_start, event_end, False
    start = max(0.0, event_start - padding)
    end = min(source_duration, event_end + padding)
    if end <= start:
        return float(candidate["start"]), float(candidate["end"]), False
    return round(start, 3), round(end, 3), True


def event_to_source_core(
    candidate: dict,
    event: dict,
    source_duration: float,
) -> dict | None:
    """Convert a validated clip-local event into a strict absolute core."""
    try:
        candidate_start = float(candidate["start"])
        candidate_end = float(candidate["end"])
        local_start = float(event["event_start"])
        local_end = float(event["event_end"])
    except (KeyError, TypeError, ValueError):
        return None
    values = (candidate_start, candidate_end, local_start, local_end, source_duration)
    if not all(math.isfinite(value) for value in values):
        return None
    candidate_duration = candidate_end - candidate_start
    if (
        candidate_start < 0
        or candidate_end <= candidate_start
        or local_start < 0
        or local_end <= local_start
        or local_end > candidate_duration
    ):
        return None
    start = max(0.0, min(source_duration, candidate_start + local_start))
    end = max(0.0, min(source_duration, candidate_start + local_end))
    if end <= start:
        return None
    return {"start": round(start, 3), "end": round(end, 3)}


def same_core_occurrence(left: dict, right: dict) -> bool:
    left_duration = float(left["end"]) - float(left["start"])
    right_duration = float(right["end"]) - float(right["start"])
    shorter = min(left_duration, right_duration)
    longer = max(left_duration, right_duration)
    if shorter <= 0:
        return False
    if candidate_iou(left, right) >= EVENT_DEDUPE_IOU:
        return True
    return (
        candidate_overlap_fraction(left, right) >= EVENT_DEDUPE_OVERLAP
        and longer / shorter <= 2.0
    )


def deduplicate_core_events(
    proposals: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Complete-link core dedupe that never chains or merges same-candidate events."""

    def proposal_key(item: dict):
        core = item["core"]
        return (
            core["end"] - core["start"],
            -item["confidence"],
            core["start"],
            core["end"],
            item["stable_id"],
        )

    clusters: list[dict] = []
    decisions = []
    for proposal in sorted(proposals, key=proposal_key):
        compatible = []
        for cluster in clusters:
            if proposal["candidate_id"] in cluster["candidate_ids"]:
                continue
            if all(
                same_core_occurrence(proposal["core"], member["core"])
                for member in cluster["members"]
            ):
                compatible.append(cluster)
        if len(compatible) == 1:
            cluster = compatible[0]
            cluster["members"].append(proposal)
            cluster["candidate_ids"].add(proposal["candidate_id"])
        elif len(compatible) > 1:
            decisions.append(
                {
                    "status": "ambiguous_bridge_suppressed",
                    "proposal": proposal["stable_id"],
                    "core": dict(proposal["core"]),
                    "compatible_cluster_count": len(compatible),
                }
            )
        else:
            clusters.append(
                {
                    "members": [proposal],
                    "candidate_ids": {proposal["candidate_id"]},
                }
            )

    normalized_clusters = []
    for cluster in clusters:
        members = sorted(cluster["members"], key=proposal_key)
        representative = members[0]
        normalized_clusters.append(
            {
                "representative": representative,
                "members": members,
            }
        )
        for member in members[1:]:
            decisions.append(
                {
                    "status": "duplicate_suppressed",
                    "proposal": member["stable_id"],
                    "representative": representative["stable_id"],
                    "core": dict(member["core"]),
                }
            )
    normalized_clusters.sort(
        key=lambda item: (
            item["representative"]["core"]["start"],
            item["representative"]["core"]["end"],
            item["representative"]["stable_id"],
        )
    )
    decisions.sort(
        key=lambda item: (
            item["core"]["start"],
            item["core"]["end"],
            item["proposal"],
        )
    )
    return normalized_clusters, decisions


def padded_bounds_for_core(
    core: dict,
    *,
    previous_core: dict | None,
    next_core: dict | None,
    padding: float,
    source_duration: float,
) -> tuple[float, float, dict]:
    """Pad a core while preventing its context from crossing a neighboring core."""
    core_start = float(core["start"])
    core_end = float(core["end"])
    min_start = 0.0
    max_end = source_duration
    conflicts = []
    if previous_core is not None:
        previous_end = float(previous_core["end"])
        if previous_end <= core_start:
            min_start = (previous_end + core_start) / 2.0
        else:
            min_start = core_start
            conflicts.append("previous_core_overlap")
    if next_core is not None:
        next_start = float(next_core["start"])
        if core_end <= next_start:
            max_end = (core_end + next_start) / 2.0
        else:
            max_end = core_end
            conflicts.append("next_core_overlap")
    start = max(min_start, core_start - padding)
    end = min(max_end, core_end + padding)
    return round(start, 3), round(end, 3), {
        "min_start": round(min_start, 3),
        "max_end": round(max_end, 3),
        "core_overlap_conflicts": conflicts,
    }


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


def detect_scene_change_hints(
    source: Path,
    *,
    threshold: float = DEFAULT_SCENE_CHANGE_THRESHOLD,
    max_hints: int = DEFAULT_MAX_SCENE_HINTS,
) -> list[dict]:
    result = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select='gt(scene,{threshold:.6f})',metadata=mode=print:file=-",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    changes = []
    current_time = None
    for line in result.stdout.splitlines():
        if "pts_time:" in line:
            try:
                current_time = float(line.rsplit("pts_time:", 1)[1].strip())
            except ValueError:
                current_time = None
        elif line.startswith("lavfi.scene_score=") and current_time is not None:
            try:
                score = float(line.split("=", 1)[1])
            except ValueError:
                continue
            changes.append(
                {"time": round(current_time, 3), "score": round(score, 4)}
            )
            current_time = None
    selected = sorted(changes, key=lambda item: item["score"], reverse=True)[
        :max_hints
    ]
    return sorted(selected, key=lambda item: item["time"])


def make_scene_change_candidates(
    hints: list[dict],
    duration: float,
    *,
    covered_by: list[dict],
    padding: float,
    max_candidates: int,
    coverage_margin: float = 0.5,
) -> list[dict]:
    candidates = []
    for hint in sorted(hints, key=lambda item: item.get("score", 0), reverse=True):
        try:
            timestamp = float(hint["time"])
            score = float(hint["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if score < MIN_SCENE_CANDIDATE_SCORE or not 0 <= timestamp <= duration:
            continue
        covered = False
        for existing in covered_by:
            try:
                start = float(existing["start"])
                end = float(existing["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if start - coverage_margin <= timestamp <= end + coverage_margin:
                covered = True
                break
        if covered:
            continue
        start = max(0.0, timestamp - min(padding, SCENE_CANDIDATE_PRE_PADDING))
        end = min(duration, timestamp + padding)
        if end <= start:
            continue
        candidate = {
            "start": round(start, 3),
            "end": round(end, 3),
            "score": round(max(0.0, min(1.0, score)), 4),
            "origin": "visual_change",
            "top_label": "scene_change",
            "change_time": round(timestamp, 3),
            "focus_time": round(timestamp - start, 3),
            "visual_evidence": (
                f"本地画面变化检测在 {timestamp:.3f} 秒发现强变化，需排除镜头切换"
            ),
            "time": f"{start:.2f}-{end:.2f}",
        }
        if any(
            candidate_iou(candidate, existing) >= SCENE_CANDIDATE_NMS_IOU
            or candidate_overlap_fraction(candidate, existing)
            >= SCENE_CANDIDATE_NMS_OVERLAP
            for existing in candidates
        ):
            continue
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return sorted(candidates, key=lambda item: item["start"])


def build_visual_scan_prompt(
    event: str,
    *,
    interval: float,
    max_candidates: int,
    time_hints: list[dict] | None = None,
    scene_hints: list[dict] | None = None,
) -> str:
    hints = []
    for item in time_hints or []:
        try:
            hints.append(f"{float(item['start']):.2f}-{float(item['end']):.2f}秒")
        except (KeyError, TypeError, ValueError):
            continue
    hint_text = (
        "声音粗筛给出的待核对时间为：" + "、".join(hints) + "。这些时间只是提示，"
        "可能误报；既要逐一核对，也要检查其外的画面。"
        if hints
        else "没有可用的声音时间提示，请独立检查全部画面。"
    )
    scene_times = []
    for item in scene_hints or []:
        try:
            scene_times.append(f"{float(item['time']):.2f}秒")
        except (KeyError, TypeError, ValueError):
            continue
    scene_hint_text = (
        "本地画面变化检测还标出了：" + "、".join(scene_times) + "。这些位置可能只是"
        "镜头切换或相机运动，不是事件证据，但必须逐一查看其前后状态。"
        if scene_times
        else "本地画面变化检测没有提供额外时间提示。"
    )
    return f"""你是视频事件粗筛器。这个视频是从原视频每隔 {interval:.3f} 秒抽取一帧组成的无声扫描视频。
每帧左上角的 source 时间是原视频时间。请只根据画面寻找可能发生“{event}”的位置。
{hint_text}
{scene_hint_text}
请完整遍历所有帧，不要找到第一处就停止。原视频可能包含多次独立事件或不同重放视角，
每一次明确的状态变化都要分别列出。重点观察事件发生前后的状态变化，不要因为已经破损的
物体一直存在就重复报候选。
返回最多 {max_candidates} 个原视频时间段；证据不足时返回空数组。只输出 JSON：
{{"candidates":[{{"start":0.0,"end":1.0,"confidence":0.0,"visual_evidence":"..."}}]}}"""


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
    time_hints: list[dict] | None = None,
    scene_hints: list[dict] | None = None,
    prompt: str | None = None,
) -> tuple[list[dict], dict]:
    if prompt is None:
        prompt = build_visual_scan_prompt(
            event,
            interval=interval,
            max_candidates=max_candidates,
            time_hints=time_hints,
            scene_hints=scene_hints,
        )
    try:
        raw, attempt = call_model_with_retries(
            scan,
            prompt,
            api_key,
            attempts=attempts,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"视觉扫描失败：{exc}") from exc
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


def normalize_confirmation_events(raw: dict, clip_duration: float) -> list[dict]:
    """Normalize strict clip-local event cores without widening invalid values."""
    top_confidence = as_confidence(raw.get("confidence"))
    if "events" in raw:
        values = raw.get("events") if isinstance(raw.get("events"), list) else []
    else:
        values = [
            {
                "event_start": raw.get("event_start"),
                "event_end": raw.get("event_end"),
                "confidence": top_confidence,
                "visual_evidence": raw.get("visual_evidence", ""),
                "audio_evidence": raw.get("audio_evidence", ""),
                "reason": raw.get("reason", ""),
            }
        ]
    normalized_by_core = {}
    tolerance = 0.25
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            start = float(value.get("event_start", value.get("start")))
            end = float(value.get("event_end", value.get("end")))
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < -tolerance
            or end > clip_duration + tolerance
            or end <= start
        ):
            continue
        start = round(max(0.0, min(clip_duration, start)), 3)
        end = round(max(0.0, min(clip_duration, end)), 3)
        if end <= start:
            continue
        event = {
            "event_start": start,
            "event_end": end,
            "confidence": as_confidence(value.get("confidence", top_confidence)),
            "visual_evidence": str(
                value.get("visual_evidence", raw.get("visual_evidence", ""))
            ),
            "audio_evidence": str(
                value.get("audio_evidence", raw.get("audio_evidence", ""))
            ),
            "reason": str(value.get("reason", raw.get("reason", ""))),
        }
        key = (start, end)
        previous = normalized_by_core.get(key)
        if previous is None or event["confidence"] > previous["confidence"]:
            normalized_by_core[key] = event
    return sorted(
        normalized_by_core.values(),
        key=lambda item: (item["event_start"], item["event_end"]),
    )[:MAX_CONFIRMATION_EVENTS]


def build_confirmation_prompt(event: str, candidate: dict) -> str:
    if candidate.get("origin") == "visual_fallback":
        source_description = (
            f"候选来自视觉粗筛，初筛置信度为 {candidate['score']}，"
            f"初筛证据为：{candidate.get('visual_evidence', '')}。"
        )
    elif candidate.get("origin") == "visual_change":
        source_description = (
            f"候选来自本地画面变化检测，变化点约在片段内 "
            f"{candidate.get('focus_time', 0):.2f} 秒，得分为 {candidate['score']}。"
            "该变化可能只是镜头切换、相机运动或非目标撞击；重点核对变化点及其后是否"
            "新发生目标事件，不要把变化点之前的旧事件算作本候选证据。"
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
    return f"""你是视频事件取证审核器。请综合画面和声音，判断候选片段中是否发生了“{event}”。
{source_description}
不要仅凭文件名或提示词判断。只有片段里的实际证据足够时才确认。
必须看到一次新的发生瞬间或从未发生到已发生的状态变化；如果片段开始时事件已经发生，
后面只有残留碎片、持续破损状态或前一次事件的余波，应判定 confirmed=false。
候选可能很宽，若其中有多次彼此独立的新事件，events 必须按时间逐次列出，不能只返回第一处；
一项只表示一次新的状态变化/撞击，重放视角中的新发生也单列，但声音尾巴、碎片余波不另列。
每项边界必须紧贴发生核心，不含上下文和余波：event_start 是首次接触、首次不可逆裂纹或
首次新破损的时刻，不能提前到物体仍完整的画面；event_end 是主要裂纹扩展或大块脱落结束，
不能在第一帧破损时过早停止，也不要延长到零星余屑。最多返回 {MAX_CONFIRMATION_EVENTS} 项。occurrence_count 等于 events 数量。
只输出 JSON：
{{"confirmed":true,"confidence":0.0,"occurrence_count":1,"events":[{{"event_start":0.0,"event_end":0.0,"confidence":0.0,"visual_evidence":"...","audio_evidence":"...","reason":"..."}}],"reason":"..."}}
无法精确定位时相应边界可填 null；没有新事件时 events=[]、occurrence_count=0。"""


def confirm_candidate(
    clip: Path,
    event: str,
    candidate: dict,
    api_key: str,
    *,
    prompt: str | None = None,
) -> dict:
    if prompt is None:
        prompt = build_confirmation_prompt(event, candidate)
    raw, attempts = call_model_with_retries(clip, prompt, api_key)
    clip_duration = max(0.0, float(candidate["end"]) - float(candidate["start"]))
    events = normalize_confirmation_events(raw, clip_duration)
    confirmed = bool(events)
    top_confidence = as_confidence(raw.get("confidence"))
    first_event = events[0] if events else {}
    return {
        "confirmed": confirmed,
        "confidence": max(
            [top_confidence, *(item["confidence"] for item in events)],
            default=top_confidence,
        ),
        "occurrence_count": len(events) if confirmed else 0,
        "events": events if confirmed else [],
        "event_start": first_event.get("event_start"),
        "event_end": first_event.get("event_end"),
        "visual_evidence": str(raw.get("visual_evidence", "")),
        "audio_evidence": str(raw.get("audio_evidence", "")),
        "reason": str(raw.get("reason", "")),
        "attempts": attempts,
        "raw": raw,
    }


def review_clip(
    clip: Path,
    event: str,
    api_key: str,
    *,
    focus_start: float | None = None,
    focus_end: float | None = None,
) -> dict:
    focus_text = ""
    if focus_start is not None and focus_end is not None:
        focus_text = (
            f"本次要审核的事件核心位于成品内 {focus_start:.3f}-{focus_end:.3f} 秒。"
            "必须核对这一处；片段其他位置即使有同类事件，也不能替代它通过审核。"
        )
    prompt = f"""你是视频剪辑质检员。检查这个成品是否完整包含“{event}”。
{focus_text}
请确认事件确实存在，关键声音没有被截断，并且画面包含足够的事件上下文。
只输出 JSON：
{{"event_present":true,"complete":true,"confidence":0.0,"needs_more_before":false,"needs_more_after":false,"reason":"..."}}"""
    raw, attempts = call_model_with_retries(clip, prompt, api_key)
    return {
        "event_present": as_bool(raw.get("event_present")),
        "complete": as_bool(raw.get("complete")),
        "confidence": as_confidence(raw.get("confidence")),
        "needs_more_before": as_bool(raw.get("needs_more_before")),
        "needs_more_after": as_bool(raw.get("needs_more_after")),
        "reason": str(raw.get("reason", "")),
        "attempts": attempts,
        "raw": raw,
    }


def review_consolidation(
    clip: Path,
    event: str,
    component_ranges: list[dict],
    api_key: str,
) -> dict:
    ranges = "、".join(
        f"{item['start']:.2f}-{item['end']:.2f}秒" for item in component_ranges
    )
    prompt = f"""你是视频事件去重审核器。这个连续片段里有两个相邻候选区间：{ranges}。
请判断它们是否只是同一次“{event}”的重复/延续证据，还是两次独立的新事件。
仅有持续破损状态、残留碎片或前一次事件余波不算新的事件；出现第二次明确的新撞击、
从完整到破损的变化或新的特征声音，则算独立事件。证据不足时不要合并。
只输出 JSON：
{{"same_occurrence":false,"occurrence_count":2,"confidence":0.0,"reason":"..."}}"""
    raw, attempts = call_model_with_retries(clip, prompt, api_key)
    try:
        occurrence_count = int(raw.get("occurrence_count"))
    except (TypeError, ValueError):
        occurrence_count = 0
    return {
        "same_occurrence": as_bool(raw.get("same_occurrence")),
        "occurrence_count": max(0, occurrence_count),
        "confidence": as_confidence(raw.get("confidence")),
        "reason": str(raw.get("reason", "")),
        "attempts": attempts,
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


def proposal_provenance(proposal: dict) -> dict:
    return {
        "proposal_id": proposal["stable_id"],
        "candidate_id": proposal["candidate_id"],
        "candidate_index": proposal["candidate_index"],
        "occurrence_index": proposal["occurrence_index"],
        "origin": proposal["candidate"].get("origin"),
        "core": dict(proposal["core"]),
        "confidence": proposal["confidence"],
    }


def materialize_event_result(
    source: Path,
    output_dir: Path,
    event: str,
    event_number: int,
    cluster: dict,
    *,
    previous_core: dict | None,
    next_core: dict | None,
    source_duration: float,
    event_padding: float,
    retry_padding: float,
    min_confidence: float,
    final_review: bool,
    api_key: str,
) -> dict:
    proposal = cluster["representative"]
    core = proposal["core"]
    start, end, limits = padded_bounds_for_core(
        core,
        previous_core=previous_core,
        next_core=next_core,
        padding=event_padding,
        source_duration=source_duration,
    )
    confirmation = {
        **proposal["event"],
        "confirmed": True,
        "attempts": proposal["candidate_confirmation"]["attempts"],
        "selected_occurrence_index": proposal["occurrence_index"],
    }
    item = {
        "candidate": proposal["candidate"],
        "candidate_index": proposal["candidate_index"],
        "occurrence_index": proposal["occurrence_index"],
        "occurrence_count": proposal["candidate_confirmation"]["occurrence_count"],
        "confirmation": confirmation,
        "event_bounds": {**core, "source": "model"},
        "dedup_members": [
            proposal_provenance(member) for member in cluster["members"]
        ],
        "boundary_refinement": {
            "used": True,
            "event_padding": event_padding,
            "start": start,
            "end": end,
            "limits": limits,
            "retry_expanded": False,
        },
        "status": "pending",
    }
    clip_name = f"{source.stem}_{safe_name(event)}_{event_number:02d}.mp4"
    clip_path = output_dir / clip_name
    cut_clip(source, clip_path, start, end)
    technical = technical_review(clip_path, end - start)
    item["technical_review"] = technical
    if not technical["passed"]:
        clip_path.unlink(missing_ok=True)
        item["status"] = "technical_review_failed"
        return item

    reviews = []
    if final_review:
        review = review_clip(
            clip_path,
            event,
            api_key,
            focus_start=float(core["start"]) - start,
            focus_end=float(core["end"]) - start,
        )
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
            expanded_start = max(
                limits["min_start"],
                start - (retry_padding if add_before else 0.0),
            )
            expanded_end = min(
                limits["max_end"],
                end + (retry_padding if add_after else 0.0),
            )
            if expanded_start < start or expanded_end > end:
                start = round(expanded_start, 3)
                end = round(expanded_end, 3)
                cut_clip(source, clip_path, start, end)
                technical = technical_review(clip_path, end - start)
                item["technical_review"] = technical
                item["boundary_refinement"].update(
                    {"start": start, "end": end, "retry_expanded": True}
                )
                if technical["passed"]:
                    review = review_clip(
                        clip_path,
                        event,
                        api_key,
                        focus_start=float(core["start"]) - start,
                        focus_end=float(core["end"]) - start,
                    )
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
            return item

    item["status"] = "completed"
    item["clip"] = {
        "path": str(clip_path),
        "start": round(start, 3),
        "end": round(end, 3),
        "time": f"{start:.2f}-{end:.2f}",
    }
    return item


def collect_event_proposals(
    source: Path,
    temp_dir: Path,
    candidates: list[dict],
    event: str,
    *,
    source_duration: float,
    source_sha256: str,
    confirmation_cache_path: Path,
    min_confidence: float,
    api_key: str,
) -> tuple[list[dict], list[dict], dict]:
    reviews = []
    proposals = []
    entries = load_confirmation_cache(
        confirmation_cache_path,
        source_sha256=source_sha256,
    )
    cache_details = {
        "path": str(confirmation_cache_path),
        "hits": 0,
        "writes": 0,
        "write_error": None,
    }
    for candidate_index, candidate in enumerate(candidates, start=1):
        prompt = build_confirmation_prompt(event, candidate)
        cache_identity = confirmation_cache_identity(
            source=source,
            source_sha256=source_sha256,
            candidate=candidate,
            prompt=prompt,
        )
        confirmation = cached_confirmation(entries, cache_identity)
        if confirmation is None:
            candidate_clip = make_candidate_clip(
                source,
                temp_dir,
                candidate_index,
                candidate["start"],
                candidate["end"],
            )
            confirmation = confirm_candidate(
                candidate_clip,
                event,
                candidate,
                api_key,
                prompt=prompt,
            )
            confirmation["cache_hit"] = False
            entries[stable_json_sha256(cache_identity)] = {
                "identity": cache_identity,
                "confirmation": confirmation,
            }
            try:
                write_confirmation_cache(
                    confirmation_cache_path,
                    source=source,
                    source_sha256=source_sha256,
                    entries=entries,
                )
                cache_details["writes"] += 1
            except OSError as exc:
                cache_details["write_error"] = str(exc)
        else:
            cache_details["hits"] += 1
        candidate_id = stable_json_sha256(candidate)[:16]
        review = {
            "candidate": candidate,
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "confirmation": confirmation,
            "accepted_proposals": [],
            "rejected_events": [],
            "status": "rejected",
        }
        for occurrence_index, occurrence in enumerate(
            confirmation.get("events", []), start=1
        ):
            if occurrence["confidence"] < min_confidence:
                review["rejected_events"].append(
                    {
                        "occurrence_index": occurrence_index,
                        "reason": "low_confidence",
                        "confidence": occurrence["confidence"],
                    }
                )
                continue
            core = event_to_source_core(candidate, occurrence, source_duration)
            if core is None:
                review["rejected_events"].append(
                    {
                        "occurrence_index": occurrence_index,
                        "reason": "invalid_event_core",
                    }
                )
                continue
            stable_id = (
                f"{candidate_id}:{occurrence_index}:"
                f"{core['start']:.3f}-{core['end']:.3f}"
            )
            proposal = {
                "stable_id": stable_id,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "occurrence_index": occurrence_index,
                "candidate": candidate,
                "candidate_confirmation": confirmation,
                "event": occurrence,
                "core": core,
                "confidence": occurrence["confidence"],
            }
            proposals.append(proposal)
            review["accepted_proposals"].append(stable_id)
        if review["accepted_proposals"]:
            review["status"] = "confirmed"
        reviews.append(review)
    cache_details["entry_count"] = len(entries)
    return reviews, proposals, cache_details


def materialize_unique_events(
    source: Path,
    output_dir: Path,
    event: str,
    proposals: list[dict],
    *,
    source_duration: float,
    event_padding: float,
    retry_padding: float,
    min_confidence: float,
    final_review: bool,
    api_key: str,
) -> tuple[list[dict], dict]:
    clusters, decisions = deduplicate_core_events(proposals)
    representative_cores = [
        cluster["representative"]["core"] for cluster in clusters
    ]
    results = []
    for event_index, cluster in enumerate(clusters, start=1):
        try:
            result = materialize_event_result(
                source,
                output_dir,
                event,
                event_index,
                cluster,
                previous_core=(
                    representative_cores[event_index - 2]
                    if event_index > 1
                    else None
                ),
                next_core=(
                    representative_cores[event_index]
                    if event_index < len(representative_cores)
                    else None
                ),
                source_duration=source_duration,
                event_padding=event_padding,
                retry_padding=retry_padding,
                min_confidence=min_confidence,
                final_review=final_review,
                api_key=api_key,
            )
        except Exception as exc:
            proposal = cluster["representative"]
            result = {
                "status": "error",
                "candidate": proposal["candidate"],
                "candidate_index": proposal["candidate_index"],
                "occurrence_index": proposal["occurrence_index"],
                "event_bounds": {**proposal["core"], "source": "model"},
                "dedup_members": [
                    proposal_provenance(member) for member in cluster["members"]
                ],
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        results.append(result)
    return results, {
        "proposal_count": len(proposals),
        "cluster_count": len(clusters),
        "decisions": decisions,
    }


def process_confirmed_candidates(
    source: Path,
    temp_dir: Path,
    output_dir: Path,
    candidates: list[dict],
    event: str,
    *,
    source_duration: float,
    source_sha256: str,
    confirmation_cache_path: Path,
    event_padding: float,
    retry_padding: float,
    min_confidence: float,
    final_review: bool,
    api_key: str,
) -> tuple[list[dict], list[dict], dict, dict]:
    reviews, proposals, confirmation_cache = collect_event_proposals(
        source,
        temp_dir,
        candidates,
        event,
        source_duration=source_duration,
        source_sha256=source_sha256,
        confirmation_cache_path=confirmation_cache_path,
        min_confidence=min_confidence,
        api_key=api_key,
    )
    results, deduplication = materialize_unique_events(
        source,
        output_dir,
        event,
        proposals,
        source_duration=source_duration,
        event_padding=event_padding,
        retry_padding=retry_padding,
        min_confidence=min_confidence,
        final_review=final_review,
        api_key=api_key,
    )
    return reviews, results, deduplication, confirmation_cache


def candidate_modalities(item: dict) -> set[str]:
    origin = item.get("candidate", {}).get("origin")
    if origin == "audio_visual":
        return {"audio", "visual"}
    if origin in {"visual_fallback", "visual_change"}:
        return {"visual"}
    if origin == "audio":
        return {"audio"}
    return set()


def completed_clip_bounds(item: dict) -> tuple[float, float] | None:
    if item.get("status") != "completed" or not isinstance(item.get("clip"), dict):
        return None
    try:
        start = float(item["clip"]["start"])
        end = float(item["clip"]["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return None
    return start, end


def group_adjacent_confirmed_results(
    results: list[dict], *, max_gap: float
) -> list[list[dict]]:
    """Group only non-overlapping, adjacent fragments from complementary modalities."""
    valid = []
    invalid = []
    for item in results:
        bounds = completed_clip_bounds(item)
        if bounds is None:
            if item.get("status") == "completed":
                invalid.append([item])
            continue
        valid.append((bounds, item))
    completed = [item for _, item in sorted(valid, key=lambda value: value[0])]
    groups: list[list[dict]] = []
    for item in completed:
        if not groups:
            groups.append([item])
            continue
        previous = groups[-1][-1]
        gap = float(item["clip"]["start"]) - float(previous["clip"]["end"])
        modalities = candidate_modalities(item)
        previous_modalities = candidate_modalities(previous)
        complementary = bool(
            modalities
            and previous_modalities
            and modalities.isdisjoint(previous_modalities)
        )
        if len(groups[-1]) == 1 and 0.0 <= gap <= max_gap and complementary:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups + invalid


def consolidate_completed_results(
    source: Path,
    results: list[dict],
    *,
    max_gap: float,
    event: str,
    min_confidence: float,
    api_key: str,
) -> list[dict]:
    consolidations = []
    for group in group_adjacent_confirmed_results(results, max_gap=max_gap):
        if len(group) < 2:
            continue
        start = min(float(item["clip"]["start"]) for item in group)
        end = max(float(item["clip"]["end"]) for item in group)
        primary = group[0]
        primary_path = Path(primary["clip"]["path"])
        temporary = primary_path.with_name(primary_path.stem + "_consolidating.mp4")
        decision = {
            "status": "kept_separate",
            "start": round(start, 3),
            "end": round(end, 3),
            "component_count": len(group),
        }
        try:
            cut_clip(source, temporary, start, end)
            technical = technical_review(temporary, end - start)
            if not technical["passed"]:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                decision["reason"] = "technical_review_failed"
                consolidations.append(decision)
                continue

            components = [
                {
                    "origin": item.get("candidate", {}).get("origin"),
                    "clip": dict(item["clip"]),
                    "event_bounds": dict(item.get("event_bounds", {})),
                }
                for item in group
            ]
            component_ranges = [
                {
                    "start": float(item["clip"]["start"]) - start,
                    "end": float(item["clip"]["end"]) - start,
                }
                for item in group
            ]
            review = review_consolidation(
                temporary,
                event,
                component_ranges,
                api_key,
            )
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            decision["reason"] = "consolidation_error"
            decision["error_type"] = type(exc).__name__
            decision["error"] = str(exc)
            consolidations.append(decision)
            continue

        should_merge = (
            review["same_occurrence"]
            and review["occurrence_count"] == 1
            and review["confidence"] >= min_confidence
        )
        if not should_merge:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            decision["review"] = review
            consolidations.append(decision)
            continue

        try:
            temporary.replace(primary_path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            decision["reason"] = "replace_failed"
            decision["error_type"] = type(exc).__name__
            decision["error"] = str(exc)
            consolidations.append(decision)
            continue

        cleanup_errors = []
        for item in group[1:]:
            try:
                Path(item["clip"]["path"]).unlink(missing_ok=True)
            except OSError as exc:
                cleanup_errors.append(str(exc))
            item["status"] = "consolidated"
            item["consolidated_into"] = str(primary_path)

        primary["technical_review"] = technical
        primary["clip"] = {
            "path": str(primary_path),
            "start": round(start, 3),
            "end": round(end, 3),
            "time": f"{start:.2f}-{end:.2f}",
        }
        primary["consolidation"] = {
            "component_count": len(components),
            "max_gap": max_gap,
            "components": components,
            "review": review,
        }
        merged_decision = {
            "status": "merged",
            "output": str(primary_path),
            "start": round(start, 3),
            "end": round(end, 3),
            "component_count": len(components),
        }
        if cleanup_errors:
            merged_decision["cleanup_errors"] = cleanup_errors
        consolidations.append(merged_decision)
    return consolidations


def write_report(output_dir: Path, video: Path, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video.stem}_pipeline_report.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: dict) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def visual_scan_cache_identity(
    *,
    source: Path,
    source_sha256: str,
    prompt: str,
    interval: float,
) -> dict:
    return {
        "cache_version": VISUAL_SCAN_CACHE_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source": str(source),
        "source_sha256": source_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "scan_interval": interval,
        "scan_playback_fps": VISUAL_SCAN_PLAYBACK_FPS,
        "model": os.environ.get("ARK_MODEL", DEFAULT_MODEL),
        "endpoint": os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
    }


def write_visual_scan_cache(
    path: Path,
    *,
    identity: dict,
    raw: dict,
    scan_media: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_version": VISUAL_SCAN_CACHE_VERSION,
        "fingerprint": stable_json_sha256(identity),
        "identity": identity,
        "raw": raw,
        "scan_media": scan_media,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def load_visual_scan_cache(
    path: Path,
    *,
    identity: dict,
) -> dict | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(cached, dict)
        or cached.get("cache_version") != VISUAL_SCAN_CACHE_VERSION
        or cached.get("fingerprint") != stable_json_sha256(identity)
        or cached.get("identity") != identity
        or not isinstance(cached.get("raw"), dict)
        or not isinstance(cached.get("scan_media"), dict)
    ):
        return None
    return {
        "raw": cached["raw"],
        "scan_media": cached["scan_media"],
    }


def confirmation_cache_identity(
    *,
    source: Path,
    source_sha256: str,
    candidate: dict,
    prompt: str,
) -> dict:
    return {
        "cache_version": CONFIRMATION_CACHE_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source": str(source),
        "source_sha256": source_sha256,
        "candidate": candidate,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model": os.environ.get("ARK_MODEL", DEFAULT_MODEL),
        "endpoint": os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
    }


def load_confirmation_cache(path: Path, *, source_sha256: str) -> dict:
    if not path.is_file():
        return {}
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(cached, dict)
        or cached.get("cache_version") != CONFIRMATION_CACHE_VERSION
        or cached.get("source_sha256") != source_sha256
        or not isinstance(cached.get("entries"), dict)
    ):
        return {}
    return cached["entries"]


def cached_confirmation(entries: dict, identity: dict) -> dict | None:
    key = stable_json_sha256(identity)
    entry = entries.get(key)
    if (
        not isinstance(entry, dict)
        or entry.get("identity") != identity
        or not isinstance(entry.get("confirmation"), dict)
    ):
        return None
    return {**entry["confirmation"], "attempts": 0, "cache_hit": True}


def write_confirmation_cache(
    path: Path,
    *,
    source: Path,
    source_sha256: str,
    entries: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cache_version": CONFIRMATION_CACHE_VERSION,
        "source": str(source),
        "source_sha256": source_sha256,
        "entries": entries,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


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
    merge_confirmed_gap: float,
    visual_fallback: bool,
    visual_scan_interval: float,
    visual_padding: float,
    visual_scan_attempts: int,
    require_visual_scan_success: bool,
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
    source_sha256 = sha256_path(source)
    source_media["sha256"] = source_sha256

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
    settings = {
        "multimodal_model": os.environ.get("ARK_MODEL", DEFAULT_MODEL),
        "audio_threshold": threshold,
        "min_confirmation_confidence": min_confidence,
        "max_candidates": max_candidates,
        "final_model_review": final_review,
        "retry_padding": retry_padding,
        "event_padding": event_padding,
        "merge_confirmed_gap": merge_confirmed_gap,
        "visual_fallback": visual_fallback,
        "visual_scan_mode": "supplemental" if visual_fallback else "disabled",
        "visual_scan_interval": visual_scan_interval,
        "visual_padding": visual_padding,
        "visual_scan_attempts": visual_scan_attempts,
        "require_visual_scan_success": require_visual_scan_success,
        "max_confirmation_candidate_seconds": MAX_CONFIRMATION_CANDIDATE_SECONDS,
        "confirmation_window_overlap_seconds": CONFIRMATION_WINDOW_OVERLAP_SECONDS,
    }
    report_path = output_dir / f"{source.stem}_pipeline_report.json"
    scan_cache_path = output_dir / f"{source.stem}_visual_scan_cache.json"
    confirmation_cache_path = (
        output_dir / f"{source.stem}_confirmation_cache.json"
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "source": str(source),
        "event": event,
        "status": "processing",
        "error": None,
        "settings": settings,
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
            "cache_hit": False,
            "cache_path": str(scan_cache_path),
            "cache_write_error": None,
            "scene_change_hints": [],
            "scene_change_candidates": [],
            "scene_change_error": None,
            "error": None,
        },
        "confirmation_cache": {
            "path": str(confirmation_cache_path),
            "hits": 0,
            "writes": 0,
            "write_error": None,
            "entry_count": 0,
        },
        "candidate_reviews": [],
        "confirmation_candidates": [],
        "results": [],
        "deduplication": {
            "strategy": "complete_link_event_core",
            "iou_threshold": EVENT_DEDUPE_IOU,
            "overlap_threshold": EVENT_DEDUPE_OVERLAP,
            "proposal_count": 0,
            "cluster_count": 0,
            "decisions": [],
        },
        "consolidation": {
            "enabled": False,
            "replaced_by": "core_deduplication",
            "max_gap": merge_confirmed_gap,
            "decisions": [],
        },
        "outputs": [],
    }

    with tempfile.TemporaryDirectory(prefix="evidence-pipeline-") as temp_name:
        temp_dir = Path(temp_name)
        if visual_fallback:
            try:
                scene_hints = detect_scene_change_hints(source)
            except (OSError, subprocess.CalledProcessError) as exc:
                scene_hints = []
                report["visual_fallback"]["scene_change_error"] = str(exc)
            report["visual_fallback"]["scene_change_hints"] = scene_hints
            scan_prompt = build_visual_scan_prompt(
                event,
                interval=visual_scan_interval,
                max_candidates=max_candidates,
                time_hints=candidates,
                scene_hints=scene_hints,
            )
            scan_cache_identity = visual_scan_cache_identity(
                source=source,
                source_sha256=source_sha256,
                prompt=scan_prompt,
                interval=visual_scan_interval,
            )
            scan_cache = load_visual_scan_cache(
                scan_cache_path,
                identity=scan_cache_identity,
            )
            cache_write_error = None
            if scan_cache is not None:
                visual_candidates = normalize_visual_candidates(
                    scan_cache["raw"],
                    source_media["duration"],
                    padding=visual_padding,
                    max_candidates=max_candidates,
                )
                visual_scan = {"raw": scan_cache["raw"], "attempts": 0}
                scan_media = scan_cache["scan_media"]
                cache_hit = True
            else:
                scan_path = temp_dir / "visual_scan.mp4"
                scan_media = None
                try:
                    make_visual_scan(source, scan_path, visual_scan_interval)
                    scan_media = probe_media(scan_path)
                    visual_candidates, visual_scan = scan_visual_candidates(
                        scan_path,
                        event,
                        source_media["duration"],
                        interval=visual_scan_interval,
                        padding=visual_padding,
                        max_candidates=max_candidates,
                        attempts=visual_scan_attempts,
                        api_key=api_key,
                        time_hints=candidates,
                        scene_hints=scene_hints,
                        prompt=scan_prompt,
                    )
                except Exception as exc:
                    report["visual_fallback"].update(
                        {
                            "used": True,
                            "scan_media": scan_media,
                            "error": str(exc),
                        }
                    )
                    if require_visual_scan_success or not candidates:
                        report["status"] = "visual_scan_failed"
                        report_path = write_report(output_dir, source, report)
                        raise RuntimeError(
                            f"补充视觉扫描失败，完整流程已停止：{exc}；报告：{report_path}"
                        ) from exc
                    visual_candidates = []
                    visual_scan = {"raw": None, "attempts": visual_scan_attempts}
                cache_hit = False
                if isinstance(visual_scan.get("raw"), dict):
                    try:
                        write_visual_scan_cache(
                            scan_cache_path,
                            identity=scan_cache_identity,
                            raw=visual_scan["raw"],
                            scan_media=scan_media,
                        )
                    except OSError as exc:
                        cache_write_error = str(exc)

            if visual_scan["raw"] is not None:
                scene_candidates = make_scene_change_candidates(
                    scene_hints,
                    source_media["duration"],
                    covered_by=[*candidates, *visual_candidates],
                    padding=visual_padding,
                    max_candidates=max_candidates,
                )
                report["visual_fallback"] = {
                    "enabled": True,
                    "mode": "supplemental",
                    "used": True,
                    "scan_media": scan_media,
                    "raw": visual_scan["raw"],
                    "candidates": visual_candidates,
                    "attempts": visual_scan["attempts"],
                    "cache_hit": cache_hit,
                    "cache_path": str(scan_cache_path),
                    "cache_write_error": cache_write_error,
                    "scene_change_hints": scene_hints,
                    "scene_change_candidates": scene_candidates,
                    "scene_change_error": report["visual_fallback"][
                        "scene_change_error"
                    ],
                    "error": None,
                }
                candidates = fuse_candidates(
                    audio_candidates,
                    [*visual_candidates, *scene_candidates],
                    max_candidates=max_candidates,
                )
        candidates = split_confirmation_candidates(candidates)
        report["confirmation_candidates"] = candidates
        if not candidates:
            report["status"] = "no_candidate"
            report_path = write_report(output_dir, source, report)
            return report, report_path

        try:
            (
                candidate_reviews,
                results,
                deduplication,
                confirmation_cache,
            ) = process_confirmed_candidates(
                source,
                temp_dir,
                output_dir,
                candidates,
                event,
                source_duration=source_media["duration"],
                source_sha256=source_sha256,
                confirmation_cache_path=confirmation_cache_path,
                event_padding=event_padding,
                retry_padding=retry_padding,
                min_confidence=min_confidence,
                final_review=final_review,
                api_key=api_key,
            )
        except Exception as exc:
            report["status"] = "processing_failed"
            report["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            report_path = write_report(output_dir, source, report)
            raise RuntimeError(
                f"候选确认或事件出片失败：{exc}；报告：{report_path}"
            ) from exc
        report["candidate_reviews"] = candidate_reviews
        report["results"] = results
        report["deduplication"].update(deduplication)
        report["confirmation_cache"].update(confirmation_cache)

        report["outputs"] = [
            item["clip"]["path"]
            for item in report["results"]
            if item.get("status") == "completed"
        ]

    if any(item.get("status") == "error" for item in report["results"]):
        report["status"] = "completed_with_errors"
    else:
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
        if args.merge_confirmed_gap < 0:
            raise RuntimeError("merge-confirmed-gap 不能小于 0")
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
            merge_confirmed_gap=args.merge_confirmed_gap,
            visual_fallback=not args.no_visual_fallback,
            visual_scan_interval=args.visual_scan_interval,
            visual_padding=args.visual_padding,
            visual_scan_attempts=args.visual_scan_attempts,
            require_visual_scan_success=False,
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
