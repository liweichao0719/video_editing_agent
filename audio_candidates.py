#!/usr/bin/env python3
"""Locate audio-event candidate windows in a video with YAMNet."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "yamnet"
DEFAULT_EVENT_CONFIG = ROOT / "event_labels.json"
SAMPLE_RATE = 16_000
FRAME_DURATION = 0.96
FRAME_HOP = 0.48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use YAMNet to locate candidate audio events in a video"
    )
    parser.add_argument(
        "--video", type=Path, required=True, help="带音轨的视频或音频文件"
    )
    parser.add_argument("--event", default="玻璃破碎")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON file or output directory; omit to print to stdout",
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--padding", type=float, default=0.50)
    parser.add_argument("--merge-gap", type=float, default=0.24)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--event-config", type=Path, default=DEFAULT_EVENT_CONFIG)
    return parser.parse_args()


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"缺少可执行程序：{name}")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def video_duration(video: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ]
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_audio(video: Path, wav_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )


def load_event_labels(config_path: Path, event: str) -> list[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalized = event.strip().casefold()
    for item in config["events"]:
        names = [item["name"], *item.get("aliases", [])]
        if normalized in {name.casefold() for name in names}:
            return item["yamnet_labels"]
    return [event]


def load_class_names(csv_path: Path) -> list[str]:
    with csv_path.open(encoding="utf-8") as file:
        return [row["display_name"] for row in csv.DictReader(file)]


def format_time(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{remainder:05.2f}"


def merge_frames(
    frames: list[dict], duration: float, padding: float, merge_gap: float
) -> list[dict]:
    merged: list[dict] = []
    for frame in frames:
        if merged and frame["start"] <= merged[-1]["end"] + merge_gap:
            current = merged[-1]
            current["end"] = max(current["end"], frame["end"])
            current["score"] = max(current["score"], frame["score"])
            current["frame_count"] += 1
            if frame["score"] >= current["score"]:
                current["top_label"] = frame["top_label"]
            for label, score in frame["label_scores"].items():
                current["label_scores"][label] = max(
                    current["label_scores"].get(label, 0.0), score
                )
        else:
            merged.append(
                {
                    "start": frame["start"],
                    "end": frame["end"],
                    "score": frame["score"],
                    "top_label": frame["top_label"],
                    "label_scores": dict(frame["label_scores"]),
                    "frame_count": 1,
                }
            )

    for candidate in merged:
        candidate["start"] = round(max(0.0, candidate["start"] - padding), 3)
        candidate["end"] = round(min(duration, candidate["end"] + padding), 3)
        candidate["score"] = round(candidate["score"], 4)
        candidate["label_scores"] = {
            label: round(score, 4)
            for label, score in sorted(candidate["label_scores"].items())
        }
        candidate["time"] = (
            f"{format_time(candidate['start'])}-{format_time(candidate['end'])}"
        )
    return merged


def detect_candidates(
    wav_path: Path,
    model_path: Path,
    labels: list[str],
    threshold: float,
    duration: float,
    padding: float,
    merge_gap: float,
) -> tuple[list[dict], dict[str, float], list[dict]]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "缺少 TensorFlow；请运行 uv pip install --python .venv/bin/python -r requirements.txt"
        ) from exc

    model = tf.saved_model.load(str(model_path))
    waveform, sample_rate = tf.audio.decode_wav(
        tf.io.read_file(str(wav_path)), desired_channels=1
    )
    if int(sample_rate.numpy()) != SAMPLE_RATE:
        raise RuntimeError(f"音频采样率不是 {SAMPLE_RATE}Hz")
    waveform = tf.squeeze(waveform, axis=-1)
    output = model.signatures["serving_default"](waveform=waveform)
    scores = output["output_0"].numpy()

    class_names = load_class_names(model_path / "assets" / "yamnet_class_map.csv")
    name_to_index = {name.casefold(): index for index, name in enumerate(class_names)}
    missing = [label for label in labels if label.casefold() not in name_to_index]
    if missing:
        raise RuntimeError("YAMNet 不包含标签：" + ", ".join(missing))

    indices = {label: name_to_index[label.casefold()] for label in labels}
    max_scores = {
        label: float(scores[:, index].max()) for label, index in indices.items()
    }
    overall_max = scores.max(axis=0)
    overall_frame = scores.argmax(axis=0)
    top_indices = overall_max.argsort()[-10:][::-1]
    top_classes = [
        {
            "label": class_names[index],
            "score": round(float(overall_max[index]), 4),
            "time": format_time(float(overall_frame[index]) * FRAME_HOP),
        }
        for index in top_indices
    ]
    frames = []
    for frame_index, row in enumerate(scores):
        label_scores = {label: float(row[index]) for label, index in indices.items()}
        top_label = max(label_scores, key=label_scores.get)
        score = label_scores[top_label]
        if score < threshold:
            continue
        start = frame_index * FRAME_HOP
        frames.append(
            {
                "start": start,
                "end": min(duration, start + FRAME_DURATION),
                "score": score,
                "top_label": top_label,
                "label_scores": label_scores,
            }
        )
    return (
        merge_frames(frames, duration, padding, merge_gap),
        {label: round(score, 4) for label, score in max_scores.items()},
        top_classes,
    )


def output_path_for(argument: Path, video: Path) -> Path:
    if argument.suffix.casefold() == ".json":
        return argument
    return argument / f"{video.stem}_candidates.json"


def locate_candidates(
    video: Path,
    event: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    event_config: Path = DEFAULT_EVENT_CONFIG,
    threshold: float = 0.10,
    padding: float = 0.50,
    merge_gap: float = 0.24,
) -> dict:
    duration = video_duration(video)
    labels = load_event_labels(event_config, event)
    with tempfile.TemporaryDirectory(prefix="audio-candidates-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        extract_audio(video, wav_path)
        candidates, max_scores, top_classes = detect_candidates(
            wav_path,
            model_path,
            labels,
            threshold,
            duration,
            padding,
            merge_gap,
        )
    return {
        "source": str(video),
        "event": event,
        "duration": round(duration, 3),
        "detector": {
            "name": "YAMNet",
            "model": "google/yamnet/1",
            "sample_rate": SAMPLE_RATE,
            "frame_duration": FRAME_DURATION,
            "frame_hop": FRAME_HOP,
            "labels": labels,
            "threshold": threshold,
            "max_scores": max_scores,
            "top_classes": top_classes,
        },
        "candidates": candidates,
    }


def main() -> int:
    args = parse_args()
    try:
        require_executable("ffmpeg")
        require_executable("ffprobe")
        video = args.video.resolve()
        if not video.is_file():
            raise RuntimeError(f"视频不存在：{video}")
        model_path = args.model_path.resolve()
        if not (model_path / "saved_model.pb").is_file():
            raise RuntimeError(f"YAMNet 模型不存在：{model_path}")
        if not 0.0 <= args.threshold <= 1.0:
            raise RuntimeError("threshold 必须在 0 到 1 之间")

        report = locate_candidates(
            video,
            args.event,
            model_path=model_path,
            event_config=args.event_config,
            threshold=args.threshold,
            padding=args.padding,
            merge_gap=args.merge_gap,
        )
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output is None:
            print(encoded)
        else:
            destination = output_path_for(args.output, video)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded + "\n", encoding="utf-8")
            print(destination)
        return 0
    except (RuntimeError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
