#!/usr/bin/env python3
"""Analyze the visual and audio content of a public video URL."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


DEFAULT_VIDEO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/5/5f/"
    "Kitchen_blender.webm"
)
DEFAULT_MODEL = "doubao-seed-2-1-turbo-260628"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze audio and video jointly")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video-url", default=None)
    source.add_argument("--video-file", type=Path, default=None)
    parser.add_argument(
        "--question",
        default="视频中发生了什么？声音与画面动作有什么对应关系？",
    )
    return parser.parse_args()


def resolve_video_url(args: argparse.Namespace) -> str:
    if args.video_file is None:
        return args.video_url or DEFAULT_VIDEO_URL

    media_type = mimetypes.guess_type(args.video_file.name)[0] or "video/webm"
    encoded = base64.b64encode(args.video_file.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def main() -> int:
    args = parse_args()
    video_url = resolve_video_url(args)
    api_key = os.environ.get("ARK_API_KEY", "").strip()
    if not api_key:
        print("缺少环境变量 ARK_API_KEY", file=sys.stderr)
        return 2

    payload = {
        "model": os.environ.get("ARK_MODEL", DEFAULT_MODEL),
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {
                        "type": "text",
                        "text": (
                            f"{args.question}\n"
                            "只输出合法 JSON："
                            '{"summary":"...","events":['
                            '{"time":"00:00-00:00","visual":"...",'
                            '"audio":"...","relation":"..."}]}'
                        ),
                    },
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
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"请求失败：{exc}", file=sys.stderr)
        return 1

    content = result["choices"][0]["message"]["content"]
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
