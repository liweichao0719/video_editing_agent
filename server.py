#!/usr/bin/env python3
"""Local API server for the multimodal data-flow demo."""

from __future__ import annotations

import base64
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_MODEL = "doubao-seed-2-1-turbo-260628"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
PUBLIC_PATHS = {
    "/web/index.html",
    "/samples/test_blender_av.webm",
    "/baselines/blender.json",
}


def extract_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"summary": cleaned, "events": []}


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def prepare_static_request(self) -> bool:
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            path = "/web/index.html"
        if path not in PUBLIC_PATHS:
            self.send_error(404, "File not found")
            return False
        self.path = path
        return True

    def do_GET(self) -> None:
        if self.prepare_static_request():
            super().do_GET()

    def do_HEAD(self) -> None:
        if self.prepare_static_request():
            super().do_HEAD()

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_json(404, {"error": "接口不存在"})
            return
        api_key = os.environ.get("ARK_API_KEY", "").strip()
        if not api_key:
            self.send_json(500, {"error": "服务端缺少 ARK_API_KEY"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_json(400, {"error": "视频为空或超过 25MB"})
            return

        video = self.rfile.read(length)
        media_type = self.headers.get("Content-Type", "video/mp4").split(";", 1)[0]
        question = unquote(self.headers.get("X-Question", "视频中发生了什么？声音与画面动作有什么对应关系？"))
        data_url = f"data:{media_type};base64," + base64.b64encode(video).decode("ascii")
        payload = {
            "model": os.environ.get("ARK_MODEL", DEFAULT_MODEL),
            "temperature": 0.1,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": data_url}},
                {"type": "text", "text": question + "\n只输出 JSON，包含 summary 和 events；events 每项包含 time、visual、audio、relation。"},
            ]}],
        }
        request = urllib.request.Request(
            os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL).rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                result = json.load(response)
            content = result["choices"][0]["message"]["content"]
            self.send_json(200, {"result": extract_json(content)})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.send_json(502, {"error": "模型调用失败", "detail": detail})
        except (urllib.error.URLError, TimeoutError) as exc:
            self.send_json(504, {"error": f"模型请求超时：{exc}"})
        except Exception as exc:
            self.send_json(500, {"error": f"处理失败：{exc}"})


if __name__ == "__main__":
    host = os.environ.get("SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("SERVER_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Demo running at http://{host}:{port}", flush=True)
    server.serve_forever()
