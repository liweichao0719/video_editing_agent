# Video Editing Agent

[![Tests](https://github.com/liweichao0719/video_editing_agent/actions/workflows/tests.yml/badge.svg)](https://github.com/liweichao0719/video_editing_agent/actions/workflows/tests.yml)

面向通用视频剪辑 Agent 的实验项目。当前阶段通过“声音事件取证”验证最小闭环：
理解素材 → 定位片段 → 执行剪辑 → 检查结果 → 必要时修正。

项目来源、定位演进、资源判断及未来公开时的致谢要求见
[项目背景](docs/PROJECT_CONTEXT.md)。

## 目录

```text
docs/        项目背景与决策记录
evaluation/  评测清单、数据来源和结果
samples/     本地演示素材（不提交视频文件）
tests/       单元测试与小型音频夹具
web/         本地演示页面
models/      下载的本地模型（不提交）
outputs/     生成的剪辑和报告（不提交）
```

## 环境准备

需要 Python 3.12、FFmpeg 和 ffprobe。创建虚拟环境并安装完整运行依赖：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

本地演示视频和 YAMNet 模型不会提交到 Git。素材下载方式见
[samples/README.md](samples/README.md)，模型安装方式见下方“音频候选定位”。

## 快速开始

```bash
source .venv/bin/activate
export ARK_API_KEY='你的方舟 API Key'
python analyze_video.py
```

分析本地文件：

```bash
python analyze_video.py --video-file samples/test_blender_av.webm
```

脚本默认使用 Wikimedia Commons 的
[Kitchen blender](https://commons.wikimedia.org/wiki/File:Kitchen_blender.webm)
作为测试素材。署名 Sounds of Changes，CC BY 3.0。

API Key 只从环境变量读取，不保存在本项目中。

启动带 API 的页面：

```bash
export ARK_API_KEY='你的方舟 API Key'
python server.py
```

服务默认只监听 `127.0.0.1:8000`，并且仅公开页面所需的静态文件。

音频候选定位：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models/yamnet
curl -L 'https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed' | tar -xz -C models/yamnet
.venv/bin/python audio_candidates.py \
  --video tests/fixtures/glass_shattering_cc0.mp3 \
  --event 玻璃破碎 \
  --output outputs/
```

候选定位使用官方 YAMNet；结果仅用于提出候选时间段，最终事件仍需多模态模型确认。
早期测试素材 `samples/test_glass_av.mp4` 的音轨实际为静音，未纳入当前评测集。

运行完整取证闭环：

```bash
export ARK_API_KEY='你的方舟 API Key'
.venv/bin/python pipeline.py \
  --video your_video.mp4 \
  --event 玻璃破碎 \
  --output outputs/
```

流程会执行音频定位、FFmpeg 本地强画面变化检测，以及每 2 秒一次的稀疏视觉扫描。
抽取帧会压缩成短视频并保留原始时间戳；声音时间和画面变化时间都会提示视觉模型完整
检查多次事件。视觉扫描的瞬时网络错误会退避重试；成功结果写入独立缓存，并校验源文件
SHA-256、实际提示、模型和接口身份后才复用，不会覆盖最终报告。逐候选确认结果也会在每次
成功响应后立即缓存，因此后续候选超时不会丢掉已经完成的远端工作。

每个候选一次返回其中所有独立事件的紧核心边界。系统先对未加上下文的核心做全局去重，
禁止链式合并，也不会合并同一宽候选里明确列出的近邻事件；去重后再加默认 1 秒上下文，
相邻事件的上下文以中点为界，避免成品互相吞并。最后执行 FFmpeg 剪辑和聚焦该核心的成品
复检。报告同时保留 `event_bounds`（事件核心）、`clip`（成品范围）和证据来源。

可用 `--visual-scan-interval 3` 调整抽帧间隔、`--event-padding 0.5` 调整事件上下文，
`--visual-scan-attempts 1` 调整尝试次数，或用 `--no-visual-scan` 关闭补充视觉扫描。
旧参数 `--no-visual-fallback` 仍然兼容。

运行测试：

```bash
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -m unittest discover -s tests -v
```

没有下载 YAMNet 或本地视频时，相关集成测试会自动跳过；其余核心测试仍会执行。
GitHub Actions 在每次推送到 `main` 以及每个 Pull Request 上运行这组核心测试。

批量评测与标注格式见 [evaluation/README.md](evaluation/README.md)。

## 当前结果

现有小型真实评测集共 10 条样本。2026-08-18 的首轮结果基于旧的 9 个宽时间段标注，
只保留作历史基线；当前清单已逐帧修订为 11 个紧事件核心，指标口径也增加了事件精确率、
严格 `IoU >= 0.5` 和独立成片覆盖率，因此两版数字不能直接比较。

新版完整 10 样本复评尚未完成：定向样本的远端确认多次超时，已完成结果会由缓存保留，
但在全量评测结束前不发布新的准确率结论。历史结果及其限制见
[evaluation/RESULTS.md](evaluation/RESULTS.md)。

## 致谢与许可证

项目在概念上受到 [OmniGAIA](https://github.com/RUC-NLPIR/OmniGAIA) 的主动多模态感知与
工具协作思路启发。原创代码采用 [MIT License](LICENSE)；测试素材、外部模型和第三方
依赖沿用各自许可证，不包含在本项目 MIT 授权范围内。素材来源和许可记录见
`evaluation/data/metadata.json`。
