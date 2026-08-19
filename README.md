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
当前 `samples/test_glass_av.mp4` 的音轨实际为静音，不能作为声音检测的正样本。

运行完整取证闭环：

```bash
export ARK_API_KEY='你的方舟 API Key'
.venv/bin/python pipeline.py \
  --video your_video.mp4 \
  --event 玻璃破碎 \
  --output outputs/
```

流程会执行音频定位和每 2 秒一次的稀疏视觉扫描。抽取帧会压缩成短视频并保留原始时间戳，
视觉扫描请求失败时默认重试一次；随后合并两种候选并进行豆包确认、
FFmpeg 剪辑和成品复检。这样即使视频中已有声音候选，也不会漏掉同一视频里声音不明显的
其他事件。模型给出的片段内事件边界会加上 1 秒上下文后用于剪辑；只有确认和复检通过的
事件才会保留，并同时生成 `pipeline_report.json`。

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

现有小型真实评测集共 10 条样本：分类结果 10/10，9 个可评估正样本中有 7 个时间段
达到当前匹配标准。样本量仍然很小且场景偏向玻璃破碎，不能视为通用准确率。

针对原先漏检的两个多事件视频，2026-08-19 的定向回归从 2/4 提升到 4/4 个事件匹配，
平均 IoU 从 0.2979 提升到 0.6131。该结果尚未替代完整 10 样本评测，详见
[evaluation/RESULTS.md](evaluation/RESULTS.md)。

## 致谢与许可证

项目在概念上受到 [OmniGAIA](https://github.com/RUC-NLPIR/OmniGAIA) 的主动多模态感知与
工具协作思路启发。原创代码采用 [MIT License](LICENSE)；测试素材、外部模型和第三方
依赖沿用各自许可证，不包含在本项目 MIT 授权范围内。素材来源和许可记录见
`evaluation/data/metadata.json`。
