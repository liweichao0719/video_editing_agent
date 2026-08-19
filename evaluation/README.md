# 批量评测

`dev_manifest.json` 只是开发冒烟集，包含一个构造正样本和一个真实负样本，不能用于宣称模型准确率。`real_manifest.json` 是首轮真实素材集，包含 5 个正样本和 5 个易混淆负样本。

真实素材不提交到仓库。下载前先查看 `source_catalog.json`，然后运行：

```bash
.venv/bin/python evaluation/download_data.py
```

下载脚本会核对 Wikimedia Commons 元数据、裁剪长视频并生成 `data/metadata.json`，其中保存来源链接、作者、许可、文件哈希和裁剪时间。无音轨或实质静音的候选素材记录在 `source_catalog.json` 的排除列表中。

透明玻璃在 240p 版本中会丢失关键碎片细节，因此 `Drones vs. Windows` 的三个裁剪样本
统一使用 480p VP9 衍生版；下载器允许的单个来源上限为 150MB，并会校验预期大小、记录哈希。

每个样本需要填写：

- `id`：唯一名称。
- `path`：相对于清单 `root` 的视频路径。
- `event`：可省略，默认使用清单的 `default_event`。
- `expected`：视频是否包含目标事件。
- `events`：人工逐帧标注的事件核心起止秒数；从首次新变化到主要变化结束，不含接近、
  稳定破损或余波。慢放/重放按视频时间轴上的 source occurrence 分开计数。
- `split`：例如 `dev`、`test`。
- `kind`：建议填写 `real` 或 `synthetic`。

只评测声音候选定位，不调用豆包：

```bash
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python evaluate.py \
  --manifest evaluation/real_manifest.json \
  --stage audio \
  --output outputs/evaluation_real_audio.json
```

评测完整闭环：

```bash
export ARK_API_KEY='你的方舟 API Key'
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python evaluate.py \
  --manifest evaluation/real_manifest.json \
  --stage full \
  --output outputs/evaluation_real_full_visual_fallback.json \
  --resume
```

程序每完成一个样本就保存一次，`--resume` 可以跳过成功结果并重试失败样本；续跑会核对
清单、媒体 SHA-256、本地模型、流程版本和远端模型身份，避免混用不可比结果。报告同时包含
样本级分类指标，以及事件级一对一匹配的精确率、召回率和 F1；事件定位默认报告
`IoU >= 0.1`，并额外报告严格的 `IoU >= 0.5`，还包括平均 IoU、起止时间误差和完整覆盖率。
IoU 使用未加上下文的 `event_bounds`，覆盖率则使用实际输出的 `clip` 范围。
排查单条素材时可重复使用 `--sample-id 样本ID` 精确选择样本。

完整流程默认启用补充视觉扫描：无论声音阶段是否已有候选，都会生成稀疏无声扫描视频，
画面左上角保留原视频时间。本地画面变化检测会补充远端粗筛漏掉的短暂撞击，并过滤已经被
声音或视觉候选覆盖的时间。每个候选一次返回其中所有独立事件的紧核心；系统使用全局
complete-link 规则去除跨候选的明确重复，禁止链式合并，并保留同一宽候选里的独立近邻。
去重后才添加上下文，相邻核心的上下文以中点为界，再进行聚焦该核心的成品复检。可通过
`--visual-scan-interval`、`--visual-padding`、`--visual-scan-attempts`、
`--event-padding` 调整，或用
`--no-visual-scan` 关闭。

完整流程评测中如果视觉扫描超时或失败，该样本会记为评测错误而不是模型漏检，之后可用
`--resume` 重试。成功的视觉粗筛写入独立 `_visual_scan_cache.json`；只有源内容、实际提示、
模型和接口身份全部一致才会命中。逐候选成功确认也立即写入 `_confirmation_cache.json`，
因此后续接口超时后重试时，不必重做已经完成的远端工作；两种缓存都不会覆盖最终报告。

当前真实集仍然很小：10 条样本中只有 5 条正样本、共 11 个 source occurrence，且只来自
两个源事件组，其中包含慢动作和重放版本；`Drones vs. Windows` 的 Commons 页面还标记为
许可待复核。因此它只适合发现明显问题，不能代表线上准确率，也不应直接作为可再分发数据集。
首轮结果见 `RESULTS.md`。
