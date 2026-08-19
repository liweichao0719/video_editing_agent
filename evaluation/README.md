# 批量评测

`dev_manifest.json` 只是开发冒烟集，包含一个构造正样本和一个真实负样本，不能用于宣称模型准确率。`real_manifest.json` 是首轮真实素材集，包含 5 个正样本和 5 个易混淆负样本。

真实素材不提交到仓库。下载前先查看 `source_catalog.json`，然后运行：

```bash
.venv/bin/python evaluation/download_data.py
```

下载脚本会核对 Wikimedia Commons 元数据、裁剪长视频并生成 `data/metadata.json`，其中保存来源链接、作者、许可、文件哈希和裁剪时间。无音轨或实质静音的候选素材记录在 `source_catalog.json` 的排除列表中。

每个样本需要填写：

- `id`：唯一名称。
- `path`：相对于清单 `root` 的视频路径。
- `event`：可省略，默认使用清单的 `default_event`。
- `expected`：视频是否包含目标事件。
- `events`：人工标注的事件起止秒数；正样本至少一个。
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

程序每完成一个样本就保存一次，`--resume` 可以跳过已有结果。报告包含准确率、精确率、召回率、F1、时间段 IoU、起止时间误差和完整覆盖率。

完整流程默认启用视觉兜底：仅当声音阶段没有候选时生成稀疏无声扫描视频，画面左上角保留原视频时间；视觉初筛得到的时间段仍需经过精确音画确认和成品复检。可通过 `--visual-scan-interval`、`--visual-padding` 调整，或用 `--no-visual-fallback` 关闭。

当前真实集仍然很小：正样本只来自两个源事件组，其中包含慢动作和重放版本；`Drones vs. Windows` 的 Commons 页面还标记为许可待复核。因此它只适合发现明显问题，不能代表线上准确率，也不应直接作为可再分发数据集。首轮结果见 `RESULTS.md`。
