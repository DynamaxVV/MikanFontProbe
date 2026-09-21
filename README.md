# MikanFontProbe

MikanFontProbe 是一个离线字体识别实验项目：从 MikanQuiz 题库取得日文漫画文字截图，保留原图证据，研究文字区域收紧、逐字裁切、日文字形匹配和字体族 Top-3 候选。题库给出的中文字体名只是粗粒度代理标签，不是日文原字体的真值。

字体样本的版权与再分发权限归原权利人所有；本仓库不对字体重新授权，详见 [FONT_SAMPLES_NOTICE.md](docs/FONT_SAMPLES_NOTICE.md)。目录边界、保留规则和本轮清理记录见 [PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md)。

## 先看什么

| 目的 | 文档 |
| --- | --- |
| 当前结果、默认入口和限制 | 本 README、[RANKED_RESULTS.md](docs/RANKED_RESULTS.md) |
| 8 类、题量超过 5 的主验证 | [CATEGORY_EXPANSION_REPORT.md](docs/CATEGORY_EXPANSION_REPORT.md) |
| 未参与调参的新题与样本审计 | [ALIGNED_HOLDOUT_REPORT.md](docs/ALIGNED_HOLDOUT_REPORT.md)、[FRESH_SAMPLE_AUDIT.md](docs/FRESH_SAMPLE_AUDIT.md) |
| 字体样本、假名优先选字 | [FONT_BASELINE.md](docs/FONT_BASELINE.md)、[GLYPH_DISCRIMINATION.md](docs/GLYPH_DISCRIMINATION.md) |
| 轻吟体失败与保守修正 | [HUMMING_RULE_REPORT.md](docs/HUMMING_RULE_REPORT.md)、[HUMMING_VISUAL_REVIEW.md](docs/HUMMING_VISUAL_REVIEW.md) |
| 切割与输入策略 | [AB_REPORT.md](docs/AB_REPORT.md)、[TYPOGRAPHIC_REPORT.md](docs/TYPOGRAPHIC_REPORT.md)、[INPUT_ABLATION.md](docs/INPUT_ABLATION.md)、[STRUCTURE_BASELINE.md](docs/STRUCTURE_BASELINE.md) |
| 可解释的局部特征实验 | [CONTOUR_FAILURE_REPORT.md](docs/CONTOUR_FAILURE_REPORT.md)、[STROKE_FEATURE_REPORT.md](docs/STROKE_FEATURE_REPORT.md)、[CORRESPONDENCE_V7_REPORT.md](docs/CORRESPONDENCE_V7_REPORT.md) |
| 共享局部 CNN 及隔离优化 | [LOCAL_ENCODER_V11_REPORT.md](docs/LOCAL_ENCODER_V11_REPORT.md)、[LOCAL_ENCODER_V13_REPORT.md](docs/LOCAL_ENCODER_V13_REPORT.md) |

## 当前结论

- 默认识别仍采用受保护的 v4 风格全字匹配；可复现入口是 `.venv/bin/python -m mikan_font_probe.rank_font_guarded`。输出中的置信度是未校准证据指数，不是正确概率。
- 8 类扩展验证的 44 道合格题中，清晰度优先为 34/44，主动选字为 32/44，主动 Top-3 为 41/44；真正新题的两种 Top-1 都是 19/26，因此没有证明主动选字稳定优于清晰度基线。
- 统一处理链的 8 道严格 7 字新题中，两种策略均为 7/8；q11 白字黑底且只有两个可靠不同字，作为域外案例保留，不计入主指标。
- 轻吟体是当前最明显的困难族。保守的有效分辨率/骨架重排在已暴露开发集上改善了部分题目，但 q106、q108、q109 仍需更多真实字重或原字体证据，不能靠规则奖励强行改判。
- v11/v13 的共享局部编码器是诊断和候选召回实验。v13 只保留字族内部 Top-2 字重聚合作为研究结论；它没有替换默认入口，也没有把合成留出结果当作真实漫画准确率。

## 快速开始

安装依赖：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

从部署中的 MikanQuiz 服务取得题库元数据和图片。凭据只在内存中使用，不写入磁盘：

```sh
MK_ADMIN_USER=... MK_ADMIN_PASS=... \
  .venv/bin/python -m mikan_font_probe.fetch_dataset
```

对已有优先样本运行区域处理和排版切分：

```sh
.venv/bin/python -m mikan_font_probe.process_regions \
  --version priority-v7 \
  --annotations data/priority-a.json data/priority-b.json \
  --ids 118 115 100 125 76 62 58 96 65 114 92 70 91 61 33 72 27 140
.venv/bin/python -m mikan_font_probe.segment_text \
  --version priority-v7 \
  --annotations data/priority-a.json data/priority-b.json
```

推荐的默认字体候选入口：

```sh
.venv/bin/python -m mikan_font_probe.rank_font_guarded --qid 46
```

全量回归：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover
```

## 处理链和证据

1. 题库截图及题号/中文标签写入 `data/manifest.json`；原始图片在 `data/images/`。
2. 视觉模型只负责给出粗定位和字符线索；OpenCV 在原图上做阈值、投影、连通域、白描边辅助和按列排版切分，并保留原图坐标与叠加图。
3. 在字符已知的离线验证中，字体匹配将源字与同字符模板等比对齐，结合二值/灰度、局部结构和有限位移；字体族合并字重，但少女/方圆/海报保持独立。
4. 假名区分表先统计不同字体在多个有效分辨率下的渲染位图差异，用于选取理论上更有区分力的字；共享或低分辨率不稳定的字不投强支持票。
5. 每个阶段都保留原图、裁片、JSON、冻结协议和哈希。报告里的 Top-1/Top-3 都是指定样本池上的实验指标，不等于整页检出率、OCR 识字率或生产概率。

## 数据和边界

`data/` 保留当前基线、留出验证、类别扩展、失败案例和已删除报告对应的历史冻结证据。`output/pdf/font-recognition-report.pdf` 是当前图文报告。字体文件位于 `font_sample/`，作为输入资产不参与目录清理。

当前实验大多使用人工预裁或人工核对的字符；尚未证明整页漫画中的框内正文、框外规整字和手写体能被稳定自动区分，也没有把 OCR 误读混入字体准确率。低分辨率、白字反色、描边、背景纹理、未知字体和真实作品间独立性仍是主要风险。对背景复杂、切割不合格或字库覆盖不足的样本，正确行为是拒判/请求复核，而不是输出高置信字体。

## 项目结构

Python 入口统一位于 `mikan_font_probe/`，测试位于 `tests/`，调用方式统一为 `python -m mikan_font_probe.<module>`。旧冻结记录中的源码相对路径由 `mikan_font_probe.paths.resolve_repo_path()` 兼容。agent 的协作约定见 [AGENTS.md](AGENTS.md)，更完整的目录和实验保留规则见 [PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md)。
