# 项目目录与实验保留规则

本文件定义离线字体实验的目录边界。整理目标是让当前可复用入口、重要验证证据和历史诊断数据分层，同时不因为删除报告而破坏冻结 JSON、原图、字体样本或代码复现能力。

## 当前结构

```text
MikanFontProbe/
├── README.md                  使用说明、当前结论和文档索引
├── AGENTS.md                  agent 协作、实验和验证约定
├── docs/                      维护中的基线、验证、失败案例和算法报告
│   ├── PROJECT_LAYOUT.md      目录边界、保留规则和清理记录
│   └── *.md                   其余维护文档
├── mikan_font_probe/          离线入口、共享算法模块和路径兼容层
├── tests/                     回归测试与共享合成夹具
├── data/
│   ├── images/                从 MikanQuiz 获取的原始题图
│   ├── processed/             各代区域/遮罩/切分结果
│   ├── local-encoder-v11/     共享局部编码器基线及评估
│   ├── local-encoder-v13/     v11 上的本地隔离诊断产物，不提交
│   └── <版本目录>/             留出、字体区分、失败案例和历史冻结证据
├── font_sample/               日文字体样本，属于输入资产
├── output/                    最终图文报告
├── .venv/                     本地依赖环境
└── .omx/                      运行时状态
```

`docs/` 内的 Markdown 按职责保留，而不是按每一次小实验各生成一份报告：

- 基础与当前结果：`docs/FONT_BASELINE.md`、`docs/RANKED_RESULTS.md`、`docs/GLYPH_DISCRIMINATION.md`。
- 新样本与范围验证：`docs/HOLDOUT_REPORT.md`、`docs/FRESH_SAMPLE_AUDIT.md`、`docs/ALIGNED_HOLDOUT_REPORT.md`、`docs/CATEGORY_EXPANSION_REPORT.md`。
- 区域/输入策略：`docs/AB_REPORT.md`、`docs/TYPOGRAPHIC_REPORT.md`、`docs/INPUT_ABLATION.md`、`docs/STRUCTURE_BASELINE.md`。
- 失败案例与可解释局部特征：`docs/HUMMING_RULE_REPORT.md`、`docs/HUMMING_VISUAL_REVIEW.md`、`docs/CONTOUR_FAILURE_REPORT.md`、`docs/STROKE_FEATURE_REPORT.md`、`docs/CORRESPONDENCE_V7_REPORT.md`。
- 学习模型诊断：`docs/LOCAL_ENCODER_V11_REPORT.md`、`docs/LOCAL_ENCODER_V13_REPORT.md`。
- 资产说明：`docs/FONT_SAMPLES_NOTICE.md`。

## 当前应保留的入口

- `mikan_font_probe/rank_font_guarded.py`：当前受保护的字体排名入口，默认仍以 v4 风格全字匹配为基础。
- `mikan_font_probe/process_regions.py`、`mikan_font_probe/segment_text.py`、`mikan_font_probe/typographic_regions.py`：原图区域、列/字候选和排版先验处理。
- `mikan_font_probe/font_family_policy.py`、`mikan_font_probe/build_glyph_discrimination.py`：字体族映射和假名优先区分表。
- `mikan_font_probe/local_encoder_v11.py`、`mikan_font_probe/train_local_encoder_v11.py`、`mikan_font_probe/evaluate_local_encoder_v11.py`：共享局部编码器基线。
- `mikan_font_probe/local_encoder_v13.py`、`mikan_font_probe/local_features_v13.py` 及对应训练/评估入口：冻结的 v13 诊断实验，不改变默认排名。

Python 模块统一位于 `mikan_font_probe/`，测试统一位于 `tests/`；命令行入口使用 `python -m mikan_font_probe.<module>`。`mikan_font_probe.paths.resolve_repo_path()` 兼容历史冻结记录中仍使用的旧源码相对路径。

## 数据保留规则

1. 当前基线、有效留出、类别扩展、失败案例和模型实验保留入口、最小回归测试、冻结协议/manifest、评估 JSON 及报告。
2. 历史诊断数据可以保留以支持哈希审计，但不要求每个已淘汰的小版本继续占用根目录文档名。
3. 只有在报告没有独有结论、没有下游文档引用、且结果已被更完整的验证或失败报告覆盖时，才删除中间报告；对应 `data/` 证据不因报告删除而自动删除。
4. 原始题图、字体样本、最终 PDF、冻结 JSON、审计文件和可复现代码不作为“缓存”清理。
5. `data/eval-v12.json` 与 `data/processed/v12/` 是早期切割实验，不等于已删除的局部编码器 v12；除非确认无引用，不按版本号误删。

## 本轮清理

此前已删除 6 份已被当前文档体系覆盖的中间报告：

- `RESULTS.md`：首轮取样结果，已由 `FONT_BASELINE.md` 和 `RANKED_RESULTS.md` 取代。
- `REVIEW_SOL_MEDIUM.md`：优先18题的旧复评，结论已进入 `AB_REPORT.md`/`TYPOGRAPHIC_REPORT.md` 的切割边界说明。
- `ACTIVE_SELECTION_REPORT.md`：旧主动选字逐题明细，汇总与留出边界已在 `ALIGNED_HOLDOUT_REPORT.md`、`CATEGORY_EXPANSION_REPORT.md` 保留。
- `CONTINUOUS_V8_REPORT.md`、`CONTRASTIVE_V9_REPORT.md`、`PAIR_ATTENTION_V10_REPORT.md`：连续局部、候选热区和无训练成对注意力的开发中间版本；当前可解释结论保留在 `CONTOUR_FAILURE_REPORT.md`、`STROKE_FEATURE_REPORT.md`、`CORRESPONDENCE_V7_REPORT.md`。

本轮另外将上述维护文档统一移入 `docs/`，根目录只保留 `README.md`。没有删除报告对应的 `data/`、源码、字体样本、原图或最终 PDF。移动后已重新检查 133 项全量测试、28 项文档示例测试和 132 个本地 Markdown 链接，均通过。

## 后续新增实验约定

- 报告开头明确标注 `baseline`、`diagnostic`、`rejected` 或 `frozen`。
- 每个可复用实验必须同时有入口、最小回归测试、manifest/评估结果和一份维护报告；临时探索放在 `data/` 之外或使用明确的实验目录，不在根目录堆积 Markdown。
- 新实验不再创建与旧数据版本重名的顶层目录；局部编码器使用 `data/local-encoder-vN/`，并在 README 和本文件登记。
- 新增 Python 入口放入 `mikan_font_probe/`，新增测试放入 `tests/`；根目录不再新增 `.py` 或报告 Markdown，维护文档统一进入 `docs/`。
