# MikanFontProbe

本仓库用于保存字体识别离线实验的源码、示例数据、实验报告和字体样本。字体样本的版权与再分发权限仍归原权利人所有；本仓库不对这些字体重新授权。详见 [FONT_SAMPLES_NOTICE.md](FONT_SAMPLES_NOTICE.md)。

An isolated offline experiment using MikanQuiz question images to test Japanese text extraction and font matching. The Chinese font answer is treated as a coarse proxy label; it is **not** a verified Japanese font family.

字体文件已提供，圆体／宋体／黑体的第一版同字匹配基线见 [FONT_BASELINE.md](FONT_BASELINE.md)。当前按三大类评估，字重保留为候选变体；包含攸望黑体的日文字形，允许 Antique AN+ 使用宋体样式假名。该基线不是 OCR 测试，也不是盲测准确率。

后续对照：[切割与输入方案](INPUT_ABLATION.md)、[结构评分与已知字体降质自检](STRUCTURE_BASELINE.md)。各版本保留，不把开发集排名当作已校准概率。

首次新题目验证见 [HOLDOUT_REPORT.md](HOLDOUT_REPORT.md)：12 张未用于评分调参的题目，冻结参数后进行配对验证。

字族 Top-3 与置信参考见 [RANKED_RESULTS.md](RANKED_RESULTS.md)，含全字库回归结果、复核提示及 `output/pdf/font-recognition-report.pdf` 图文报告。参考置信度未校准，不是正确概率。

最新实验：[假名字形区分表](GLYPH_DISCRIMINATION.md)及[候选驱动选字、迭代补字图文验证](ACTIVE_SELECTION_REPORT.md)。同预算开发集对照未证明整体提升，保留清晰度基线，自动早停默认关闭；逐字支持/反证解释不等于校准概率。

后续冻结验证：[统一处理链的新样本报告](ALIGNED_HOLDOUT_REPORT.md)及[样本审计](FRESH_SAMPLE_AUDIT.md)。新取9题168裁片，8题满足严格7字对照；统一归一化/匹配距离后两策略均7/8，3/5字主动策略略差，仍未证明稳定提升。q11白字黑底及少字作为域外样本单独保留。旧数据与首次新评估结果均不覆盖。

最新类别扩展：[8类48题图文报告](CATEGORY_EXPANSION_REPORT.md)。覆盖题量>5且有日文字体的8类，每类固定6题，619个原像素裁片；44题符合7字主指标，清晰度34/44、主动32/44。真正新题26题两者均19/26；轻吟体是明显薄弱项。数据、暴露分层及冻结协议保存在 `data/category-expansion-v3/`，不是94道可选题的全量评测。

轻吟体修正：[视觉诊断与保护规则报告](HUMMING_RULE_REPORT.md)、[独立视觉复核](HUMMING_VISUAL_REVIEW.md)。推荐的新离线入口为 `rank_font_guarded.py --qid 46`：7字清晰度选择、仅歧义候选进行有效分辨率/骨架重排，明确领先者不被覆盖，所有结果仍需复核。开发回归轻吟体0/6→3/6、总体34/44→38/44；后三题仍未解决。4题Folk控制经保护后与原结果相同（仅1/4正确），未隐瞒原有失败；这些控制也已用于规则设计，不能当新盲测。旧入口、原图、字体、冻结记录均未改写。

Run `MK_ADMIN_USER=... MK_ADMIN_PASS=... python3 fetch_dataset.py` to fetch question metadata and images from the deployed service. Credentials are used only in memory and are not written to disk. The script uses the system `curl` for this machine's proxy compatibility.

局部轮廓 v5 实验：[失败案例与规则作用图文报告](CONTOUR_FAILURE_REPORT.md)。新实验入口 `rank_font_contours.py --qid 109` 增加模板端点/角点窗口、同字轮廓比对和每对最多两字的补字；回归仍为38/44、轻吟3/6，未改善，未新增错误。局部距离仍混入粗细和整体形状，不能宣称已提取纯圆润特征；默认继续使用 v4。最终证据位于 `data/contours-v5/final/`，全部样本已经暴露，不是新盲测。

分部位 v6 实验：[自由笔端、连接角点与方形正证据报告](STROKE_FEATURE_REPORT.md)。入口 `rank_font_features.py --qid 109` 将自由端点与外轮廓连接角点分开建模，用邻接笔宽归一化，并保留灰度和方/圆/unknown分类。q108/q109多个字的局部软证据转向Humming，但角点覆盖率低、部位对应与保护门槛仍限制作用；主指标仍38/44、轻吟3/6，无新增错误，不替换v4。正式实验保存于 `data/features-v6/final/`。

v7对应实验：[短笔段、闭合框与比例变化报告](CORRESPONDENCE_V7_REPORT.md)。入口 `rank_font_correspondence.py --qid 109` 增加横纵比例下的部位映射、孤立笔段线索、按角角色对应闭合框。q109「こ」从弃权恢复一张Humming对Skip的圆端票；q108「知」内孔能与Humming/Skip/Folk定位对应，角形仍unknown。全类别回归仍38/44；非Humming候选仅多1张无助于正确类别的票，暂不替换v4。数据位于 `data/correspondence-v7/final/`。

Generated data lives in `data/`. `manifest.json` records the question ID, coarse label, difficulty, image URL, image filename, and download status. Images are grouped by label using a manifest rather than duplicate files.

## Current experiment

All 137 question images were downloaded and decoded successfully; 28 answer labels are represented. `data/annotations-a.json` and `data/annotations-b.json` contain one representative crop per label, visually classified and locally transcribed by GPT-5.6 Luna at high reasoning effort. Bounding boxes are approximate. `data/line-rois-hard.json` contains extra coarse line boxes for difficult multi-line samples.

The active priority is recorded in `data/priority.json`: six coarse labels with at least four examples and at least two one-star examples. Eighteen samples (three per label, seventeen one-star) are the current validation batch. One- or two-example labels are deferred. `data/priority-a.json` and `data/priority-b.json` contain GPT-5.6 Luna high's per-column visual localization and short OCR excerpts for those 18 images. Several ROI estimates were corrected after inspecting the original images (including q100, q33, q72, q65, q125, q76, q91, q140); these are still experimental annotations, not a ground-truth dataset.

For local processing, install dependencies in a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python process_regions.py --version priority-v7 --annotations data/priority-a.json data/priority-b.json --ids 118 115 100 125 76 62 58 96 65 114 92 70 91 61 33 72 27 140
.venv/bin/python segment_text.py --version priority-v7 --annotations data/priority-a.json data/priority-b.json
```

`process_regions.py` writes a candidate ink mask and an original-image overlay, plus separate masks for each proposed text column; `segment_text.py` uses those per-column masks to avoid joining nearby columns and writes candidate line/character boxes. These are visual candidates, not verified ground truth. Several iterations are retained in `data/processed/`; `data/eval-priority-a.json` and `data/eval-priority-b.json` contain Luna's first visual scores. The later `priority-v2`–`priority-v7` outputs apply a conservative ROI guard to reject artwork admitted by padded boxes and separately segment each text column. `priority-v7` is the current output; earlier versions are retained as comparison evidence.

Run `python3 compile_scorecard.py` to merge the latest Luna visual reviews into `data/scorecard-priority-v7.json`. Of the 18 priority samples, 14 are currently usable *crop candidates*, 3 need manual review (q114, q72, q140), and 1 is rejected (q27). The gate requires text coverage and background rejection; it is not a font-identity confidence or measured accuracy rate.

The independent Sol-medium visual review in `REVIEW_SOL_MEDIUM.md` found additional cropped line endings, so prefer that report over the older Luna scorecard for diagnosing misses. The follow-up `AB_REPORT.md` compares v7 against `adaptive_regions.py` and the optional comic-text-detector ONNX raw outputs on all 18 samples. `data/ab-comparison/` contains side-by-side originals and overlays. `adaptive_regions.py` needs only the existing OpenCV dependency; `ctd_regions.py` takes an externally downloaded ONNX model and does not add PyTorch to this project.

## Limits

- The question-bank labels identify expected **Chinese** fonts, not the exact Japanese source font or weight.
- This dataset consists mostly of preselected crops; it cannot measure full-page text detection recall.
- Some crops contain large illustration areas or white text on screened black backgrounds. Region refinement is still experimental.
- q27 has bold outlined text over dense speed lines; current thresholding cannot cleanly isolate glyphs. Reject/ask for review rather than treating its noisy mask as font evidence.
- OCR excerpts and mask scores are model judgments, not manually verified labels. Review them before using them as a benchmark.

# v8 连续局部形状实验

见 [CONTINUOUS_V8_REPORT.md](CONTINUOUS_V8_REPORT.md)。该实验只导出局部形状、横竖笔宽和选字诊断，不改变默认字体排名。

## v9 候选对差异热区

见 [CONTRASTIVE_V9_REPORT.md](CONTRASTIVE_V9_REPORT.md)。含视觉失败案例、模板差异热区与机器学习文献对照；仍为只读诊断实验。

## v10 无训练成对注意力

见 [PAIR_ATTENTION_V10_REPORT.md](PAIR_ATTENTION_V10_REPORT.md)。先冻结稳定度、部位约束和弃权规则，再对已暴露的 52 题做开发回归；未改变默认排名。
