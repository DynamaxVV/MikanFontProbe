# Luna high 日文 OCR 与离线题库字体判断（diagnostic）

本轮范围是宋、黑、圆、方圆、少女、海报、雅士黑、方新书、轻吟体中，先前没有核准日文原文的 111 道离线题。六个 `gpt-6-luna`、`high` 子代理分别查看原图，选择单列日文并记录原图 SHA、坐标、文字及存疑原因。首轮直接读图；第二轮允许参考已有 ONNX 结果提出候选，再回到原图核对。主代理抽查并纠正了多处跨列、漏字和误读。因此 Luna 与 ONNX 一致不能算独立复核。

**OCR 输入是题库保存的整张题目图片 `data/images/q<ID>.<ext>`，并非下方字体分析生成的 `region.png`。** 子代理在题目图片上转写并给出文字区域坐标；之后程序才从同一张图片裁出 `region.png` 用于切字和字体匹配。题库图片本身可能是出题时从漫画截取的局部，本轮没有取得更大的漫画页面。

结果见 [逐题 HTML](../output/REAL_QUESTION_LUNA_OCR_V3.html) 和 [结构化 JSON](../data/real-question-luna-validation-v3/results.json)。六组原始转写保存在 `data/real-question-luna-ocr-v1/lane-0.json` 至 `lane-5.json`；每题保留原图链接、所选区域和字体候选。上一轮结果保留在 v1/v2 目录，本轮没有覆盖。

| 阶段 | 题数 |
| --- | ---: |
| 范围内题目 | 111 |
| Luna 标为可转写 | 58 |
| Luna 存疑 / 不可辨 | 52 / 1 |
| 转写与切分字数不符 | 39 |
| 字数对齐但有效支持字不足 | 2 |
| 产生待复核字体排名 | 17 |

17 道排名题按题库中文粗标签计算，Top-1 为 11/17、Top-3 为 15/17。这些数值只描述当前可对齐子集，不能推广为 111 题的字体准确率；中文答案也不是日文字体真值。字符数一致尚不能证明字与裁片逐个对应，现有排名和区域仍需人工验收。题目已暴露于策略研发，不是新盲测。字体距离未经概率校准。

复现命令：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m mikan_font_probe.evaluate_offline_luna_ocr \
  --ocr-dir data/real-question-luna-ocr-v1 \
  --output-dir data/real-question-luna-validation-v4 \
  --html output/REAL_QUESTION_LUNA_OCR_V4.html
```

运行时必须使用新的版本目录，以保留已有 OCR、裁片与排名证据。下一步应优先检查 39 道字数不符题的区域、标点和切分，再对 17 道排名题逐字核验；单纯重复整图 OCR 无法证明字体判断改善。

提交边界：本报告、题目原图、Luna 转写、ONNX 对照、v3 JSON/HTML 和报告引用的原生裁片可公开核对。字体样本和本地生成的字体模板不随报告重新分发；完整匹配复跑仍需要本机字体资产与对应代码环境。v1/v2 中间版本及可再生训练产物保留在本地。
