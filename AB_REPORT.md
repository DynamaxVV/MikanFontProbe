# 漫画文字截取三路 A/B（首轮）

本轮没有改写先前的 v7 基线。对同一批 18 张预裁题库图片运行：

1. `data/processed/priority-v7/`：原有 LLM 粗框 + 固定护栏 OpenCV。
2. `data/processed/adaptive-v3/`：LLM 框仅作种子，沿文字方向搜索并按侧向位置与大间隙排除独立插画；仍使用 OpenCV 的阈值/连通域作字形候选。
3. `data/processed/ctd-raw-v2/`：`comic-text-detector` 的 ONNX 模型，OpenCV DNN 推理得到原始文字像素概率、文字列概率和文字块候选；图像遮罩阈值为 0.3。此处展示**原始模型结果**，未运行上游项目的完整 PyTorch 后处理和 mask-refinement 链路。

每题原图及三路叠加图放在 `data/ab-comparison/q<题号>-comparison.png`；第三路还有 `q<题号>-blocks.png` 可看文字块候选。绿像素是候选字形，蓝框是候选列；它们不是字体匹配结果。

| 难例 | v7 基线 | 动态 OpenCV | 漫画模型原始输出 |
| --- | --- | --- | --- |
| [q70](data/ab-comparison/q70-comparison.png) | 上方「冒」顶部及多列尾字漏截 | 首字顶部和「で」「を」、下方三列末字均延伸纳入，背景未明显增加 | 像素遮罩覆盖七列且很干净，但文字块分支只给两个框，漏覆盖上方一列 |
| [q92](data/ab-comparison/q92-comparison.png) | 中列末字「く」不完整 | 「く」恢复；右上气泡弧线仍有少量误收 | 部分细字笔画只被概率模型选中一部分，两个文字块也没覆盖所有列 |
| [q72](data/ab-comparison/q72-comparison.png) | 尾字保留，下方手机未纳入 | 尾字保留；按侧向偏移排除了手机（最初版本曾误收，v3 已修复） | 粗体字形有大面积漏笔画，且没有给出文字块 |
| [q140](data/ab-comparison/q140-comparison.png) | 末「は」保留，邻近边线残留 | v3 保留与前字隔得较远的「は」；仍有少量边线残留 | 未证明优于 OpenCV，边缘截断内容不可从截图补全 |
| [q27](data/ab-comparison/q27-comparison.png) | 粗描边字与速度线粘连 | 没有解决速度线和字形分离 | 原始分割也不可靠、无文字块；三路均应拒判 |

其余 [q125](data/ab-comparison/q125-comparison.png)、[q76](data/ab-comparison/q76-comparison.png)、[q61](data/ab-comparison/q61-comparison.png)、[q114](data/ab-comparison/q114-comparison.png) 的并排图可检查动态路线的回归和模型风格差异。模型文字块在 18 张中有 4 张完全没有候选（q76、q96、q72、q27）；这不等同于模型像素头完全没有输出。

目前没有人工逐像素真值，也没有足量未裁切原页，因此**不能报告三路的总体 IoU、召回率或整页检出率**。上述是可回看的局部视觉结论；动态路线主要修复粗框首尾漏字，未解决所有分字问题，也不自动区分框外手写字。

## 复现

当前环境：Python 3.14.7，OpenCV 5.0.0；没有新增项目依赖。模型取自 [manga-image-translator beta-0.2.1 官方发布](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1) 的 `comictextdetector.pt.onnx`（94,669,756 字节，本地 SHA-256 `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`），此次存放在项目外 `/tmp/mikanfont-ctd.iS5wMS/`，没有打包进项目。上游 [comic-text-detector 仓库](https://github.com/dmMaze/comic-text-detector) 标注 GPL-3.0；若后续产品集成其代码或模型，应单独核对授权。仅运行本地 OpenCV 路线不需要该模型。

```sh
.venv/bin/python adaptive_regions.py --version adaptive-v3 --annotations data/priority-a.json data/priority-b.json
.venv/bin/python segment_text.py --version adaptive-v3 --annotations data/priority-a.json data/priority-b.json
.venv/bin/python ctd_regions.py --version ctd-raw-v2 --model /path/to/comictextdetector.pt.onnx
.venv/bin/python compare_overlays.py
.venv/bin/python -m unittest test_adaptive_regions.py -v
```
