# v13 五步隔离优化报告

## 结论

本轮没有把“更大的模型”直接当作答案，而是按单变量顺序完成五步并保留每一步的证据。当前推荐的默认组合是：

```text
v11 robust 编码器
+ 原有 quality_order 选字
加权字族内部 Top-2 face 聚合
```

它只替换同一中文答案下多个日文字重的聚合方式，不替换编码器，不启用未经校准的局部重排。

## 五步记录

### 1. 冻结 v11 robust 基线

冻结记录见 `data/local-encoder-v13/step1-baseline/baseline.json`，基线权重保存在 `data/local-encoder-v11/robust/checkpoint.pt`。为避免把 q11 的白字极性问题混进暗字比较，主要比较使用类别扩展集的共同 47 题：

| 指标 | v11 robust |
|---|---:|
| 类别扩展 Top-1 | 36/47 |
| 类别扩展 Top-3 | 43/47 |
| fresh holdout Top-1 | 7/8 |
| fresh holdout Top-3 | 8/8 |
| folk controls Top-1 / Top-3 | 4/4 / 4/4 |

### 2. 只改字重聚合

固定 v11 robust 权重、原有选字和真实样本，只比较一个字族中多个 face 的聚合方式。`max` 是原规则，`top2` 取该字族最高两个 face 相似度的平均，再对选中字取均值。

| 聚合 | 共同 47 题 Top-1 / Top-3 | fresh 9 题 Top-1 / Top-3 | folk 4 题 Top-1 / Top-3 |
|---|---:|---:|---:|
| max | 36/47 / 43/47 | 7/9 / 9/9 | 4/4 / 4/4 |
| **top2** | **37/47 / 45/47** | 6/9 / 9/9 | 4/4 / 4/4 |
| top3 | 33/47 / 44/47 | 4/9 / 9/9 | 4/4 / 4/4 |
| all-face mean | 30/47 / 43/47 | 2/9 / 8/9 | 4/4 / 4/4 |
| category prototype | 34/47 / 44/47 | 7/9 / 9/9 | 4/4 / 4/4 |

这一步是本轮唯一在共同类别扩展集上稳定超过基线的改动；它也说明 `max face` 的问题确实存在：某一个偶然相似的字重会把整类抬高。Top-2 仍然不是经过概率校准的置信度。

### 3. 弱化形态增强

使用与 v11 相同的损失、批采样和网络，只降低腐蚀/膨胀概率，保留描边、模糊、压缩、噪声和极性变化。结果低于基线：类别扩展共同样本为 34/47 Top-1、42/47 Top-3，故不采用。

### 4. 加入类别原型损失

在第 3 步条件上加入可学习的类别中心，只作为训练正则，推理仍只使用编码器向量。结果进一步下降到 32/47 Top-1、44/47 Top-3；它把不同字重收拢得过早，牺牲了当前截图里有用的细微差异，故不采用。

### 5. 端点/拐角局部证据

实现了一个可弃权的局部分支：

- 端点使用阈值扰动下稳定的平头/圆头几何；
- 拐角使用快速候选点与局部宽度/响应；
- 只有全局 Top-2 都落在 `轻吟体 / 方新书 / 雅士黑` 且差距不超过 0.06 时才启用；
- 无足够稳定对应点时返回中性分数，不强行改变全局结果。

该分支在全量留出集上保持 Top-2 聚合的 37/47、45/47，不产生额外提升。因此它现在应当作为诊断证据和后续训练标注来源，而不是默认加权项。局部特征的输出是相似度证据，不是概率。

## 为什么没有继续扩大 CNN

问题已经不是单纯的容量不足：v12 的多尺度 CNN 合成字体族指标变好，但真实 Top-1 回退；v13 的弱增强和类别原型也未能跨过截图域差距。当前最有证据的收益来自推理层的 face 聚合，而不是网络变大。

轻吟体的失败样本仍集中在 `q106/q108/q109` 一类：全局向量能把候选放入前三，但不足以在低分辨率下完成稳定的 Top-1 区分。下一步应该为这些混淆对积累更高原生分辨率的假名、端点和转角标注，再训练局部分支；目前不应把没有稳定证据的局部得分强行写入默认排序。

## 复现产物

v13 的 checkpoint、manifest 和逐题 `evaluation.json` 是可再生的大型本地产物，未提交到 GitHub；目录 `data/local-encoder-v13/` 已加入 `.gitignore`。本报告保留汇总结果，代码保留在仓库中，可在本地重新生成：

```sh
.venv/bin/python -m mikan_font_probe.evaluate_local_encoder_v13 \
  --checkpoint data/local-encoder-v11/robust/checkpoint.pt \
  --output data/local-encoder-v13/step2-aggregation/evaluation.json
.venv/bin/python -m mikan_font_probe.train_local_encoder_v13 \
  --output data/local-encoder-v13/step3-weak-augmentation-v2
```

五步实现：[evaluate_local_encoder_v13.py](../mikan_font_probe/evaluate_local_encoder_v13.py)、[train_local_encoder_v13.py](../mikan_font_probe/train_local_encoder_v13.py)、[local_features_v13.py](../mikan_font_probe/local_features_v13.py)。
