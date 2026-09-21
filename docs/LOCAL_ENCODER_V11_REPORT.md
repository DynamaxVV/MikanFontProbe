# v11 共享局部形状编码器报告

## 结论先行

第一版可训练链路已经跑通：原始漫画裁片和字体模板使用同一个 5 层小 CNN，输入统一为 64×64 灰度局部图，并额外输入原生像素尺寸，输出 64 维 L2 归一化形状向量。训练只使用现有字体文件生成的模板，没有把真实漫画像素混进训练集。

模型规模符合原先的设计区间：推理编码器本体 399,864 个参数；训练时加上族、具体字脸、字重三个辅助头共 402,724 个参数；FP32 checkpoint 约 1.5 MB。`final` 和加入白描边/JPEG增强的 `robust` 两个权重都保留，暂不把其中一版宣称为生产模型。

真实样本上，当前模型已经能达到“可作为候选召回器”的程度，但还没有证据证明它优于现有几何匹配器。最重要的价值目前是把模板和漫画局部统一到一个可训练表示空间，并留下可复现的训练、留出和失败样本协议。

## 模型结构

```text
灰度局部图 1×64×64       原生像素 n
        │                     │
  Conv 32, s2                log(n/64) → 8d
  Conv 64, s2                     │
  Conv 96, s2                     ├─ concat → 168d
  Conv 128, s2                    │
  Conv 160, s1                    │
  adaptive average pool           │
        └──────────────→ 128 → 64 → L2 normalize
```

每层使用 `3×3 Conv + GroupNorm + SiLU`。前四层下采样，最后一层保留局部结构；因此它不是把所有信息压成单纯的墨量或字框面积。原生分辨率只作为条件输入，不通过插值伪造细节。

## 训练数据和目标

字体目录当前有 23 个具体字脸，每个字脸从模板目录提供 205 个字形，共冻结 4,715 条“字脸—字符”记录：

| 项目 | 当前设置 |
| --- | --- |
| 训练记录 | 3,887 |
| 字符留出验证记录 | 828 |
| 留出方式 | 按字符哈希留出约 20%，同一字符的所有字脸同时进入验证 |
| 正例 | 同字、同字脸的两次独立成像变体 |
| 难负例 | 同字、不同字体族，批次内显式加入 |
| 跨字同字体正例 | 暂不加入；当前没有可靠的笔段对应标注 |
| 辅助监督 | 字体族、具体字脸、字重桶 |
| 真实漫画训练像素 | 0 |

少女／方圆／海报仍作为三个独立字体族参与族监督；没有把它们的粗细差异合并成可以任意抹掉的噪声。训练时仍保留具体字脸和字重头。

训练增强包含漫画缩放、低分辨率栅格化、轻微模糊、形态学粗细变化、背景变化、噪声、白描边暗背景变体和 JPEG 压缩。对少女／方圆／海报降低形态学和模糊强度，避免增强本身覆盖字重证据。

## 两个权重版本

`final` 是先完成的干净增强版本；`robust` 是补入显式白描边和 JPEG 压缩后的版本。两者都由同一套 12 epoch、MPS 训练流程得到。

### 合成留出验证

这是“未见字符”的字体监督验证，不是实际漫画准确率：

| 权重 | 最佳 epoch | 字体族 | 具体字脸 | 字重桶 |
| --- | ---: | ---: | ---: | ---: |
| `final` | 11 | 19.1% | 12.9% | 49.3% |
| `robust` | 11 | 18.5% | 9.7% | 45.9% |

随机猜测不是本实验的唯一基准，因为训练集有族/字脸不均衡；这些数字只说明编码器是否从合成字体中学到可迁移的字体信息，不能直接解释为漫画识别率。`final` 的合成监督略好，说明增强强度仍需要继续调。

### 真实漫画裁片

题库类别比较使用了固定答案映射：例如字体目录的“少女体”计为题库的“少女”，“雅士黑”计为“润圆/雅士黑”。原始目录类别仍保存在 JSON 中。

| 评估池 | 题量 | 可评估 | `final` Top-1 / Top-3 | `robust` Top-1 / Top-3 | 旧几何基线 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fresh holdout | 9 | 8 | 7/8，8/8 | 7/8，8/8 | 7/8，8/8 |
| category expansion | 48 | 47 | 37/47，44/47 | 36/47，43/47 | 37/48，45/48 |
| Folk controls | 4 | 4 | 0/4，4/4 | 4/4，4/4 | 原报告未作为同一编码器指标 |

`fresh holdout` 的 q11 是白字黑底，当前入口按输入协议弃权；这不是把它算成错误。`category expansion` 已被此前项目实验使用，不是盲测；因此只能作为回归池。`Folk controls` 也曾参与规则设计，不能当作新的无偏测试集。现阶段较稳妥的判断是：`final` 与旧几何方案在现有开发数据上相当；`robust` 对这 4 个方新书控制样本的误吸附明显更少，但合成留出略退步。

## 失败样本和原因

- q11：白字黑底输入被统一标记为 `unsupported_light_input`。下一步应在编码器前加入极性/描边归一化分支，并单独评估白字样本，不能直接把黑字模板反相后当成同分布。
- q21：低分辨率下黑体和圆体的整体字框、墨量相近，`final` 与 `robust` 都偏到圆体，说明仅靠全局 embedding 仍会把共同骨架当成证据。
- q106、q108、q109：轻吟体仍是困难族，预测分别偏向润圆/雅士黑或方新书；这与先前几何、端点、闭合框实验观察一致。需要把相邻笔画粗度、自由笔端和连接角点作为显式局部 token 或后级证据，而不是要求一个 64d 全局向量独立解决。
- q4：润圆/雅士黑仍被方新书吸附。它提示“类别答案正确”之外还需要保留具体日文字脸和候选间隔，不能只看类别 Top-1。

## 代码、权重与复现

- 模型定义：[mikan_font_probe/local_encoder_v11.py](../mikan_font_probe/local_encoder_v11.py)
- 训练入口：[mikan_font_probe/train_local_encoder_v11.py](../mikan_font_probe/train_local_encoder_v11.py)
- 真实裁片检索：[mikan_font_probe/evaluate_local_encoder_v11.py](../mikan_font_probe/evaluate_local_encoder_v11.py)
- 训练数据冻结清单：[data/local-encoder-v11/robust/manifest.json](../data/local-encoder-v11/robust/manifest.json)
- 当前增强版本权重：[data/local-encoder-v11/robust/checkpoint.pt](../data/local-encoder-v11/robust/checkpoint.pt)
- 干净增强对照权重：[data/local-encoder-v11/final/checkpoint.pt](../data/local-encoder-v11/final/checkpoint.pt)
- 训练记录：[data/local-encoder-v11/robust/training.json](../data/local-encoder-v11/robust/training.json)
- 真实评估：[data/local-encoder-v11/robust/evaluation.json](../data/local-encoder-v11/robust/evaluation.json)
- 回归测试：[tests/test_local_encoder_v11.py](../tests/test_local_encoder_v11.py)

```sh
.venv/bin/python -m mikan_font_probe.train_local_encoder_v11 --epochs 12 --batch-size 128 --output data/local-encoder-v11/<new-run>
.venv/bin/python -m mikan_font_probe.evaluate_local_encoder_v11 --checkpoint data/local-encoder-v11/<new-run>/checkpoint.pt --output data/local-encoder-v11/<new-run>/evaluation.json
.venv/bin/python -m unittest -q
```

## 下一步建议

1. 先冻结 `final` 作为高合成验证线、`robust` 作为描边/压缩鲁棒线，不凭当前 4 个控制样本强行二选一。
2. 加入极性归一化和“白描边是否存在”的显式输入，再对 q11 及新的白字样本做按作品留出验证。
3. 缓存每个字脸—字符的模板向量，输出具体字脸 Top-3、类别 Top-3、支持字数、Top-1/Top-2 间隔和弃权原因；这些比未经校准的单个 cosine 分数更适合给审核者看。
4. 只有在有可靠笔段/角点对应时，才加入同字体跨字局部正例；下一版可以把局部 token 编码器与当前全局向量并行，而不是让全局 CNN 猜测“口”角或自由笔端。
5. 增补按作品或页面留出的真实漫画，且将“原图裁切误差、描边、背景纹理、极性”分别标注，验证合成到截图的域差距。
