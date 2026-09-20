# 日文字体字形区分表 v1

本目录是一个可复现的、以渲染位图为证据的假名优先区分表。它服务于后续选字查找，不给出字体“真正矢量字形相同”的结论，也不把距离转换成概率。

## 当前接口

构建后只依赖 NumPy 的主环境可以这样加载：

```python
from build_glyph_discrimination import load_index

index = load_index("data/glyph-discrimination-v1")
result = index.pair_distance("shinmaru", "antique", "な", 24)
# {
#   "distance": 0.0..1.0 or None,
#   "covered": True/False,
#   "shared": True/False/None,
#   "status": "exact_bitmap" | "low_distance" | "different" |
#             "missing" | "same_family" | "unknown_resolution",
#   "low_resolution_approximation": bool,
#   "evidence_basis": "rendered_raster",
#   "vector_glyph_equal": None,
# }
```

`pair_distance(family_a, family_b, char, resolution)` 的家族距离是两个家族已覆盖字重变体之间的最小距离，因此对选字是保守的。相同家族的查询返回 `same_family` 和零距离；需要同一字体零差断言时使用 `face_distance(face_id, face_id, char, resolution)`。若查询分辨率不是档位，loader 取不高于请求值的最大档位（例如请求 20 使用 16）；请求低于最小 16px 时不回退到更高清档，而返回 `unknown_resolution`、`distance=None`、`shared=None`（`covered` 仍只表示 cmap 是否覆盖）。结果同时返回 `requested_resolution` 与实际 `resolution`。家族键来自现有 `font_family_policy.py`：共 14 个家族、23 个字体文件；新丸ゴ/リュウミン/Antique/YW/Skip 的字重按家族合并，但 `RodinHappyPro-L/B/UB` 保持三个家族。Antique 与 YW 虽然答案标签同为“黑体”，在表中仍是两个家族。

缺字不是 tofu：覆盖由字体 cmap 元数据核对，`covered=False` 时没有渲染位图，距离为 `None`，`shared=None`。

## 文件与 schema

`data/glyph-discrimination-v1/index.json` 是可读 manifest：

- `characters`：稳定的 Unicode 字符顺序；包括显式平假名/片假名、长音、小字、浊音、半浊音、`ゔ/ヴ` 等，baseline、holdout、`data/active-selection-v1/characters.txt` 的字符，以及少量固定常见汉字验证集。
- `fonts`：23 个字体的 id、相对路径、PostScript、字重、家族键、cmap 覆盖字符。
- `families`：14 个家族及其成员字体 id。
- `resolutions`：默认 `16,24,32,48,64`；`source_resolution=128` 是给同字匹配使用的源模板。
- `status_codes`、`low_distance_threshold`、渲染和距离协议、输入文件 SHA-256、覆盖/状态计数。

`data/glyph-discrimination-v1/glyphs.npz` 使用 `allow_pickle=False`，包含：

- `glyph_r16`、`glyph_r24`、`glyph_r32`、`glyph_r48`、`glyph_r64`、`glyph_r128`：`uint8[face, character, height, width]`，白底孤立字形；缺字槽位虽为白底，但必须结合 `coverage` 使用。
- `coverage`：`bool[face, character]`。
- `face_distance`：`float32[face, face, resolution, character]`，缺字为 NaN。
- `family_distance`：`float32[family, family, resolution, character]`，缺字为 NaN。
- `family_status`：`uint8[family, family, resolution, character]`。

因此主代理不需要 PIL，也不需要十万张 PNG；`image(face_id, char, 128)` 可以直接取压缩 NPZ 中的 128px 模板。loader 初始化时缓存覆盖、face/family 距离和状态数组，重复 `pair_distance` 不会反复解压这些 NPZ 成员；字形栅格按分辨率首次访问时缓存。`data/glyph-discrimination-v1/evidence/` 仅保存一张小型对照图和其 JSON 选择记录，不是全量渲染缓存。

## 渲染和距离边界

构建脚本只在生成阶段懒加载 bundled Pillow。每个孤立字在目标字号绘制到固定正方形，并按可见 bbox 居中；没有 fallback 或字符串 shaping。距离先把黑色墨迹转成 `[0,1]`，尝试中心及上下左右一像素的固定局部位移，取 `1-soft ink IoU` 的最小值。由于非负墨迹上 `L1/union = 1-intersection/union`，两者不是独立特征，代码不重复加权。`exact_bitmap` 只表示持久化渲染位图逐像素相同；`low_distance` 是工程检索带（当前阈值 0.12），在 16/24px 时另标 `low_resolution_approximation`；`different` 只是该距离带之外。它们都不是概率，也不是轮廓相等证明。

## 精确复现

在本目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 \
  build_glyph_discrimination.py \
  --output data/glyph-discrimination-v1 \
  --extra-chars data/active-selection-v1/characters.txt

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest -v test_glyph_discrimination.py
```

`--extra-chars` 可重复传入 UTF-8 字符文件，也接受不含路径的 literal 字符串；默认构建已经自动纳入当前 active-selection 文件，显式传入是为了让复现命令自描述。若主代理后续提供真实多字字符文件，只需追加该参数即可，不会改变已有代码接口。

## 当前结果和证据边界

运行生成的 `index.json` 中 `summary` 是本次样本数、覆盖量和各分辨率的家族对状态计数；`evidence/examples.json` 是从实际矩阵中挑选的同字体零差、跨家族低距离/共享候选、高距离例子，`comparison-examples.png` 是对应视觉核验。汉字/假名是否共享必须读取 `pair_distance` 的计算结果，不能按中文答案标签或“同字”概括推断。

当前 active-selection 文件为 67 个字符；视觉核对后池中保留 127 个可用裁片、67 个不重复字符。它被视为已暴露开发图的配对对照，不是盲测集。当前生成结果的跨家族 `exact_bitmap` 计数在 16/24/32/48/64px 均为 0；低距离是实际 raster 计算结果，例如 `humming`/`skip` 的 `リ` 在 16px 为 0.083829（低距离），而 `shinmaru`/`antique` 的 `な` 在 16px 为 0.403006（不同）。该表覆盖的字形仍受本地 23 款字体 cmap 限制；例如当前 YWHeiTi 两个变体不覆盖 `ゔ`，查询返回 `missing` 而不会匹配 tofu。位图距离还受 Pillow/FreeType 版本、字号栅格化和居中策略影响；换渲染器或实际设备后需重新构建并重新核验。
