# MikanFontProbe Agent Instructions

本文件是 MikanFontProbe 的项目级协作约定。若与系统、开发者或上级目录规则冲突，以更高优先级规则为准。

## 项目目标

这是一个离线日文漫画字体识别实验项目。主要任务包括：

- 从题库截图中收紧文字区域、分列、分字，并保留原图证据；
- 使用已知字符和本地日文字体样本进行字体族匹配；
- 研究假名优先选字、笔端/角点等可解释特征，以及局部编码器；
- 用未参与调参的样本验证稳定性。

题库中文答案只是粗粒度代理标签，不等于日文原字体真值。任何实验结论都必须注明这一边界。

## 目录约定

- 根目录只保留 `README.md`、`AGENTS.md` 和项目配置文件；维护文档统一放在 `docs/`。
- Python 入口统一放在 `mikan_font_probe/`，测试统一放在 `tests/`，根目录不新增 `.py` 文件。
- `data/` 保存原图、裁片、模板、冻结协议、评估 JSON 和实验输出；除非用户明确要求，不删除或覆盖其中的证据。
- 大型且可再生的训练/评估产物可以保持本地，但必须加入 `.gitignore`，并在报告中说明生成方式和未提交边界。
- `font_sample/` 是输入资产，涉及字体版权，不移动、重新分发或清理字体文件。
- `output/` 保存最终图文报告；已有 PDF 不作为缓存删除。
- `.venv/` 和 `.omx/` 是本地环境/运行时目录，不纳入实验结果，也不要提交其中内容。

详细目录说明见 `docs/PROJECT_LAYOUT.md`。

## 常用命令

在仓库根目录执行：

```sh
# 安装依赖
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 全量回归
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover

# 单项测试使用 dotted module 路径
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_font_baseline -v

# 默认受保护字体入口
.venv/bin/python -m mikan_font_probe.rank_font_guarded --qid 46
```

运行实验模块时使用 `python -m mikan_font_probe.<module>`，不要依赖已经迁移走的根目录旧模块路径。冻结记录中的旧路径由 `mikan_font_probe.paths.resolve_repo_path()` 兼容。

## 实验与数据规则

- 先区分 `baseline`、`diagnostic`、`rejected`、`frozen` 状态，再决定是否保留结果。
- 不覆盖既有 `freeze.json`、`protocol*.json`、原图、裁片或评估结果；新的运行使用新的明确版本目录。
- 已经看过或参与调参的样本不能称为新盲测。报告必须区分开发集、留出集、控制集和域外样本。
- 报告 Top-1、Top-3、距离、confidence 和支持票时，明确它们是否校准；默认不得把未校准证据写成概率。
- 局部特征、OCR、视觉模型和字体匹配结果要分开记录。OCR 误读不能直接当字体识别错误，背景复杂或切割不合格时应允许拒判。
- 任何视觉结论都尽量同时保留原图、原生裁片、坐标、处理后图和结构化 JSON，不用放大图替代原生证据。

## 修改与清理

- 保留用户已有的未提交修改；不要使用 `git reset --hard`、`git checkout --`、宽范围 `rm` 或清空目录。
- 代码和文档编辑使用 `apply_patch`；批量机械路径改写必须限定在已确认的文件集合内。
- 不把账号、密码、访问令牌或外部服务响应中的敏感信息写入代码、JSON、报告或日志。
- 删除中间报告前，确认没有独有结论、下游引用或复现职责；优先保留 `data/` 冻结证据和源码，删除后修正文档链接。
- 不自动暂存、提交、推送或发布。完成后报告变更文件、验证结果和未处理风险。

## 完成前验证

代码变更至少执行：

1. 与变更相关的 targeted tests；
2. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover`；
3. `git diff --check`；
4. 受影响文档的本地链接检查和必要的 CLI `--help`/smoke check。

最终说明必须区分“代码测试通过”“冻结结果可复现”“真实漫画质量已验证”这三种不同证据，不能用前两者替代第三者。
