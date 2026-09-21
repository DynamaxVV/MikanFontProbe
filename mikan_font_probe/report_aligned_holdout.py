"""Generate a paired holdout report without changing the frozen selector."""
import json
import math

import cv2
import numpy as np

from mikan_font_probe.process_regions import ROOT, imread, imwrite
from mikan_font_probe.paths import resolve_repo_path

OUTPUT = ROOT/"data/aligned-holdout-v2"


def main():
    result = json.loads((OUTPUT/"results.json").read_text())
    count = result["question_count"]
    lines = ["# 统一字形处理后的冻结新样本验证", "",
             "## 结论", "",
             "严格7字对照的8张新题中，两种策略均为7/8，没有配对改善或退步。3字为清晰度7/8、主动6/8；5字为6/8、5/8。Top-3在支持域8题均包含中文答案。统一处理链消除了度量不一致，但本轮仍未证明主动选字优于清晰度基线，不能宣布稳定提升或替换默认策略。",
             "q10主动策略在3/5字时误判轻吟体，7字补回圆体；q21两策略所有预算均误判圆体，增加至全部裁片也未修正，属于不能仅靠选字解决的剩余失败。没有足够证据在本轮将其归因于某一种具体原因。",
             "实验迭代在q1因无已知区分字停止，候选错误但证据指数为0；这应表示需要复核，而不是已识别成功。数值置信仍未校准，自动早停保持默认关闭。q11是白字黑底及少字的域外案例，原样保留但不计入严格主指标。",
             "下一步应先独立核验共同失败样本的实际源字体/字形覆盖和输入模型，再设计新实验；不利用这次测试结果继续调参并重新声称盲测。", "",
             "## 处理方式", "",
             "模拟查询由同一128px字体模板裁紧灰度墨迹、降采样到16/24/32/48/64px，再调用实际截图所用normalize和score_pair。模板端与真实匹配相同。区分度采用双向跨字体混合距离减去同字体降采样距离，再取所有字重组合的最小非负值；不再采用固定em的Soft-IoU。",
             "近似共享门槛改用已有匹配分差0.015，而不是旧Soft-IoU阈值0.12。预算、质量权重、候选数等不再调参。合成降采样不模拟所有真实压缩、描边和背景，仍非完美成像模型。",
             "## 验证边界", "",
             "策略与字体文件在新样本匹配前冻结。采样审计见[FRESH_SAMPLE_AUDIT.md](FRESH_SAMPLE_AUDIT.md)。字符和裁片经视觉标注；这不是OCR端到端测试。题库提供中文答案大类，不能据此断言识别了精确日文字体。独立题目也不必然是独立漫画来源。",
             "主指标预先指定为7字预算的中文答案Top-1一致率，3/5字和Top-3是次指标。自动早停只作探索性记录，不默认启用。", "",
             "## 同预算结果", "", "| 策略 | Top-1 | Top-3 | 总取字数 |", "|---|---:|---:|---:|"]
    for name, values in result["summary"].items():
        lines.append(f"| {name} | {values['top1']}/{count} | {values['top3']}/{count} | {values['glyphs']} |")
    eligible = [row for row in result["samples"] if len({g["character"]
                for g in row["glyphs"] if g["quality"]["usable"]}) >= 7]
    lines += ["", "上表为全部题目的最多预算结果；少字题可能不足预算或出现重复补字，不视作严格等预算比较。",
              f"按匹配前补充约定，严格7字主指标纳入{len(eligible)}题（≥7个不同可用字符）。", "",
              "| 字数预算 | 清晰度优先 Top-1 | 统一处理的主动选字 Top-1 |", "|---|---:|---:|"]
    for budget in (3, 5, 7):
        correct = [sum(row["variants"][f"{strategy}_{budget}"]["ranking"][0]["answer"]
                       == row["expected"] for row in eligible) for strategy in ("quality", "active")]
        lines.append(f"| {budget} | {correct[0]}/{len(eligible)} | {correct[1]}/{len(eligible)} |")
    wins, losses = [], []
    for row in eligible:
        good = {key: row["variants"][key]["ranking"][0]["answer"] == row["expected"]
                for key in ("quality_7", "active_7")}
        if good["active_7"] and not good["quality_7"]:
            wins.append(row["qid"])
        if good["quality_7"] and not good["active_7"]:
            losses.append(row["qid"])
    discordant = len(wins)+len(losses)
    probability = min(1., 2*sum(math.comb(discordant, k)
        for k in range(min(len(wins), len(losses))+1))/2**discordant) if discordant else 1.
    lines += ["", f"主指标配对变化：改善题目 {wins}；退步题目 {losses}。",
              f"对不一致配对作双侧精确符号检验：p={probability:.4f}。小样本且可能同源，不能把未显著或一次提升解读成稳定泛化。", "",
              "## 原始裁片与取字轨迹", "",
              "下面按裁片池索引展示实际输入；U+为字符码。三种同预算策略均使用这些输入，不按识别成败筛除题目。"]
    for row in result["samples"]:
        glyphs = row["glyphs"]
        canvas = np.full((math.ceil(len(glyphs)/8)*100, 800, 3), 255, np.uint8)
        for index, glyph in enumerate(glyphs):
            image = imread(resolve_repo_path(glyph["path"]))
            scale = 65/max(image.shape[:2])
            image = cv2.resize(image, (max(1, round(image.shape[1]*scale)),
                                      max(1, round(image.shape[0]*scale))))
            x, y = index%8*100, index//8*100
            canvas[y+25:y+25+image.shape[0], x:x+image.shape[1]] = image
            cv2.putText(canvas, f"{index} U+{ord(glyph['character']):04X}", (x,y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, .32, (0,0,0), 1)
        path = OUTPUT/"review"/f"q{row['qid']}.png"
        imwrite(path, canvas)
        lines += ["", f"### q{row['qid']} / {row['expected']}", "",
                  "| 策略 | 选字顺序 | 首选 | Top-3家族 |", "|---|---|---|---|"]
        for key in ("quality_3", "active_3", "quality_5", "active_5", "quality_7", "active_7", "iterative"):
            record = row["variants"][key]
            ranking = record["ranking"]
            lines.append(f"| {key} | {record['characters']} | {ranking[0]['answer']} | "
                         + " / ".join(x["name"] for x in ranking[:3]) + " |")
        lines += ["", f"![q{row['qid']}原始裁片]({path})"]
    lines += ["", "## 置信解释与复现", "",
              "confidence字段仍是未校准证据指数，不是正确概率；共享近似字形不投支持票、重复字不重复计票。即使多个字一致，也可能共享裁切或缺库误差。完整每字支持/反证、每轮候选与选择理由见results.json。",
              "冻结文件：data/aligned-holdout-v2/protocol-freeze.json；结果：data/aligned-holdout-v2/results.json；按需生成的字体区分表：discrimination.json。旧版数据未覆盖。",
              "评估入口validate_aligned_holdout.py拒绝覆盖第一次结果，拒绝在冻结代码/字体改变后评估。报告可运行 .venv/bin/python report_aligned_holdout.py 重新生成。", ""]
    (ROOT/"ALIGNED_HOLDOUT_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
