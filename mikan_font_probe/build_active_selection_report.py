"""Paired source overlays and an honest, data-driven regression report."""

import json

import cv2
import numpy as np

from mikan_font_probe.process_regions import ROOT, imread, imwrite

DIRECTORY = ROOT/"data/active-selection-v1"


def main():
    result = json.loads((DIRECTORY/"results.json").read_text())
    lines = ["# 候选驱动选字与迭代补字验证", "",
             "## 结论", "",
             "本轮没有证明候选驱动选字比清晰度选字稳定更准。3字预算为5/9对6/9，5字均为6/9，7字为6/9对8/9；3字Top-3覆盖由8/9升至9/9，但样本很小且已暴露。暂不替换清晰度基线。",
             "q72受益于针对性补字，但q100、q58出现退步。实验早停在q91、q96错误收敛，故默认关闭自动早停；证据指数不能用于自动放行。",
             "字形表按固定em像素渲染比较，而截图匹配会收紧墨迹并归一化尺寸，两者的距离语义尚未对齐。这可能让选字器偏好实际匹配中已被抹去的差异，是下一步需验证的假设，不是已确认根因。优先统一归一化与距离，并在冻结策略后用新样本验证，而不是继续对这9题调参。",
             "字体表覆盖254字符（84平假名、88片假名），23字体、14家族及16/24/32/48/64px五档。近似共享基于栅格距离，不是矢量轮廓共享证明；本轮未发现跨家族零距离，也不代表不存在共享字形。", "",
             "## 评估边界", "",
             "同一9张已暴露开发图，扩展为127个人工核对的字符裁片。此轮不是新盲测、不是自动OCR，也不是整页端到端测试。",
             "每种策略使用同一23款字体、14个候选家族、同一固定混合匹配距离与同一个字符池。清晰度基线也去重相同字符，避免把去重收益误归给区分表。",
             "同字匹配可预先缓存，但选下一个字的函数不接收未选字符的匹配分；它只使用既有候选、字符、裁片质量、分辨率和字体区分表。", "",
             "## 同预算配对结果", "", "| 策略 | Top-1中文大类一致 | Top-3包含中文答案 | 总取字数 |", "|---|---:|---:|---:|"]
    for key,row in result["summary"].items():
        lines.append(f"| {key} | {row['top1']}/9 | {row['top3']}/9 | {row['glyphs']} |")
    lines += ["", "all_pool_reference只是用全部可用字符的参考，不是理论上限，也不是精确源字体真值。", "",
              "## 每题选字轨迹和实际截图", "",
              "每张图从左到右：最清晰5字、候选驱动5字、自动停止的迭代结果。框内数字为查询顺序；字符读法见表。"]
    for sample in result["samples"]:
        qid = sample["qid"]
        source = ROOT/f"data/processed/typographic-v3/q{qid}-source.png"
        original = imread(source)
        panels = []
        modes = [("quality_5","Quality k=5"),("active_5","Candidate-driven k=5"),("iterative","Iterative")]
        for mode,title in modes:
            panel = original.copy()
            for order,index in enumerate(sample["variants"][mode]["selected"]):
                x,y,w,h = sample["glyphs"][index]["bbox"]
                cv2.rectangle(panel,(x,y),(x+w,y+h),(0,130,255),1)
                cv2.putText(panel,str(order+1),(x,y+10),cv2.FONT_HERSHEY_SIMPLEX,.35,(210,30,20),1)
            scale = min(1.,440/panel.shape[0])
            panel = cv2.resize(panel,(round(panel.shape[1]*scale),round(panel.shape[0]*scale)))
            width = max(230,panel.shape[1])
            canvas = np.full((470,width,3),255,np.uint8)
            canvas[30:30+panel.shape[0],:panel.shape[1]] = panel
            cv2.putText(canvas,title,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
            panels.append(canvas)
        imwrite(DIRECTORY/"comparisons"/f"q{qid}.png",np.hstack(panels))
        lines += ["",f"### q{qid} · 题库答案：{sample['expected']}","", "| 策略 | 选中字符 | 第一候选 / 中文答案 |", "|---|---|---|"]
        for mode,_ in modes:
            record = sample["variants"][mode]
            winner = record["ranking"][0]
            lines.append(f"| {mode} | {record['characters']} | {winner['name']} / {winner['answer']} |")
        iterative = sample["variants"]["iterative"]
        lines += ["",f"停止原因：`{iterative['stop_reason']}`。证据状态：`{iterative['explanation']['state']}`。", ""]
        lead = iterative["explanation"]["top3"][0]
        for pair in lead["pairwise_evidence"]:
            by_reason = {}
            for evidence in pair["evidence"]:
                by_reason.setdefault(evidence["reason"],[]).append(evidence["character"])
            lines.append(f"- 对比 `{pair['rival']}`："+"；".join(f"{reason}={''.join(chars)}" for reason,chars in by_reason.items()))
        lines += ["",f"![q{qid} 同预算与迭代选字](data/active-selection-v1/comparisons/q{qid}.png)"]
    lines += ["", "## 置信解释的变化", "",
        "同一个字重复出现只算一次证据。字体表在当前分辨率下判为共享或近似的字，只算中性证据，即便微小噪声使它在图像匹配中给某候选更低分，也不能给第一名投票。",
        "逐字列出：支持候选、支持竞争者、共享/近似、低质量、差异不足、表缺失。汉字与假名标签均保留在JSON。数值仍只是未校准的区分性证据指数，不是正确概率。",
        "对手始终是当前相近候选，而不是题库答案。至少两个不同字符支持且无反证只是初始停止启发式；不代表两份统计独立证据，也不能证明字体在库。",
        "所有方法仍要求已知字符。裁片误读、混入邻字或实际字体缺库会使选择器优先查询了理论好字，却仍然比较失败。", "",
        "## 复现与文件", "",
        "- 字体表：GLYPH_DISCRIMINATION.md 及 data/glyph-discrimination-v1/。",
        "- 池与评分：data/active-selection-v1/pools.json、match_scores.npz、match_freeze.json。",
        "- 完整轨迹：data/active-selection-v1/results.json，含每轮候选、备选字效用、对手和每字证据。",
        "- 策略代码：informative_selection.py。重跑评估入口：run_active_selection.py。",
        "- 本报告图像生成：.venv/bin/python build_active_selection_report.py。", ""]
    (ROOT/"ACTIVE_SELECTION_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
