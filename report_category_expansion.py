"""Report balanced category coverage with explicit exposure and eligibility."""
from collections import Counter
import json

from process_regions import ROOT

OUT = ROOT/"data/category-expansion-v3"


def metrics(rows, key):
    return (sum(r["variants"][key]["ranking"][0]["answer"] == r["expected"] for r in rows),
            sum(r["expected"] in [x["answer"] for x in r["variants"][key]["ranking"][:3]] for r in rows))


def main():
    protocol = json.loads((OUT/"protocol.json").read_text())
    result = json.loads((OUT/"results.json").read_text())
    rows = result["samples"]
    eligible = [r for r in rows if r["eligible_seven"]]
    lines = ["# 字体类别扩展验证", "", "## 结论", "",
             "44道合格题的7字对照：清晰度优先34/44，主动选字32/44；主动Top-3为41/44。仅看26道真正的新题，两者Top-1均为19/26，主动Top-3为23/26。主动选字有局部收益但没有稳定优势，继续保留清晰度基线，自动早停不启用。",
             "新增类别中，方圆5/5、少女5/6、润圆/雅士黑4/5、竹体4/5；轻吟体仅1/6，主要混淆为方新书或润圆/雅士黑。优先核验轻吟体样本与所提供Humming文件的对应关系、字重覆盖及成像归一化，现有证据不足以认定题库错标或确定具体根因。少女q75两种策略都判方圆，同样应检查实际字形/字重证据，不自动改答案。",
             "7字预算主动策略只改善q30，退步q100、q58、q83；新题中的q30改善与q83退步抵消。整体分数不能与之前8题结果直接比较，因为本轮扩大了类别且混有已见样本。以下结果保留为首轮冻结评估，没有用其继续调参。", "",
             "## 范围与口径", "",
             "筛选按已约定的字体家族：普通字重合并，少女/方圆/海报分开。题库总题量严格大于5且存在映射日文字体者纳入。8类共94道可选题，本轮每类固定6题，共48题；这不是94题全量测试。竹带体无对应字体、雷盖/雷鬼体仅5题，不纳入。",
             "核心三类复用既有裁片，其余五类按题号升序取前6题，不按评分换题。新增标注按阅读顺序取最多14个不同可读常规字；不足字数、白字等限制保留。评估仍是已知字符的裁片字体匹配，不是自动OCR或整页检测。",
             "沿用已冻结的统一归一化、同一模板端与混合距离、0.015区分门槛、23字体14家族，不利用扩展结果调参。比较双方在每题使用同一输入池。字库中的其他家族仍作为竞争候选，没有只保留答案涉及的8类。",
             "严格主指标要求黑字且至少7个不同可用字符。少字/白字题保留在完整结果但不混入主指标；不能因为主指标排除而宣称这些输入已经支持。", "",
             f"样本暴露分层：{dict(Counter(r['exposure'] for r in rows))}。严格主指标共{len(eligible)}题。未见只指本地历史记录，无作品级独立性保证。", "",
             "## 每类结果：7字预算", "",
             "| 类别 | 题库题数 | 本轮题数 | 主指标题数 | 清晰度 Top-1 | 主动 Top-1 | 主动 Top-3 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group in protocol["groups"]:
        subset = [r for r in eligible if r["expected"] == group["answer"]]
        quality, _ = metrics(subset, "quality_7")
        active, top3 = metrics(subset, "active_7")
        n = len(subset)
        lines.append(f"| {group['answer']} | {group['available']} | 6 | {n} | {quality}/{n} | {active}/{n} | {top3}/{n} |")
    lines += ["", "## 暴露分层与同预算结果", "",
              "| 样本层 | 合格题数 | 字数 | 清晰度 Top-1 | 主动 Top-1 | 主动 Top-3 |",
              "|---|---:|---:|---:|---:|---:|"]
    for exposure in ("all", "new_question", "previous_annotation_only", "previous_font_evaluation"):
        subset = eligible if exposure == "all" else [r for r in eligible if r["exposure"] == exposure]
        for budget in (3, 5, 7):
            quality, _ = metrics(subset, f"quality_{budget}")
            active, top3 = metrics(subset, f"active_{budget}")
            n = len(subset)
            lines.append(f"| {exposure} | {n} | {budget} | {quality}/{n} | {active}/{n} | {top3}/{n} |")
    improved, worsened = [], []
    for row in eligible:
        q = row["variants"]["quality_7"]["ranking"][0]["answer"] == row["expected"]
        a = row["variants"]["active_7"]["ranking"][0]["answer"] == row["expected"]
        if a and not q:
            improved.append(row["qid"])
        if q and not a:
            worsened.append(row["qid"])
    lines += ["", f"同题配对变化：主动选字改善 {improved}；退步 {worsened}。", "",
              "## 主动策略的主要误判", "", "| 题库类别 → 首选类别 | 数量 |", "|---|---:|"]
    confusions = Counter((r["expected"], r["variants"]["active_7"]["ranking"][0]["answer"])
                         for r in eligible if r["expected"] != r["variants"]["active_7"]["ranking"][0]["answer"])
    for (expected, actual), count in confusions.most_common():
        lines.append(f"| {expected} → {actual} | {count} |")
    lines += ["", "## 全部题目（含不合格样本）", "",
              "| qid | 答案 | 暴露 | 合格 | 主动取字 | 首选 | Top-3家族 |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        record = row["variants"]["active_7"]
        lines.append(f"| {row['qid']} | {row['expected']} | {row['exposure']} | {row['eligible_seven']} | {record['characters']} | "
                     + record["ranking"][0]["answer"] + " | " + "/".join(x["name"] for x in record["ranking"][:3])+" |")
    lines += ["", "## 新增类别原始裁片示例", "",
              "裁片均来自原图，含原图bbox及哈希。每类展示第一题；完整review在三个lane目录。"]
    for group in protocol["groups"][3:]:
        row = next(r for r in rows if r["qid"] == group["selected"][0])
        directory = (ROOT/row["annotation_origin"]).parent
        lines += ["", f"### {group['answer']} · q{row['qid']}", "",
                  f"![原始裁片]({directory/'review'/('q'+str(row['qid'])+'-glyphs.png')})"]
    lines += ["", "## 复现与限制", "",
              "完整结果：data/category-expansion-v3/results.json；名单/历史暴露/冻结哈希：protocol.json；字体表：discrimination.json。构建及评估入口expand_font_categories.py，报告入口report_category_expansion.py。首次结果拒绝覆盖，旧测试和字体文件保留。",
              "置信字段仍为未校准证据指数，不是正确概率，不能据此自动放行。Top-3命中是中文答案大类命中，不证明日文字体精确身份；未训练专用识别模型。类别均衡取样也不代表真实生产分布。", ""]
    (ROOT/"CATEGORY_EXPANSION_REPORT.md").write_text("\n".join(lines))
    print({"eligible": len(eligible), "quality7": metrics(eligible, "quality_7"),
           "active7": metrics(eligible, "active_7"), "improved": improved, "worsened": worsened})


if __name__ == "__main__":
    main()
