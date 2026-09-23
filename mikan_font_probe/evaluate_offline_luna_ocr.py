"""Evaluate Luna-reviewed Japanese columns in the scoped offline bank."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path

from mikan_font_probe.evaluate_real_question_ocr import TARGET_ANSWERS, digest, evaluate_one, load_assignments
from mikan_font_probe.process_regions import ROOT


def box_iou(first: list[int] | None, second: list[int] | None) -> float:
    if not first or not second:
        return 0.0
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = width * height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def matcher_layout(orientation: str) -> str:
    return "vertical" if orientation == "vertical-rtl" else orientation


def load_lanes(ocr_dir: Path, assignments: dict[int, dict]) -> dict[int, dict]:
    records = {}
    for lane in range(6):
        payload = json.loads((ocr_dir / f"lane-{lane}.json").read_text())
        if payload.get("lane") != lane or payload.get("language") != "Japanese":
            raise ValueError(f"Wrong lane or language: {lane}")
        expected = {qid for qid, question in assignments.items()
                    if question["lane"] == lane and question["coarse_label"] in TARGET_ANSWERS}
        observed = set()
        for record in payload["records"]:
            qid = record["qid"]
            if qid in observed or qid not in expected:
                raise ValueError(f"Unexpected or duplicate q{qid} in lane {lane}")
            if record["image_sha256"] != assignments[qid]["image_sha256"]:
                raise ValueError(f"Image SHA mismatch: q{qid}")
            if record["status"] not in {"transcribed", "uncertain", "illegible"}:
                raise ValueError(f"Unknown OCR status: q{qid}")
            observed.add(qid)
            records[qid] = record
        if observed != expected:
            raise ValueError(f"Missing qids in lane {lane}: {sorted(expected - observed)}")
    return records


def run(ocr_dir: Path, output_dir: Path, html_path: Path,
        previous_dir: Path = ROOT / "data/real-question-ocr-v1") -> dict:
    if output_dir.exists() or html_path.exists():
        raise FileExistsError("Use new versioned output paths")
    assignments = load_assignments(previous_dir)
    records = load_lanes(ocr_dir, assignments)
    onnx_path = ROOT / "data/real-question-onnx-ocr-v4/recognition.json"
    onnx_by_qid = {row["qid"]: row for row in json.loads(onnx_path.read_text())["rows"]}
    if records.keys() != onnx_by_qid.keys():
        raise ValueError("Luna and ONNX question inventories differ")
    output_dir.mkdir(parents=True)
    rows = []
    for qid in sorted(records):
        record = records[qid]
        row = evaluate_one(assignments[qid], {**record, "orientation": matcher_layout(record["orientation"])},
                           output_dir)
        row["source_orientation"] = record["orientation"]
        row["luna_status"] = record["status"]
        row["luna_notes"] = record.get("notes", "")
        row["luna_text"] = record.get("text", "")
        onnx = onnx_by_qid[qid]
        row["onnx_text"] = onnx["rec_text"]
        row["onnx_region_iou"] = box_iou(record.get("region_bbox"), onnx["ocr_bbox"])
        row["same_onnx_region"] = row["onnx_region_iou"] >= 0.6
        row["text_agrees_with_onnx"] = (record.get("text", "") == onnx["rec_text"]
                                         if record["status"] == "transcribed" and row["same_onnx_region"]
                                         else None)
        rows.append(row)
    matched = [row for row in rows if row["status"] == "matched"]
    summary = {"total": len(rows), "luna_status": dict(Counter(row["luna_status"] for row in rows)),
               "judgment_status": dict(Counter(row["status"] for row in rows)),
               "font_ranked": len(matched),
               "proxy_top1": sum(row["proxy_top1_hit"] for row in matched),
               "proxy_top3": sum(row["proxy_top3_hit"] for row in matched),
               "ranked_text_agrees_with_onnx": sum(row["text_agrees_with_onnx"] is True for row in matched)}
    result = {"protocol": {"status": "development_exposed_visual_ocr_diagnostic",
                           "language": "Japanese", "ocr_model": "gpt-6-luna high, six lanes",
                           "ocr_review": "initial visual transcription was independent; second pass could inspect ONNX suggestions as hypotheses",
                           "ocr_source_sha256": {f"lane-{i}.json": digest(ocr_dir / f"lane-{i}.json") for i in range(6)},
                           "onnx_comparison_sha256": digest(onnx_path),
                           "limits": ["Chinese quiz answers are proxy labels, not source font truth",
                                      "Luna text and boxes require human acceptance",
                                      "Luna and ONNX agreement is not independent when ONNX was seen during second-pass review",
                                      "character-count alignment does not prove per-glyph alignment",
                                      "font distances are not calibrated probabilities"]},
              "summary": summary, "rows": rows}
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_html(rows, summary, html_path, result_path, ocr_dir)
    return summary


def write_html(rows: list[dict], summary: dict, html_path: Path, result_path: Path,
               ocr_dir: Path) -> None:
    cards = []
    for row in rows:
        qid = row["qid"]
        image = html.escape(row["image_path"], quote=True)
        region = row.get("region_path")
        crop = (f'<a href="../{html.escape(region, quote=True)}"><img class="region" '
                f'src="../{html.escape(region, quote=True)}" loading="lazy" alt="q{qid} 文字裁片"></a>'
                if region else "")
        ranks = "".join(
            f'<li>{index}. {html.escape(candidate["answer"])} · '
            f'{html.escape(candidate["representative_face"])} · {candidate["appearance_distance"]:.4f}</li>'
            for index, candidate in enumerate(row.get("display_top3", []), 1))
        agreement = (str(row["text_agrees_with_onnx"]) if row["text_agrees_with_onnx"] is not None
                     else "不同区域或未可靠转录")
        cards.append(
            f'<article><h3>q{qid} · {html.escape(row["coarse_label"])} → '
            f'{html.escape(row["target_answer"])} <small>{html.escape(row["status"])}</small></h3>'
            f'<div class="case"><a href="../{image}"><img src="../{image}" '
            f'loading="lazy" alt="q{qid} 原图"></a><div>{crop}'
            f'<p>Luna 日文：<strong>{html.escape(row["luna_text"]) or "（未可靠转录）"}</strong> '
            f'· 状态 {html.escape(row["luna_status"])}</p>'
            f'<p>ONNX 对照：{html.escape(row["onnx_text"]) or "（空）"} · '
            f'区域 IoU {row["onnx_region_iou"]:.2f} · 同区域文字完全一致：{agreement}</p>'
            f'<p>坐标：{html.escape(str(row.get("region_bbox")))} · '
            f'候选字框：{row.get("candidate_count", "—")} · '
            f'选用字：{html.escape(row.get("selected_characters", "")) or "（无）"}</p>'
            f'<p>OCR 说明：{html.escape(row["luna_notes"]) or "（无）"}</p>'
            f'<ol>{ranks}</ol></div></div></article>')
    json_link = html.escape(str(result_path.relative_to(ROOT)), quote=True)
    lane_links = " · ".join(
        f'<a href="../{html.escape(str((ocr_dir / f"lane-{i}.json").relative_to(ROOT)), quote=True)}">lane {i}</a>'
        for i in range(6))
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Luna high 日文 OCR 与字体判断</title><style>
body{{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Noto Sans SC",sans-serif;color:#17242a;background:#f4f6f5;margin:0}}main{{max-width:1100px;margin:auto;padding:28px}}h1{{font-size:30px}}.notice,article{{background:#fff;border:1px solid #d8e2e0;border-radius:12px;padding:18px;margin:16px 0}}.case{{display:grid;grid-template-columns:200px 1fr;gap:18px}}.case img{{max-width:100%;max-height:260px;object-fit:contain}}.case .region{{max-height:150px}}small{{font-size:12px;color:#667}}ol{{padding-left:22px}}@media(max-width:600px){{.case{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Luna high：离线题库日文 OCR 与字体判断</h1><div class="notice">
<p>范围：宋、黑、圆、方圆、少女、海报、雅士黑、方新书、轻吟体中此前没有已核准日文原文的 {summary["total"]} 题。六个 Luna high 子代理按题号分工，首轮直接查看日文原图并选一列文字；部分存疑题的第二轮复核允许将 ONNX 结果作为待验证提示，最终文字仍须以原图为准。</p>
<p>OCR 状态：{html.escape(str(summary["luna_status"]))}。字体判断状态：{html.escape(str(summary["judgment_status"]))}。产生待复核字体排名 {summary["font_ranked"]} 题；中文代理标签 Top-1 {summary["proxy_top1"]}/{summary["font_ranked"]}，Top-3 {summary["proxy_top3"]}/{summary["font_ranked"]}。仅在两个区域重叠 IoU ≥ 0.6 时比较文字；其中 {summary["ranked_text_agrees_with_onnx"]} 题与 ONNX 输出完全相同，此一致数不是独立验证率。</p>
<p>文字、区域和逐字切分仍需人工验收；字数相同不能证明逐字对应。题库中文答案不是日文字体真值，这批题已用于研发而非盲测。字体距离不是正确概率。</p>
<p><a href="../{json_link}">结构化判断</a> · {lane_links}</p></div>{''.join(cards)}</main></body></html>'''
    html_path.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.ocr_dir.resolve(), args.output_dir.resolve(), args.html.resolve()),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
