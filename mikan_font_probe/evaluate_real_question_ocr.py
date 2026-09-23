"""Evaluate Japanese OCR regions in the scoped offline question bank."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

from mikan_font_probe.analyze_region import analyze, candidate_boxes_from_mask
from mikan_font_probe.process_regions import ROOT, extract, imread, imwrite


TARGET_ANSWERS = {
    "细宋": "宋体", "中宋": "宋体", "粗宋": "宋体", "雅宋": "宋体",
    "细黑": "黑体", "中黑": "黑体", "粗黑": "黑体",
    "细圆": "圆体", "中圆": "圆体", "粗圆": "圆体",
    "方圆": "方圆", "少女": "少女", "海报体": "海报体",
    "润圆/雅士黑": "润圆/雅士黑", "方新书": "方新书", "轻吟体": "轻吟体",
}
PUNCTUATION = frozenset("。、！？!?・…―〜～（）()「」『』，,．.：:；;\"' ")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alignment_text(text: str, candidate_count: int) -> str | None:
    compact = "".join(text.split())
    if len(compact) == candidate_count:
        return compact
    without_punctuation = "".join(c for c in compact if c not in PUNCTUATION)
    if len(without_punctuation) == candidate_count:
        return without_punctuation
    return None


def load_assignments(ocr_dir: Path) -> dict[int, dict]:
    assignments = {}
    for lane in range(6):
        source = json.loads((ocr_dir / f"assignment-{lane}.json").read_text())
        for question in source["questions"]:
            qid = question["qid"]
            if qid in assignments:
                raise ValueError(f"Duplicate assignment: q{qid}")
            assignments[qid] = {**question, "lane": lane}
    return assignments


def load_ocr(ocr_dir: Path, assignments: dict[int, dict]) -> dict[int, dict]:
    records = {}
    for lane in range(6):
        payload = json.loads((ocr_dir / f"lane-{lane}.json").read_text())
        if payload["lane"] != lane or payload["language"] != "Japanese":
            raise ValueError(f"Wrong lane or language in lane-{lane}.json")
        for record in payload["records"]:
            qid = record["qid"]
            if qid in records or qid not in assignments or assignments[qid]["lane"] != lane:
                raise ValueError(f"Unexpected OCR record: q{qid}")
            if record["image_sha256"] != assignments[qid]["image_sha256"]:
                raise ValueError(f"Image SHA mismatch in OCR: q{qid}")
            records[qid] = record
    if records.keys() != assignments.keys():
        raise ValueError(f"Missing OCR records: {sorted(assignments.keys() - records.keys())}")
    return records


def load_reviewed(result_path: Path) -> list[dict]:
    source = json.loads(result_path.read_text())
    labels = {x["qid"]: x["proxy_answer"] for x in source["inventory"]}
    selected = {}
    for sample in source["samples"]:
        qid = sample["qid"]
        if labels[qid] not in TARGET_ANSWERS or qid in selected:
            continue
        variant = sample.get("variants", {}).get("hybrid_selected", {})
        selected[qid] = {
            "qid": qid, "coarse_label": labels[qid],
            "target_answer": TARGET_ANSWERS[labels[qid]], "source_image": sample["source_image"],
            "status": sample["status"], "cohort": sample["cohort"],
            "characters": "".join(g["character"] for g in sample["glyphs"]),
            "top3": variant.get("display_ranking", [])[:3],
            "proxy_top1_hit": variant.get("proxy_top1_hit"),
            "proxy_top3_hit": variant.get("proxy_top3_hit"),
        }
    return [selected[qid] for qid in sorted(selected)]


def evaluate_one(question: dict, ocr: dict, output_dir: Path) -> dict:
    qid = question["qid"]
    result = {"qid": qid, "coarse_label": question["coarse_label"],
              "target_answer": TARGET_ANSWERS.get(question["coarse_label"]),
              "image_path": question["image_path"], "image_sha256": question["image_sha256"],
              "ocr_status": ocr["status"], "ocr_text": ocr.get("text", ""),
              "region_bbox": ocr.get("region_bbox"), "orientation": ocr.get("orientation"),
              "ocr_notes": ocr.get("notes", ""), "status": "out_of_scope"}
    if result["target_answer"] is None:
        return result
    if ocr["status"] not in {"transcribed", "uncertain"} or not ocr.get("text"):
        result["status"] = "ocr_unavailable"
        return result
    if ocr["status"] == "uncertain":
        result["status"] = "ocr_uncertain"
        return result
    bbox = ocr.get("region_bbox")
    image_path = ROOT / question["image_path"]
    if digest(image_path) != question["image_sha256"]:
        raise ValueError(f"Original image SHA mismatch: q{qid}")
    image = imread(image_path)
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(x, int) for x in bbox):
        result["status"] = "bbox_missing"
        return result
    x, y, width, height = bbox
    if x < 0 or y < 0 or width < 12 or height < 12 or x + width > image.shape[1] or y + height > image.shape[0]:
        result["status"] = "bbox_invalid"
        return result
    if ocr.get("orientation") not in {"vertical", "horizontal"}:
        result["status"] = "layout_unknown"
        return result
    case_dir = output_dir / f"q{qid}"
    case_dir.mkdir()
    crop_path = case_dir / "region.png"
    crop = image[y:y + height, x:x + width]
    imwrite(crop_path, crop)
    mask, _ = extract(crop)
    boxes = candidate_boxes_from_mask(mask, ocr["orientation"])
    text = alignment_text(ocr["text"], len(boxes))
    result["candidate_count"] = len(boxes)
    result["region_path"] = str(crop_path.relative_to(ROOT))
    result["alignment_text"] = text
    if text is None:
        result["status"] = "character_count_mismatch"
        return result
    report = analyze(crop_path, text, layout=ocr["orientation"], output_directory=case_dir / "analysis")
    result["analysis_path"] = str((case_dir / "analysis/analysis.json").relative_to(ROOT))
    result["status"] = report["state"]
    if report["original_polarity"] == "light":
        result["status"] = "light_text_review_required"
        return result
    if report["state"] != "matched":
        return result
    match = report["match"]
    result["selected_characters"] = match.get("characters", "")
    result["selected_count"] = len(match.get("selected_indices", []))
    result["display_top3"] = match.get("display_top3", [])
    result["auto_accept"] = bool(match.get("auto_accept"))
    result["match_state"] = match.get("state")
    if result["selected_count"] < 3:
        result["status"] = "insufficient_supported_glyphs"
        return result
    result["proxy_top1_hit"] = bool(result["display_top3"] and
                                    result["display_top3"][0]["answer"] == result["target_answer"])
    result["proxy_top3_hit"] = result["target_answer"] in [x["answer"] for x in result["display_top3"]]
    return result


def write_html(rows: list[dict], reviewed: list[dict], output_path: Path, result_path: Path) -> None:
    counts = Counter(x["status"] for x in rows)
    target = [x for x in rows if x["target_answer"] is not None]
    matched = [x for x in target if x["status"] == "matched"]
    reviewed_scored = [x for x in reviewed if x["status"] == "scored"]
    reviewed_cards = []
    for row in reviewed:
        path = html.escape(row["source_image"], quote=True)
        ranks = "".join(f'<li>{i}. {html.escape(x["answer"])} · {html.escape(x["representative_face"])} · {x["distance"]:.4f}</li>'
                        for i, x in enumerate(row["top3"], 1))
        reviewed_cards.append(f'<article><h3>q{row["qid"]} · {html.escape(row["coarse_label"])} → {html.escape(row["target_answer"])}'
                              f' <small>{html.escape(row["status"])}</small></h3><div class="case"><a href="../{path}"><img src="../{path}" loading="lazy" alt="q{row["qid"]} 原图"></a>'
                              f'<div><p>人工已核对字：{html.escape(row["characters"])}</p><ol>{ranks}</ol></div></div></article>')
    cards = []
    for row in target:
        top = row.get("display_top3", [])
        ranking = "".join(f'<li>{i}. {html.escape(x["answer"])} · {html.escape(x["representative_face"])} · {x["appearance_distance"]:.4f}</li>'
                          for i, x in enumerate(top, 1))
        path = html.escape(row["image_path"], quote=True)
        region = (f'<img class="region" src="../{html.escape(row["region_path"],quote=True)}" alt="OCR 所选文字区域">'
                  if row.get("region_path") else "")
        cards.append(f'<article><h3>q{row["qid"]} · {html.escape(row["coarse_label"])} → {html.escape(row["target_answer"])}'
                     f' <small>{html.escape(row["status"])}</small></h3><div class="case"><a href="../{path}"><img src="../{path}" loading="lazy" alt="q{row["qid"]} 原图"></a>'
                     f'<div>{region}<p>日文 OCR：<strong>{html.escape(row["ocr_text"])}</strong></p>'
                     f'<p>选用字：{html.escape(row.get("selected_characters", ""))}；候选框：{row.get("candidate_count", "—")}</p>'
                     f'<ol>{ranking}</ol></div></div></article>')
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实题库日文 OCR 与字体验证</title><style>
body{{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Noto Sans SC",sans-serif;color:#16242b;background:#f4f6f5;margin:0}}main{{max-width:1100px;margin:auto;padding:28px}}h1{{font-size:32px}}.notice,article{{background:white;border:1px solid #d8e2e0;border-radius:12px;padding:18px;margin:16px 0}}.case{{display:grid;grid-template-columns:200px 1fr;gap:18px}}.case img{{max-width:100%;max-height:260px;object-fit:contain}}.case .region{{max-height:170px}}small{{font-size:12px;color:#667}}ol{{padding-left:22px}}@media(max-width:600px){{.case{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>真实题库：日文 OCR 与字体验证</h1>
<div class="notice"><p>本轮只看宋、黑、圆、方圆、少女、海报、雅士黑、方新书、轻吟体对应的 {len(target)+len(reviewed)} 题。人工已有文字与裁片 {len(reviewed)} 题；新增日文 OCR {len(target)} 题。日文初转录由 gpt-6-luna 子代理完成；主代理复核了所有标为可转录的文字，纠正或拒绝了与原图不符的记录。OCR 题由本地 26-face 同字图库和当前受保护策略判断。</p>
<p>人工已核对组：{len(reviewed_scored)} 题有评分，中文代理 Top-1 {sum(x["proxy_top1_hit"] for x in reviewed_scored)}/{len(reviewed_scored)}，Top-3 {sum(x["proxy_top3_hit"] for x in reviewed_scored)}/{len(reviewed_scored)}；这是同图同字 27 种配准下的 hybrid 选变换对照结果，详情见 <a href="REAL_SELECTOR_DEVELOPMENT_V5.html">完整对照报告</a>。</p>
<p>新增 OCR 组：成功对齐并匹配 {len(matched)} 题；其余状态：{html.escape(str(dict(counts)))}。匹配题中的中文代理 Top-1 为 {sum(x["proxy_top1_hit"] for x in matched)}/{len(matched)}，Top-3 为 {sum(x["proxy_top3_hit"] for x in matched)}/{len(matched)}。中文答案不是日文字体真值；OCR、选框和切字尚未逐题人工核准，排名是待复核候选。这些题都已暴露于模型或策略研发，不是独立盲测。</p>
<p><a href="../{html.escape(str(result_path.relative_to(ROOT)),quote=True)}">下载结构化结果</a> · <a href="../docs/CURRENT_FONT_SUPPORT.md">可识别字体与限制</a></p></div><h2>人工已核对题</h2>{''.join(reviewed_cards)}<h2>新增日文 OCR 题</h2>{''.join(cards)}</main></body></html>'''
    output_path.write_text(document)


def run(ocr_dir: Path, output_dir: Path, html_path: Path, reviewed_path: Path) -> dict:
    ocr_dir = ocr_dir.resolve()
    output_dir = output_dir.resolve()
    html_path = html_path.resolve()
    reviewed_path = reviewed_path.resolve()
    if output_dir.exists() or html_path.exists():
        raise FileExistsError("Use new versioned result and HTML paths")
    assignments = load_assignments(ocr_dir)
    records = load_ocr(ocr_dir, assignments)
    reviewed = load_reviewed(reviewed_path)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())["entries"]
    expected = {q["question_id"] for q in manifest if q["coarse_label"] in TARGET_ANSWERS}
    observed = {q["qid"] for q in reviewed} | {qid for qid, q in assignments.items()
                                               if q["coarse_label"] in TARGET_ANSWERS}
    if observed != expected or {q["qid"] for q in reviewed} & assignments.keys():
        raise ValueError("Scoped target questions are missing or duplicated")
    output_dir.mkdir(parents=True)
    rows = [evaluate_one(assignments[qid], records[qid], output_dir) for qid in sorted(assignments)]
    result = {"protocol": {"status": "development_exposed_ocr_diagnostic",
                           "language": "Japanese", "ocr_model": "gpt-6-luna",
                           "ocr_review": "primary agent reviewed all transcribed regions against original images",
                           "reviewed_result_sha256": digest(reviewed_path),
                           "ocr_source_sha256": {f"lane-{i}.json": digest(ocr_dir / f"lane-{i}.json") for i in range(6)},
                           "limits": ["Chinese question labels are proxy labels, not Japanese source font truth",
                                      "OCR and region boxes have not been manually certified",
                                      "only exact character-count alignment with at least three supported glyphs is scored"]},
              "reviewed": reviewed, "rows": rows}
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_html(rows, reviewed, html_path, result_path)
    return {"target": sum(x["target_answer"] is not None for x in rows),
            "reviewed": len(reviewed),
            "matched": sum(x["status"] == "matched" for x in rows),
            "status": dict(Counter(x["status"] for x in rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-dir", type=Path, default=ROOT / "data/real-question-ocr-v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/real-question-validation-v1")
    parser.add_argument("--html", type=Path, default=ROOT / "output/REAL_QUESTION_VALIDATION_V1.html")
    parser.add_argument("--reviewed-results", type=Path,
                        default=ROOT / "data/real-selector-development-v5/results.json")
    args = parser.parse_args()
    print(json.dumps(run(args.ocr_dir, args.output_dir, args.html, args.reviewed_results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
