"""Prepare visually annotated real holdout crops; never import font matchers.

Run with .venv/bin/python -B prepare_fresh_holdout.py --audit, then --build.
The audit freezes the cohort before cropping. Build never substitutes questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from mikan_font_probe.paths import ROOT, resolve_repo_path
DATA = ROOT / "data"
OUT = DATA / "fresh-holdout-v2"
GROUPS = {
    "圆体": ["细圆", "中圆", "粗圆"],
    "宋体": ["细宋", "中宋", "粗宋"],
    "黑体": ["细黑", "中黑", "粗黑"],
}
QID_FIELDS = {"qid", "question_id", "qids", "question_ids", "sample_ids"}
QREF = re.compile(r"(?<![A-Za-z0-9])q(\d+)(?!\d)")
# Concurrent evaluator preparation is prospective, not prior exposure. Exempt
# only these two explicitly identified protocol documents, never result files.
PROSPECTIVE_PROTOCOLS = {
    "data/aligned-holdout-v2/protocol-freeze.json",
    "data/aligned-holdout-v2/eligibility-addendum.json",
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_image(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def qid_references(value, key="", pointer="$", found=None):
    """Only collect ID fields and literal qNN references, never numeric scores.

    In particular, numeric column keys and resolution lists are not question IDs.
    """
    found = [] if found is None else found
    if isinstance(value, dict):
        for name, child in value.items():
            for match in QREF.finditer(name):
                found.append((int(match[1]), f"{pointer}/{name} (key)"))
            qid_references(child, name, f"{pointer}/{name}", found)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            qid_references(child, key, f"{pointer}/{index}", found)
    elif key in QID_FIELDS and re.fullmatch(r"q?\d+", str(value)):
        found.append((int(str(value).removeprefix("q")), pointer))
    elif isinstance(value, str):
        found.extend((int(m[1]), pointer) for m in QREF.finditer(value))
    return found


def history_kind(path):
    name = str(path)
    if any(token in name for token in ("baseline", "holdout", "ranked", "ablation", "active-selection", "active_selection")):
        return "font_development_or_validation"
    if any(token in name for token in ("processed", "annotations", "priority", "eval-", "line-rois", "scorecard")):
        return "segmentation_annotation_or_priority"
    return "other_historical_artifact"


def audit_history():
    evidence = defaultdict(list)
    files = []
    for path in sorted(DATA.rglob("*")):
        if not path.is_file() or OUT in path.parents or "images" in path.relative_to(DATA).parts:
            continue
        if path == DATA / "manifest.json":
            continue
        if str(path.relative_to(ROOT)) in PROSPECTIVE_PROTOCOLS:
            continue
        refs = [(int(m[1]), "artifact filename") for m in QREF.finditer(path.name)]
        if path.suffix == ".json":
            refs.extend(qid_references(read_json(path)))
        elif path.suffix == ".txt":
            refs.extend((int(m[1]), "text qNN reference") for m in QREF.finditer(path.read_text()))
        elif path.suffix == ".npz":
            # Inspect archive member NAMES ONLY: never load a score array.
            with np.load(path, allow_pickle=False) as archive:
                refs.extend((int(m[1]), f"archive key {key}") for key in archive.files for m in QREF.finditer(key))
        relative = str(path.relative_to(ROOT))
        if refs or path.suffix in {".json", ".txt", ".npz"}:
            files.append({"path": relative, "sha256": digest(path.read_bytes()),
                          "kind": history_kind(path), "qids": sorted({q for q, _ in refs})})
        for qid, pointer in refs:
            evidence[qid].append({"path": relative, "location": pointer, "kind": history_kind(path)})
    return files, evidence


def source_inventory(entries):
    rows = []
    for entry in sorted(entries, key=lambda e: e["question_id"]):
        if not entry.get("image_file"):
            rows.append({**entry, "unavailable": True})
            continue
        path = DATA / entry["image_file"]
        # Unchanged decoding also detects identical pixels across containers.
        pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise ValueError(f"Cannot decode {path}")
        rows.append({**entry, "source_path": str(path.relative_to(ROOT)),
                     "source_sha256": digest(path.read_bytes()),
                     "pixel_sha256": digest(str(pixels.shape).encode() + pixels.tobytes()),
                     "shape": list(pixels.shape)})
    return rows


def select_cohort(rows, evidence):
    excluded = set(evidence)
    seen_hashes = {r[k] for r in rows if r["question_id"] in excluded for k in ("source_sha256", "pixel_sha256") if k in r}
    eligible = []
    decisions = []
    for row in rows:
        qid = row["question_id"]
        group = next((g for g, labels in GROUPS.items() if row["coarse_label"] in labels), None)
        if group is None:
            continue
        reason = "eligible"
        if qid in excluded:
            reason = "historical_qid"
        elif row.get("unavailable"):
            reason = "source_unavailable"
        elif any(row[k] in seen_hashes for k in ("source_sha256", "pixel_sha256")):
            reason = "duplicate_historical_or_earlier_source"
        else:
            eligible.append({**row, "expected": group})
            seen_hashes.update(row[k] for k in ("source_sha256", "pixel_sha256"))
        decisions.append({"qid": qid, "group": group, "coarse_label": row["coarse_label"], "decision": reason})
    # First three per group, then ascending-QID supplement to nine if a group is scarce.
    selected = []
    for group in GROUPS:
        selected.extend([r for r in eligible if r["expected"] == group][:3])
    selected_ids = {r["question_id"] for r in selected}
    for row in eligible:
        if len(selected) >= 9:
            break
        if row["question_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["question_id"])
    return sorted(selected, key=lambda r: r["question_id"]), decisions


def audit():
    target = OUT / "audit.json"
    if target.exists():
        raise ValueError("Audit is already frozen; do not overwrite an existing cohort")
    files, evidence = audit_history()
    rows = source_inventory(read_json(DATA / "manifest.json")["entries"])
    selected, decisions = select_cohort(rows, evidence)
    duplicates = {}
    for key in ("source_sha256", "pixel_sha256"):
        groups = defaultdict(list)
        for row in rows:
            if key in row:
                groups[row[key]].append(row["question_id"])
        duplicates[key] = [qids for qids in groups.values() if len(qids) > 1]
    document = {
        "scope": "Local data artifacts before this cohort; no claim about unrecorded external use or manga-author independence",
        "selection_rule": "Union of all historical QIDs excluded; exact byte/pixel duplicate exclusion; ascending numeric QID, first 3 per core group, then ascending QID supplementation to 9. Never replace for difficulty or score.",
        "label_groups": GROUPS, "manifest_sha256": digest((DATA / "manifest.json").read_bytes()),
        "prospective_protocol_exceptions": [{"path": name, "sha256": digest((resolve_repo_path(name)).read_bytes()), "reason": "用户本轮在匹配前冻结的协议及q11资格补充，不是此前使用证据；不豁免任何结果文件"} for name in sorted(PROSPECTIVE_PROTOCOLS) if (resolve_repo_path(name)).exists()],
        "historical_json_count": sum(p["path"].endswith(".json") for p in files),
        "excluded_qids": sorted(evidence), "exclusion_evidence": {str(q): evidence[q] for q in sorted(evidence)},
        "historical_files": files, "original_count": len(rows), "originals": rows,
        "duplicate_groups": duplicates, "core_candidate_decisions": decisions, "selected": selected,
    }
    write_json(target, document)
    print(json.dumps({"excluded_qids": document["excluded_qids"], "selected": [r["question_id"] for r in selected], "duplicates": duplicates}, ensure_ascii=False))


def crop_quality(image, polarity):
    """Input-only quality, same components as existing pool quality; no fonts."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kind = cv2.THRESH_BINARY if polarity == "light" else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(gray, 0, 255, kind + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if not len(xs) or len(xs) == mask.size:
        raise ValueError("Blank/solid annotated crop")
    edge = np.zeros(mask.shape, bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    edge_ratio = float(np.count_nonzero((mask > 0) & edge) / len(xs))
    contrast = float((np.percentile(gray, 95) - np.percentile(gray, 5)) / 255)
    resolution = min(1., min(gray.shape) / 40)
    score = .35 * resolution + .35 * contrast + .30 * max(0., 1 - edge_ratio / .08)
    flags = []
    if edge_ratio > .015:
        flags.append("墨迹触边，可能截断")
    if contrast < .5:
        flags.append("低对比")
    if min(gray.shape) < 24:
        flags.append("分辨率偏低")
    quality = {"score": round(score, 4), "resolution": resolution, "contrast": contrast,
               "edge_ink_fraction": edge_ratio, "flags": flags, "usable": score >= .25}
    return quality, int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def glyph_crop(image, cell, polarity):
    """Tighten a manually assigned original-coordinate cell, preserve pixels."""
    x, y, w, h = cell
    ih, iw = image.shape[:2]
    if not (0 <= x < x + w <= iw and 0 <= y < y + h <= ih):
        raise ValueError(f"Invalid original-image cell {cell}")
    region = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    ink = gray > 160 if polarity == "light" else gray < 155
    ys, xs = np.nonzero(ink)
    if not len(xs):
        raise ValueError(f"No ink in {cell}")
    left, top = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
    right, bottom = min(w, int(xs.max()) + 3), min(h, int(ys.max()) + 3)
    bbox = [x + left, y + top, right - left, bottom - top]
    return bbox, image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]


def reviews(qid, original, glyphs):
    sheet = np.full((((len(glyphs) + 7) // 8) * 112, 8 * 112, 3), 255, np.uint8)
    overlay = original.copy()
    for index, glyph in enumerate(glyphs):
        crop = read_image(resolve_repo_path(glyph["path"]))
        scale = 76 / max(crop.shape[:2])
        thumb = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
        sx, sy = index % 8 * 112 + 4, index // 8 * 112 + 28
        sheet[sy:sy + thumb.shape[0], sx:sx + thumb.shape[1]] = thumb
        label = f"{index} U+{ord(glyph['character']):04X}"
        cv2.putText(sheet, label, (sx, sy - 9), cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 0, 0), 1)
        x, y, w, h = glyph["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (0, 160, 255), 1)
        cv2.putText(overlay, str(index), (x, max(8, y)), cv2.FONT_HERSHEY_SIMPLEX, .3, (200, 0, 0), 1)
    write_image(OUT / "review" / f"q{qid}-glyphs.png", sheet)
    write_image(OUT / "review" / f"q{qid}-source-bboxes.png", overlay)


def build():
    if (OUT / "pools.json").exists():
        raise ValueError("Pool already exists: immutable build; no automatic overwrite")
    audit_doc = read_json(OUT / "audit.json")
    annotations = read_json(OUT / "annotations.json")
    selected = audit_doc["selected"]
    if {a["qid"] for a in annotations["samples"]} != {r["question_id"] for r in selected}:
        raise ValueError("Annotations must retain every frozen selected QID")
    by_id = {a["qid"]: a for a in annotations["samples"]}
    samples = []
    for row in selected:
        qid = row["question_id"]
        annotation = by_id[qid]
        path = ROOT / row["source_path"]
        if digest(path.read_bytes()) != row["source_sha256"]:
            raise ValueError(f"q{qid}: original changed since audit")
        original = read_image(path)
        polarity = annotation.get("polarity", "dark")
        glyphs = []
        for column_index, column in enumerate(annotation["columns"]):
            text, edges = column["text"], column["y_edges"]
            if len(edges) != len(text) + 1 or edges != sorted(set(edges)):
                raise ValueError(f"q{qid}: invalid column {column_index}")
            for index, character in enumerate(text):
                if character == "_":
                    continue
                x0, x1 = column["x_range"]
                cell = [x0, edges[index], x1 - x0, edges[index + 1] - edges[index]]
                bbox, crop = glyph_crop(original, cell, polarity)
                sid = f"q{qid}-c{column_index}-g{index}"
                target = OUT / "crops" / f"{sid}.png"
                write_image(target, crop)
                quality, resolution = crop_quality(crop, polarity)
                glyphs.append({"id": sid, "character": character, "path": str(target.relative_to(ROOT)),
                               "sha256": digest(target.read_bytes()), "bbox": bbox, "cell_bbox": cell,
                               "quality": quality, "resolution": resolution,
                               "script": "kana" if 0x3040 <= ord(character) <= 0x30ff else "kanji",
                               "polarity": polarity, "annotation_method": "visual_original_manual_cell"})
        unique = len({g["character"] for g in glyphs})
        if unique < 7 and not annotation.get("shortfall_reason"):
            raise ValueError(f"q{qid}: unreported character shortfall")
        reviews(qid, original, glyphs)
        samples.append({"qid": qid, "expected": row["expected"], "coarse_label": row["coarse_label"],
                        "source_path": row["source_path"], "source_absolute_path": str(path),
                        "source_sha256": row["source_sha256"], "source_pixel_sha256": row["pixel_sha256"],
                        "polarity": polarity, "glyphs": glyphs, "distinct_characters": unique,
                        "meets_seven_distinct": unique >= 7, "shortfall_reason": annotation.get("shortfall_reason"),
                        "exclusions": annotation["exclusions"], "notes": annotation.get("notes", [])})
    document = {"status": "fresh local-evidence holdout; visually annotated; never font-matched by preparer",
                "schema": "active-selection-v1 compatible, source provenance and shortfall fields added",
                "bbox_format": "original image [x,y,width,height], half-open; crop pixels unmodified",
                "audit_sha256": digest((OUT / "audit.json").read_bytes()),
                "annotations_sha256": digest((OUT / "annotations.json").read_bytes()),
                "samples": samples}
    write_json(OUT / "pools.json", document)
    characters = sorted({g["character"] for r in samples for g in r["glyphs"]})
    (OUT / "characters.txt").write_text("".join(characters) + "\n")
    print(json.dumps({"questions": len(samples), "glyphs": sum(len(r["glyphs"]) for r in samples),
                      "characters": len(characters), "counts": [{"qid": r["qid"], "glyphs": len(r["glyphs"]), "distinct": r["distinct_characters"]} for r in samples]}, ensure_ascii=False))


def verify():
    document = read_json(OUT / "pools.json")
    audit_doc = read_json(OUT / "audit.json")
    assert document["audit_sha256"] == digest((OUT / "audit.json").read_bytes())
    assert document["annotations_sha256"] == digest((OUT / "annotations.json").read_bytes())
    assert [s["qid"] for s in document["samples"]] == [r["question_id"] for r in audit_doc["selected"]]
    assert not set(audit_doc["excluded_qids"]) & {s["qid"] for s in document["samples"]}
    crop_ids, hashes, characters = set(), set(), set()
    for sample in document["samples"]:
        source = ROOT / sample["source_path"]
        assert digest(source.read_bytes()) == sample["source_sha256"]
        assert sample["source_sha256"] not in hashes
        hashes.add(sample["source_sha256"])
        image = read_image(source)
        for glyph in sample["glyphs"]:
            assert glyph["id"] not in crop_ids
            crop_ids.add(glyph["id"])
            assert len(glyph["character"]) == 1
            characters.add(glyph["character"])
            path = resolve_repo_path(glyph["path"])
            assert digest(path.read_bytes()) == glyph["sha256"]
            x, y, w, h = glyph["bbox"]
            assert np.array_equal(read_image(path), image[y:y + h, x:x + w])
            assert glyph["resolution"] > 0
        assert sample["distinct_characters"] == len({g["character"] for g in sample["glyphs"]})
        assert sample["distinct_characters"] >= 7 or sample["shortfall_reason"]
    assert (OUT / "characters.txt").read_text() == "".join(sorted(characters)) + "\n"
    print(f"Verified {len(document['samples'])} questions, {len(crop_ids)} original-pixel crops, {len(characters)} characters; no matching executed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit()
    elif args.build:
        build()
    else:
        verify()
