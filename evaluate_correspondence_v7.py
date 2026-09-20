"""Frozen exposed-data regression for aspect and closed-box correspondence."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from evaluate_features_v6 import panel
from process_regions import ROOT, imread, imwrite
from rank_font_correspondence import augment
from stroke_correspondence_v7 import (CORRESPONDENCE_POLICY, FeatureReferences,
                                      extract, gray_canvas, pair_features)

_REFERENCES = {}


def references(directory):
    key = str(directory)
    if key not in _REFERENCES:
        _REFERENCES[key] = FeatureReferences(directory)
    return _REFERENCES[key]


def evaluate_one(job):
    group, base, sample, baseline, output = job
    refs = references(base/"templates")
    result = augment(sample, base/"templates", baseline["result"], references=refs)
    return {"qid": sample["qid"], "group": group, "expected": baseline["expected"],
            "eligible": baseline["eligible"], "baseline": baseline["result"]["ranking"][0]["answer"],
            "result": result}


def font_table_one(char):
    refs = references(ROOT/"data/category-expansion-v3/templates")
    fonts = {f["postscript"]: f for f in refs.fonts}
    target = fonts["HummingStd-M"]
    rows = []
    for face in ("SkipStd-M", "SkipStd-B", "FolkPro-Bold"):
        rival = fonts[face]
        if char not in rival["templates"]:
            continue
        for pixels in (24, 48, 64):
            a, b = (refs.reference(f, char, pixels) for f in (target, rival))
            rows.append({"a": target["postscript"], "b": face, "character": char,
                         "pixels": pixels, **pair_features(a, b)})
    return rows


def focus_image(sample, refs, result, output, characters):
    faces = {f["postscript"]: f for f in refs.fonts}
    active = {r["key"]: r["representative_face"] for r in result["ranking"]}
    names = [active.get(key, default) for key, default in (("humming", "HummingStd-M"),
                                                            ("skip", "SkipStd-M"),
                                                            ("folk", "FolkPro-Bold"))]
    images, records = [], []
    for char in characters:
        glyph = next((g for g in sample["glyphs"] if g["character"] == char), None)
        if glyph is None:
            continue
        gray, pixels = gray_canvas(imread(ROOT/glyph["path"]))
        source = {"gray": gray, "features": extract(gray, pixels)}
        templates = [refs.reference(faces[name], char, pixels) for name in names]
        strip = np.full((172, 576, 3), 255, np.uint8)
        for i, view in enumerate([source]+templates):
            strip[29:157, 144*i:144*i+128] = panel(view["gray"], view["features"])
        cv2.putText(strip, f"q{sample['qid']} U+{ord(char):04X} {pixels}px | source | {names[0]} | {names[1]} | {names[2]}",
                    (2, 18), cv2.FONT_HERSHEY_SIMPLEX, .29, (0, 0, 0), 1)
        images.append(strip)
        records.append({"character": char, "pixels": pixels, "source": source["features"],
                        "templates": {name: view["features"] for name, view in zip(names, templates)}})
    if images:
        imwrite(output/f"q{sample['qid']}-focus.png", np.vstack(images))
        (output/f"q{sample['qid']}-focus.json").write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n")


def main():
    from closed_box_features import BOX_POLICY
    from stroke_corner_features import CORNER_POLICY
    from stroke_terminal_features import TERMINAL_POLICY
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/correspondence-v7")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/"results.json").exists():
        raise ValueError("Preserve completed v7 experiment")
    code = ("closed_box_features.py", "stroke_correspondence_v7.py", "rank_font_correspondence.py",
            "evaluate_correspondence_v7.py")
    inputs = ("data/category-expansion-v3/pools.json", "data/category-expansion-v3/templates/catalog.json",
              "data/humming-v4/folk-controls/pools.json", "data/humming-v4/folk-controls/templates/catalog.json",
              "data/humming-v4/guarded-results.json", "data/features-v6/final/results.json")
    freeze = {"code": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code},
              "source_snapshot": {p: (ROOT/p).read_text() for p in code},
              "inputs": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},
              "policies": {"correspondence": CORRESPONDENCE_POLICY, "box": BOX_POLICY,
                           "terminal": TERMINAL_POLICY, "corner": CORNER_POLICY},
              "status": "all real cases already exposed; development regression"}
    (output/"freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2)+"\n")
    prior = json.loads((ROOT/"data/humming-v4/guarded-results.json").read_text())
    jobs, pools_by_group = [], {}
    for group, name in (("samples", "data/category-expansion-v3"),
                        ("controls", "data/humming-v4/folk-controls")):
        base = ROOT/name
        pools = {s["qid"]: s for s in json.loads((base/"pools.json").read_text())["samples"]}
        pools_by_group[group] = (pools, base/"templates")
        jobs += [(group, base, pools[r["qid"]], r, output) for r in prior[group]]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(evaluate_one, jobs):
            results.append(row)
            top = row["result"]["ranking"][0]["answer"] if row["result"]["ranking"] else None
            print(row["qid"], row["expected"], top, row["result"]["correspondence_state"], flush=True)
        (output/"results.json").write_text(json.dumps({"status": "development regression", "samples": results},
                                               ensure_ascii=False, indent=2)+"\n")
        refs = references(ROOT/"data/category-expansion-v3/templates")
        humming = next(f for f in refs.fonts if f["postscript"] == "HummingStd-M")
        chars = sorted(c for c in humming["templates"] if '\u3040' <= c <= '\u30ff')
        table = [row for records in executor.map(font_table_one, chars) for row in records]
        (output/"kana-table.json").write_text(json.dumps(table, ensure_ascii=False, indent=2)+"\n")
    for qid, chars in ((108, "知"), (109, "当こ"), (106, "け感"), (21, "当こ"), (75, "しも")):
        pools, directory = pools_by_group["samples"]
        sample = pools.get(qid)
        if sample:
            result = next(r["result"] for r in results if r["qid"] == qid)
            focus_image(sample, references(directory), result, output, chars)
    print("font table", len(table), "informative", sum(r["utility"] > 0 for r in table), flush=True)


if __name__ == "__main__":
    main()
