"""Frozen-policy, exposed-data regression and local failure evidence export."""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from mikan_font_probe.geometry_font_matcher import GeometryReferences
from mikan_font_probe.local_contour_matcher import POLICY, compare, template_evidence
from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import ROOT, imread, imwrite
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.rank_font_contours import augment

OUT = ROOT/"data/contours-v5"


def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def failure_image(sample, refs, result):
    pair = result["contour_pairs"][0]
    if sample.get("expected") == "轻吟体":  # Display only; never used in matching.
        pair = next((p for p in result["contour_pairs"] if {p["a"], p["b"]} == {"humming", "folk"}), pair)
    faces = {f["postscript"]: f for f in refs.fonts}
    fa, fb = faces[pair["face_a"]], faces[pair["face_b"]]
    tiles = []
    for row in pair["evidence"]:
        glyph = sample["glyphs"][row["index"]]
        src = normalize(imread(resolve_repo_path(glyph["path"])))
        a, b = (refs.reference(f, row["character"], row["pixels"])["ink"] for f in (fa, fb))
        _, (aa, bb, mask) = compare(src, a, b, row["pixels"])
        cells = []
        for ink in (src, aa, bb):
            cell = cv2.cvtColor(255-ink, cv2.COLOR_GRAY2BGR)
            cell[mask] = (.75*cell[mask]+.25*np.array([0, 180, 255])).astype(np.uint8)
            cells.append(cv2.resize(cell, (128, 128), interpolation=cv2.INTER_NEAREST))
        tile = np.full((158, 580, 3), 255, np.uint8)
        tile[:128, :384] = np.hstack(cells)
        text = f"U+{ord(row['character']):04X} {row['pixels']}px vote={row['vote']}"
        cv2.putText(tile, text, (5, 149), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        cv2.putText(tile, f"A {row['distance_a']:.3f}", (395, 40), 0, .5, (0, 0, 0), 1)
        cv2.putText(tile, f"B {row['distance_b']:.3f}", (395, 70), 0, .5, (0, 0, 0), 1)
        tiles.append(tile)
    if tiles:
        imwrite(OUT/f"q{sample['qid']}-local.png", np.vstack(tiles))
    return {"a": pair["face_a"], "b": pair["face_b"], "characters": [e["character"] for e in pair["evidence"]]}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"results.json").exists():
        raise ValueError("Preserve completed v5 experiment")
    code = ("local_contour_matcher.py", "rank_font_contours.py", "evaluate_contours_v5.py")
    inputs = ("data/category-expansion-v3/pools.json", "data/category-expansion-v3/templates/catalog.json",
              "data/humming-v4/folk-controls/pools.json", "data/humming-v4/folk-controls/templates/catalog.json",
              "data/humming-v4/guarded-results.json")
    save("freeze.json", {"policy": POLICY, "status": "all real samples already exposed; not blind validation",
                         "inputs": {p: hashlib.sha256((resolve_repo_path(p)).read_bytes()).hexdigest() for p in inputs},
                         "code": {p: hashlib.sha256((resolve_repo_path(p)).read_bytes()).hexdigest() for p in code},
                         "rerank": "top two only, both within old .015 appearance band; >=3 votes, >=80% agreement",
                         "additional_queries": "at most 2 distinct chars per pair, font-only utility, fixed face per sample"})
    templates = ROOT/"data/category-expansion-v3/templates"
    refs = GeometryReferences(templates)
    groups = {k: [f for f in refs.fonts if f["family_group"]["key"] == k]
              for k in ("humming", "skip", "folk")}
    table = []
    for a, b in (("humming", "skip"), ("humming", "folk")):
        for fa, fb in product(groups[a], groups[b]):
            for char in sorted(set(fa["templates"]) & set(fb["templates"])):
                if not '\u3040' <= char <= '\u30ff':
                    continue
                for pixels in (16, 24, 32, 48, 64):
                    x, y = (refs.reference(f, char, pixels)["ink"] for f in (fa, fb))
                    table.append({"a": fa["postscript"], "b": fb["postscript"], "character": char,
                                  "pixels": pixels, **template_evidence(x, y, pixels)})
    save("kana-table.json", table)
    print("table", len(table), flush=True)
    old = json.loads((ROOT/"data/humming-v4/guarded-results.json").read_text())
    results = []
    for group, pool_path, template_path in (
        ("samples", ROOT/"data/category-expansion-v3/pools.json", templates),
        ("controls", ROOT/"data/humming-v4/folk-controls/pools.json", ROOT/"data/humming-v4/folk-controls/templates")):
        pools = {s["qid"]: s for s in json.loads(pool_path.read_text())["samples"]}
        refs = GeometryReferences(template_path)
        for baseline in old[group]:
            sample = pools[baseline["qid"]]
            result = augment(sample, template_path, baseline["result"])
            row = {"qid": sample["qid"], "group": group, "expected": baseline["expected"],
                   "eligible": baseline["eligible"], "baseline": baseline["result"]["ranking"][0]["answer"],
                   "result": result}
            predicted = result["ranking"][0]["answer"] if result.get("ranking") else None
            if predicted != row["expected"] and result.get("contour_pairs"):
                row["image"] = failure_image(sample, refs, result)
            results.append(row)
            print(sample["qid"], row["baseline"], predicted, result["contour_state"], flush=True)
    save("results.json", {"status": "development regression, not independent validation", "samples": results})


if __name__ == "__main__":
    main()
