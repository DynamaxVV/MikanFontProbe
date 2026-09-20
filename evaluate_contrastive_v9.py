"""Evaluate pair-driven patch saliency without reranking any candidate."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from contrastive_patch_v9 import compare
from evaluate_continuous_v8 import _source, _view
from process_regions import ROOT, imwrite
from stroke_correspondence_v7 import FeatureReferences


def evaluate_one(job):
    base_name, sample, prior = job
    refs = FeatureReferences(ROOT/base_name/"templates")
    faces = {face["postscript"]: face for face in refs.fonts}
    glyphs = sample["glyphs"]
    pairs = []
    for pair in prior["pairs"]:
        a, b = faces[pair["face_a"]], faces[pair["face_b"]]
        evidence = []
        for item in pair["evidence"]:
            glyph = glyphs[item["index"]]
            source = _source(glyph)
            views = [_view(refs, face, glyph["character"], source["pixels"])
                     for face in (a, b)]
            modes = {mode: compare(source["gray"], views[0]["gray"], views[1]["gray"],
                                   source["pixels"], mode)
                     for mode in ("ink", "edge", "skeleton")}
            boxes = [feature for feature in source["features"] if feature["kind"] == "box_corner"]
            if boxes:
                box = max(boxes, key=lambda row: (row["box_bbox"][2]-row["box_bbox"][0]) *
                          (row["box_bbox"][3]-row["box_bbox"][1]))
                x0, y0, x1, y1 = box["box_bbox"]
                pad = int(round(1.5 * box["width"]))
                region = (x0-pad, y0-pad, x1+pad, y1+pad)
                modes.update({f"box_{mode}": compare(source["gray"], views[0]["gray"],
                                                    views[1]["gray"], source["pixels"], mode, region)
                              for mode in ("ink", "edge", "skeleton")})
            evidence.append({"character": glyph["character"], "index": item["index"],
                             "added": item["added"], "pixels": source["pixels"], "modes": modes})
        pairs.append({"a": pair["a"], "b": pair["b"], "face_a": pair["face_a"],
                      "face_b": pair["face_b"], "evidence": evidence})
    return {"qid": sample["qid"], "group": prior["group"], "expected": prior["expected"],
            "baseline_answer": prior["baseline_answer"], "pairs": pairs}


def focus_image(row, sample, refs, char, pair_keys, output):
    pair = next((p for p in row["pairs"] if (p["a"], p["b"]) == pair_keys), None)
    if pair is None:
        return
    entry = next((g for g in pair["evidence"] if g["character"] == char), None)
    if entry is None:
        return
    face_map = {face["postscript"]: face for face in refs.fonts}
    source = _source(sample["glyphs"][entry["index"]])
    views = [_view(refs, face_map[name], char, source["pixels"])["gray"]
             for name in (pair["face_a"], pair["face_b"])]
    panels = []
    for i, gray in enumerate([source["gray"]]+views):
        panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if i == 0:
            for patch in entry["modes"]["skeleton"]["patches"]:
                x, y = patch["center"]
                radius = patch["size"] // 2
                cv2.rectangle(panel, (x-radius, y-radius), (x+radius, y+radius), (0, 0, 255), 1)
        panels.append(panel)
    imwrite(output/f"q{row['qid']}-{ord(char):04x}-patches.png", cv2.hconcat(panels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/contrastive-v9/final")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if (args.output/"results.json").exists():
        raise ValueError("Preserve completed experiment")
    prior = json.loads((ROOT/"data/continuous-v8/diagnostic-diverse.json").read_text())
    jobs, pools = [], {}
    for group, name in (("samples", "data/category-expansion-v3"),
                        ("controls", "data/humming-v4/folk-controls")):
        pool = {s["qid"]: s for s in json.loads((ROOT/name/"pools.json").read_text())["samples"]}
        pools[group] = (pool, name)
        jobs += [(name, pool[row["qid"]], row) for row in prior["samples"] if row["group"] == group]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(evaluate_one, jobs))
    args.output.mkdir(parents=True, exist_ok=True)
    code = ("contrastive_patch_v9.py", "evaluate_contrastive_v9.py")
    provenance = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code}
    provenance["data/continuous-v8/diagnostic-diverse.json"] = hashlib.sha256(
        (ROOT/"data/continuous-v8/diagnostic-diverse.json").read_bytes()).hexdigest()
    (args.output/"results.json").write_text(json.dumps({"status": "diagnostic_only_exposed_samples",
                                                     "provenance": provenance, "samples": rows},
                                                    ensure_ascii=False, indent=2)+"\n")
    pool, name = pools["samples"]
    refs = FeatureReferences(ROOT/name/"templates")
    for qid, char, keys in ((108, "知", ("humming", "folk")),
                            (109, "こ", ("humming", "skip"))):
        focus_image(next(row for row in rows if row["qid"] == qid), pool[qid],
                    refs, char, keys, args.output)
    print(f"evaluated {len(rows)} cases -> {args.output}")


if __name__ == "__main__":
    main()
