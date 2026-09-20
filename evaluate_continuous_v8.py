"""Read-only diagnostic regression on exposed category and control pools."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from continuous_evidence_v8 import axis_width, local_evidence
from evaluate_active_selection import quality_order
from process_regions import ROOT, imread
from stroke_correspondence_v7 import FeatureReferences, extract, gray_canvas, pair_features


def _source(glyph):
    path = ROOT / glyph["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != glyph["sha256"]:
        raise ValueError("Original crop changed")
    gray, pixels = gray_canvas(imread(path))
    return {"gray": gray, "pixels": pixels, "features": extract(gray, pixels)}


def _view(refs, face, char, pixels):
    row = refs.reference(face, char, pixels)
    return {**row, "pixels": pixels}


def _width_evidence(source, a, b):
    values = [axis_width(row["gray"], source["pixels"]) for row in (source, a, b)]
    if any(row["ratio"] is None for row in values):
        return {"values": values, "winner": None, "margin": 0.}
    from math import log
    distances = [abs(log(values[0]["ratio"] / values[i]["ratio"])) for i in (1, 2)]
    return {"values": values, "distance_a": distances[0], "distance_b": distances[1],
            "winner": "a" if distances[0] < distances[1] else "b",
            "margin": abs(distances[0] - distances[1])}


def evaluate_one(job):
    group, base_name, sample, prior = job
    refs = FeatureReferences(ROOT/base_name/"templates")
    ranks = prior["result"]["ranking"]
    faces = {f["postscript"]: f for f in refs.fonts}
    by_key = {r["key"]: r for r in ranks}
    pairs = [(ranks[0]["key"], ranks[1]["key"])] if len(ranks) > 1 else []
    for pair in (("humming", "folk"), ("humming", "skip")):
        if all(key in by_key for key in pair) and set(pair) not in [set(p) for p in pairs]:
            pairs.append(pair)
    order = quality_order(sample["glyphs"])
    outcomes = []
    for key_a, key_b in pairs:
        fa = faces[by_key[key_a]["representative_face"]]
        fb = faces[by_key[key_b]["representative_face"]]
        available = [i for i in order if sample["glyphs"][i]["character"] in fa["templates"]
                     and sample["glyphs"][i]["character"] in fb["templates"]
                     and sample["glyphs"][i].get("polarity") != "light"]
        selected = available[:7]
        candidates = []
        for i in available[7:]:
            glyph = sample["glyphs"][i]
            source = _source(glyph)
            pixels = source["pixels"]
            a = _view(refs, fa, glyph["character"], pixels)
            b = _view(refs, fb, glyph["character"], pixels)
            # Font-only local separability; never reads the source match or answer.
            template = local_evidence(a, a, b)
            utility = max((r["margin"] for r in template), default=0.)
            measurable = local_evidence(source, a, b)
            native_width = max((r["native_width"] for r in measurable), default=0.)
            width_a, width_b = axis_width(a["gray"], pixels), axis_width(b["gray"], pixels)
            if width_a["ratio"] and width_b["ratio"]:
                utility += min(.1, abs(width_a["ratio"] - width_b["ratio"]) / 5)
            raw = utility * glyph["quality"]["score"] * min(1, pixels / 48)
            distinct = pair_features(a, b)["utility"]
            simple = (distinct * min(native_width, 12) / 12 * glyph["quality"]["score"]
                      / max(1, len(a["features"])) if measurable else 0.)
            candidates.append({"index": i, "utility": raw, "simple": simple,
                               "measurable_regions": len(measurable),
                               "template_distinct_regions": distinct})
        first = max(candidates, key=lambda row: (row["utility"], -row["index"]), default=None)
        remaining = [row for row in candidates if row is not first]
        second = max(remaining, key=lambda row: (row["simple"], row["utility"], -row["index"]), default=None)
        added = [row["index"] for row in (first, second) if row and (row["utility"] or row["simple"])]
        evidence = []
        for i in selected + added:
            glyph = sample["glyphs"][i]
            source = _source(glyph)
            a = _view(refs, fa, glyph["character"], source["pixels"])
            b = _view(refs, fb, glyph["character"], source["pixels"])
            evidence.append({"character": glyph["character"], "index": i, "added": i in added,
                             "pixels": source["pixels"], "local": local_evidence(source, a, b),
                             "axis_width": _width_evidence(source, a, b)})
        outcomes.append({"a": key_a, "b": key_b, "face_a": fa["postscript"],
                         "face_b": fb["postscript"], "evidence": evidence,
                         "selection": [row for row in (first, second) if row]})
    return {"qid": sample["qid"], "group": group, "expected": prior["expected"],
            "baseline_answer": ranks[0]["answer"] if ranks else None, "pairs": outcomes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/continuous-v8/diagnostic.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing experiment")
    prior = json.loads((ROOT/"data/correspondence-v7/final/results.json").read_text())
    jobs = []
    for group, name in (("samples", "data/category-expansion-v3"),
                        ("controls", "data/humming-v4/folk-controls")):
        pools = {s["qid"]: s for s in json.loads((ROOT/name/"pools.json").read_text())["samples"]}
        jobs += [(group, name, pools[row["qid"]], row) for row in prior["samples"] if row["group"] == group]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(evaluate_one, jobs))
    provenance = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        "continuous_evidence_v8.py", "evaluate_continuous_v8.py",
        "data/correspondence-v7/final/results.json", "data/category-expansion-v3/pools.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "diagnostic_only_exposed_samples_no_rerank",
                                       "provenance": provenance, "samples": rows}, ensure_ascii=False, indent=2)+"\n")
    print(f"evaluated {len(rows)} cases -> {args.output}")


if __name__ == "__main__":
    main()
