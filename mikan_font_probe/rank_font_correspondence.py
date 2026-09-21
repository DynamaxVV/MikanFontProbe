"""Opt-in v7: bounded aspect mapping and closed-box correspondence."""
import argparse
import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path

from mikan_font_probe.evaluate_active_selection import quality_order
from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.rank_font_guarded import rank_sample
from mikan_font_probe.stroke_correspondence_v7 import (FeatureReferences, compare_features,
                                      decide, extract, gray_canvas, pair_features)


def augment(sample, templates, baseline, budget=7, references=None):
    result = copy.deepcopy(baseline)
    rows = result.get("ranking", [])
    indices = quality_order(sample.get("glyphs", []))[:budget]
    if not rows or sample.get("polarity") == "light" or any(
            sample["glyphs"][i].get("polarity") == "light" for i in indices):
        return {**result, "ranking": [], "top3": [], "correspondence_state": "unsupported_input",
                "correspondence_pairs": [], "correspondence_changed": False, "auto_accept": False}
    refs = references or FeatureReferences(templates)
    faces = {f["postscript"]: f for f in refs.fonts}
    by_key = {r["key"]: r for r in rows}
    sources = {}

    def source(i, need_features=True):
        if i not in sources:
            glyph = sample["glyphs"][i]
            path = resolve_repo_path(glyph["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != glyph["sha256"]:
                raise ValueError("Original crop changed")
            gray, pixels = gray_canvas(imread(path))
            sources[i] = {"gray": gray, "pixels": pixels}
        if need_features and "features" not in sources[i]:
            sources[i]["features"] = extract(sources[i]["gray"], sources[i]["pixels"])
        return sources[i]

    candidates = [r["key"] for r in rows[:3]]
    pairs = list(combinations(candidates, 2))
    for pair in (("humming", "skip"), ("humming", "folk")):
        if set(pair) <= by_key.keys() and not any(set(pair) == set(p) for p in pairs):
            pairs.append(pair)
    details = []
    for a, b in pairs:
        fa, fb = (faces[by_key[k]["representative_face"]] for k in (a, b))
        available = [i for i in quality_order(sample["glyphs"])
                     if sample["glyphs"][i].get("polarity") != "light"]
        seen = {sample["glyphs"][i]["character"] for i in indices}
        computed, utilities = {}, []
        for i in available:
            glyph = sample["glyphs"][i]
            char = glyph["character"]
            if char not in fa["templates"] or char not in fb["templates"]:
                continue
            pixels = source(i, False)["pixels"]
            x, y = (refs.reference(f, char, pixels) for f in (fa, fb))
            potential = pair_features(x, y)
            computed[i] = (x, y, potential)
            if char not in seen and potential["utility"]:
                utilities.append((potential["utility"]*glyph["quality"]["score"], i))
        selected = indices+[i for _, i in sorted(utilities, reverse=True)[:2]]
        evidence = []
        for i in selected:
            if i not in computed:
                continue
            x, y, potential = computed[i]
            observed = source(i)
            evidence.append({"index": i, "character": sample["glyphs"][i]["character"],
                             "pixels": observed["pixels"], "added": i not in indices,
                             **compare_features(observed, x, y, potential)})
        winner, reason = decide(evidence)
        details.append({"a": a, "b": b, "face_a": fa["postscript"], "face_b": fb["postscript"],
                        "winner": a if winner == 1 else b if winner == -1 else None,
                        "reason": reason, "evidence": evidence})
    minimum = min(row["appearance_distance"] for row in rows)
    close = {row["key"] for row in rows if row["appearance_distance"] <= minimum+.015}
    first = details[0] if details else None
    state = "correspondence_inconclusive"
    if first and first["winner"]:
        if len(indices) < 3 or minimum > .35 or not {first["a"], first["b"]} <= close:
            state = "correspondence_diagnostic_only_guard"
        elif first["winner"] == rows[1]["key"]:
            rows[0], rows[1] = rows[1], rows[0]
            state = "correspondence_top_two_reranked"
        else:
            state = "correspondence_support_leader"
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return {**result, "ranking": rows, "top3": rows[:3], "correspondence_state": state,
            "correspondence_pairs": details,
            "correspondence_changed": rows[0]["key"] != baseline["ranking"][0]["key"],
            "source_features": {str(i): {"pixels": s["pixels"], "features": s.get("features", [])}
                                for i, s in sources.items() if "features" in s},
            "auto_accept": False, "confidence_kind": "uncalibrated_geometry_evidence"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", type=int, required=True)
    parser.add_argument("--pools", type=Path, default=ROOT/"data/category-expansion-v3/pools.json")
    parser.add_argument("--templates", type=Path, default=ROOT/"data/category-expansion-v3/templates")
    args = parser.parse_args()
    sample = next(s for s in json.loads(args.pools.read_text())["samples"] if s["qid"] == args.qid)
    print(json.dumps(augment(sample, args.templates, rank_sample(sample, args.templates)),
                     ensure_ascii=False, indent=2))
