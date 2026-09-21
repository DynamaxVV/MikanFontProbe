"""Opt-in v5: bounded contour reranking and explicit contradictory evidence."""
import argparse
import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path

from mikan_font_probe.evaluate_active_selection import quality_order
from mikan_font_probe.geometry_font_matcher import GeometryReferences
from mikan_font_probe.glyph_geometry import effective_resolution
from mikan_font_probe.local_contour_matcher import compare, decide, template_evidence
from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.rank_font_guarded import rank_sample


def augment(sample, templates, baseline, budget=7):
    result = copy.deepcopy(baseline)
    rows = result.get("ranking", [])
    if not rows or sample.get("polarity") == "light" or any(
            g.get("polarity") == "light" for g in sample.get("glyphs", [])):
        return {**result, "ranking": [], "top3": [], "contour_state": "unsupported_input",
                "contour_pairs": [], "contour_changed": False, "auto_accept": False}
    refs = GeometryReferences(templates)
    faces = {f["postscript"]: f for f in refs.fonts}
    by_key = {r["key"]: r for r in rows}
    indices = quality_order(sample["glyphs"])[:budget]
    images, pixels = {}, {}

    def source(i):
        if i not in images:
            glyph = sample["glyphs"][i]
            path = resolve_repo_path(glyph["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != glyph["sha256"]:
                raise ValueError("Original crop changed")
            image = imread(path)
            images[i] = normalize(image)
            pixels[i] = effective_resolution(image)["effective"]
        return images[i]

    # Runtime contenders, plus a declared Humming/Skip/Folk diagnostic panel.
    # Diagnostic pairs are fixed regardless of the expected label, never boosted.
    candidates = [r["key"] for r in rows[:3]]
    pairs = list(combinations(candidates, 2))
    for pair in (("humming", "skip"), ("humming", "folk")):
        if set(pair) <= by_key.keys() and not any(set(pair) == set(p) for p in pairs):
            pairs.append(pair)
    details = []
    for a, b in pairs:
        fa, fb = (faces[by_key[k]["representative_face"]] for k in (a, b))
        selected = list(indices)
        utilities = []
        seen = {sample["glyphs"][i]["character"] for i in selected}
        for i in quality_order(sample["glyphs"]):
            glyph = sample["glyphs"][i]
            char = glyph["character"]
            if char in seen or char not in fa["templates"] or char not in fb["templates"]:
                continue
            source(i)  # Resolution/quality only: no candidate-vs-source score here.
            x, y = (refs.reference(f, char, pixels[i])["ink"] for f in (fa, fb))
            evidence = template_evidence(x, y, pixels[i])
            utility = max(0., evidence["separation"]-evidence["threshold"])*glyph["quality"]["score"]
            if evidence["reliable"]:
                utilities.append((utility, i))
        selected += [i for _, i in sorted(utilities, reverse=True)[:2]]
        evidence_rows = []
        for i in selected:
            glyph = sample["glyphs"][i]
            char = glyph["character"]
            if char not in fa["templates"] or char not in fb["templates"]:
                continue
            src = source(i)
            x, y = (refs.reference(f, char, pixels[i])["ink"] for f in (fa, fb))
            evidence, _ = compare(src, x, y, pixels[i])
            evidence_rows.append({"index": i, "character": char, "pixels": pixels[i],
                                  "added": i not in indices, **evidence})
        decision = decide([r["vote"] for r in evidence_rows])
        details.append({"a": a, "b": b, "face_a": fa["postscript"],
                        "face_b": fb["postscript"], "evidence": evidence_rows,
                        "winner": a if decision == 1 else b if decision == -1 else None})

    best_appearance = min(r["appearance_distance"] for r in rows)
    close = {r["key"] for r in rows if r["appearance_distance"] <= best_appearance+.015}
    # Only top-two reversal is authorized; all other pairs are diagnostic.
    first = details[0] if details else None
    state = "contour_inconclusive"
    if first and first["winner"]:
        if len(indices) < 3 or best_appearance > .35:
            state = "contour_guard_preserved"
        elif not {first["a"], first["b"]} <= close:
            state = "contour_diagnostic_only_clear_appearance_lead"
        elif first["winner"] == rows[1]["key"]:
            rows[0], rows[1] = rows[1], rows[0]
            state = "contour_top_two_reranked"
        else:
            state = "contour_supports_current_leader"
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return {**result, "ranking": rows, "top3": rows[:3], "contour_state": state,
            "contour_changed": rows[0]["key"] != baseline["ranking"][0]["key"],
            "contour_pairs": details, "auto_accept": False,
            "confidence_kind": "uncalibrated_local_evidence_not_probability"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", type=int, required=True)
    parser.add_argument("--pools", type=Path, default=ROOT/"data/category-expansion-v3/pools.json")
    parser.add_argument("--templates", type=Path, default=ROOT/"data/category-expansion-v3/templates")
    args = parser.parse_args()
    sample = next(s for s in json.loads(args.pools.read_text())["samples"] if s["qid"] == args.qid)
    print(json.dumps(augment(sample, args.templates, rank_sample(sample, args.templates)),
                     ensure_ascii=False, indent=2))
