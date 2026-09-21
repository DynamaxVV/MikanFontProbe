"""Frozen category-coverage extension, not a new algorithm tuning run."""
import argparse
import hashlib
import json
from collections import Counter

from mikan_font_probe import cache_selection_scores
from mikan_font_probe import evaluate_active_selection
from mikan_font_probe.aligned_discrimination import AlignedSeparation
from mikan_font_probe.process_regions import ROOT
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.validate_aligned_holdout import hashes

OUT = ROOT/"data/category-expansion-v3"
GROUPS = {
    "圆体": ["细圆", "中圆", "粗圆"],
    "宋体": ["细宋", "中宋", "粗宋"],
    "黑体": ["细黑", "中黑", "粗黑"],
    "少女": ["少女"], "方圆": ["方圆"],
    "润圆/雅士黑": ["润圆/雅士黑"], "竹体": ["竹体"], "轻吟体": ["轻吟体"],
}
CORE = {"圆体": [5, 8, 10, 28, 91, 100], "宋体": [2, 9, 11, 72, 92, 114],
        "黑体": [1, 21, 26, 58, 96, 125]}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def freeze():
    target = OUT/"protocol.json"
    if target.exists():
        raise ValueError("Protocol exists")
    old = read(ROOT/"data/aligned-holdout-v2/protocol-freeze.json")
    assert hashes() == old["hashes"]
    entries = read(ROOT/"data/manifest.json")["entries"]
    catalog = read(ROOT/"data/aligned-holdout-v2/templates/catalog.json")["fonts"]
    answers = {f["family_group"]["answer"] for f in catalog}
    groups = []
    for answer, labels in GROUPS.items():
        ids = sorted(e["question_id"] for e in entries if e["coarse_label"] in labels)
        assert len(ids) > 5 and answer in answers
        groups.append({"answer": answer, "labels": labels, "available": len(ids),
                       "all_qids": ids, "selected": CORE.get(answer, ids[:6])})
    prior_annotation = set(read(ROOT/"data/fresh-holdout-v2/audit.json")["excluded_qids"])
    prior_font = set()
    for name in ("active-selection-v1/pools.json", "fresh-holdout-v2/pools.json"):
        prior_font.update(s["qid"] for s in read(ROOT/"data"/name)["samples"])
    prior_font.update(s["qid"] for s in read(ROOT/"data/holdout_samples_v1.json")["samples"])
    prior_annotation |= prior_font
    write(target, {"status": "fixed before expanded matching", "groups": groups,
        "matcher_hashes": hashes(), "prior_annotation_qids": sorted(prior_annotation),
        "prior_font_evaluation_qids": sorted(prior_font),
        "primary": "per-class quality_7 versus active_7 on dark >=7-distinct-usable samples",
        "budgets": [3, 5, 7], "early_stop": "exploratory; default off",
        "sampling": "core reviewed-input reuse; other families ascending first six qids; no replacements",
        "new_annotation": "reading order first at most 14 distinct readable regular characters",
        "limits": "category coverage, not all questions; known text, not OCR; coarse Chinese labels"})
    print([(g["answer"], g["available"], g["selected"]) for g in groups])


def collect():
    protocol = read(OUT/"protocol.json")
    assert hashes() == protocol["matcher_hashes"]
    if (OUT/"pools.json").exists():
        raise ValueError("Pool already frozen")
    samples = {}
    for name in ("active-selection-v1/pools.json", "fresh-holdout-v2/pools.json"):
        for sample in read(ROOT/"data"/name)["samples"]:
            samples[sample["qid"]] = {**sample, "annotation_origin": name}
    for lane in ("a", "b", "c"):
        path = OUT/f"lane-{lane}/pools.json"
        for sample in read(path)["samples"]:
            samples[sample["qid"]] = {**sample, "annotation_origin": str(path.relative_to(ROOT))}
    rows = []
    for group in protocol["groups"]:
        for qid in group["selected"]:
            sample = samples[qid]
            assert sample["expected"] == group["answer"]
            for glyph in sample["glyphs"]:
                assert hashlib.sha256((resolve_repo_path(glyph["path"])).read_bytes()).hexdigest() == glyph["sha256"]
            unique = {g["character"] for g in sample["glyphs"] if g["quality"]["usable"]}
            rows.append({**sample, "exposure": "previous_font_evaluation" if qid in protocol["prior_font_evaluation_qids"]
                else "previous_annotation_only" if qid in protocol["prior_annotation_qids"] else "new_question",
                "eligible_seven": len(unique) >= 7 and sample.get("polarity", "dark") == "dark"})
    assert len(rows) == 48 and len({r["qid"] for r in rows}) == 48
    write(OUT/"pools.json", {"status": "mixed_exposure_category_coverage", "samples": rows})
    (OUT/"characters.txt").write_text("".join(sorted({g["character"] for r in rows for g in r["glyphs"]}))+"\n")
    print(dict(Counter(r["exposure"] for r in rows)))


def evaluate():
    protocol = read(OUT/"protocol.json")
    assert hashes() == protocol["matcher_hashes"]
    if (OUT/"results.json").exists():
        raise ValueError("Preserve first evaluation")
    cache_selection_scores.OUTPUT = OUT
    cache_selection_scores.main()
    lookup = AlignedSeparation(OUT/"templates", OUT/"discrimination.json")
    evaluate_active_selection.OUTPUT = OUT
    result = evaluate_active_selection.evaluate_with_table(lookup, [OUT/"protocol.json", resolve_repo_path("aligned_discrimination.py")])
    lookup.save()
    result["status"] = "mixed_exposure_frozen_category_coverage"
    write(OUT/"results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "collect", "evaluate"))
    mode = parser.parse_args().mode
    {"freeze": freeze, "collect": collect, "evaluate": evaluate}[mode]()
