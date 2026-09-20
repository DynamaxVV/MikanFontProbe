"""Freeze v10 first, then replay on previously exposed category/control cases."""

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evaluate_continuous_v8 import _source, _view
from pair_attention_v10 import POLICY, compare, decide
from process_regions import ROOT
from stroke_correspondence_v7 import FeatureReferences


def evaluate_one(job):
    base_name, sample, prior = job
    refs = FeatureReferences(ROOT/base_name/"templates")
    faces = {face["postscript"]: face for face in refs.fonts}
    pairs = []
    for pair in prior["pairs"]:
        a, b = faces[pair["face_a"]], faces[pair["face_b"]]
        evidence = []
        for item in pair["evidence"]:
            glyph = sample["glyphs"][item["index"]]
            source = _source(glyph)
            left = _view(refs, a, glyph["character"], source["pixels"])
            right = _view(refs, b, glyph["character"], source["pixels"])
            evidence.append({"index": item["index"], "character": glyph["character"],
                             "pixels": source["pixels"], "added": item["added"],
                             **compare(source, left, right)})
        pairs.append({"a": pair["a"], "b": pair["b"],
                      "face_a": pair["face_a"], "face_b": pair["face_b"],
                      "decision": decide(evidence), "evidence": evidence})
    return {"qid": sample["qid"], "group": prior["group"], "expected": prior["expected"],
            "baseline_answer": prior["baseline_answer"], "pairs": pairs}


def summary(rows, prior_rows):
    previous = {row["qid"]: row for row in prior_rows}
    totals = defaultdict(int)
    classes = defaultdict(lambda: defaultdict(int))
    decisions = []
    for row in rows:
        labels = {candidate["key"]: candidate["answer"] for candidate
                  in previous[row["qid"]]["result"]["ranking"]}
        for pair in row["pairs"]:
            a, b = pair["a"], pair["b"]
            truth = 1 if labels[a] == row["expected"] and labels[b] != row["expected"] else (
                -1 if labels[b] == row["expected"] and labels[a] != row["expected"] else None)
            if truth is None:
                continue
            vote = pair["decision"]["vote"]
            outcome = "correct" if vote == truth else "wrong" if vote else "abstained"
            totals[outcome] += 1
            classes[row["expected"]][outcome] += 1
            decisions.append({"qid": row["qid"], "a": a, "b": b,
                              "expected": row["expected"], "outcome": outcome,
                              "reason": pair["decision"]["reason"]})
    return {"status": "development_regression_not_blind_or_calibrated",
            "questions": len(rows), "eligible_pairs": len(decisions),
            "pair_outcomes": dict(totals),
            "per_class": {key: dict(value) for key, value in classes.items()},
            "decisions": decisions,
            "note": "Pair counts are correlated within question; no ranking changed."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/pair-attention-v10/final")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing frozen v10 run")
    prior_path = ROOT/"data/continuous-v8/diagnostic-diverse.json"
    prior = json.loads(prior_path.read_text())
    old = json.loads((ROOT/"data/correspondence-v7/final/results.json").read_text())
    jobs = []
    for group, name in (("samples", "data/category-expansion-v3"),
                        ("controls", "data/humming-v4/folk-controls")):
        pool = {s["qid"]: s for s in json.loads((ROOT/name/"pools.json").read_text())["samples"]}
        jobs += [(name, pool[row["qid"]], row) for row in prior["samples"] if row["group"] == group]
    code = ("pair_attention_v10.py", "evaluate_pair_attention_v10.py")
    inputs = ("data/continuous-v8/diagnostic-diverse.json",
              "data/correspondence-v7/final/results.json",
              "data/category-expansion-v3/pools.json",
              "data/category-expansion-v3/templates/catalog.json",
              "data/humming-v4/folk-controls/pools.json",
              "data/humming-v4/folk-controls/templates/catalog.json")
    freeze = {"status": "frozen_before_full_evaluation; all cases previously exposed",
              "code": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code},
              "source_snapshot": {p: (ROOT/p).read_text() for p in code},
              "inputs": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},
              "policy": POLICY,
              "protocol": "52 exposed cases; same v8 selected glyphs/faces; no answer in inference; "
                          "pair decision needs two distinct characters and >=80% agreement; "
                          "no Top-1 mutation; outcomes reported separately for abstentions"}
    args.output.mkdir(parents=True)
    (args.output/"freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2)+"\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(evaluate_one, jobs):
            rows.append(row)
            print(row["qid"], len(row["pairs"]), flush=True)
    (args.output/"results.json").write_text(json.dumps({"status": "frozen_development_regression",
                                                  "samples": rows}, ensure_ascii=False, indent=2)+"\n")
    report = summary(rows, old["samples"])
    (args.output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(report["pair_outcomes"])


if __name__ == "__main__":
    main()
