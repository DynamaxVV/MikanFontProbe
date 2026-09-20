"""Paired equal-budget query comparison over the same reviewed glyph pools."""

import hashlib
import json
from pathlib import Path

import numpy as np

from informative_selection import run_selection, ranking_for, explain_confidence
from process_regions import ROOT

OUTPUT = ROOT/"data/active-selection-v1"


def quality_order(glyphs):
    """Fair baseline: highest quality, also avoid repeating identical text."""
    order,seen = [],set()
    for i,glyph in sorted(enumerate(glyphs),key=lambda row:(-row[1]["quality"]["score"],row[0])):
        if glyph["quality"]["usable"] and glyph["character"] not in seen:
            order.append(i)
            seen.add(glyph["character"])
    return order


def evaluate_with_table(separation, table_files):
    freeze = json.loads((OUTPUT/"match_freeze.json").read_text())
    for relative,expected in freeze.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Matcher inputs changed: {relative}")
    pools = json.loads((OUTPUT/"pools.json").read_text())["samples"]
    fonts = json.loads((OUTPUT/"templates/catalog.json").read_text())["fonts"]
    matrices = np.load(OUTPUT/"match_scores.npz")
    rows = []
    for sample in pools:
        qid,glyphs = sample["qid"],sample["glyphs"]
        scores = matrices[f"q{qid}"]
        if scores.shape != (len(fonts),len(glyphs)):
            raise ValueError("Mismatched font/glyph order")
        variants = {}
        clear_order = quality_order(glyphs)
        active = run_selection(fonts,scores,glyphs,separation,budget=7)
        for budget in (3,5,7):
            for strategy,selected in (("quality",clear_order[:budget]),("active",active["selected"][:budget])):
                ranking = ranking_for(fonts,scores,glyphs,selected)
                variants[f"{strategy}_{budget}"] = {
                    "selected":selected,"characters":"".join(glyphs[i]["character"] for i in selected),
                    "ranking":ranking,"explanation":explain_confidence(fonts,scores,glyphs,selected,ranking,separation)}
        iterative = run_selection(fonts,scores,glyphs,separation,budget=7,adaptive_stop=True)
        variants["iterative"] = {**iterative,"characters":"".join(glyphs[i]["character"] for i in iterative["selected"])}
        all_selected = [i for i,g in enumerate(glyphs) if g["quality"]["usable"]]
        variants["all_pool_reference"] = {"selected":all_selected,
            "ranking":ranking_for(fonts,scores,glyphs,all_selected)}
        rows.append({**sample,"variants":variants,"active_trace":active["trace"]})
    summary = {}
    for key in rows[0]["variants"]:
        summary[key] = {"top1":sum(r["variants"][key]["ranking"][0]["answer"]==r["expected"] for r in rows),
                        "top3":sum(r["expected"] in [x["answer"] for x in r["variants"][key]["ranking"][:3]] for r in rows),
                        "glyphs":sum(len(r["variants"][key]["selected"]) for r in rows)}
    paths = [ROOT/"informative_selection.py",ROOT/"evaluate_active_selection.py",OUTPUT/"pools.json",OUTPUT/"match_scores.npz",*map(Path,table_files)]
    provenance = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = {"status":"paired_development_regression_not_holdout","question_count":len(rows),
              "provenance":provenance,"summary":summary,"samples":rows,
              "notes":["known text and reviewed crops; not OCR end-to-end", "same full font library and fixed matcher for every query policy",
                       "quality comparator also deduplicates characters", "cached unselected scores never enter choose_next"]}
    (OUTPUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False))
    for row in rows:
        print(row["qid"],row["expected"],{k:v["ranking"][0]["answer"] for k,v in row["variants"].items()})
    return result
