"""Freeze before fresh scoring, then run the unchanged paired query protocol."""
import argparse
import hashlib
import json
import shutil

from mikan_font_probe import cache_selection_scores
from mikan_font_probe import evaluate_active_selection
from mikan_font_probe.aligned_discrimination import AlignedSeparation
from mikan_font_probe.paths import ROOT, resolve_repo_path

OUTPUT = ROOT / "data/aligned-holdout-v2"
FILES = ["aligned_discrimination.py", "informative_selection.py",
         "match_font_baseline.py", "structure_font_baseline.py",
         "font_family_policy.py", "rank_font_candidates.py",
         "cache_selection_scores.py", "evaluate_active_selection.py",
         "render_rank_templates.py", "validate_aligned_holdout.py"]


def hashes():
    result = {name: hashlib.sha256(resolve_repo_path(name).read_bytes()).hexdigest()
              for name in FILES}
    fonts = sorted(p for p in (ROOT/"font_sample").rglob("*")
                   if p.suffix.lower() in (".otf", ".ttf"))
    result.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in fonts})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/"protocol-freeze.json"
    if args.freeze:
        if path.exists():
            raise ValueError("Protocol already frozen; do not overwrite")
        path.write_text(json.dumps({"hashes": hashes(), "budgets": [3, 5, 7],
            "primary": "quality_7 versus active_7 top1 Chinese label agreement",
            "secondary": "3/5 glyphs and Top3; iterative early stop experimental only",
            "threshold": .015, "no_tuning_after_fresh_results": True,
            "scope": "known-character cropped Japanese text; not OCR or exact font truth"},
            ensure_ascii=False, indent=2)+"\n")
        print("Frozen before fresh matching")
        return
    if json.loads(path.read_text())["hashes"] != hashes():
        raise ValueError("Frozen protocol changed")
    if (OUTPUT/"results.json").exists():
        raise ValueError("Fresh results already exist; preserve first evaluation")
    source = ROOT/"data/fresh-holdout-v2/pools.json"
    shutil.copyfile(source, OUTPUT/"pools.json")
    cache_selection_scores.OUTPUT = OUTPUT
    cache_selection_scores.main()
    lookup = AlignedSeparation(OUTPUT/"templates", OUTPUT/"discrimination.json")
    evaluate_active_selection.OUTPUT = OUTPUT
    result = evaluate_active_selection.evaluate_with_table(lookup, [path, resolve_repo_path("aligned_discrimination.py")])
    lookup.save()
    result["status"] = "frozen_policy_new_question_holdout_known_characters"
    result["protocol"] = json.loads(path.read_text())
    result["sample_audit"] = "FRESH_SAMPLE_AUDIT.md"
    (OUTPUT/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
