"""Versioned geometry-rule regression; preserve all old frozen artifacts."""
import argparse
import copy
import hashlib
import json

import numpy as np

import evaluate_active_selection
from geometry_font_matcher import GeometryReferences, prepare, score
from glyph_geometry import effective_resolution
from match_font_baseline import normalize
from process_regions import ROOT, imread

BASE = ROOT/"data/category-expansion-v3"
OUT = ROOT/"data/humming-v4/regression"
CODE = ["geometry_font_matcher.py", "glyph_geometry.py", "evaluate_humming_v4.py",
        "informative_selection.py", "match_font_baseline.py", "structure_font_baseline.py"]


def fingerprints():
    files = [ROOT/name for name in CODE]+[BASE/"pools.json", BASE/"templates/catalog.json"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "run"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = OUT/"freeze.json"
    if args.mode == "freeze":
        if freeze.exists():
            raise ValueError("Already frozen")
        freeze.write_text(json.dumps({"hashes": fingerprints(),
            "rule": "50% existing appearance + 50% bounded centerline geometry; one shared alignment",
            "resolution": "exact periodic duplicate detection in both axes; equal-resolution references",
            "status": "six Humming questions used for diagnosis; 48-question regression is not blind",
            "no_family_specific_bonus": True}, indent=2)+"\n")
        print("Candidate rules frozen before full regression")
        return
    assert json.loads(freeze.read_text())["hashes"] == fingerprints()
    if (OUT/"results.json").exists():
        raise ValueError("Preserve first regression result")
    references = GeometryReferences(BASE/"templates")
    pools = copy.deepcopy(json.loads((BASE/"pools.json").read_text()))
    matrices = {}
    for row in pools["samples"]:
        matrix = np.zeros((len(references.fonts), len(row["glyphs"])))
        for index, glyph in enumerate(row["glyphs"]):
            image = imread(ROOT/glyph["path"])
            metadata = effective_resolution(image)
            glyph["resolution_model"] = metadata
            glyph["resolution"] = metadata["effective"]
            source = prepare(normalize(image))
            for fi, font in enumerate(references.fonts):
                target = references.reference(font, glyph["character"], metadata["effective"])
                matrix[fi,index] = score(source, target)
        matrices[f"q{row['qid']}"] = matrix
        print(f"q{row['qid']} scored; duplicate factors: "
              + str(sorted({g['resolution_model']['repeat_scale'] for g in row['glyphs']})), flush=True)
    (OUT/"pools.json").write_text(json.dumps(pools,ensure_ascii=False,indent=2)+"\n")
    np.savez_compressed(OUT/"match_scores.npz", **matrices)
    (OUT/"templates").mkdir(exist_ok=True)
    (OUT/"templates/catalog.json").write_text((BASE/"templates/catalog.json").read_text())
    paths = [OUT/"pools.json", OUT/"templates/catalog.json", *[ROOT/name for name in CODE]]
    (OUT/"match_freeze.json").write_text(json.dumps({str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+"\n")
    evaluate_active_selection.OUTPUT = OUT
    result = evaluate_active_selection.evaluate_with_table(references,[freeze,ROOT/"geometry_font_matcher.py",ROOT/"glyph_geometry.py"])
    result["status"] = "geometry_rule_development_regression_not_new_holdout"
    # This changed score distribution has no calibrated numeric confidence.
    result["confidence_warning"] = "legacy evidence index only; uncalibrated after scoring change; no automatic acceptance"
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (OUT/"discrimination.json").write_text(json.dumps({json.dumps(k,ensure_ascii=False):v
        for k,v in references.distances.items()},ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
