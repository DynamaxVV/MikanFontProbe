"""Fixed-policy v6 exposed-data regression with separate feature overlays."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from mikan_font_probe.process_regions import ROOT, imread, imwrite
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.rank_font_features import augment
from mikan_font_probe.stroke_feature_matcher import FEATURE_POLICY, FeatureReferences, gray_canvas, pair_features
from mikan_font_probe.stroke_terminal_features import TERMINAL_POLICY

_REFERENCES = {}


def references(directory):
    key = str(directory)
    if key not in _REFERENCES:
        _REFERENCES[key] = FeatureReferences(directory)
    return _REFERENCES[key]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def panel(gray, features):
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = {"square": (220, 90, 0), "rounded": (0, 150, 0), "unknown": (160, 160, 160)}
    for feature in features:
        x, y = map(lambda v: int(round(v)), feature["point"])
        color = colors[feature["state"]]
        if feature["kind"] == "terminal":
            cv2.circle(image, (x, y), 4, color, 1)
        else:
            cv2.rectangle(image, (x-4, y-4), (x+4, y+4), color, 1)
    return image


def evidence_image(sample, result, refs, output):
    families = {f["family_group"]["key"]: f for f in refs.fonts}
    faces = {f["postscript"]: f for f in refs.fonts}
    candidates = {r["key"]: r["representative_face"] for r in result["ranking"]}
    fonts = [faces[candidates[k]] if k in candidates else families[k]
             for k in ("humming", "skip", "folk")]
    tiles = []
    records = []
    for index, glyph in enumerate(sample["glyphs"]):
        if str(index) not in result.get("source_features", {}):
            continue
        source = result["source_features"][str(index)]
        raw = imread(resolve_repo_path(glyph["path"]))
        gray, pixels = gray_canvas(raw)
        references = [refs.reference(f, glyph["character"], pixels) for f in fonts]
        cols = [panel(gray, source["features"])]+[panel(r["gray"], r["features"]) for r in references]
        row = np.full((165, 576, 3), 255, np.uint8)
        for j, cell in enumerate(cols):
            row[24:152, j*144:j*144+128] = cell
        title = f"U+{ord(glyph['character']):04X}  native extent {pixels}px  source | Humming | Skip | Folk"
        cv2.putText(row, title, (2, 16), 0, .34, (0, 0, 0), 1)
        tiles.append(row)
        records.append({"character": glyph["character"], "source": source,
                        "templates": [{"face": f["postscript"], "features": r["features"]}
                                      for f, r in zip(fonts, references)]})
    if tiles:
        imwrite(output/f"q{sample['qid']}-features.png", np.vstack(tiles))
    return {"faces": [f["postscript"] for f in fonts], "characters": records}


def evaluate_one(job):
    group, base, sample, baseline, output = job
    refs = references(base/"templates")
    result = augment(sample, base/"templates", baseline["result"], references=refs)
    row = {"qid": sample["qid"], "group": group, "eligible": baseline["eligible"],
           "expected": baseline["expected"], "baseline": baseline["result"]["ranking"][0]["answer"],
           "result": result}
    if sample["qid"] in (30, 46, 47, 106, 108, 109, 4, 127, 128):
        row["visual"] = evidence_image(sample, result, refs, output)
    return row


def table_one(char):
    refs = references(ROOT/"data/category-expansion-v3/templates")
    fonts = {f["postscript"]: f for f in refs.fonts}
    humming = fonts["HummingStd-M"]
    table = []
    for face in ("SkipStd-M", "SkipStd-B", "FolkPro-Bold"):
        rival = fonts[face]
        if char not in rival["templates"]:
            continue
        for pixels in (24, 48, 64):
            a, b = (refs.reference(f, char, pixels) for f in (humming, rival))
            table.append({"a": "HummingStd-M", "b": face, "character": char,
                          "pixels": pixels, **pair_features(a, b)})
    return table


def main():
    from mikan_font_probe.stroke_corner_features import CORNER_POLICY
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/features-v6")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/"results.json").exists():
        raise ValueError("Preserve completed experiment")
    code = ("stroke_terminal_features.py", "stroke_corner_features.py", "stroke_feature_matcher.py",
            "rank_font_features.py", "evaluate_features_v6.py")
    inputs = ("data/category-expansion-v3/pools.json", "data/category-expansion-v3/templates/catalog.json",
              "data/humming-v4/folk-controls/pools.json", "data/humming-v4/folk-controls/templates/catalog.json",
              "data/humming-v4/guarded-results.json")
    write_json(output/"freeze.json", {"code": {p: hashlib.sha256((resolve_repo_path(p)).read_bytes()).hexdigest() for p in code},
                                    "source_snapshot": {p: (resolve_repo_path(p)).read_text() for p in code},
                                    "inputs": {p: hashlib.sha256((resolve_repo_path(p)).read_bytes()).hexdigest() for p in inputs},
                                    "terminal_policy": TERMINAL_POLICY, "corner_policy": CORNER_POLICY,
                                    "feature_policy": FEATURE_POLICY,
                                    "status": "all real data exposed; development regression, not new holdout"})
    old = json.loads((ROOT/"data/humming-v4/guarded-results.json").read_text())
    results = []
    jobs = []
    for group, name in (("samples", "data/category-expansion-v3"), ("controls", "data/humming-v4/folk-controls")):
        base = resolve_repo_path(name)
        pools = {s["qid"]: s for s in json.loads((base/"pools.json").read_text())["samples"]}
        for baseline in old[group]:
            sample = pools[baseline["qid"]]
            jobs.append((group, base, sample, baseline, output))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(evaluate_one, jobs):
            result = row["result"]
            results.append(row)
            prediction = result["ranking"][0]["answer"] if result["ranking"] else None
            print(row["qid"], prediction, result["feature_state"], flush=True)
        write_json(output/"results.json", {"status": "development regression", "samples": results})
        # Font-only snapshot: exact same features/correspondence as inference.
        refs = references(ROOT/"data/category-expansion-v3/templates")
        humming = next(f for f in refs.fonts if f["postscript"] == "HummingStd-M")
        chars = sorted(c for c in humming["templates"] if '\u3040' <= c <= '\u30ff')
        table = [row for rows in executor.map(table_one, chars) for row in rows]
        write_json(output/"kana-table.json", table)
        print("font table", len(table), "informative", sum(r["utility"] > 0 for r in table), flush=True)


if __name__ == "__main__":
    main()
