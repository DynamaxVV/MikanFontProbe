"""Focused diagnostic figures; these extra inspections do not vote in ranking."""
import json
from pathlib import Path

import cv2
import numpy as np

from mikan_font_probe.evaluate_features_v6 import panel
from mikan_font_probe.process_regions import ROOT, imread, imwrite
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.stroke_feature_matcher import FeatureReferences, extract, gray_canvas
from mikan_font_probe.testing_shapes import bend
from mikan_font_probe.testing_shapes import shaft


def main():
    output = ROOT/"data/features-v6/final"
    controls = [("flat cap", shaft()), ("round cap", shaft(True)),
                ("square connection", bend()), ("round connection", bend(radius=12))]
    sheet = np.full((172, 640, 3), 255, np.uint8)
    for i, (name, gray) in enumerate(controls):
        sheet[28:156, i*160:i*160+128] = panel(gray, extract(gray, 104))
        cv2.putText(sheet, name, (i*160+2, 17), 0, .38, (0, 0, 0), 1)
    imwrite(output/"synthetic-models.png", sheet)
    pools = {s["qid"]: s for s in json.loads((ROOT/"data/category-expansion-v3/pools.json").read_text())["samples"]}
    refs = FeatureReferences(ROOT/"data/category-expansion-v3/templates")
    fonts = {f["postscript"]: f for f in refs.fonts}
    cases = []
    panels = []
    for qid, char in ((108, "知"), (109, "当"), (109, "こ"), (106, "け")):
        glyph = next(g for g in pools[qid]["glyphs"] if g["character"] == char)
        gray, pixels = gray_canvas(imread(resolve_repo_path(glyph["path"])))
        source = {"gray": gray, "features": extract(gray, pixels)}
        faces = ("HummingStd-M", "SkipStd-M", "FolkPro-Bold")
        templates = [refs.reference(fonts[face], char, pixels) for face in faces]
        row = np.full((174, 576, 3), 255, np.uint8)
        for i, item in enumerate([source]+templates):
            row[30:158, i*144:i*144+128] = panel(item["gray"], item["features"])
        cv2.putText(row, f"q{qid} U+{ord(char):04X} {pixels}px: source / Humming-M / Skip-M / Folk-B",
                    (2, 18), 0, .34, (0, 0, 0), 1)
        panels.append(row)
        cases.append({"qid": qid, "character": char, "pixels": pixels, "source": source["features"],
                      "templates": {face: template["features"] for face, template in zip(faces, templates)}})
    imwrite(output/"focused-cases.png", np.vstack(panels))
    (output/"focused-cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
