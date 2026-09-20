"""Render frozen v10 part decisions without modifying inference results."""

import json

import cv2
import numpy as np

from evaluate_continuous_v8 import _source, _view
from process_regions import ROOT, imwrite
from stroke_correspondence_v7 import FeatureReferences


def render(qid, character, keys, output):
    rows = json.loads((output/"results.json").read_text())["samples"]
    row = next(item for item in rows if item["qid"] == qid)
    pair = next(item for item in row["pairs"] if (item["a"], item["b"]) == keys)
    glyph = next(item for item in pair["evidence"] if item["character"] == character)
    base = ROOT/("data/category-expansion-v3" if row["group"] == "samples"
                 else "data/humming-v4/folk-controls")
    pool = next(item for item in json.loads((base/"pools.json").read_text())["samples"]
                if item["qid"] == qid)
    source = _source(pool["glyphs"][glyph["index"]])
    refs = FeatureReferences(base/"templates")
    faces = {face["postscript"]: face for face in refs.fonts}
    templates = [_view(refs, faces[name], character, source["pixels"])["gray"]
                 for name in (pair["face_a"], pair["face_b"])]
    panels = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
              for image in (source["gray"], *templates)]
    colors = {1: (255, 120, 0), -1: (0, 90, 255), 0: (150, 150, 150)}
    for region in glyph["regions"]:
        x0, y0, x1, y1 = region["bbox"]
        x0, y0, x1, y1 = [int(v) for v in (x0, y0, x1, y1)]
        cv2.rectangle(panels[0], (max(0, x0), max(0, y0)),
                      (min(127, x1), min(127, y1)), colors[region["vote"]], 1)
    image = np.full((156, 384, 3), 255, np.uint8)
    image[28:, :] = cv2.hconcat(panels)
    caption = f"q{qid} U+{ord(character):04X}: source | {pair['a']} | {pair['b']}"
    cv2.putText(image, caption, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 1)
    imwrite(output/f"q{qid}-{ord(character):04x}-parts.png", image)


if __name__ == "__main__":
    output = ROOT/"data/pair-attention-v10/final"
    for qid, character, keys in ((108, "知", ("humming", "folk")),
                                  (100, "夜", ("shinmaru", "humming")),
                                  (109, "こ", ("humming", "skip"))):
        render(qid, character, keys, output)
