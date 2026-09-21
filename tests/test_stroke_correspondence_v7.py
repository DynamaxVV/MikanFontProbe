"""Cases that nearest-after-uniform alignment previously missed."""
import json
import unittest

import cv2
import numpy as np

from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.stroke_correspondence_v7 import correspondence, pair_features, compare_features, extract
from mikan_font_probe.stroke_feature_matcher import gray_canvas


def image(left, right):
    gray = np.full((128, 128), 255, np.uint8)
    cv2.rectangle(gray, (left, 12), (right, 115), 0, -1)
    return gray


def endpoint(x, y, dx=1):
    return {"kind": "terminal", "point": [x, y], "direction": [dx, 0],
            "state": "rounded", "native_width": 8, "reason": "synthetic"}


def boxcorner(x, y, role, center):
    return {"kind": "box_corner", "point": [x, y], "role": role,
            "box_center": center, "box_bbox": [x, y, x+12, y+36],
            "width": 8, "state": "square", "native_width": 8,
            "reason": "synthetic"}


class CorrespondenceTests(unittest.TestCase):
    def test_independent_axis_map_recovers_narrower_glyph(self):
        source = {"gray": image(19, 109), "features": [endpoint(91, 22)]}
        target = {"gray": image(12, 115), "features": [endpoint(103, 16)]}
        self.assertEqual(correspondence(source, target)[0], [(0, 0)])

    def test_incompatible_direction_and_excessive_stretch_abstain(self):
        source = {"gray": image(19, 109), "features": [endpoint(91, 22)]}
        wrong = {"gray": image(12, 115), "features": [endpoint(103, 16, -1)]}
        self.assertEqual(correspondence(source, wrong)[0], [])
        distorted = {"gray": image(50, 80), "features": [endpoint(70, 22)]}
        self.assertEqual(correspondence(source, distorted)[1], "aspect_outside_bounded_range")

    def test_closed_box_requires_same_corner_role_and_center(self):
        source = {"gray": image(19, 109), "features": [boxcorner(85, 40, "tl", [90, 60])]}
        target = {"gray": image(12, 115), "features": [boxcorner(93, 40, "tr", [99, 60])]}
        self.assertEqual(correspondence(source, target)[0], [])
        target["features"][0]["role"] = "tl"
        self.assertEqual(correspondence(source, target)[0], [(0, 0)])
        target["features"][0]["box_center"] = [30, 30]
        self.assertEqual(correspondence(source, target)[0], [])

    def test_isolated_strokes_with_different_centers_do_not_mix(self):
        left = endpoint(90, 50)
        right = endpoint(93, 50)
        left.update(segment_center=[65, 20], segment_span=70)
        right.update(segment_center=[65, 75], segment_span=70)
        source = {"gray": image(19, 109), "features": [left]}
        target = {"gray": image(12, 115), "features": [right]}
        self.assertEqual(correspondence(source, target)[0], [])

    def test_unknown_box_geometry_is_diagnostic_only(self):
        source = {"gray": image(19, 109), "features": [boxcorner(85, 40, "tl", [90, 60])]}
        a = {"gray": image(12, 115), "features": [boxcorner(93, 40, "tl", [99, 60])]}
        b = {"gray": image(12, 115), "features": [boxcorner(93, 40, "tl", [99, 60])]}
        for item in (source, a, b):
            item["features"][0]["state"] = "unknown"
        result = compare_features(source, a, b)
        self.assertEqual(result["vote"], 0)
        self.assertEqual(len(result["box_geometry"]), 1)
        self.assertEqual(result["box_geometry"][0]["reason"], "shape_diagnostic_only")

    def test_q109_ko_round_cap_matches_same_template_stroke(self):
        pools = json.loads((ROOT/"data/category-expansion-v3/pools.json").read_text())["samples"]
        sample = next(s for s in pools if s["qid"] == 109)
        glyph = next(g for g in sample["glyphs"] if g["character"] == "こ")
        gray, pixels = gray_canvas(imread(ROOT/glyph["path"]))
        source = {"gray": gray, "features": extract(gray, pixels)}
        fonts = json.loads((ROOT/"data/category-expansion-v3/templates/catalog.json").read_text())["fonts"]
        references = []
        for face in ("HummingStd-M", "SkipStd-B"):
            font = next(f for f in fonts if f["postscript"] == face)
            gray, _ = gray_canvas(imread(ROOT/"data/category-expansion-v3/templates"/font["templates"]["こ"]), pixels)
            references.append({"gray": gray, "features": extract(gray, pixels)})
        comparison = compare_features(source, *references, pair_features(*references))
        self.assertEqual(comparison["vote"], 1)
        self.assertEqual(comparison["round_support_a"], 1)
        self.assertEqual(comparison["square_support_b"], 0)


if __name__ == "__main__":
    unittest.main()
