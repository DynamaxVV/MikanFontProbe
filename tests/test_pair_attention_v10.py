"""Regression for structural proposals and conservative abstention."""

import json
import unittest

import cv2
import numpy as np

from mikan_font_probe.evaluate_continuous_v8 import _source, _view
from mikan_font_probe.pair_attention_v10 import compare, decide, rectangular_holes
from mikan_font_probe.process_regions import ROOT
from mikan_font_probe.stroke_correspondence_v7 import FeatureReferences


class PairAttentionTests(unittest.TestCase):
    def test_rectangular_hole_not_circular_hole(self):
        box = np.full((128, 128), 255, np.uint8)
        ring = box.copy()
        cv2.rectangle(box, (35, 20), (80, 105), 0, 8)
        cv2.circle(ring, (64, 64), 31, 0, 8)
        self.assertEqual(len(rectangular_holes(box)), 1)
        self.assertEqual(rectangular_holes(ring), [])

    def test_real_q108_box_and_kana_loops(self):
        pools = json.loads((ROOT/"data/category-expansion-v3/pools.json").read_text())
        sample = next(row for row in pools["samples"] if row["qid"] == 108)
        for char, count in (("知", 1), ("る", 0), ("よ", 0), ("す", 0)):
            glyph = next(row for row in sample["glyphs"] if row["character"] == char)
            self.assertEqual(len(rectangular_holes(_source(glyph)["gray"])), count)

    def test_low_resolution_and_pair_abstention(self):
        blank = np.full((128, 128), 255, np.uint8)
        source = {"gray": blank, "pixels": 18, "features": []}
        self.assertEqual(compare(source, {"gray": blank}, {"gray": blank})["vote"], 0)
        self.assertEqual(decide([{"vote": 1}])["vote"], 0)
        self.assertEqual(decide([{"vote": 1}, {"vote": -1}])["vote"], 0)
        self.assertEqual(decide([{"vote": 1}, {"vote": 1}])["vote"], 1)

    def test_real_q109_ko_structure(self):
        pools = json.loads((ROOT/"data/category-expansion-v3/pools.json").read_text())
        sample = next(row for row in pools["samples"] if row["qid"] == 109)
        glyph = next(row for row in sample["glyphs"] if row["character"] == "こ")
        source = _source(glyph)
        refs = FeatureReferences(ROOT/"data/category-expansion-v3/templates")
        fonts = {face["postscript"]: face for face in refs.fonts}
        a = _view(refs, fonts["HummingStd-M"], "こ", source["pixels"])
        b = _view(refs, fonts["SkipStd-B"], "こ", source["pixels"])
        result = compare(source, a, b)
        self.assertTrue(any(row["kind"] == "free_end" for row in result["regions"]))


if __name__ == "__main__":
    unittest.main()
