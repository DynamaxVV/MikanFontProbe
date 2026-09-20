"""Synthetic checks for diagnostic shape and width measurements."""

import unittest
import json

import cv2
import numpy as np

from continuous_evidence_v8 import _distance, _sample_patch, axis_width, local_evidence
from process_regions import ROOT
from stroke_correspondence_v7 import FeatureReferences
from evaluate_continuous_v8 import _source, _view


class ContinuousEvidenceTests(unittest.TestCase):
    def test_patch_prefers_matching_corner(self):
        square = np.full((128, 128), 255, np.uint8)
        rounded = square.copy()
        cv2.rectangle(square, (42, 42), (72, 72), 0, 6)
        cv2.rectangle(rounded, (42, 42), (72, 72), 0, 6)
        cv2.circle(rounded, (42, 42), 5, 255, -1)
        source = _sample_patch(square, (42, 42), 11)
        same = _sample_patch(square, (42, 42), 11)
        other = _sample_patch(rounded, (42, 42), 11)
        self.assertLess(_distance(source, same), _distance(source, other))

    def test_axis_width_ratio(self):
        gray = np.full((128, 128), 255, np.uint8)
        cv2.line(gray, (20, 30), (105, 30), 0, 3)
        cv2.line(gray, (60, 46), (60, 115), 0, 9)
        result = axis_width(gray, 80)
        self.assertEqual(result["reason"], "measured")
        self.assertGreater(result["ratio"], 1.7)

    def test_axis_width_abstains_on_empty(self):
        result = axis_width(np.full((128, 128), 255, np.uint8), 80)
        self.assertIsNone(result["ratio"])

    def test_real_q109_ko_local_evidence(self):
        pools = json.loads((ROOT/"data/category-expansion-v3/pools.json").read_text())
        sample = next(row for row in pools["samples"] if row["qid"] == 109)
        glyph = next(row for row in sample["glyphs"] if row["character"] == "こ")
        source = _source(glyph)
        refs = FeatureReferences(ROOT/"data/category-expansion-v3/templates")
        faces = {row["postscript"]: row for row in refs.fonts}
        a = _view(refs, faces["HummingStd-M"], "こ", source["pixels"])
        b = _view(refs, faces["SkipStd-B"], "こ", source["pixels"])
        rows = local_evidence(source, a, b)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["winner"], "a")


if __name__ == "__main__":
    unittest.main()
