"""Contract regressions; these do not establish real font accuracy."""

import json
import unittest

import cv2
import numpy as np

from mikan_font_probe.match_font_baseline import match, normalize
from mikan_font_probe.process_regions import ROOT


class FontBaselineTests(unittest.TestCase):
    def test_normalization_preserves_aspect_and_source(self):
        image = np.full((80,40),255,np.uint8)
        image[10:70,10:30] = 0
        original = image.copy()
        result = normalize(image)
        ys,xs = np.nonzero(result)
        self.assertAlmostEqual((xs.max()-xs.min()+1)/(ys.max()-ys.min()+1), 1/3, delta=.025)
        np.testing.assert_array_equal(image,original)

    def test_blank_is_not_a_font_candidate(self):
        with self.assertRaises(ValueError):
            normalize(np.full((32,32),255,np.uint8))

    def test_self_match_beats_different_geometry(self):
        a = np.zeros((64,64),np.uint8)
        cv2.circle(a,(32,32),20,255,4)
        b = np.zeros_like(a)
        cv2.rectangle(b,(12,12),(52,52),255,4)
        self.assertLess(match(a,a)[0],1e-6)
        self.assertGreater(match(a,b)[0],.05)

    def test_baseline_contains_three_groups_and_no_weight_labels(self):
        config = json.loads((ROOT/"data/baseline_samples.json").read_text())
        samples = config["samples"]
        self.assertEqual({s["group"] for s in samples},{"圆体","宋体","黑体"})
        self.assertEqual(len({s["qid"] for s in samples}),len(samples))
        self.assertTrue(all(len(char)==1 for s in samples for _,_,char in s["glyphs"]))


if __name__ == "__main__":
    unittest.main()
