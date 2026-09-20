import unittest

import numpy as np

from stroke_profile_matcher import hybrid_distance, stroke_template
from structure_font_baseline import score_pair


class StrokeProfileTests(unittest.TestCase):
    def test_fast_hybrid_is_exact_original_branch(self):
        a = np.zeros((64, 64), np.uint8)
        a[10:50, 25:29] = 255
        b = np.roll(a, 2, axis=0)
        self.assertAlmostEqual(hybrid_distance(a, b), score_pair(a, b)[0]["hybrid"], places=12)

    def test_x_growth_widens_vertical_stroke_without_extending_height(self):
        source = np.zeros((64, 64), np.uint8)
        source[10:50, 25:29] = 255
        grown = stroke_template(source, 1.5, 0)
        ys, xs = np.nonzero(grown)
        self.assertEqual((ys.min(), ys.max()), (10, 49))
        self.assertGreater(xs.max()-xs.min()+1, 4)
        self.assertEqual(int(source.sum()), 40*4*255)

    def test_identity_preserves_input(self):
        source = np.arange(4096, dtype=np.uint8).reshape(64, 64)
        np.testing.assert_array_equal(source, stroke_template(source, 0, 0))
