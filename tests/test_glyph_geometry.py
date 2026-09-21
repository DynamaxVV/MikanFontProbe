import unittest
import cv2
import numpy as np
from mikan_font_probe.glyph_geometry import thin, shape_loss, repeated_scale, effective_resolution
from mikan_font_probe.geometry_font_matcher import prepare, score


class GeometryTests(unittest.TestCase):
    def test_exact_upscale_detected(self):
        rng = np.random.default_rng(2026)
        native = rng.integers(0, 256, (24,24), dtype=np.uint8)
        large = np.repeat(np.repeat(native, 2, axis=0), 2, axis=1)
        self.assertEqual(repeated_scale(large), 2)
        self.assertEqual(repeated_scale(native), 1)
        self.assertEqual(repeated_scale(np.full((48,48),255,np.uint8)), 1)

    def test_one_axis_repeat_does_not_trigger(self):
        native = np.random.default_rng(9).integers(0,256,(24,24),dtype=np.uint8)
        self.assertEqual(repeated_scale(np.repeat(native,2,axis=0)), 1)

    def test_thinning_preserves_ring_hole_and_input(self):
        ink = np.zeros((64,64), np.uint8)
        cv2.circle(ink, (32,32), 15, 255, 5)
        before = ink.copy()
        skeleton = thin(ink)
        self.assertEqual(cv2.connectedComponents(skeleton)[0], 2)
        self.assertEqual(skeleton[32,32], 0)
        self.assertLess(skeleton.sum(), (ink>0).sum())
        np.testing.assert_array_equal(ink, before)

    def test_identical_reference_zero(self):
        ink = np.zeros((64,64), np.uint8)
        ink[10:50,25:29] = 255
        ref = prepare(ink)
        self.assertAlmostEqual(score(ref,ref), 0.)
        self.assertEqual(shape_loss(ref['skeleton'],ref['skeleton']),0.)
