"""Structural-feature contract checks, not image classification acceptance."""

import unittest

import cv2
import numpy as np

from mikan_font_probe.structure_font_baseline import structure_features, structural_distance, score_pair


class StructureTests(unittest.TestCase):
    def test_self_distance_is_zero(self):
        image = np.zeros((64,64),np.uint8)
        cv2.rectangle(image,(12,12),(52,52),255,3)
        feature = structure_features(image)
        self.assertLess(structural_distance(feature,feature),1e-7)
        values,_ = score_pair(image,image)
        self.assertTrue(all(v < 1e-6 for v in values.values()))

    def test_local_orientation_distinguishes_horizontal_vertical(self):
        a = np.zeros((64,64),np.uint8)
        a[28:35,12:52] = 255
        b = a.T.copy()
        self.assertGreater(structural_distance(structure_features(a),structure_features(b)),.1)

    def test_feature_does_not_modify_input(self):
        image = np.arange(4096,dtype=np.uint8).reshape(64,64)
        before = image.copy()
        structure_features(image)
        np.testing.assert_array_equal(image,before)
        with self.assertRaises(ValueError):
            structure_features(image[:32])


if __name__ == "__main__":
    unittest.main()
