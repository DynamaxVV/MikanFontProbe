"""Checks for template-difference saliency and missing-landmark bypass."""

import unittest

import cv2
import numpy as np

from mikan_font_probe.contrastive_patch_v9 import compare


class ContrastivePatchTests(unittest.TestCase):
    def test_distinct_shapes_prefer_matching_template(self):
        a = np.full((128, 128), 255, np.uint8)
        b = a.copy()
        cv2.line(a, (22, 35), (99, 35), 0, 7)
        cv2.line(b, (22, 35), (99, 35), 0, 7)
        cv2.circle(a, (99, 35), 5, 0, -1)
        cv2.rectangle(b, (95, 31), (104, 39), 0, -1)
        cv2.line(a, (22, 80), (98, 80), 0, 6)
        cv2.line(b, (22, 80), (98, 80), 0, 6)
        result = compare(a, a, b, 72, "ink")
        self.assertEqual(result["winner"], "a")
        self.assertTrue(result["patches"])

    def test_low_resolution_abstains(self):
        image = np.full((128, 128), 255, np.uint8)
        cv2.line(image, (20, 50), (105, 50), 0, 5)
        self.assertIsNone(compare(image, image, image, 18)["winner"])

    def test_same_templates_have_no_saliency(self):
        image = np.full((128, 128), 255, np.uint8)
        cv2.line(image, (20, 50), (105, 50), 0, 5)
        self.assertEqual(compare(image, image, image, 72)["reason"],
                         "templates_near_shared")

    def test_region_confines_patch_centers(self):
        a = np.full((128, 128), 255, np.uint8)
        b = a.copy()
        cv2.rectangle(a, (25, 30), (95, 90), 0, 5)
        cv2.rectangle(b, (25, 30), (95, 90), 0, 5)
        cv2.line(b, (25, 35), (25, 85), 0, 7)
        result = compare(a, a, b, 72, "ink", (20, 20, 44, 100))
        self.assertTrue(result["patches"])
        self.assertTrue(all(20 <= row["center"][0] <= 44 for row in result["patches"]))


if __name__ == "__main__":
    unittest.main()
