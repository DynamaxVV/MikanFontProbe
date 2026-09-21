"""Crop safety and grayscale-input contracts, independent of class rankings."""

import unittest

import numpy as np

from mikan_font_probe.refine_font_inputs import refine_box, soft_normalize, soft_match


class InputTests(unittest.TestCase):
    def test_internal_gap_cannot_shrink_old_cell(self):
        image = np.full((80,80,3),255,np.uint8)
        image[20:24,20:60] = 0
        image[32:55,25:55] = 0
        box = [18,18,44,42]
        x,y,w,h = refine_box(image,box)
        self.assertLessEqual(x,20)
        self.assertLessEqual(y,20)
        self.assertGreaterEqual(x+w,60)
        self.assertGreaterEqual(y+h,58)

    def test_crop_is_in_bounds(self):
        image = np.full((32,32,3),255,np.uint8)
        image[2:30,2:30] = 0
        x,y,w,h = refine_box(image,[0,0,32,32])
        self.assertGreaterEqual(min(x,y),0)
        self.assertLessEqual(x+w,32)
        self.assertLessEqual(y+h,32)

    def test_soft_input_preserves_gray_interior(self):
        image = np.full((40,40),255,np.uint8)
        image[5:35,5:35] = 80
        original = image.copy()
        result = soft_normalize(image)
        self.assertEqual(int(result[32,32]),175)
        np.testing.assert_array_equal(image,original)
        self.assertLess(soft_match(result,result)[0],1e-6)


if __name__ == "__main__":
    unittest.main()
