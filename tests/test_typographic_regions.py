"""Independent layout fixtures and real outlined-print anchor regressions."""

import unittest

import cv2
import numpy as np

from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.typographic_regions import column_bands, fit_grid, outlined_ink


class TypographyTests(unittest.TestCase):
    def test_grid_keeps_detached_parts_in_six_regular_cells(self):
        mask = np.zeros((180, 24), np.uint8)
        for top in range(8, 170, 28):
            mask[top:top+7, 2:22] = 255
            mask[top+12:top+22, 2:22] = 255
        cells, _ = fit_grid(mask, 24)
        self.assertEqual(len(cells), 6)
        for top in range(8, 170, 28):
            self.assertTrue(any(a <= top and b >= top+22 for a,b,_ in cells))
        self.assertFalse(any(crossing for _,_,crossing in cells))

    def test_columns_do_not_depend_on_number_of_llm_seeds(self):
        mask = np.zeros((150, 120), np.uint8)
        mask[10:130, 10:35] = 255
        mask[10:130, 55:80] = 255
        self.assertEqual(column_bands(mask, 50), [(10,35), (55,80)])

    def test_q27_keeps_last_glyphs_and_rejects_bottom_speedlines(self):
        image = imread(ROOT/"data/images/q27.jpg")
        before = image.copy()
        mask, _ = outlined_ink(image, 134)
        # Manually inspected terminal glyph regions, not an IoU ground truth.
        for x,y,w,h in [(490,525,130,120), (320,940,115,130), (150,525,105,120)]:
            self.assertGreater(cv2.countNonZero(mask[y:y+h,x:x+w]), 2500)
        self.assertLess(cv2.countNonZero(mask[1100:]), 100)
        np.testing.assert_array_equal(image, before)

    def test_edge_connected_background_is_not_detached_by_roi(self):
        image = np.full((120, 120, 3), 255, np.uint8)
        image[:, 50:70] = 0
        mask, _ = outlined_ink(image, 30)
        self.assertEqual(cv2.countNonZero(mask), 0)


if __name__ == "__main__":
    unittest.main()
