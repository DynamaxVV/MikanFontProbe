import unittest

import cv2
import numpy as np

from mikan_font_probe.stroke_terminal_features import extract_terminals
from mikan_font_probe.testing_shapes import shaft


class TerminalTests(unittest.TestCase):
    def test_square_caps(self):
        rows = extract_terminals(shaft(), 104)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["state"] == "square" for r in rows))

    def test_round_is_not_square(self):
        rows = extract_terminals(shaft(True), 104)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["state"] == "rounded" for r in rows))

    def test_low_native_width_abstains(self):
        rows = extract_terminals(shaft(), 24)
        self.assertTrue(all(r["state"] == "unknown" for r in rows))

    def test_width_reference(self):
        for width in (10, 20):
            rows = extract_terminals(shaft(width=width), 104)
            self.assertTrue(all(r["state"] == "square" for r in rows))

    def test_rotation(self):
        rows = extract_terminals(shaft(angle=25), 104)
        self.assertTrue(rows)
        self.assertFalse(any(r["state"] == "rounded" for r in rows))

    def test_loop_has_no_caps(self):
        gray = np.full((128, 128), 255, np.uint8)
        cv2.circle(gray, (64, 64), 30, 0, 10)
        self.assertEqual(extract_terminals(gray, 104), [])

    def test_blurred_caps_do_not_swap_classes(self):
        from mikan_font_probe.stroke_feature_matcher import gray_canvas
        for rounded in (False, True):
            for width in (10, 20):
                for pixels in (32, 64):
                    for sigma in (.5, 1., 1.5):
                        small = cv2.resize(shaft(rounded, width), (pixels, pixels), interpolation=cv2.INTER_AREA)
                        blurred = cv2.GaussianBlur(small, (0, 0), sigma)
                        gray, native = gray_canvas(blurred)
                        wrong = "square" if rounded else "rounded"
                        self.assertFalse(any(r["state"] == wrong for r in extract_terminals(gray, native)))


if __name__ == "__main__":
    unittest.main()
