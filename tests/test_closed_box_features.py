"""Synthetic controls for conservative closed-box corner evidence."""

import json
import unittest

import cv2
import numpy as np

from mikan_font_probe.closed_box_features import extract_boxes


def _round_rect(image, lo, hi, radius, color):
    x0, y0 = lo
    x1, y1 = hi
    cv2.rectangle(image, (x0 + radius, y0), (x1 - radius, y1), color, -1)
    cv2.rectangle(image, (x0, y0 + radius), (x1, y1 - radius), color, -1)
    for x in (x0 + radius, x1 - radius):
        for y in (y0 + radius, y1 - radius):
            cv2.circle(image, (x, y), radius, color, -1)


def ring(shape="square", side=64, width=10, radius=11, angle=0):
    scale = 4
    image = np.full((512, 512), 255, np.uint8)
    outer = side // 2
    center = 64 * scale
    cv2.rectangle(image, (center - outer * scale, center - outer * scale),
                  (center + outer * scale, center + outer * scale), 0, -1)
    inner = outer - width
    low = (center - inner * scale, center - inner * scale)
    high = (center + inner * scale, center + inner * scale)
    if shape == "rounded":
        _round_rect(image, low, high, radius * scale, 255)
    elif shape == "circle":
        cv2.circle(image, (center, center), inner * scale, 255, -1)
    else:
        cv2.rectangle(image, low, high, 255, -1)
    if angle:
        matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
        image = cv2.warpAffine(image, matrix, (512, 512), borderValue=255)
    return cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)


class ClosedBoxTests(unittest.TestCase):
    def test_square_and_rounded_holes(self):
        for shape, state in (("square", "square"), ("rounded", "rounded")):
            with self.subTest(shape=shape):
                rows = extract_boxes(ring(shape), 104)
                self.assertEqual(len(rows), 4)
                self.assertEqual({row["role"] for row in rows}, {"tl", "tr", "br", "bl"})
                self.assertEqual({row["state"] for row in rows}, {state})
                self.assertTrue(all(row["width"] > 5 for row in rows))
                self.assertTrue(all(row["square_score"] > 0 for row in rows)
                                if state == "square" else
                                all(row["square_score"] == 0 for row in rows))

    def test_short_sides_and_rotation(self):
        for side in (36, 44, 64):
            for angle in (0, 17, 36):
                with self.subTest(side=side, angle=angle):
                    square = extract_boxes(ring(side=side, width=7, angle=angle), 104)
                    rounded = extract_boxes(ring("rounded", side=side, width=7,
                                                  radius=6, angle=angle), 104)
                    self.assertFalse(any(row["state"] == "rounded" for row in square))
                    self.assertFalse(any(row["state"] == "square" for row in rounded))
                    if angle == 0 and side >= 44:
                        self.assertEqual(len(square), 4)

    def test_circle_and_open_frame(self):
        circle = extract_boxes(ring("circle"), 104)
        self.assertFalse(any(row["state"] == "rounded" for row in circle))
        image = ring()
        cv2.rectangle(image, (55, 28), (73, 54), 255, -1)
        self.assertEqual(extract_boxes(image, 104), [])

    def test_long_narrow_hole_survives_insufficient_short_flanks(self):
        image = np.full((128, 128), 255, np.uint8)
        cv2.rectangle(image, (40, 12), (76, 116), 0, -1)
        cv2.rectangle(image, (51, 24), (64, 104), 255, -1)
        rows = extract_boxes(image, 104)
        self.assertEqual(len(rows), 4)
        self.assertGreater(rows[0]["box_bbox"][3] - rows[0]["box_bbox"][1], 70)
        self.assertLess(rows[0]["box_bbox"][2] - rows[0]["box_bbox"][0], 20)
        self.assertTrue(all(row["state"] == "unknown" for row in rows))

    def test_island_grandchild_is_not_a_white_hole(self):
        image = ring()
        cv2.circle(image, (64, 64), 8, 0, -1)
        rows = extract_boxes(image, 104)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["box_bbox"][2] - row["box_bbox"][0] > 35
                            for row in rows))

    def test_low_resolution_and_blur_abstain(self):
        for shape in ("square", "rounded"):
            source = ring(shape)
            for size in (16, 24, 32):
                low = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
                up = cv2.resize(low, (128, 128), interpolation=cv2.INTER_LINEAR)
                rows = extract_boxes(up, size)
                self.assertFalse(any(row["state"] != "unknown" for row in rows))
            blurred = cv2.GaussianBlur(source, (0, 0), 2.4)
            rows = extract_boxes(blurred, 24)
            self.assertFalse(any(row["state"] != "unknown" for row in rows))

    def test_schema_determinism_and_invalid_input(self):
        image = ring(angle=17)
        before = image.copy()
        rows = extract_boxes(image, 80)
        self.assertEqual(rows, json.loads(json.dumps(rows, allow_nan=False)))
        self.assertEqual(rows, extract_boxes(image, 80))
        np.testing.assert_array_equal(before, image)
        for row in rows:
            self.assertEqual(row["kind"], "box_corner")
            self.assertEqual(len(row["point"]), 2)
            self.assertEqual(len(row["box_center"]), 2)
            self.assertEqual(len(row["box_bbox"]), 4)
            self.assertAlmostEqual(row["native_width"], row["width"] * 80 / 104)
            self.assertEqual(len(row["metrics"]["threshold_states"]), 3)
        self.assertEqual(extract_boxes(np.full((128, 128), 255, np.uint8), 104), [])
        for gray, pixels in ((np.zeros((64, 64)), 104),
                             (np.zeros((128, 128)), 0),
                             (np.full((128, 128), np.nan), 104)):
            with self.assertRaises(ValueError):
                extract_boxes(gray, pixels)


if __name__ == "__main__":
    unittest.main()
