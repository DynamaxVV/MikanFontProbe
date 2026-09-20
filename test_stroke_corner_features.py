"""Standalone synthetic geometric controls: python test_stroke_corner_features.py."""

import json
import unittest

import cv2
import numpy as np

from stroke_corner_features import CORNER_POLICY, extract_corners


def bend(width=12, radius=0, angle=0):
    """Supersampled L with flat caps and an optional circular connection."""
    scale = 4
    image = np.full((128 * scale, 128 * scale), 255, np.uint8)
    half = width / 2
    center = np.array([90.0 - radius, 90.0 - radius])
    if radius:
        theta = np.linspace(0, np.pi / 2, 101)
        outer = center + (radius + half) * np.column_stack([np.cos(theta), np.sin(theta)])
        inner = center + (radius - half) * np.column_stack([np.cos(theta[::-1]), np.sin(theta[::-1])])
        polygon = np.vstack([[90 + half, 20], outer, [20, 90 + half],
                             [20, 90 - half], inner, [90 - half, 20]])
    else:
        polygon = np.array([[90 + half, 20], [90 + half, 90 + half],
                            [20, 90 + half], [20, 90 - half],
                            [90 - half, 90 - half], [90 - half, 20]])
    cv2.fillPoly(image, [np.rint(polygon * scale).astype(np.int32)], 0)
    if angle:
        transform = cv2.getRotationMatrix2D((64 * scale, 64 * scale), angle, 1)
        image = cv2.warpAffine(image, transform, image.shape[::-1], borderValue=255)
    return cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)


class CornerTests(unittest.TestCase):
    def test_square_l(self):
        rows = extract_corners(bend(), 104)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "square")
        self.assertLess(np.linalg.norm(np.array(rows[0]["point"]) - [96, 96]), 3)

    def test_smooth_bend_has_positive_round_evidence(self):
        rows = extract_corners(bend(radius=12), 104)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "rounded")
        self.assertEqual(rows[0]["square_score"], 0)

    def test_width_and_rotation(self):
        positives = 0
        for width in (8, 12, 18):
            for angle in (0, 17, 45, 83, 135):
                with self.subTest(width=width, angle=angle):
                    rows = extract_corners(bend(width=width, angle=angle), 104)
                    self.assertLessEqual(len(rows), 1)
                    self.assertFalse(any(row["state"] == "rounded" for row in rows))
                    for row in rows:
                        self.assertLess(abs(row["width"] - width), 3)
                        positives += row["state"] == "square"
                    if angle == 0:
                        self.assertEqual([row["state"] for row in rows], ["square"])
        # Raster skeletons can lose a connection after rotation. Abstention is
        # valid, but the matrix must also exercise actual rotated detections.
        self.assertGreaterEqual(positives, 5)

    def test_rotated_smooth_bends_not_square(self):
        for width in (8, 12, 18):
            for angle in (0, 17, 45, 83):
                with self.subTest(width=width, angle=angle):
                    rows = extract_corners(bend(width, radius=width, angle=angle), 104)
                    self.assertFalse(any(row["state"] == "square" for row in rows))

    def test_circles_are_not_connection_corners(self):
        for width in (5, 10, 18):
            gray = np.full((128, 128), 255, np.uint8)
            cv2.circle(gray, (64, 64), 34, 0, width, cv2.LINE_AA)
            with self.subTest(width=width):
                self.assertEqual(extract_corners(gray, 104), [])

    def test_caps_and_crossings_are_not_connections(self):
        for shape in ("shaft", "cross", "tee", "disk", "box"):
            gray = np.full((128, 128), 255, np.uint8)
            if shape == "disk":
                cv2.circle(gray, (64, 64), 30, 0, -1)
            elif shape == "box":
                cv2.rectangle(gray, (30, 30), (98, 98), 0, -1)
            else:
                cv2.rectangle(gray, (20, 58), (108, 70), 0, -1)
                if shape != "shaft":
                    cv2.rectangle(gray, (58, 20), (70, 108 if shape == "cross" else 64), 0, -1)
            for angle in (0, 29):
                transformed = cv2.warpAffine(gray, cv2.getRotationMatrix2D((64, 64), angle, 1),
                                             (128, 128), borderValue=255)
                with self.subTest(shape=shape, angle=angle):
                    self.assertEqual(extract_corners(transformed, 104), [])

    def test_native_resolution_gate(self):
        rows = extract_corners(bend(), 24)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "unknown")
        self.assertEqual(rows[0]["reason"], "native_stroke_too_thin")
        self.assertEqual(rows[0]["square_score"], 0)

    def test_low_resolution_curves_never_become_square(self):
        for size in (16, 24, 32):
            for interpolation in (cv2.INTER_NEAREST, cv2.INTER_LINEAR):
                gray = cv2.resize(bend(radius=12), (size, size), interpolation=cv2.INTER_AREA)
                gray = cv2.resize(gray, (128, 128), interpolation=interpolation)
                rows = extract_corners(gray, size * 80 / 128)
                self.assertFalse(any(row["state"] == "square" for row in rows))

    def test_schema_no_mutation_and_determinism(self):
        gray = bend(angle=17)
        before = gray.copy()
        rows = extract_corners(gray, 80)
        self.assertEqual(rows, json.loads(json.dumps(rows, allow_nan=False)))
        self.assertEqual(rows, extract_corners(gray, 80))
        np.testing.assert_array_equal(gray, before)
        for row in rows:
            self.assertEqual(row["kind"], "corner")
            self.assertAlmostEqual(row["native_width"], row["width"] * 80 / 104)
            self.assertTrue(0 <= row["square_score"] <= 1)
            self.assertEqual(len(row["metrics"]["threshold_states"]), 3)
        self.assertLessEqual(len(rows), CORNER_POLICY["max_features"])

    def test_blank_and_invalid_input(self):
        self.assertEqual(extract_corners(np.full((128, 128), 255, np.uint8), 104), [])
        for image, pixels in ((np.zeros((64, 64)), 104), (np.zeros((128, 128)), 0),
                              (np.full((128, 128), np.nan), 104)):
            with self.assertRaises(ValueError):
                extract_corners(image, pixels)


if __name__ == "__main__":
    unittest.main()
