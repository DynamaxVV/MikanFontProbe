"""Pipeline equivalence and no invented low-resolution evidence."""
import unittest
from pathlib import Path
import numpy as np
from mikan_font_probe.aligned_discrimination import synthetic_query, AlignedSeparation
from mikan_font_probe.match_font_baseline import normalize


class AlignedTests(unittest.TestCase):
    def test_native_size_uses_exact_normalizer(self):
        image = np.full((40, 40), 255, np.uint8)
        image[5:35, 12:20] = 0
        np.testing.assert_array_equal(synthetic_query(image, 30), normalize(image))

    def test_blank_rejected(self):
        with self.assertRaises(ValueError):
            synthetic_query(np.full((20, 20), 255, np.uint8), 16)

    def test_small_resolution_unknown(self):
        lookup = object.__new__(AlignedSeparation)
        self.assertIsNone(lookup("a", "b", "あ", 15))

    def test_same_font_has_zero_excess_loss(self):
        root = Path(__file__).resolve().parents[1]
        lookup = AlignedSeparation(root/"data/active-selection-v1/templates",
                                   root/"data/aligned-holdout-v2/test-unused.json")
        self.assertEqual(lookup("shinmaru", "shinmaru", "な", 24), 0.)

    def test_direction_symmetry_and_floor(self):
        root = Path(__file__).resolve().parents[1]
        lookup = AlignedSeparation(root/"data/active-selection-v1/templates",
                                   root/"data/aligned-holdout-v2/test-unused.json")
        first = lookup("shinmaru", "antique", "な", 20)
        self.assertEqual(first, lookup("antique", "shinmaru", "な", 16))
        self.assertEqual(len(lookup.cache), 1)

    def test_missing_character_has_no_evidence(self):
        root = Path(__file__).resolve().parents[1]
        lookup = AlignedSeparation(root/"data/active-selection-v1/templates",
                                   root/"data/aligned-holdout-v2/test-unused.json")
        self.assertIsNone(lookup("shinmaru", "antique", "𠮷", 24))
