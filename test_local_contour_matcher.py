import unittest
import json
from unittest.mock import patch

import cv2
import numpy as np

from local_contour_matcher import compare, decide, local_loss, region_mask, template_evidence
from rank_font_contours import augment
from process_regions import ROOT


class LocalContourTests(unittest.TestCase):
    def setUp(self):
        self.square = np.zeros((64, 64), np.uint8)
        cv2.rectangle(self.square, (15, 27), (48, 36), 255, -1)
        self.round = np.zeros_like(self.square)
        cv2.line(self.round, (20, 32), (43, 32), 255, 9)

    def test_identity(self):
        self.assertEqual(local_loss(self.round, self.round, region_mask(self.round)), 0)

    def test_outline_difference(self):
        self.assertGreater(local_loss(self.square, self.round,
                                     region_mask(self.square, self.round)), .01)

    def test_low_resolution_abstains(self):
        result, _ = compare(self.round, self.round, self.square, 24)
        self.assertEqual(result["vote"], 0)
        self.assertEqual(result["reason"], "low_resolution")

    def test_shared_abstains(self):
        self.assertFalse(template_evidence(self.round, self.round, 64)["reliable"])

    def test_same_mask_and_correct_direction(self):
        result, (_, _, mask) = compare(self.round, self.round, self.square, 64)
        self.assertGreater(result["margin"], 0)
        self.assertTrue(mask.any())
        self.assertLess(mask.mean(), .6)

    def test_conservative_vote(self):
        self.assertEqual(decide([1, 1]), 0)
        self.assertEqual(decide([1, 1, 1, -1]), 0)
        self.assertEqual(decide([1, 1, 1, 0]), 1)
        self.assertEqual(decide([-1, -1, -1]), -1)

    def test_light_text_does_not_leak_cached_ranking(self):
        result = augment({"polarity": "light"}, "unused", {"ranking": [{"key": "wrong"}]})
        self.assertEqual(result["ranking"], [])
        self.assertFalse(result["auto_accept"])

    def test_empty_input(self):
        self.assertEqual(augment({}, "unused", {"ranking": []})["top3"], [])

    def test_live_rerank_branch_and_clear_lead_guard(self):
        # Force local votes, not the final decision, to exercise integration.
        sample = next(s for s in json.loads((ROOT/"data/category-expansion-v3/pools.json").read_text())
                      ["samples"] if s["qid"] == 30)
        rows = [{"key": "humming", "representative_face": "HummingStd-M", "appearance_distance": .1},
                {"key": "folk", "representative_face": "FolkPro-Bold", "appearance_distance": .11}]
        with patch("rank_font_contours.compare", return_value=({"vote": -1}, None)), \
                patch("rank_font_contours.template_evidence", return_value={
                    "separation": 0., "threshold": .04, "reliable": False}):
            result = augment(sample, ROOT/"data/category-expansion-v3/templates", {"ranking": rows}, budget=3)
            self.assertTrue(result["contour_changed"])
            self.assertEqual(result["ranking"][0]["key"], "folk")
            self.assertEqual(rows[0]["key"], "humming")  # No baseline mutation.
            rows[1]["appearance_distance"] = .2
            guarded = augment(sample, ROOT/"data/category-expansion-v3/templates", {"ranking": rows}, budget=3)
            self.assertFalse(guarded["contour_changed"])
            self.assertEqual(guarded["ranking"][0]["key"], "humming")


if __name__ == "__main__":
    unittest.main()
