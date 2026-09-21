"""Regression contracts for the isolated v13 experiments."""

import unittest

import numpy as np
import torch

from mikan_font_probe.evaluate_local_encoder_v13 import aggregate
from mikan_font_probe.local_encoder_v13 import CategoryPrototypeBank
from mikan_font_probe.local_features_v13 import compare_local, effective_state, local_reference
from mikan_font_probe.process_regions import ROOT
from mikan_font_probe.train_local_encoder_v13 import augment_weak
from mikan_font_probe.train_local_encoder_v11 import build_records, tight_gray


class LocalEncoderV13Tests(unittest.TestCase):
    def test_top2_is_face_aggregation_not_question_max(self):
        self.assertAlmostEqual(aggregate([.9, .8, .2], "max"), .9)
        self.assertAlmostEqual(aggregate([.9, .8, .2], "top2"), .85)
        self.assertAlmostEqual(aggregate([.9, .8, .2], "top3"), .6333333333,
                               places=5)

    def test_category_prototype_bank_returns_logits(self):
        bank = CategoryPrototypeBank(13, 64)
        logits = bank.logits(torch.randn(4, 64))
        self.assertEqual(tuple(logits.shape), (4, 13))
        norms = torch.linalg.vector_norm(bank.prototypes.detach(), dim=1)
        self.assertTrue(torch.all(norms > 0))

    def test_weak_augmentation_is_deterministic_when_requested(self):
        records, _ = build_records()
        crop = tight_gray(ROOT / records[0]["template"])
        first = augment_weak(crop, np.random.default_rng(17), native=48,
                             deterministic=True)
        second = augment_weak(crop, np.random.default_rng(17), native=48,
                              deterministic=True)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], 48)
        self.assertEqual(first[0].shape, (64, 64))

    def test_local_feature_path_abstains_or_returns_bounded_score(self):
        image = np.full((64, 64), 255, np.uint8)
        image[18:46, 22:42] = 0
        reference = local_reference(image, 40)
        result = compare_local(reference, reference)
        self.assertGreaterEqual(result["score"], 0.)
        self.assertLessEqual(result["score"], 1.)
        self.assertIn(result["reason"], {
            "stable_local_correspondence", "insufficient_stable_local_features",
            "no_unique_local_correspondence",
        })

    def test_effective_state_requires_threshold_majority(self):
        self.assertEqual(effective_state({"state": "unknown", "metrics": {
            "threshold_states": ["rounded", "rounded", "unknown"]}}), "rounded")
        self.assertEqual(effective_state({"state": "unknown", "metrics": {
            "threshold_states": ["rounded", "square", "unknown"]}}), "unknown")


if __name__ == "__main__":
    unittest.main()
