"""Regression contracts for the trainable local glyph encoder."""

import unittest

import numpy as np
import torch

from local_encoder_v11 import LocalShapeEncoder, parameter_count
from process_regions import ROOT
from train_local_encoder_v11 import augment, build_records, tight_gray


class LocalEncoderV11Tests(unittest.TestCase):
    def test_model_shape_resolution_and_parameter_budget(self):
        model = LocalShapeEncoder(embedding_dim=64, family_count=14,
                                  face_count=23, weight_count=7)
        self.assertGreaterEqual(parameter_count(model), 300_000)
        self.assertLessEqual(parameter_count(model), 1_000_000)
        output = model(torch.rand(3, 1, 64, 64), torch.tensor([24., 48., 80.]),
                       heads=True)
        self.assertEqual(tuple(output["embedding"].shape), (3, 64))
        self.assertEqual(tuple(output["family"].shape), (3, 14))
        self.assertEqual(tuple(output["face"].shape), (3, 23))
        self.assertEqual(tuple(output["weight"].shape), (3, 7))
        norms = torch.linalg.vector_norm(output["embedding"], dim=1)
        torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_font_records_are_character_split_and_provenanced(self):
        records, labels = build_records()
        self.assertEqual(len(records), 4715)
        self.assertEqual(labels["summary"]["split_counts"],
                         {"train": 3887, "val": 828})
        train_chars = {row["character"] for row in records if row["split"] == "train"}
        val_chars = {row["character"] for row in records if row["split"] == "val"}
        self.assertTrue(train_chars.isdisjoint(val_chars))
        self.assertTrue(all(row["source_font"].startswith("font_sample/")
                            for row in records))
        self.assertEqual(labels["summary"]["faces"],
                         sorted({row["face"] for row in records}))

    def test_same_seed_render_is_repeatable_and_keeps_native_resolution(self):
        records, _ = build_records()
        row = records[0]
        crop = tight_gray(ROOT / row["template"])
        first = augment(crop, np.random.default_rng(1234), native=48,
                        deterministic=True)
        second = augment(crop, np.random.default_rng(1234), native=48,
                         deterministic=True)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], 48)
        self.assertEqual(first[0].shape, (64, 64))
        self.assertGreaterEqual(float(first[0].min()), 0.)
        self.assertLessEqual(float(first[0].max()), 1.)


if __name__ == "__main__":
    unittest.main()
