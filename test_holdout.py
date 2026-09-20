"""Holdout integrity checks, not assertions that predictions must be correct."""

import hashlib
import json
import unittest
from collections import Counter

from process_regions import ROOT


class HoldoutTests(unittest.TestCase):
    def test_balanced_unseen_question_ids(self):
        samples = json.loads((ROOT/"data/holdout_samples_v1.json").read_text())["samples"]
        previous = []
        for name in ("priority-a","priority-b"):
            previous += json.loads((ROOT/f"data/{name}.json").read_text())
        self.assertFalse({s["qid"] for s in samples} & {s["question_id"] for s in previous})
        self.assertEqual(Counter(s["group"] for s in samples),{"圆体":4,"宋体":4,"黑体":4})
        self.assertEqual(sum(len(s["glyphs"]) for s in samples),36)

    def test_freeze_matches_inputs_and_scoring_code(self):
        freeze = json.loads((ROOT/"data/holdout-v1/freeze.json").read_text())
        for path,digest in freeze["input_files"].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest)

    def test_all_fonts_cover_all_evaluation_characters(self):
        samples = json.loads((ROOT/"data/holdout_samples_v1.json").read_text())["samples"]
        required = {g[2] for s in samples for g in s["glyphs"]}
        fonts = json.loads((ROOT/"data/holdout-v1/templates/catalog.json").read_text())["fonts"]
        self.assertEqual(len(fonts),12)
        for font in fonts:
            self.assertTrue(required <= set(font["templates"]))


if __name__ == "__main__":
    unittest.main()
