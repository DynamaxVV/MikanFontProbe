"""Selection scope and preservation of the previously frozen matcher."""
import unittest

from mikan_font_probe.expand_font_categories import OUT, read
from mikan_font_probe.validate_aligned_holdout import hashes


class CategoryExpansionTests(unittest.TestCase):
    def test_eight_eligible_groups_six_unique_questions_each(self):
        protocol = read(OUT/"protocol.json")
        groups = protocol["groups"]
        self.assertEqual(len(groups), 8)
        ids = []
        for group in groups:
            self.assertGreater(group["available"], 5)
            self.assertEqual(len(group["selected"]), 6)
            self.assertTrue(set(group["selected"]) <= set(group["all_qids"]))
            ids.extend(group["selected"])
        self.assertEqual(len(set(ids)), 48)

    def test_frozen_matching_implementation_unchanged(self):
        self.assertEqual(hashes(), read(OUT/"protocol.json")["matcher_hashes"])

    def test_new_families_use_first_six_without_score_filter(self):
        for group in read(OUT/"protocol.json")["groups"][3:]:
            self.assertEqual(group["selected"], sorted(group["all_qids"])[:6])
