import unittest
from resolve_font_ambiguity import resolve


def ranking(pairs):
    return [{"key": name, "name": name, "answer": name, "distance": distance}
            for name, distance in pairs]


class AmbiguityTests(unittest.TestCase):
    def test_clear_lead_not_overridden(self):
        result = resolve(ranking([("a",.1),("b",.13)]), ranking([("b",.05),("a",.2)]),7)
        self.assertEqual(result["ranking"][0]["key"], "a")
        self.assertFalse(result["changed"])

    def test_near_tie_reranked_without_family_bonus(self):
        result = resolve(ranking([("a",.1),("b",.107)]), ranking([("b",.05),("a",.2)]),7)
        self.assertEqual(result["ranking"][0]["key"], "b")
        self.assertFalse(result["auto_accept"])

    def test_outside_winner_creates_conflict_not_forced_promotion(self):
        result = resolve(ranking([("a",.1),("b",.107),("c",.2)]),
                         ranking([("c",.03),("b",.05),("a",.2)]),7)
        self.assertEqual(result["ranking"][0]["key"], "a")
        self.assertEqual(result["state"], "conflicting_geometry_review_required")

    def test_low_evidence_keeps_original_order(self):
        result = resolve(ranking([("a",.1),("b",.107)]), ranking([("b",.05),("a",.2)]),2)
        self.assertEqual(result["ranking"][0]["key"], "a")
