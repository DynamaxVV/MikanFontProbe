import unittest
from mikan_font_probe.resolve_font_ambiguity import resolve


def ranking(pairs):
    return [{"key": name, "name": name, "answer": name, "distance": distance}
            for name, distance in pairs]


class AmbiguityTests(unittest.TestCase):
    def test_reranked_candidate_displays_geometry_face_and_preserves_both_sources(self):
        appearance = ranking([("a", .1), ("b", .107)])
        geometry = ranking([("b", .05), ("a", .2)])
        appearance[0]["representative_face"] = "a-appearance"
        appearance[1]["representative_face"] = "b-appearance"
        geometry[0]["representative_face"] = "b-geometry"
        geometry[1]["representative_face"] = "a-geometry"

        result = resolve(appearance, geometry, 7)

        self.assertEqual(result["state"], "ambiguous_candidates_geometry_reranked")
        self.assertEqual(result["ranking"][0]["representative_face"], "b-geometry")
        self.assertEqual(result["ranking"][0]["appearance_representative_face"], "b-appearance")
        self.assertEqual(result["ranking"][0]["geometry_representative_face"], "b-geometry")
        self.assertEqual(result["ranking"][0]["representative_face_basis"], "geometry")

    def test_preserved_appearance_lead_keeps_appearance_face(self):
        appearance = ranking([("a", .1), ("b", .13)])
        geometry = ranking([("b", .05), ("a", .2)])
        appearance[0]["representative_face"] = "a-appearance"
        geometry[1]["representative_face"] = "a-geometry"

        result = resolve(appearance, geometry, 7)

        self.assertEqual(result["ranking"][0]["representative_face"], "a-appearance")
        self.assertEqual(result["ranking"][0]["geometry_representative_face"], "a-geometry")
        self.assertEqual(result["ranking"][0]["representative_face_basis"], "appearance")

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
