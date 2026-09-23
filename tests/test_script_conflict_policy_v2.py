import unittest

from mikan_font_probe.script_conflict_policy_v2 import arbitrate_family_rankings
from mikan_font_probe.rank_font_guarded import _script_family_evidence, _select_script_balanced
from mikan_font_probe.rank_font_style_v2 import (
    _apply_script_family_evidence,
    _replacement_candidates_for_decision,
)


def ranking(*pairs):
    return [{"key": key, "name": key, "distance": distance}
            for key, distance in pairs]


class ScriptConflictPolicyV2Tests(unittest.TestCase):
    def test_confident_kana_anchors_source_family_over_kanji(self):
        result = arbitrate_family_rankings(
            ranking(("gothic", .10), ("mincho", .20)),
            ranking(("mincho", .10), ("gothic", .14)),
            ranking(("gothic", .01), ("mincho", .30)),
            kana_glyph_count=3,
        )

        self.assertEqual(result["state"], "kana_anchor_on_script_conflict")
        self.assertEqual(result["decision_family"], "mincho")
        self.assertEqual(result["ranking"][0]["key"], "mincho")
        self.assertFalse(result["auto_accept"])

    def test_weak_kana_conflict_never_displays_kanji_winner_as_resolved_top1(self):
        result = arbitrate_family_rankings(
            ranking(("gothic", .10), ("mincho", .20)),
            ranking(("mincho", .100), ("gothic", .110)),
            ranking(("gothic", .01), ("mincho", .30)),
            kana_glyph_count=3,
        )

        self.assertEqual(result["state"], "script_conflict_needs_review")
        self.assertIsNone(result["decision_family"])
        self.assertEqual(result["ranking"][0]["key"], "mincho")

    def test_one_kana_is_not_enough_to_resolve_a_script_conflict(self):
        result = arbitrate_family_rankings(
            ranking(("gothic", .10), ("mincho", .20)),
            ranking(("mincho", .10), ("gothic", .20)),
            ranking(("gothic", .01), ("mincho", .30)),
            kana_glyph_count=1,
        )

        self.assertEqual(result["state"], "script_conflict_needs_review")
        self.assertIsNone(result["decision_family"])
        self.assertEqual(result["ranking"][0]["key"], "mincho")

    def test_script_agreement_preserves_mixed_ranking(self):
        result = arbitrate_family_rankings(
            ranking(("a", .1), ("b", .2)),
            ranking(("a", .1), ("b", .3)),
            ranking(("a", .2), ("b", .3)),
            kana_glyph_count=2,
        )

        self.assertEqual(result["state"], "script_agreement")
        self.assertEqual(result["decision_family"], "a")
        self.assertEqual(result["ranking"][0]["key"], "a")

    def test_no_kana_uses_kanji_only_as_a_nonautomatic_fallback(self):
        result = arbitrate_family_rankings(
            ranking(("b", .1), ("a", .2)),
            [],
            ranking(("b", .1), ("a", .2)),
            kana_glyph_count=0,
        )

        self.assertEqual(result["state"], "kanji_only")
        self.assertEqual(result["decision_family"], "b")
        self.assertFalse(result["auto_accept"])

    def test_quality_selection_reserves_available_kana(self):
        glyphs = [
            {"character": "字", "quality": {"score": .99, "usable": True}},
            {"character": "あ", "quality": {"score": .60, "usable": True}},
            {"character": "本", "quality": {"score": .98, "usable": True}},
            {"character": "い", "quality": {"score": .59, "usable": True}},
            {"character": "学", "quality": {"score": .97, "usable": True}},
        ]

        selected = _select_script_balanced(glyphs, budget=3)

        self.assertEqual(selected, [1, 3, 0])

    def test_family_evidence_reports_hiragana_katakana_and_excludes_marks(self):
        import numpy as np
        from mikan_font_probe.font_style_policy import font_style_policy

        fonts = []
        for postscript in ("FolkPro-Bold", "SkipStd-B", "YWHeiTi-Light", "PAntiqueANpProN-Medium"):
            policy = font_style_policy(postscript, metadata={"weight": 700})
            fonts.append({
                "id": postscript,
                "postscript": postscript,
                "family_group": policy["source_family"],
            })
        glyphs = [{"character": character, "quality": {"score": 1.0}}
                  for character in ("し", "シ", "字", "゛")]
        scores = np.asarray([
            [.10, .35, .45, .10],  # FolkPro
            [.15, .40, .42, .12],  # Skip, merged with Folk
            [.30, .12, .11, .11],  # YWHeiTi
            [.40, .50, .20, .13],  # Antique AN+
        ], dtype=float)

        evidence = _script_family_evidence(fonts, scores, glyphs)

        self.assertEqual(evidence["kana_subscript_glyph_counts"], {
            "hiragana": 1, "katakana": 1, "kana_mark": 0,
        })
        self.assertEqual(evidence["kana_subscript_rankings"]["hiragana"][0]["key"],
                         "folk_skip")
        self.assertEqual(evidence["kana_subscript_rankings"]["katakana"][0]["key"],
                         "ywheiti")
        self.assertEqual(evidence["glyph_counts"]["kana"], 2)

    def test_style_v2_marks_weak_conflict_as_review_and_suppresses_replacement(self):
        hypotheses = [
            {"key": "ywheiti", "appearance_distance": .1},
            {"key": "ryumin", "appearance_distance": .2},
        ]
        evidence = {
            "state": "script_conflict_needs_review",
            "decision_family": None,
            "kana_winner": "ryumin",
            "kanji_winner": "ywheiti",
            "mixed_winner": "ywheiti",
            "kana_margin": .01,
            "kana_glyph_count": 2,
            "ranking": ranking(("ryumin", .1), ("ywheiti", .2)),
            "kana_ranking": ranking(("ryumin", .1), ("ywheiti", .2)),
        }

        rows, decision, needs_review, state = _apply_script_family_evidence(
            hypotheses, evidence, by_face={})

        self.assertEqual(state, "script_conflict_needs_review")
        self.assertTrue(needs_review)
        self.assertIsNone(decision["decision_family"])
        self.assertEqual(rows[0]["key"], "ryumin")
        self.assertEqual(_replacement_candidates_for_decision(hypotheses, decision), [])

    def test_style_v2_anchors_confident_kana_and_keeps_decision_nonautomatic(self):
        hypotheses = [
            {"key": "ywheiti", "appearance_distance": .1},
            {"key": "ryumin", "appearance_distance": .2},
        ]
        evidence = {
            "state": "kana_anchor_on_script_conflict",
            "decision_family": "ryumin",
            "kana_winner": "ryumin",
            "kanji_winner": "ywheiti",
            "mixed_winner": "ywheiti",
            "kana_margin": .04,
            "kana_glyph_count": 3,
            "ranking": ranking(("ryumin", .1), ("ywheiti", .2)),
            "kana_ranking": ranking(("ryumin", .1), ("ywheiti", .2)),
        }

        rows, decision, needs_review, state = _apply_script_family_evidence(
            hypotheses, evidence, by_face={})

        self.assertEqual(state, "kana_anchor_on_script_conflict")
        self.assertFalse(needs_review)
        self.assertEqual(decision["decision_family"], "ryumin")
        self.assertFalse(decision["auto_accept"])
        self.assertEqual(rows[0]["key"], "ryumin")


if __name__ == "__main__":
    unittest.main()
