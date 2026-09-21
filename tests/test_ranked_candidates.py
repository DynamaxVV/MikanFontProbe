"""Family-policy, input quality and uncalibrated-confidence contracts."""

import unittest
import numpy as np

from mikan_font_probe.font_family_policy import family_policy
from mikan_font_probe.rank_font_candidates import crop_quality, family_ranking, confidence_details


def face(name):
    return {"id":name,"postscript":name,"family_group":family_policy(name,"")}


class RankingTests(unittest.TestCase):
    def test_weight_variants_collapse_but_answer_aliases_do_not(self):
        fonts = [face(n) for n in ["PAntiqueANpProN-Light","PAntiqueANpProN-Bold","YWHeiTi-Bold"]]
        ranked = family_ranking(fonts,np.array([[.1,.2],[.2,.1],[.3,.3]]),[1,1])
        self.assertEqual(len(ranked),2)
        self.assertEqual({r["answer"] for r in ranked},{"黑体"})

    def test_rodin_three_answer_exception(self):
        rows = [family_policy("RodinHappyPro-"+w,"") for w in ("L","B","UB")]
        self.assertEqual(len({r["key"] for r in rows}),3)
        self.assertEqual([r["answer"] for r in rows],["少女","方圆","海报体"])

    def test_blank_rejected_and_edge_penalized(self):
        blank = np.full((48,48,3),255,np.uint8)
        self.assertFalse(crop_quality(blank)["usable"])
        clean = blank.copy()
        clean[10:38,10:38] = 0
        cut = blank.copy()
        cut[:28,10:38] = 0
        self.assertGreater(crop_quality(clean)["score"],crop_quality(cut)["score"])

    def test_no_per_character_best_weight_cherrypicking(self):
        fonts = [face("PShinMGoPr6N-Light"),face("PShinMGoPr6N-Bold")]
        result = family_ranking(fonts,np.array([[0,1],[1,0]]),[1,1])
        self.assertEqual(result[0]["distance"],.5)

    def test_ambiguity_is_not_high_confidence(self):
        fonts = [face("PShinMGoPr6N-Light"),face("YWHeiTi-Light")]
        d = np.array([[.2,.2,.2],[.201,.201,.201]])
        ranked = family_ranking(fonts,d,[1,1,1])
        top,warnings = confidence_details(ranked,fonts,d,[1,1,1],1)
        self.assertLess(top[0]["confidence"],40)
        self.assertIn("前两名过于接近",warnings)
        self.assertTrue(all(t["confidence_kind"]=="uncalibrated_evidence_index" for t in top))

    def test_missing_input_has_no_confident_result(self):
        self.assertEqual(family_ranking([],np.zeros((0,0)),[]),[])
        top,warnings = confidence_details([],[],np.zeros((0,0)),[],0)
        self.assertEqual(top,[])
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
