"""Active-query tests with controlled tables, not real-font accuracy claims."""

import unittest
import numpy as np

from mikan_font_probe.informative_selection import choose_next, explain_confidence, explain_pair, run_selection


def glyph(char,quality=1,resolution=32):
    return {"character":char,"quality":{"score":quality,"usable":True},"resolution":resolution,"script":"kana"}


def fonts():
    return [{"id":f,"postscript":f,"family_group":{"key":f,"name":f,"answer":f}} for f in ("A","B","C")]


RANKING = [{"key":"A","distance":.1},{"key":"B","distance":.11},{"key":"C","distance":.12}]


class InformativeSelectionTests(unittest.TestCase):
    def test_discriminating_kana_beats_clear_shared_shape(self):
        items = [glyph("一"),glyph("日",1),glyph("あ",.7)]
        table = lambda a,b,c,r: .01 if c in "一日" else .3
        index,_ = choose_next(items,[0],RANKING,table)
        self.assertEqual(index,2)

    def test_low_resolution_can_remove_discrimination(self):
        items = [glyph("一"),glyph("あ",1,16),glyph("め",.8,32)]
        index,_ = choose_next(items,[0],RANKING,lambda a,b,c,r:.01 if r==16 else .2)
        self.assertEqual(index,2)

    def test_same_character_is_not_repeated_independent_evidence(self):
        items = [glyph("あ"),glyph("あ"),glyph("め",.8)]
        index,_ = choose_next(items,[0],RANKING,lambda *args:.2)
        self.assertEqual(index,2)
        scores = np.array([[.1,.1,.1],[.3,.3,.3],[.4,.4,.4]])
        evidence = explain_pair(fonts(),scores,items,[0,1],"A","B",lambda *args:.2)
        self.assertEqual(evidence["support"],1)

    def test_shared_shapes_do_not_create_high_confidence(self):
        items = [glyph(c) for c in "あいう"]
        scores = np.array([[.1]*3,[.2]*3,[.3]*3])
        explanation = explain_confidence(fonts(),scores,items,[0,1,2],RANKING,lambda *args:0.)
        self.assertEqual(explanation["top3"][0]["confidence"],0.)
        self.assertEqual(explanation["state"],"needs_more_discriminating_glyphs")

    def test_unqueried_match_values_do_not_choose_next_character(self):
        items = [glyph("あ",1),glyph("い",.8),glyph("う",.7)]
        scores = np.array([[.1,.2,.3],[.2,.3,.2],[.4,.2,.1]])
        changed = scores.copy()
        changed[:,1:] = 1-scores[:,1:]
        table = lambda a,b,c,r:.1 if c=="い" else .3
        first = run_selection(fonts(),scores,items,table,budget=2)
        second = run_selection(fonts(),changed,items,table,budget=2)
        self.assertEqual(first["selected"],second["selected"])

    def test_no_useful_glyphs_stops_without_confidence(self):
        items = [glyph(c) for c in "あいうえお"]
        scores = np.array([[.1]*5,[.2]*5,[.3]*5])
        result = run_selection(fonts(),scores,items,lambda *args:0.,adaptive_stop=True)
        self.assertEqual(result["stop_reason"],"no_known_discriminating_glyph_in_pool")
        self.assertEqual(result["explanation"]["top3"][0]["confidence"],0.)


if __name__ == "__main__":
    unittest.main()
