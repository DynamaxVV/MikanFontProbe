import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from process_regions import ROOT
from rank_font_correspondence import augment
from test_stroke_terminal_features import shaft


class RankTests(unittest.TestCase):
    def setUp(self):
        sample = next(s for s in json.loads((ROOT/"data/category-expansion-v3/pools.json").read_text())
                      ["samples"] if s["qid"] == 30)
        self.sample = copy.deepcopy(sample)
        self.sample["glyphs"] = self.sample["glyphs"][:3]
        fonts = [{"postscript": face, "templates": {g["character"]: "unused" for g in self.sample["glyphs"]}}
                 for face in ("A", "B")]
        self.refs = SimpleNamespace(fonts=fonts, reference=lambda *args: {"gray": shaft(), "features": []})
        self.baseline = {"ranking": [{"key": "a", "representative_face": "A", "appearance_distance": .1},
                                     {"key": "b", "representative_face": "B", "appearance_distance": .11}]}

    def test_unused_light_glyph_does_not_erase_ranking(self):
        self.sample["glyphs"].append({"character": "無", "polarity": "light",
                                      "quality": {"usable": False, "score": 0}})
        with patch("rank_font_correspondence.extract", return_value=[]):
            result = augment(self.sample, "unused", self.baseline, references=self.refs)
        self.assertEqual(result["ranking"][0]["key"], "a")

    def test_guard_reverses_only_near_top_two(self):
        with patch("rank_font_correspondence.extract", return_value=[]), \
                patch("rank_font_correspondence.pair_features", return_value={"utility": 0}), \
                patch("rank_font_correspondence.compare_features", return_value={"vote": -1, "square_anchor": True}):
            result = augment(self.sample, "unused", self.baseline, references=self.refs)
            self.assertEqual(result["ranking"][0]["key"], "b")
            self.assertEqual(self.baseline["ranking"][0]["key"], "a")
            self.baseline["ranking"][1]["appearance_distance"] = .13
            guarded = augment(self.sample, "unused", self.baseline, references=self.refs)
            self.assertEqual(guarded["ranking"][0]["key"], "a")

    def test_selected_white_is_rejected(self):
        self.sample["glyphs"][0]["polarity"] = "light"
        result = augment(self.sample, "unused", self.baseline, references=self.refs)
        self.assertEqual(result["ranking"], [])
        self.assertFalse(result["auto_accept"])


if __name__ == "__main__":
    unittest.main()
