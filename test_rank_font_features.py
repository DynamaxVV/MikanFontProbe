import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from process_regions import ROOT
from rank_font_features import augment
from test_stroke_terminal_features import shaft


class EntryTests(unittest.TestCase):
    def test_unselected_light_glyph_does_not_erase_valid_ranking(self):
        sample = next(s for s in json.loads((ROOT/"data/category-expansion-v3/pools.json").read_text())
                      ["samples"] if s["qid"] == 30)
        sample = copy.deepcopy(sample)
        sample["glyphs"] = sample["glyphs"][:3]
        unavailable = {"character": "無", "polarity": "light", "quality": {"usable": False, "score": 0}}
        sample["glyphs"].append(unavailable)
        fonts = [{"postscript": face, "templates": {g["character"]: "unused" for g in sample["glyphs"]}}
                 for face in ("A", "B")]
        refs = SimpleNamespace(fonts=fonts, reference=lambda *args: {"gray": shaft(), "features": []})
        baseline = {"ranking": [{"key": "a", "representative_face": "A", "appearance_distance": .1},
                                {"key": "b", "representative_face": "B", "appearance_distance": .11}]}
        with patch("rank_font_features.extract", return_value=[]):
            result = augment(sample, "unused", baseline, references=refs)
        self.assertEqual(result["ranking"][0]["key"], "a")
        self.assertFalse(result["feature_changed"])

    def test_selected_light_input_is_rejected(self):
        sample = {"glyphs": [{"character": "あ", "polarity": "light",
                              "quality": {"usable": True, "score": 1}}]}
        result = augment(sample, "unused", {"ranking": [{"key": "a"}]})
        self.assertEqual(result["ranking"], [])
