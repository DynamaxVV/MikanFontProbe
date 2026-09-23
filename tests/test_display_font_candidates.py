"""User-facing answer deduplication without changing internal rankings."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mikan_font_probe.display_font_candidates import display_candidates
from mikan_font_probe.font_family_policy import family_policy
from mikan_font_probe.rank_font_guarded import rank_sample


def candidate(postscript, distance):
    return {
        **family_policy(postscript, ""),
        "distance": distance,
        "representative_face": postscript,
    }


class DisplayCandidateTests(unittest.TestCase):
    def test_seurat_and_shinmaru_share_chinese_proxy_answer(self):
        ranking = [
            {"answer": "Seurat", "representative_face": "SeuratProN-M", "distance": .10},
            candidate("PShinMGoPr6N-Medium", .11),
            candidate("HummingStd-Medium", .12),
        ]
        display = display_candidates(ranking)
        self.assertEqual([row["answer"] for row in display], ["圆体", "轻吟体"])
        self.assertEqual(display[0]["representative_face"], "SeuratProN-M")
        self.assertEqual(ranking[0]["answer"], "Seurat")

    def test_answer_aliases_keep_only_the_highest_ranked_candidate(self):
        ranking = [
            candidate("PAntiqueANpProN-Light", .10),
            candidate("YWHeiTi-Bold", .11),
            candidate("FolkPro-Bold", .12),
        ]

        display = display_candidates(ranking)

        self.assertEqual([row["answer"] for row in display], ["黑体", "方新书"])
        self.assertEqual(display[0]["representative_face"], "PAntiqueANpProN-Light")

    def test_scans_full_ranking_to_fill_three_distinct_answers(self):
        ranking = [
            candidate("PAntiqueANpProN-Light", .10),
            candidate("YWHeiTi-Bold", .11),
            candidate("FolkPro-Bold", .12),
            candidate("HummingStd-Medium", .13),
        ]

        display = display_candidates(ranking)

        self.assertEqual([row["answer"] for row in display], ["黑体", "方新书", "轻吟体"])

    def test_rodin_answers_remain_distinct(self):
        ranking = [candidate(f"RodinHappyPro-{weight}", distance)
                   for weight, distance in (("L", .10), ("B", .11), ("UB", .12))]

        display = display_candidates(ranking)

        self.assertEqual([row["answer"] for row in display], ["少女", "方圆", "海报体"])

    def test_helper_does_not_mutate_ranking_or_legacy_top3(self):
        ranking = [
            candidate("PAntiqueANpProN-Light", .10),
            candidate("YWHeiTi-Bold", .11),
            candidate("FolkPro-Bold", .12),
            candidate("HummingStd-Medium", .13),
        ]
        original = copy.deepcopy(ranking)
        legacy_top3 = ranking[:3]

        display = display_candidates(ranking)
        display[0]["answer"] = "changed only in display"

        self.assertEqual(ranking, original)
        self.assertEqual(legacy_top3, original[:3])

    def test_guarded_entrypoint_keeps_ranking_and_top3(self):
        resolved_ranking = [
            {"rank": 1, **candidate("PAntiqueANpProN-Light", .10)},
            {"rank": 2, **candidate("YWHeiTi-Bold", .11)},
            {"rank": 3, **candidate("FolkPro-Bold", .12)},
            {"rank": 4, **candidate("HummingStd-Medium", .13)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glyph.png"
            path.write_bytes(b"original crop")
            sample = {"glyphs": [{
                "character": "あ",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }]}
            references = SimpleNamespace(fonts=[{
                "templates": {"あ": "unused.png"},
            }])
            with patch("mikan_font_probe.rank_font_guarded.GeometryReferences", return_value=references), \
                    patch("mikan_font_probe.rank_font_guarded.quality_order", return_value=[0]), \
                    patch("mikan_font_probe.rank_font_guarded.imread", return_value=np.zeros((8, 8), np.uint8)), \
                    patch("mikan_font_probe.rank_font_guarded.normalize", return_value=np.zeros((8, 8), np.uint8)), \
                    patch("mikan_font_probe.rank_font_guarded.prepare", return_value={}), \
                    patch("mikan_font_probe.rank_font_guarded.effective_resolution", return_value={"effective": 8}), \
                    patch("mikan_font_probe.rank_font_guarded.hybrid_distance", return_value=.1), \
                    patch("mikan_font_probe.rank_font_guarded.ranking_for", return_value=[
                        {"key": "first", "distance": .1},
                        {"key": "second", "distance": .2},
                    ]), \
                    patch("mikan_font_probe.rank_font_guarded.resolve", return_value={
                        "ranking": resolved_ranking,
                        "state": "appearance_lead_preserved",
                        "auto_accept": False,
                    }):
                result = rank_sample(sample, directory)

        self.assertEqual(result["ranking"], resolved_ranking)
        self.assertEqual(result["top3"], resolved_ranking[:3])
        self.assertEqual([row["answer"] for row in result["display_top3"]],
                         ["黑体", "方新书", "轻吟体"])

    def test_empty_and_light_inputs_have_empty_display_candidates(self):
        references = SimpleNamespace(fonts=[])
        with patch("mikan_font_probe.rank_font_guarded.GeometryReferences", return_value=references), \
                patch("mikan_font_probe.rank_font_guarded.quality_order", return_value=[]):
            empty = rank_sample({"glyphs": []}, "unused")
        with patch("mikan_font_probe.rank_font_guarded.GeometryReferences", return_value=references), \
                patch("mikan_font_probe.rank_font_guarded.quality_order", return_value=[0]):
            light = rank_sample({"polarity": "light", "glyphs": [{}]}, "unused")

        self.assertEqual(empty["display_top3"], [])
        self.assertEqual(light["display_top3"], [])


if __name__ == "__main__":
    unittest.main()
