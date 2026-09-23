"""Thin arbitrary-region entrypoint contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from mikan_font_probe.analyze_region import (analyze, candidate_boxes_from_mask,
                                             clean_text, parse_exclude)
from mikan_font_probe.process_regions import imwrite


class AnalyzeRegionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "region.png"
        imwrite(self.image_path, np.full((90, 180, 3), 255, np.uint8))
        self.mask = np.zeros((90, 180), np.uint8)
        self.boxes = [[10, 20, 30, 40], [70, 20, 30, 40], [130, 20, 30, 40]]

    def tearDown(self):
        self.temporary.cleanup()

    def test_text_and_exclude_parsing(self):
        self.assertEqual(clean_text("あ い\nう"), "あいう")
        self.assertEqual(parse_exclude("3,1,3"), [1, 3])
        with self.assertRaises(ValueError):
            parse_exclude("-1")

    def test_without_text_writes_numbered_segmentation_artifacts(self):
        output = self.root / "no-text"
        with patch("mikan_font_probe.analyze_region.extract", return_value=(self.mask, {"text_box": None})), \
                patch("mikan_font_probe.analyze_region.candidate_boxes_from_mask", return_value=self.boxes):
            result = analyze(self.image_path, output_directory=output)

        self.assertEqual(result["state"], "text_required_for_matching")
        self.assertEqual(result["candidate_count"], 3)
        self.assertTrue((output / "analysis.json").exists())
        self.assertTrue((output / "source.png").exists())
        self.assertTrue((output / "segments.png").exists())

    def test_count_mismatch_stops_before_matching(self):
        output = self.root / "mismatch"
        with patch("mikan_font_probe.analyze_region.extract", return_value=(self.mask, {})), \
                patch("mikan_font_probe.analyze_region.candidate_boxes_from_mask", return_value=self.boxes), \
                patch("mikan_font_probe.analyze_region.rank_sample") as rank:
            result = analyze(self.image_path, "あい", output_directory=output)

        self.assertEqual(result["state"], "character_count_mismatch")
        rank.assert_not_called()

    def test_exclusion_aligns_text_and_preserves_matcher_result(self):
        output = self.root / "matched"
        fake_result = {
            "ranking": [{"answer": "轻吟体"}],
            "display_top3": [{"answer": "轻吟体"}],
            "selected_indices": [0, 1],
            "state": "poor_library_fit_review_required",
        }
        with patch("mikan_font_probe.analyze_region.extract", return_value=(self.mask, {})), \
                patch("mikan_font_probe.analyze_region.candidate_boxes_from_mask", return_value=self.boxes), \
                patch("mikan_font_probe.analyze_region.rank_sample", return_value=fake_result), \
                patch("mikan_font_probe.analyze_region.save_match_preview"):
            result = analyze(self.image_path, "あい", exclude=[1], output_directory=output)

        self.assertEqual(result["state"], "matched")
        self.assertEqual(result["selected_candidate_indices"], [0, 2])
        self.assertEqual(result["match"], fake_result)
        saved = json.loads((output / "analysis.json").read_text())
        self.assertEqual(saved["excluded_candidates"], [1])
        self.assertEqual([glyph["character"] for glyph in saved["glyphs"]], ["あ", "い"])
        self.assertEqual([row.get("character") for row in saved["candidates"]], ["あ", None, "い"])

    def test_vertical_pitch_grid_splits_touching_character_groups(self):
        mask = np.zeros((180, 90), np.uint8)
        for x in (10, 55):
            for y in (10, 50, 90):
                cv2.circle(mask, (x + 12, y + 12), 9, 255, 3)

        boxes = candidate_boxes_from_mask(mask, "vertical")

        self.assertEqual(len(boxes), 6)
        self.assertGreater(boxes[0][0], boxes[3][0])

    def test_horizontal_pitch_grid_preserves_reading_order(self):
        mask = np.zeros((90, 180), np.uint8)
        for x in (10, 50, 90):
            cv2.circle(mask, (x + 12, 30), 9, 255, 3)

        boxes = candidate_boxes_from_mask(mask, "horizontal")

        self.assertEqual(len(boxes), 3)
        self.assertLess(boxes[0][0], boxes[1][0])

    def test_light_text_is_inverted_only_for_matcher_input(self):
        image = np.zeros((90, 180, 3), np.uint8)
        imwrite(self.image_path, image)
        output = self.root / "light"
        fake_result = {"ranking": [{"answer": "圆体"}], "display_top3": [],
                       "selected_indices": [0], "state": "review"}
        with patch("mikan_font_probe.analyze_region.extract", return_value=(self.mask, {})), \
                patch("mikan_font_probe.analyze_region.candidate_boxes_from_mask",
                      return_value=[self.boxes[0]]), \
                patch("mikan_font_probe.analyze_region.rank_sample", return_value=fake_result), \
                patch("mikan_font_probe.analyze_region.save_match_preview"):
            result = analyze(self.image_path, "あ", output_directory=output)

        self.assertEqual(result["original_polarity"], "light")
        self.assertEqual(result["glyphs"][0]["input_transform"], "inverted_from_light_text")
        source = cv2.imread(str(output / "crops/candidate-00-source.png"))
        matcher_input = cv2.imread(str(output / "crops/candidate-00-input.png"))
        self.assertEqual(int(source.mean()), 0)
        self.assertEqual(int(matcher_input.mean()), 255)

    def test_existing_output_directory_is_not_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            analyze(self.image_path, output_directory=output)


if __name__ == "__main__":
    unittest.main()
