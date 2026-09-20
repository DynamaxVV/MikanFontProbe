"""Small real-image regressions for column growth and artwork rejection."""

import json
import unittest

from adaptive_regions import grow_line
from process_regions import ROOT, imread


class AdaptiveRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        entries = json.loads((ROOT / "data/manifest.json").read_text())["entries"]
        cls.images = {entry["question_id"]: ROOT / "data" / entry["image_file"] for entry in entries}
        annotations = json.loads((ROOT / "data/priority-a.json").read_text())
        annotations += json.loads((ROOT / "data/priority-b.json").read_text())
        cls.annotations = {entry["question_id"]: entry for entry in annotations}

    def extent(self, qid: int, line_index: int) -> tuple[int, int]:
        image = imread(self.images[qid])
        bbox = self.annotations[qid]["lines"][line_index]["bbox"]
        _, meta = grow_line(image, tuple(bbox), horizontal=False)
        return tuple(meta["grown_extent"])

    def test_q70_recovers_first_and_last_characters(self) -> None:
        self.assertLess(self.extent(70, 0)[0], 60)
        self.assertGreater(self.extent(70, 1)[1], 355)
        self.assertGreater(self.extent(70, 5)[1], 550)

    def test_q92_recovers_middle_column_last_kana(self) -> None:
        self.assertGreater(self.extent(92, 1)[1], 240)

    def test_q72_does_not_extend_into_phone_artwork(self) -> None:
        self.assertLess(self.extent(72, 0)[1], 365)

    def test_q140_keeps_separated_final_character(self) -> None:
        self.assertGreater(self.extent(140, 0)[1], 250)

    def test_q100_stops_before_lower_right_illustration(self) -> None:
        self.assertLess(self.extent(100, 0)[1], 550)


if __name__ == "__main__":
    unittest.main()
