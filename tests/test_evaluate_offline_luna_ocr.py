import json
import tempfile
import unittest
from pathlib import Path

from mikan_font_probe.evaluate_offline_luna_ocr import box_iou, load_lanes, matcher_layout


class LunaOcrLaneTest(unittest.TestCase):
    def test_region_comparison_requires_actual_overlap(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 0, 10, 10]), 0)
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)

    def test_vertical_reading_order_uses_vertical_glyph_segmenter(self):
        self.assertEqual(matcher_layout("vertical-rtl"), "vertical")
        self.assertEqual(matcher_layout("horizontal"), "horizontal")

    def test_each_scoped_question_appears_once_with_matching_image_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = {lane: {"qid": lane, "lane": lane, "coarse_label": "细宋",
                                  "image_sha256": str(lane)} for lane in range(6)}
            for lane in range(6):
                record = {"qid": lane, "image_sha256": str(lane), "status": "transcribed",
                          "region_bbox": [1, 2, 3, 4], "orientation": "vertical", "text": "日本語"}
                (root / f"lane-{lane}.json").write_text(json.dumps(
                    {"lane": lane, "language": "Japanese", "records": [record]}))
            self.assertEqual(len(load_lanes(root, assignments)), 6)
            path = root / "lane-2.json"
            payload = json.loads(path.read_text())
            payload["records"][0]["image_sha256"] = "changed"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Image SHA mismatch"):
                load_lanes(root, assignments)


if __name__ == "__main__":
    unittest.main()
