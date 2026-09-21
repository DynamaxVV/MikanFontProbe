"""Regression tests for the generated glyph discrimination index."""

import unittest

from mikan_font_probe.build_glyph_discrimination import DEFAULT_RESOLUTIONS, load_index


class GlyphDiscriminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = load_index()

    def test_same_font_is_zero(self):
        face = next(face for face in self.index.faces if "な" in face["characters"])
        for resolution in DEFAULT_RESOLUTIONS:
            result = self.index.face_distance(face["id"], face["id"], "な", resolution)
            self.assertTrue(result["covered"])
            self.assertEqual(result["distance"], 0.0)
            self.assertEqual(result["status"], "exact_bitmap")

    def test_weight_variants_merge_but_rodin_variants_do_not(self):
        shinmaru = self.index.families[self.index.family_index["shinmaru"]]
        self.assertEqual(len(shinmaru["members"]), 3)
        self.assertEqual(self.index.family_index["RodinHappyPro-L"], self.index.family_index["RodinHappyPro-L"])
        self.assertNotEqual(self.index.family_index["RodinHappyPro-L"], self.index.family_index["RodinHappyPro-B"])
        result = self.index.pair_distance("shinmaru", "shinmaru", "な", 32)
        self.assertEqual(result["status"], "same_family")
        self.assertTrue(result["shared"])

    def test_missing_glyph_is_not_tofu(self):
        yw_faces = self.index.families[self.index.family_index["ywheiti"]]["members"]
        for face in yw_faces:
            self.assertIsNone(self.index.image(face, "ゔ", 128))
        result = self.index.pair_distance("ywheiti", "antique", "ゔ", 32)
        self.assertFalse(result["covered"])
        self.assertIsNone(result["distance"])
        self.assertIsNone(result["shared"])
        self.assertEqual(result["status"], "missing")

    def test_all_measured_resolutions_and_source_template_are_indexed(self):
        self.assertEqual(self.index.resolutions, DEFAULT_RESOLUTIONS)
        face = next(face for face in self.index.faces if "か" in face["characters"])
        for resolution in (*DEFAULT_RESOLUTIONS, 128):
            image = self.index.image(face["id"], "か", resolution)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape, (resolution, resolution))

    def test_pair_distance_is_symmetric(self):
        for char in ("な", "あ", "日", "魔"):
            forward = self.index.pair_distance("shinmaru", "antique", char, 24)
            reverse = self.index.pair_distance("antique", "shinmaru", char, 24)
            self.assertEqual(forward["covered"], reverse["covered"])
            self.assertEqual(forward["status"], reverse["status"])
            self.assertEqual(forward["distance"], reverse["distance"])

    def test_non_catalog_resolution_uses_floor_band(self):
        result = self.index.pair_distance("shinmaru", "antique", "な", 20)
        self.assertEqual(result["requested_resolution"], 20)
        self.assertEqual(result["resolution"], 16)
        self.assertEqual(result["distance"], self.index.pair_distance("shinmaru", "antique", "な", 16)["distance"])

    def test_below_minimum_resolution_is_unknown(self):
        result = self.index.pair_distance("shinmaru", "antique", "な", 15)
        self.assertEqual(result["requested_resolution"], 15)
        self.assertIsNone(result["resolution"])
        self.assertIsNone(result["distance"])
        self.assertTrue(result["covered"])
        self.assertIsNone(result["shared"])
        self.assertEqual(result["status"], "unknown_resolution")

    def test_active_selection_characters_are_present(self):
        active = set((self.index.root.parent / "active-selection-v1/characters.txt").read_text(encoding="utf-8").strip())
        self.assertTrue(active.issubset(set(self.index.characters)))


if __name__ == "__main__":
    unittest.main()
