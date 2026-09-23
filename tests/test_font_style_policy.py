import unittest

from mikan_font_probe.font_style_policy import font_style_policy, replacement_candidates


class FontStylePolicyTests(unittest.TestCase):
    def test_folk_and_skip_share_recognition_family_and_keep_weight(self):
        folk = font_style_policy("FolkPro-Bold", metadata={"weight": 700})
        skip = font_style_policy("SkipStd-B", metadata={"weight": 700})
        skip_medium = font_style_policy("SkipStd-M", metadata={"weight": 500})
        self.assertEqual(folk["source_family"], skip["source_family"])
        self.assertEqual(folk["source_family"], skip_medium["source_family"])
        self.assertEqual(folk["source_family"]["key"], "folk_skip")
        self.assertEqual(folk["weight_class"], "bold")
        self.assertEqual(skip["weight_class"], "bold")
        self.assertEqual(skip_medium["weight_class"], "regular")
        self.assertIn("folk-skip-family", folk["replacement_groups"])
        self.assertIn("folk-skip-family", skip["replacement_groups"])
        self.assertEqual(folk["replacement_candidates"], skip["replacement_candidates"])
        self.assertEqual([row["name"] for row in folk["replacement_candidates"]], ["润圆/雅士黑"])

    def test_rodin_faces_share_family_but_keep_weight(self):
        rows = [font_style_policy("RodinHappyPro-" + weight) for weight in ("L", "B", "UB")]
        self.assertEqual({r["source_family"]["key"] for r in rows}, {"rodin_happy"})
        self.assertEqual([r["weight_class"] for r in rows], ["light", "bold", "ultrabold"])
        self.assertTrue(all(not r["replacement_candidates"] for r in rows))

    def test_humming_reports_partial_vendor_coverage(self):
        row = font_style_policy("HummingStd-M", metadata={"weight": 500})
        self.assertEqual(row["coverage"]["vendor"], "partial")
        self.assertIn("vendor_variant_uncovered", row["coverage_warnings"])

    def test_rodin_replacement_uses_observed_style_not_family_identity(self):
        policy = font_style_policy("RodinHappyPro-L")
        rows = replacement_candidates(policy, {"weight_class": "ultrabold"})
        self.assertEqual(rows[0], {"name": "海报体", "role": "preferred"})

    def test_new_font_assets_are_distinct_families(self):
        rows = [
            font_style_policy("SeuratProN-M", metadata={"weight": 500}),
            font_style_policy("YWJiaoMinTi-Regular", metadata={"weight": 400}),
            font_style_policy("YWJiangYuanTi-Bold", metadata={"weight": 700}),
        ]
        self.assertEqual(
            [row["source_family"]["key"] for row in rows],
            ["seurat", "yw_jiaoming", "yw_jiangyuan"],
        )
        self.assertEqual(
            [row["weight_class"] for row in rows],
            ["regular", "regular", "bold"],
        )
        self.assertEqual(rows[0]["replacement_candidates"],
                         [{"name": "圆体", "role": "preferred"}])

    def test_heiti_family_name_does_not_assign_script_styles(self):
        # Both the explicit Heiti family and the user's "black" category label
        # are family metadata, not proof that kana are Gothic-shaped.
        rows = [font_style_policy("YWHeiTi-Light", metadata={"weight": 300}),
                font_style_policy("PAntiqueANpProN-Medium")]
        self.assertTrue(all(row["script_style"] == {"kana": [], "kanji": []}
                            for row in rows))
        self.assertEqual([row["source_family"]["key"] for row in rows],
                         ["ywheiti", "antique"])

    def test_explicit_face_name_wins_over_nonstandard_os2_weight(self):
        row = font_style_policy(
            "PRyuminPr6N-Bold",
            metadata={"style": "B-KL", "weight": 550},
        )
        self.assertEqual(row["weight_value"], 550)
        self.assertEqual(row["weight_class"], "bold")
        self.assertEqual(row["weight_source"], "sfnt_style_name")


if __name__ == "__main__":
    unittest.main()
