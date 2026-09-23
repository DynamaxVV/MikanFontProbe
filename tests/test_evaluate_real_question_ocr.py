import unittest

from mikan_font_probe.evaluate_real_question_ocr import TARGET_ANSWERS, alignment_text


class RealQuestionOcrTest(unittest.TestCase):
    def test_requested_proxy_categories(self):
        self.assertEqual(TARGET_ANSWERS["雅宋"], "宋体")
        self.assertEqual(TARGET_ANSWERS["粗黑"], "黑体")
        self.assertEqual(TARGET_ANSWERS["细圆"], "圆体")
        self.assertEqual(TARGET_ANSWERS["方圆"], "方圆")
        self.assertEqual(TARGET_ANSWERS["轻吟体"], "轻吟体")
        self.assertNotIn("降圆体", TARGET_ANSWERS)

    def test_punctuation_can_be_omitted_only_when_count_matches(self):
        self.assertEqual(alignment_text("消します！", 4), "消します")
        self.assertEqual(alignment_text("消します！", 5), "消します！")
        self.assertIsNone(alignment_text("消します！", 3))


if __name__ == "__main__":
    unittest.main()
