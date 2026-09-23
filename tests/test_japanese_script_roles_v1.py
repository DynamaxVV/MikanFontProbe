import unittest

from mikan_font_probe.japanese_script_roles_v1 import kana_subscript, script_role


class JapaneseScriptRolesV1Tests(unittest.TestCase):
    def test_kana_letters_iteration_marks_and_prolonged_mark_are_kana(self):
        for character in "あアゔヷーｰゝゞヽヾ":
            with self.subTest(character=character):
                self.assertEqual(script_role(character), "kana")

    def test_kana_related_symbols_are_not_counted_as_kana_letters(self):
        for character in "゛゜・･":
            with self.subTest(character=character):
                self.assertEqual(script_role(character), "punctuation_or_other")

    def test_kanji_and_unsegmented_sequences_are_separated(self):
        self.assertEqual(script_role("字"), "kanji")
        self.assertEqual(script_role("が"), "kana")
        self.assertEqual(script_role("か\u3099"), "punctuation_or_other")

    def test_kana_subscript_distinguishes_letters_from_marks(self):
        for character in ("あ", "っ"):
            with self.subTest(character=character):
                self.assertEqual(kana_subscript(character), "hiragana")
        for character in ("ア", "ッ", "ｶ"):
            with self.subTest(character=character):
                self.assertEqual(kana_subscript(character), "katakana")
        for character in ("ー", "ｰ", "ゝ", "ヽ"):
            with self.subTest(character=character):
                self.assertEqual(kana_subscript(character), "kana_mark")
        for character in ("゛", "゜", "字", "か\u3099"):
            with self.subTest(character=character):
                self.assertIsNone(kana_subscript(character))


if __name__ == "__main__":
    unittest.main()
