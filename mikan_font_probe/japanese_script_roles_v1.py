"""Unicode-aware Japanese script roles for glyph evidence aggregation."""

from __future__ import annotations

import unicodedata


KANA_ITERATION_MARKS = frozenset("ゝゞヽヾ")


def script_role(character: str) -> str:
    """Classify a single code point without mistaking kana-related marks for kana.

    Spacing voiced/semivoiced sound marks and middle dots are punctuation or
    symbols, not ordinary kana letters. The prolonged sound mark and the four
    hiragana/katakana iteration marks remain useful kana-script evidence.
    """
    if len(character) != 1:
        return "punctuation_or_other"
    if character in {"ー", "ｰ"} or character in KANA_ITERATION_MARKS:
        return "kana"
    name = unicodedata.name(character, "")
    if "HIRAGANA LETTER" in name or "KATAKANA LETTER" in name:
        return "kana"
    if "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name:
        return "kanji"
    return "punctuation_or_other"


def kana_subscript(character: str) -> str | None:
    """Distinguish kana evidence subtypes without changing the coarse role.

    Hiragana and katakana can have different family-discrimination behavior.
    Long-vowel and iteration marks remain kana evidence, but are kept separate
    from letters so they cannot silently stand in for either syllabary.
    """
    if script_role(character) != "kana":
        return None
    if character in {"ー", "ｰ", *KANA_ITERATION_MARKS}:
        return "kana_mark"
    name = unicodedata.name(character, "")
    if "HIRAGANA" in name:
        return "hiragana"
    if "KATAKANA" in name:
        return "katakana"
    return "kana_mark"
