"""Conservative arbitration over source-family distance rankings by script."""

from __future__ import annotations


DEFAULT_MIN_KANA_DISTANCE_MARGIN = 0.015
MIN_KANA_GLYPHS_FOR_ANCHOR = 2


def arbitrate_family_rankings(mixed: list[dict], kana: list[dict],
                              kanji: list[dict], kana_glyph_count: int,
                              min_kana_margin: float = DEFAULT_MIN_KANA_DISTANCE_MARGIN) -> dict:
    """Expose one source-family decision without letting kanji reverse kana.

    Rankings use distances (lower is better). A weak cross-script conflict has
    no decision family; its display order is kana-first only to avoid presenting
    the kanji-supported family as a resolved answer. Callers must honor the
    review state and must not emit automatic replacement advice for it.
    """
    if min_kana_margin < 0:
        raise ValueError("min_kana_margin must be nonnegative")
    if kana_glyph_count < 0:
        raise ValueError("kana_glyph_count must be nonnegative")

    mixed = _with_ranks(mixed)
    kana = _with_ranks(kana)
    kanji = _with_ranks(kanji)
    mixed_winner = _winner(mixed)
    kana_winner = _winner(kana)
    kanji_winner = _winner(kanji)

    if kana_winner is None and kanji_winner is None:
        return _result("no_script_evidence", None, mixed, mixed, kana, kanji,
                       mixed_winner, kana_winner, kanji_winner, None,
                       kana_glyph_count, min_kana_margin)
    if kana_winner is None:
        return _result("kanji_only", kanji_winner, kanji, mixed, kana, kanji,
                       mixed_winner, kana_winner, kanji_winner, None,
                       kana_glyph_count, min_kana_margin)
    if kanji_winner is None:
        return _result("kana_only", kana_winner, kana, mixed, kana, kanji,
                       mixed_winner, kana_winner, kanji_winner, _margin(kana),
                       kana_glyph_count, min_kana_margin)

    if kana_winner == kanji_winner:
        if mixed_winner != kana_winner:
            return _result("mixed_aggregation_conflict_needs_review", None, kana,
                           mixed, kana, kanji, mixed_winner, kana_winner,
                           kanji_winner, _margin(kana), kana_glyph_count,
                           min_kana_margin)
        return _result("script_agreement", kana_winner, mixed, mixed, kana,
                       kanji, mixed_winner, kana_winner, kanji_winner,
                       _margin(kana), kana_glyph_count, min_kana_margin)

    margin = _margin(kana)
    if kana_glyph_count >= MIN_KANA_GLYPHS_FOR_ANCHOR and margin is not None \
            and margin >= min_kana_margin:
        return _result("kana_anchor_on_script_conflict", kana_winner, kana,
                       mixed, kana, kanji, mixed_winner, kana_winner,
                       kanji_winner, margin, kana_glyph_count, min_kana_margin)
    return _result("script_conflict_needs_review", None, kana, mixed, kana,
                   kanji, mixed_winner, kana_winner, kanji_winner, margin,
                   kana_glyph_count, min_kana_margin)


def _winner(rows: list[dict]) -> str | None:
    return rows[0]["key"] if rows else None


def _margin(rows: list[dict]) -> float | None:
    if len(rows) < 2:
        return None
    return float(rows[1]["distance"]) - float(rows[0]["distance"])


def _with_ranks(rows: list[dict]) -> list[dict]:
    return [{**row, "rank": rank} for rank, row in enumerate(rows, 1)]


def _result(state: str, decision_family: str | None, ranking: list[dict],
            mixed: list[dict], kana: list[dict], kanji: list[dict],
            mixed_winner: str | None, kana_winner: str | None,
            kanji_winner: str | None, kana_margin: float | None,
            kana_glyph_count: int, min_kana_margin: float) -> dict:
    return {
        "state": state,
        "decision_family": decision_family,
        "ranking": _with_ranks(ranking),
        "mixed_ranking": mixed,
        "kana_ranking": kana,
        "kanji_ranking": kanji,
        "mixed_winner": mixed_winner,
        "kana_winner": kana_winner,
        "kanji_winner": kanji_winner,
        "kana_margin": kana_margin,
        "kana_glyph_count": int(kana_glyph_count),
        "minimum_kana_margin": float(min_kana_margin),
        "distance_unit": "hybrid_distance_lower_is_better",
        "auto_accept": False,
    }
