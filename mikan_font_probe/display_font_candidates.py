"""User-facing font candidates derived without changing internal rankings."""


def proxy_answer(candidate):
    """Map source faces to the current question-bank Chinese answer vocabulary."""
    if candidate.get("representative_face", "").startswith("SeuratProN-"):
        return "圆体"
    return candidate["answer"]


def display_candidates(ranking, limit=3):
    """Keep the highest-ranked candidate for each displayed answer."""
    candidates = []
    seen_answers = set()
    for candidate in ranking:
        answer = proxy_answer(candidate)
        if answer in seen_answers:
            continue
        candidates.append({**candidate, "answer": answer})
        seen_answers.add(answer)
        if len(candidates) == limit:
            break
    return candidates
