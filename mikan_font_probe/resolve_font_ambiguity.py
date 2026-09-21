"""Conservative second-stage ranking; never change an established clear lead."""
from mikan_font_probe.informative_selection import OBSERVED_MARGIN


def resolve(appearance, geometry, distinct_characters):
    if not appearance:
        return {"ranking": [], "state": "no_input", "changed": False}
    geometries = {row["key"]: row for row in geometry}
    limit = appearance[0]["distance"]+OBSERVED_MARGIN
    close = [row for row in appearance if row["distance"] <= limit]
    order = appearance
    state = "appearance_lead_preserved"
    if distinct_characters < 3:
        state = "insufficient_distinct_characters"
    elif appearance[0]["distance"] > .35:
        state = "poor_library_fit_review_required"
    elif len(close) > 1 and geometry:
        close_keys = {row["key"] for row in close}
        if geometry[0]["key"] not in close_keys:
            state = "conflicting_geometry_review_required"
        elif not close_keys <= geometries.keys():
            state = "incomplete_geometry_review_required"
        else:
            order = sorted(close, key=lambda r: geometries[r["key"]]["distance"])
            order += [row for row in appearance if row["key"] not in close_keys]
            state = "ambiguous_candidates_geometry_reranked"
    rows = []
    for rank, item in enumerate(order, 1):
        rows.append({"rank": rank, "key": item["key"], "name": item["name"],
                     "answer": item["answer"], "appearance_distance": item["distance"],
                     "geometry_distance": geometries.get(item["key"], {}).get("distance"),
                     "representative_face": item.get("representative_face")})
    return {"ranking": rows, "state": state, "changed": rows[0]["key"] != appearance[0]["key"],
            "appearance_gap": appearance[1]["distance"]-appearance[0]["distance"] if len(appearance)>1 else None,
            "ambiguity_band": OBSERVED_MARGIN, "confidence_kind": "uncalibrated_evidence_only",
            "auto_accept": False}
