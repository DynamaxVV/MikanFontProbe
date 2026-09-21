"""Collect the latest visual reviews for the prioritized image batch.

These are subjective LLM review scores, not measured OCR/font accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

from mikan_font_probe.paths import ROOT
SOURCES = [
    "eval-priority-a.json",
    "eval-priority-b.json",
    "eval-priority-v2.json",
    "eval-priority-v3-spot.json",
    "eval-priority-v3-q65.json",
    "eval-priority-v4-q100.json",
    "eval-priority-v4-rest.json",
    "eval-priority-v5-q33.json",
    "eval-priority-v7-fixes.json",
]


def main() -> None:
    data = ROOT / "data"
    priority = json.loads((data / "priority.json").read_text())
    expected = {qid: group["label"] for group in priority["groups"] for qid in group["sample_ids"]}
    latest: dict[int, dict] = {}
    for filename in SOURCES:
        path = data / filename
        if not path.exists():
            continue
        entries = json.loads(path.read_text())
        for entry in entries if isinstance(entries, list) else [entries]:
            qid = int(entry["question_id"])
            if qid in expected:
                latest[qid] = {
                    "question_id": qid,
                    "coarse_label": expected[qid],
                    "scores": {key: entry[key] for key in ("text_coverage", "background_rejection", "box_tightness")},
                    "score_source": filename,
                }
    missing = sorted(set(expected) - set(latest))
    if missing:
        raise ValueError(f"Missing scores for prioritized questions: {missing}")
    for entry in latest.values():
        scores = entry["scores"]
        if scores["text_coverage"] < 4 or scores["background_rejection"] < 3 or scores["box_tightness"] < 3:
            entry["status"] = "reject"
        elif min(scores.values()) < 4:
            entry["status"] = "review"
        else:
            entry["status"] = "candidate"
    # Two independent reviewers disagreed on q27 but both found severe speed-line leakage.
    latest[27]["status"] = "reject"
    result = {
        "note": "Visual LLM judgments on per-column masks; not ground-truth OCR or font-ID accuracy. candidate means usable for a later font-file experiment, not a correct font match.",
        "version": "priority-v7",
        "entries": [latest[qid] for qid in expected],
    }
    path = data / "scorecard-priority-v7.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    counts = {status: sum(item["status"] == status for item in result["entries"]) for status in ("candidate", "review", "reject")}
    print(json.dumps({"count": len(latest), "statuses": counts, "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
