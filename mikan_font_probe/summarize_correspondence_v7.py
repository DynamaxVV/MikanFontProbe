"""Compare v6/v7 for every covered class and candidate pair."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from mikan_font_probe.process_regions import ROOT


def prediction(row):
    ranking = row["result"]["ranking"]
    return ranking[0]["answer"] if ranking else None


def oriented_votes(row):
    result = {}
    for pair in row["result"].get("correspondence_pairs", row["result"].get("feature_pairs", [])):
        key = tuple(sorted((pair["a"], pair["b"])))
        multiplier = 1 if pair["a"] == key[0] else -1
        result[key] = {(e["index"], e["character"]): multiplier*e["vote"] for e in pair["evidence"]}
    return result


def summarize(old, new):
    previous = {(r["group"], r["qid"]): r for r in old}
    groups = defaultdict(list)
    vote_counts = defaultdict(Counter)
    examples = []
    for row in new:
        baseline = previous[(row["group"], row["qid"])]
        if row["eligible"]:
            groups[row["expected"]].append((prediction(baseline), prediction(row)))
        old_votes, new_votes = oriented_votes(baseline), oriented_votes(row)
        answers = {r["key"]: r["answer"] for r in row["result"]["ranking"]}
        for pair, votes in new_votes.items():
            old_pair = old_votes.get(pair, {})
            is_humming = "humming" in pair
            section = "humming_pairs" if is_humming else "other_font_pairs"
            for identity in sorted(set(votes) | set(old_pair)):
                current = votes.get(identity, 0)
                prior = old_pair.get(identity, 0)
                counts = vote_counts[section]
                counts["compared_character_pairs"] += 1
                counts["previous_votes"] += bool(prior)
                counts["new_votes"] += bool(current)
                counts["gained_votes"] += bool(current) and not prior
                counts["lost_votes"] += bool(prior) and not current
                counts["flipped_votes"] += bool(prior) and bool(current) and prior != current
                if not is_humming and current != prior:
                    examples.append({"qid": row["qid"], "expected": row["expected"],
                                     "pair": pair, "character": identity[1],
                                     "previous": prior, "new": current,
                                     "selected_in_both": identity in old_pair})
                expected = row["expected"]
                if answers.get(pair[0]) == expected and answers.get(pair[1]) != expected:
                    direction = 1
                elif answers.get(pair[1]) == expected and answers.get(pair[0]) != expected:
                    direction = -1
                else:
                    direction = 0
                if direction:
                    counts["prior_supports_expected"] += prior == direction
                    counts["prior_opposes_expected"] += prior == -direction
                    counts["new_supports_expected"] += current == direction
                    counts["new_opposes_expected"] += current == -direction
    per_class = {name: {"n": len(rows),
                        "previous_correct": sum(before == name for before, _ in rows),
                        "new_correct": sum(after == name for _, after in rows)}
                 for name, rows in groups.items()}
    return {"status": "all inputs previously exposed; diagnostic vote counts are not probabilities",
            "per_class": per_class, "pair_votes": {k: dict(v) for k, v in vote_counts.items()},
            "other_font_changes": examples,
            "top1_changes": [{"qid": row["qid"], "expected": row["expected"],
                              "before": prediction(previous[(row["group"], row["qid"])]),
                              "after": prediction(row)} for row in new
                             if prediction(previous[(row["group"], row["qid"])]) != prediction(row)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/correspondence-v7")
    args = parser.parse_args()
    path = args.output/"summary.json"
    if path.exists():
        raise ValueError("Preserve completed summary")
    prior = json.loads((ROOT/"data/features-v6/final/results.json").read_text())["samples"]
    current = json.loads((args.output/"results.json").read_text())["samples"]
    report = summarize(prior, current)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "other_font_changes"}, ensure_ascii=False, indent=2))
