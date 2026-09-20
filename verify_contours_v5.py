"""Verify live inference, frozen inputs/code and answer-label independence."""
import hashlib
import json

from process_regions import ROOT
from rank_font_contours import augment
from rank_font_guarded import rank_sample


def main():
    directory = ROOT/"data/contours-v5/final"
    freeze = json.loads((directory/"freeze.json").read_text())
    for section in ("code", "inputs"):
        for path, sha in freeze[section].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha, path
    pools = {}
    for group, name in (("samples", "data/category-expansion-v3"),
                        ("controls", "data/humming-v4/folk-controls")):
        folder = ROOT/name
        pools[group] = ({s["qid"]: s for s in json.loads((folder/"pools.json").read_text())["samples"]},
                        folder/"templates")
    records = json.loads((directory/"results.json").read_text())["samples"]
    for record in records:
        samples, templates = pools[record["group"]]
        sample = dict(samples[record["qid"]])
        sample["expected"] = "deliberately invalid answer; must not affect inference"
        result = augment(sample, templates, rank_sample(sample, templates))
        saved = record["result"]
        assert [r["key"] for r in result["top3"]] == [r["key"] for r in saved["top3"]], record["qid"]
        assert result["contour_pairs"] == saved["contour_pairs"], record["qid"]
        assert result["contour_state"] == saved["contour_state"], record["qid"]
        assert result["auto_accept"] is False
        print("verified", record["qid"], flush=True)
    print(f"PASS: {len(records)} live decisions/evidence; altered labels have no effect; frozen hashes match")


if __name__ == "__main__":
    main()
