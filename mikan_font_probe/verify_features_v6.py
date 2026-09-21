"""Fresh end-to-end inference, label-independence and preserved-artifact audit."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mikan_font_probe.process_regions import ROOT
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.rank_font_features import augment
from mikan_font_probe.rank_font_guarded import rank_sample


def verify_one(job):
    sample, templates, expected = job
    sample = {**sample, "expected": "not a font label"}
    actual = augment(sample, templates, rank_sample(sample, templates))
    for key in ("feature_state", "feature_changed", "feature_pairs", "source_features"):
        assert actual.get(key) == expected.get(key), (sample["qid"], key)
    assert [r["key"] for r in actual["top3"]] == [r["key"] for r in expected["top3"]], sample["qid"]
    assert actual["auto_accept"] is False
    return sample["qid"]


def check_hashes(path):
    freeze = json.loads(path.read_text())
    for section in ("code", "inputs"):
        for name, sha in freeze[section].items():
            assert hashlib.sha256((resolve_repo_path(name)).read_bytes()).hexdigest() == sha, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/features-v6")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    check_hashes(args.output/"freeze.json")
    check_hashes(ROOT/"data/contours-v5/final/freeze.json")
    records = json.loads((args.output/"results.json").read_text())["samples"]
    regressions = [r["qid"] for r in records if r["eligible"] and r["baseline"] == r["expected"]
                   and (not r["result"]["ranking"] or r["result"]["ranking"][0]["answer"] != r["expected"])]
    assert not regressions, ("new regressions", regressions)
    pools = {}
    for group, name in (("samples", "data/category-expansion-v3"), ("controls", "data/humming-v4/folk-controls")):
        directory = resolve_repo_path(name)
        pools[group] = ({s["qid"]: s for s in json.loads((directory/"pools.json").read_text())["samples"]},
                        directory/"templates")
    jobs = [(pools[r["group"]][0][r["qid"]], pools[r["group"]][1], r["result"]) for r in records]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for qid in executor.map(verify_one, jobs):
            print("verified", qid, flush=True)
    print(f"PASS {len(records)} fresh inferences and complete evidence; labels inert; v5/v6 hashes unchanged")


if __name__ == "__main__":
    main()
