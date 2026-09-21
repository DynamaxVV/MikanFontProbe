"""Replay v7 from real input while changing expected labels deliberately."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mikan_font_probe.process_regions import ROOT
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.rank_font_correspondence import augment
from mikan_font_probe.rank_font_guarded import rank_sample


def verify_one(job):
    sample, directory, saved = job
    sample = {**sample, "expected": "deliberately invalid label"}
    result = augment(sample, directory, rank_sample(sample, directory))
    for key in ("correspondence_state", "correspondence_pairs", "source_features",
                "correspondence_changed"):
        assert result.get(key) == saved.get(key), (sample["qid"], key)
    assert [r["key"] for r in result["top3"]] == [r["key"] for r in saved["top3"]]
    assert result["auto_accept"] is False
    return sample["qid"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/correspondence-v7")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    freeze = json.loads((args.output/"freeze.json").read_text())
    for section in ("code", "inputs"):
        for name, sha in freeze[section].items():
            assert hashlib.sha256((resolve_repo_path(name)).read_bytes()).hexdigest() == sha, name
    records = json.loads((args.output/"results.json").read_text())["samples"]
    pools = {}
    for group, name in (("samples", "data/category-expansion-v3"),
                        ("controls", "data/humming-v4/folk-controls")):
        directory = resolve_repo_path(name)
        pools[group] = ({s["qid"]: s for s in json.loads((directory/"pools.json").read_text())["samples"]},
                        directory/"templates")
    jobs = [(pools[r["group"]][0][r["qid"]], pools[r["group"]][1], r["result"]) for r in records]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for qid in executor.map(verify_one, jobs):
            print("verified", qid, flush=True)
    print(f"PASS {len(records)} live decisions; all saved feature evidence reproduced; labels inert; input/code hashes match")


if __name__ == "__main__":
    main()
