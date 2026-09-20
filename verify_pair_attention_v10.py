"""Fresh replay and integrity check for the frozen v10 diagnostic run."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evaluate_pair_attention_v10 import evaluate_one
from process_regions import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/pair-attention-v10/final")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    freeze = json.loads((args.output/"freeze.json").read_text())
    for group in ("code", "inputs"):
        for path, digest in freeze[group].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Changed {group}: {path}")
    prior = json.loads((ROOT/"data/continuous-v8/diagnostic-diverse.json").read_text())
    jobs = []
    for group, name in (("samples", "data/category-expansion-v3"),
                        ("controls", "data/humming-v4/folk-controls")):
        pool = {s["qid"]: s for s in json.loads((ROOT/name/"pools.json").read_text())["samples"]}
        jobs += [(name, pool[row["qid"]], row) for row in prior["samples"] if row["group"] == group]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        fresh = list(executor.map(evaluate_one, jobs))
    saved = json.loads((args.output/"results.json").read_text())["samples"]
    if json.loads(json.dumps(fresh, ensure_ascii=False)) != saved:
        raise ValueError("Live result differs from frozen record")
    print(f"PASS: {len(fresh)} fresh cases and all frozen hashes")


if __name__ == "__main__":
    main()
