"""Check frozen protocol and all source crop hashes without rescoring."""
import hashlib
import json

from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.validate_aligned_holdout import OUTPUT, hashes


def main():
    freeze = json.loads((OUTPUT/"protocol-freeze.json").read_text())
    assert freeze["hashes"] == hashes(), "Frozen implementation or fonts changed"
    pools = json.loads((ROOT/"data/fresh-holdout-v2/pools.json").read_text())
    ids = [s["qid"] for s in pools["samples"]]
    assert len(ids) == len(set(ids)), "Duplicate question"
    old = json.loads((ROOT/"data/active-selection-v1/pools.json").read_text())
    old_ids = {s["qid"] for s in old["samples"]}
    old_ids.update(s["qid"] for s in json.loads((ROOT/"data/holdout_samples_v1.json").read_text())["samples"])
    assert not old_ids.intersection(ids), "Prior font-evaluation question reused"
    for sample in pools["samples"]:
        for glyph in sample["glyphs"]:
            path = resolve_repo_path(glyph["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == glyph["sha256"]
            image = imread(path)
            assert image.size and len(glyph["character"]) == 1
    if (OUTPUT/"results.json").exists():
        result = json.loads((OUTPUT/"results.json").read_text())
        for path, expected in result["provenance"].items():
            assert hashlib.sha256((resolve_repo_path(path)).read_bytes()).hexdigest() == expected, path
        assert ids == [s["qid"] for s in result["samples"]]
    print({"protocol": "unchanged", "questions": ids,
           "glyphs": sum(len(s["glyphs"]) for s in pools["samples"]),
           "crop_hashes": "verified", "old_font_evaluation_overlap": 0})


if __name__ == "__main__":
    main()
