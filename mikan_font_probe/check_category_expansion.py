"""Validate sample identity, raw crop evidence and first-result provenance."""
import hashlib
import json

import numpy as np

from mikan_font_probe.expand_font_categories import OUT, read
from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.validate_aligned_holdout import hashes


def main():
    protocol = read(OUT/"protocol.json")
    assert hashes() == protocol["matcher_hashes"]
    pools = read(OUT/"pools.json")["samples"]
    expected = [qid for group in protocol["groups"] for qid in group["selected"]]
    assert [s["qid"] for s in pools] == expected
    entries = {e["question_id"]: e for e in read(ROOT/"data/manifest.json")["entries"]}
    for row in pools:
        source = imread(ROOT/"data"/entries[row["qid"]]["image_file"])
        for glyph in row["glyphs"]:
            path = resolve_repo_path(glyph["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == glyph["sha256"]
            x, y, w, h = glyph["bbox"]
            assert np.array_equal(imread(path), source[y:y+h, x:x+w]), (row["qid"], glyph["id"])
    if (OUT/"results.json").exists():
        result = read(OUT/"results.json")
        assert [s["qid"] for s in result["samples"]] == expected
        for name, sha in result["provenance"].items():
            assert hashlib.sha256((resolve_repo_path(name)).read_bytes()).hexdigest() == sha, name
    print(json.dumps({"questions": len(pools), "glyphs": sum(len(s["glyphs"]) for s in pools),
                      "protocol_unchanged": True, "original_pixels_verified": True}))


if __name__ == "__main__":
    main()
