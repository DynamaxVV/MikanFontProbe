"""Verify frozen Folk control inputs without loading fonts or experiment scores."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED = {
    79: "屋根閉めます",
    127: "ってどうしまた息荒いですけ",
    128: "ヒビキさん今どこですかもう約",
    132: "スケッチしてたんですよね",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_image(path):
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    sources = {e["question_id"]: e for e in manifest["entries"]}
    rows = json.loads((HERE / "pools.json").read_text())["samples"]
    assert [r["qid"] for r in rows] == list(EXPECTED)
    crop_paths = set()
    summary = []
    for row in rows:
        qid = row["qid"]
        assert row["expected"] == sources[qid]["coarse_label"] == "方新书"
        assert row["source_path"] == "data/" + sources[qid]["image_file"]
        source = ROOT / row["source_path"]
        assert digest(source) == row["source_sha256"]
        original = read_image(source)
        glyphs = row["glyphs"]
        chars = "".join(g["character"] for g in glyphs)
        assert chars == EXPECTED[qid]
        assert len(chars) == len(set(chars)) <= 14
        assert len(chars) == row["distinct_characters"]
        assert row["meets_seven_distinct"] == (len(chars) >= 7)
        assert len(chars) >= 7 or row["shortfall_reason"]
        for glyph in glyphs:
            path = ROOT / glyph["path"]
            assert path.parent == HERE / "crops"
            assert digest(path) == glyph["sha256"]
            x, y, w, h = glyph["bbox"]
            assert 0 <= x < x+w <= original.shape[1]
            assert 0 <= y < y+h <= original.shape[0]
            assert np.array_equal(read_image(path), original[y:y+h, x:x+w])
            assert glyph["polarity"] == row["polarity"] == "dark"
            crop_paths.add(path)
        for suffix in ("glyphs", "source-bboxes"):
            assert (HERE / "review" / f"q{qid}-{suffix}.png").is_file()
        summary.append({"qid": qid, "characters": chars, "count": len(chars),
                        "source_sha256": digest(source), "polarity": "dark"})
    assert crop_paths == set((HERE / "crops").glob("*.png"))
    assert (HERE / "characters.txt").read_text().strip() == "".join(sorted(set("".join(EXPECTED.values()))))
    files = {str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob("*"))
             if p.is_file() and p.name != "freeze.json"}
    frozen = {"status": "fixed inputs; visually reviewed; no experiment scores read",
              "pixel_check": "all 45 crops equal decoded source bbox pixels exactly",
              "samples": summary, "files_sha256": files}
    target = HERE / "freeze.json"
    if target.exists():
        assert json.loads(target.read_text()) == frozen, "Frozen input changed"
    else:
        target.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"verified": True, "samples": summary, "crop_count": len(crop_paths)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
