"""Controlled effective-resolution diagnostic; not production label rules."""
import json
import numpy as np
from mikan_font_probe.aligned_discrimination import synthetic_query
from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.stroke_profile_matcher import hybrid_distance

BASE = ROOT/"data/category-expansion-v3"


def main():
    rows = json.loads((BASE/"pools.json").read_text())["samples"]
    fonts = json.loads((BASE/"templates/catalog.json").read_text())["fonts"]
    records = []
    for row in rows:
        if row["expected"] != "轻吟体":
            continue
        ranking = []
        for font in fonts:
            if font["family_group"]["key"] not in ("humming", "skip", "folk"):
                continue
            for factor in (1, 2, 3, 4):
                scores = []
                for glyph in row["glyphs"]:
                    source = normalize(imread(resolve_repo_path(glyph["path"])))
                    image = imread(BASE/"templates"/font["templates"][glyph["character"]])
                    reference = synthetic_query(image, max(8, round(glyph["resolution"]/factor)))
                    scores.append(hybrid_distance(source, reference))
                ranking.append({"font": font["postscript"], "factor": factor, "mean": float(np.mean(scores))})
        ranking.sort(key=lambda r: r["mean"])
        records.append({"qid": row["qid"], "ranking": ranking})
        print(row["qid"], ranking[:6], flush=True)
    (ROOT/"data/humming-v4/resolution-probe.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
