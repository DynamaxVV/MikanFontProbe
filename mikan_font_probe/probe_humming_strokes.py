"""Visual-hypothesis probe: compare fixed stroke profiles, no answer bonus."""
import json

import numpy as np

from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.stroke_profile_matcher import hybrid_distance, stroke_template

BASE = ROOT/"data/category-expansion-v3"


def main():
    rows = json.loads((BASE/"pools.json").read_text())["samples"]
    fonts = json.loads((BASE/"templates/catalog.json").read_text())["fonts"]
    records = []
    for row in rows:
        if row["expected"] != "轻吟体":
            continue
        results = []
        for font in fonts:
            if font["family_group"]["key"] not in ("humming", "folk", "skip"):
                continue
            for x, y in ((0., 0.), (1.5, 0.), (0., 1.5), (1.5, 1.5), (-1.5, 0.)):
                scores = []
                for glyph in row["glyphs"]:
                    source = normalize(imread(resolve_repo_path(glyph["path"])))
                    reference = normalize(imread(BASE/"templates"/font["templates"][glyph["character"]]))
                    scores.append(hybrid_distance(source, stroke_template(reference, x, y)))
                results.append({"font": font["postscript"], "x": x, "y": y,
                                "mean": float(np.mean(scores))})
        results.sort(key=lambda item: item["mean"])
        records.append({"qid": row["qid"], "results": results})
        print(row["qid"], results[:5], flush=True)
    (ROOT/"data/humming-v4/stroke-probe.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
