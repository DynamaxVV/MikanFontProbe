"""Untuned real Folk negative controls for the Humming-focused changes."""
import json
import numpy as np

from mikan_font_probe.evaluate_active_selection import quality_order
from mikan_font_probe.geometry_font_matcher import GeometryReferences, prepare, score
from mikan_font_probe.glyph_geometry import effective_resolution
from mikan_font_probe.informative_selection import ranking_for
from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.stroke_profile_matcher import hybrid_distance

OUT = ROOT/"data/humming-v4/folk-controls"


def main():
    if (OUT/"results.json").exists():
        raise ValueError("Preserve first negative-control evaluation")
    pools = json.loads((OUT/"pools.json").read_text())["samples"]
    references = GeometryReferences(OUT/"templates")
    fonts = references.fonts
    rows = []
    for sample in pools:
        glyphs = sample["glyphs"]
        old, new = (np.zeros((len(fonts),len(glyphs))) for _ in (0,1))
        for index, glyph in enumerate(glyphs):
            image = imread(resolve_repo_path(glyph["path"]))
            normalized = normalize(image)
            source = prepare(normalized)
            pixels = effective_resolution(image)["effective"]
            for fi, font in enumerate(fonts):
                raw = imread(OUT/"templates"/font["templates"][glyph["character"]])
                old[fi,index] = hybrid_distance(normalized, normalize(raw))
                new[fi,index] = score(source, references.reference(font,glyph["character"],pixels))
        selected = quality_order(glyphs)[:7]
        records = {key: ranking_for(fonts, values, glyphs, selected)
                   for key, values in (("baseline",old),("geometry",new))}
        rows.append({"qid": sample["qid"], "expected": sample["expected"],
                     "selected": selected, "eligible_seven": len(selected)==7,
                     "ranking": records})
        print(sample["qid"], {key: value[0]["answer"] for key,value in records.items()},flush=True)
    (OUT/"results.json").write_text(json.dumps({"status":"first untuned real negative controls",
        "samples":rows},ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
