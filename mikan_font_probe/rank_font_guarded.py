"""Usable v4 entry point: quality-selected text, conservative geometry rerank."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mikan_font_probe.evaluate_active_selection import quality_order
from mikan_font_probe.geometry_font_matcher import GeometryReferences, prepare, score
from mikan_font_probe.glyph_geometry import effective_resolution
from mikan_font_probe.informative_selection import OBSERVED_MARGIN, ranking_for
from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.resolve_font_ambiguity import resolve
from mikan_font_probe.stroke_profile_matcher import hybrid_distance


def rank_sample(sample, template_directory, budget=7):
    references = GeometryReferences(template_directory)
    fonts = references.fonts
    indices = quality_order(sample["glyphs"])[:budget]
    glyphs = [sample["glyphs"][i] for i in indices]
    if not glyphs:
        return resolve([], [], 0)
    if sample.get("polarity") == "light" or any(g.get("polarity") == "light" for g in glyphs):
        return {"ranking": [], "state": "unsupported_light_text_review_required", "auto_accept": False}
    old_scores = np.zeros((len(fonts),len(glyphs)))
    sources, resolutions = [], []
    for index, glyph in enumerate(glyphs):
        path = resolve_repo_path(glyph["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != glyph["sha256"]:
            raise ValueError("Original crop changed")
        image = imread(path)
        source = normalize(image)
        sources.append(prepare(source))
        resolutions.append(effective_resolution(image))
        for fi, font in enumerate(fonts):
            target = normalize(imread(Path(template_directory)/font["templates"][glyph["character"]]))
            old_scores[fi,index] = hybrid_distance(source,target)
    local_indices = list(range(len(glyphs)))
    appearance = ranking_for(fonts,old_scores,glyphs,local_indices)
    geometry = []
    if len(appearance)>1 and appearance[1]["distance"]-appearance[0]["distance"] <= OBSERVED_MARGIN:
        new_scores = np.zeros_like(old_scores)
        for index,glyph in enumerate(glyphs):
            for fi,font in enumerate(fonts):
                target = references.reference(font,glyph["character"],resolutions[index]["effective"])
                new_scores[fi,index] = score(sources[index],target)
        geometry = ranking_for(fonts,new_scores,glyphs,local_indices)
    result = resolve(appearance,geometry,len({g["character"] for g in glyphs}))
    return {**result,"selected_indices":indices,"characters":"".join(g["character"] for g in glyphs),
            "resolutions":resolutions,"top3":result["ranking"][:3]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid",type=int,required=True)
    parser.add_argument("--pools",type=Path,default=ROOT/"data/category-expansion-v3/pools.json")
    parser.add_argument("--templates",type=Path,default=ROOT/"data/category-expansion-v3/templates")
    args = parser.parse_args()
    sample = next(s for s in json.loads(args.pools.read_text())["samples"] if s["qid"] == args.qid)
    print(json.dumps(rank_sample(sample,args.templates),ensure_ascii=False,indent=2))
