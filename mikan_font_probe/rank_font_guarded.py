"""Usable v4 entry point: quality-selected text, conservative geometry rerank."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mikan_font_probe.display_font_candidates import display_candidates
from mikan_font_probe.evaluate_active_selection import quality_order
from mikan_font_probe.font_style_policy import font_style_policy
from mikan_font_probe.geometry_font_matcher import GeometryReferences, prepare, score
from mikan_font_probe.glyph_geometry import effective_resolution
from mikan_font_probe.informative_selection import OBSERVED_MARGIN, ranking_for
from mikan_font_probe.japanese_script_roles_v1 import kana_subscript, script_role
from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.resolve_font_ambiguity import resolve
from mikan_font_probe.script_conflict_policy_v2 import arbitrate_family_rankings
from mikan_font_probe.stroke_profile_matcher import hybrid_distance

DEFAULT_TEMPLATES = ROOT / "data/default-gallery-v1/templates"


def _select_script_balanced(glyphs, budget):
    """Reserve up to three distinct kana before filling by input quality."""
    ordered = quality_order(glyphs)
    if budget <= 0:
        return []
    kana = [index for index in ordered
            if script_role(glyphs[index].get("character", "")) == "kana"]
    reserved = kana[:min(3, budget)]
    selected = reserved + [index for index in ordered if index not in reserved]
    return selected[:budget]


def _source_family_ranking(font_ranking, fonts):
    """Collapse source-family aliases (Rodin and Skip/Folk) for v2 evidence."""
    by_face = {font["postscript"]: font for font in fonts}
    families = {}
    for row in font_ranking:
        font = by_face.get(row.get("representative_face"))
        if font is None:
            continue
        policy = font_style_policy(font["postscript"], metadata=font)
        source_family = policy["source_family"]
        candidate = {
            "key": source_family["key"],
            "name": source_family["name"],
            "distance": float(row["distance"]),
            "representative_face": row["representative_face"],
            "legacy_key": row["key"],
        }
        previous = families.get(candidate["key"])
        if previous is None or candidate["distance"] < previous["distance"]:
            families[candidate["key"]] = candidate
    return sorted(families.values(), key=lambda row: (row["distance"], row["key"]))


def _script_family_evidence(fonts, scores, glyphs):
    # Lightweight callers and test doubles may omit font identity or character
    # metadata. Keep the legacy ranking path usable without fabricating family
    # evidence from incomplete inputs.
    if any("postscript" not in font for font in fonts):
        return None
    if any("character" not in glyph for glyph in glyphs):
        return None
    selected = list(range(len(glyphs)))
    mixed = _source_family_ranking(ranking_for(fonts, scores, glyphs, selected), fonts)
    script_rows = {}
    counts = {}
    for script in ("kana", "kanji"):
        indices = [index for index, glyph in enumerate(glyphs)
                   if script_role(glyph["character"]) == script]
        counts[script] = len(indices)
        script_rows[script] = _source_family_ranking(
            ranking_for(fonts, scores, glyphs, indices), fonts) if indices else []
    evidence = arbitrate_family_rankings(
        mixed, script_rows["kana"], script_rows["kanji"], counts["kana"])
    evidence["glyph_counts"] = counts
    evidence["kana_subscript_glyph_counts"] = {}
    evidence["kana_subscript_rankings"] = {}
    for subtype in ("hiragana", "katakana", "kana_mark"):
        indices = [index for index, glyph in enumerate(glyphs)
                   if kana_subscript(glyph["character"]) == subtype]
        evidence["kana_subscript_glyph_counts"][subtype] = len(indices)
        evidence["kana_subscript_rankings"][subtype] = (
            _source_family_ranking(ranking_for(fonts, scores, glyphs, indices), fonts)
            if indices else [])
    return evidence


def rank_sample(sample, template_directory, budget=7):
    references = GeometryReferences(template_directory)
    fonts = references.fonts
    indices = _select_script_balanced(sample["glyphs"], budget)
    glyphs = [sample["glyphs"][i] for i in indices]
    if not glyphs:
        return {**resolve([], [], 0), "display_top3": []}
    if sample.get("polarity") == "light" or any(g.get("polarity") == "light" for g in glyphs):
        return {"ranking": [], "state": "unsupported_light_text_review_required", "auto_accept": False,
                "display_top3": []}
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
    final_scores = new_scores if geometry else old_scores
    source_family_evidence = _script_family_evidence(fonts, final_scores, glyphs)
    return {**result,"selected_indices":indices,"characters":"".join(g["character"] for g in glyphs),
            "resolutions":resolutions,"top3":result["ranking"][:3],
            "display_top3":display_candidates(result["ranking"]),
            "source_family_evidence":source_family_evidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid",type=int,required=True)
    parser.add_argument("--pools",type=Path,default=ROOT/"data/category-expansion-v3/pools.json")
    parser.add_argument("--templates",type=Path,default=DEFAULT_TEMPLATES)
    args = parser.parse_args()
    sample = next(s for s in json.loads(args.pools.read_text())["samples"] if s["qid"] == args.qid)
    print(json.dumps(rank_sample(sample,args.templates),ensure_ascii=False,indent=2))
