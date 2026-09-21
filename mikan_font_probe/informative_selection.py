"""Candidate-driven querying and pairwise evidence, independent of answers.

separation(a, b, character, pixels) -> float or None. Distances come from a
font-only table, never from unread image-match scores or answer labels.
"""

from itertools import combinations

import numpy as np

from mikan_font_probe.rank_font_candidates import family_ranking

# Font-table retrieval band, fixed from font-only evidence, not image answers.
NEAR_SHARED = .12
OBSERVED_MARGIN = .015


def ranking_for(fonts, scores, glyphs, selected):
    return family_ranking(fonts, scores[:,selected],
                          [glyphs[i]["quality"]["score"] for i in selected])


def choose_next(glyphs, selected, ranking, separation):
    """Choose an unseen glyph without reading its image-match result."""
    available = [i for i,g in enumerate(glyphs)
                 if i not in selected and g["quality"]["usable"]]
    if not available:
        return None, []
    if not ranking:
        best = max(available,key=lambda i:(glyphs[i]["quality"]["score"],-i))
        return best,[{"index":best,"reason":"quality_seed"}]
    contenders = ranking[:5]
    closeness = np.exp(-np.array([r["distance"]-contenders[0]["distance"] for r in contenders])/.06)
    repeated = {glyphs[i]["character"] for i in selected}
    utilities = []
    for i in available:
        glyph = glyphs[i]
        total,denominator,known = 0.,0.,0
        pair_details = []
        for a,b in combinations(range(len(contenders)),2):
            weight = float(closeness[a]*closeness[b])
            value = separation(contenders[a]["key"],contenders[b]["key"],
                               glyph["character"],glyph["resolution"])
            if value is None:
                continue
            known += 1
            # Near-shared shapes cannot be treated as informative votes.
            effective = max(0.,value-getattr(separation, "near_shared", NEAR_SHARED))
            total += weight*effective
            denominator += weight
            pair_details.append({"a":contenders[a]["key"],"b":contenders[b]["key"],"distance":value})
        novelty = .15 if glyph["character"] in repeated else 1.
        gain = total/denominator if denominator else 0.
        utilities.append({"index":i,"character":glyph["character"],"gain":gain,
                          "utility":gain*glyph["quality"]["score"]*novelty,
                          "quality":glyph["quality"]["score"],"novelty":novelty,
                          "known_pairs":known,"pair_details":pair_details})
    utilities.sort(key=lambda a:(-a["utility"],-a["quality"],a["index"]))
    return utilities[0]["index"],utilities


def family_char_distance(fonts,scores,family,index):
    values = [scores[j,index] for j,f in enumerate(fonts) if f["family_group"]["key"]==family]
    return min(values) if values else None


def explain_pair(fonts,scores,glyphs,selected,a,b,separation):
    """Shared/near-shared glyphs are neutral, not winner-take-all votes."""
    seen = set()
    rows = []
    for i in sorted(selected,key=lambda n:-glyphs[n]["quality"]["score"]):
        glyph = glyphs[i]
        char = glyph["character"]
        if char in seen:
            continue
        seen.add(char)
        potential = separation(a,b,char,glyph["resolution"])
        da = family_char_distance(fonts,scores,a,i)
        db = family_char_distance(fonts,scores,b,i)
        reason = "inconclusive"
        if potential is None or da is None or db is None:
            reason = "table_missing"
        elif potential <= getattr(separation, "near_shared", NEAR_SHARED):
            reason = "shared_or_near_at_resolution"
        elif glyph["quality"]["score"] < .5:
            reason = "low_quality"
        elif db-da > OBSERVED_MARGIN:
            reason = "supports_candidate"
        elif da-db > OBSERVED_MARGIN:
            reason = "supports_rival"
        rows.append({"index":i,"character":char,"script":glyph["script"],
                     "resolution":glyph["resolution"],"separation":potential,
                     "observed_margin":float(db-da) if da is not None and db is not None else None,
                     "reason":reason,"quality":glyph["quality"]["score"]})
    return {"candidate":a,"rival":b,"evidence":rows,
            "support":sum(r["reason"]=="supports_candidate" for r in rows),
            "opposition":sum(r["reason"]=="supports_rival" for r in rows),
            "neutral":sum(r["reason"] not in ("supports_candidate","supports_rival") for r in rows)}


def explain_confidence(fonts,scores,glyphs,selected,ranking,separation):
    if not ranking:
        return {"state":"no_usable_input","top3":[]}
    top = []
    for candidate in ranking[:3]:
        rivals = [r for r in ranking[:3] if r["key"]!=candidate["key"]]
        pairs = [explain_pair(fonts,scores,glyphs,selected,candidate["key"],r["key"],separation) for r in rivals]
        # Evidence must distinguish the candidate from EACH nearby rival.
        support = min((p["support"] for p in pairs),default=0)
        opposition = max((p["opposition"] for p in pairs),default=0)
        quality = float(np.mean([glyphs[i]["quality"]["score"] for i in selected]))
        fit = float(np.clip(1-candidate["distance"]/.5,0,1))
        confidence = 100*fit*quality*min(1,support/2)/(1+opposition)
        confidence = min(85.,confidence)
        top.append({**candidate,"confidence":round(confidence,1),
                    "confidence_kind":"uncalibrated_discriminating_evidence_index",
                    "pairwise_evidence":pairs,
                    "explanation":"至少两个不同字符支持且无明显反证" if support>=2 and opposition==0
                    else "区分性证据不足或存在反证；共享字形不算支持票"})
    leading = top[0]
    enough = all(p["support"]>=2 and p["opposition"]==0 for p in leading["pairwise_evidence"])
    state = "discriminating_evidence_present" if enough else "needs_more_discriminating_glyphs"
    if ranking[0]["distance"]>.35:
        state = "library_match_insufficient"
    return {"state":state,"top3":top}


def run_selection(fonts,scores,glyphs,separation,budget=7,adaptive_stop=False):
    selected,trace = [],[]
    ranking = []
    previous = None
    stop_reason = "budget_exhausted"
    for _ in range(budget):
        index,utilities = choose_next(glyphs,selected,ranking,separation)
        if index is None:
            stop_reason = "pool_exhausted"
            break
        if adaptive_stop and len(selected)>=3 and utilities and utilities[0].get("gain",0)<=0:
            stop_reason = "no_known_discriminating_glyph_in_pool"
            break
        selected.append(index)
        ranking = ranking_for(fonts,scores,glyphs,selected)
        explanation = explain_confidence(fonts,scores,glyphs,selected,ranking,separation)
        trace.append({"step":len(selected),"selected_index":index,"character":glyphs[index]["character"],
                      "selection":utilities[:3],"ranking":ranking[:5],"evidence":explanation})
        if adaptive_stop and len(selected)>=3 and previous==ranking[0]["key"] and explanation["state"]=="discriminating_evidence_present":
            stop_reason = "enough_discriminating_evidence"
            break
        previous = ranking[0]["key"]
    explanation = explain_confidence(fonts,scores,glyphs,selected,ranking,separation)
    return {"selected":selected,"trace":trace,"ranking":ranking,"explanation":explanation,"stop_reason":stop_reason}
