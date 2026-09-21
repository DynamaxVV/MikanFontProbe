"""Quality-weighted family ranking with explicitly uncalibrated confidence."""

import hashlib
import json

import cv2
import numpy as np

from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import ROOT, imread, imwrite
from mikan_font_probe.structure_font_baseline import score_pair

OUTPUT = ROOT/"data/ranked-v1"


def crop_quality(image):
    """Assess input without using candidate fonts or expected answer labels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    area = int(np.count_nonzero(mask))
    if area == 0 or area == mask.size:
        return {"score":0.,"flags":["空白或全黑裁片"],"usable":False}
    edge = np.zeros(mask.shape, bool)
    edge[[0,-1],:] = True
    edge[:,[0,-1]] = True
    edge_ratio = float(np.count_nonzero((mask>0) & edge)/area)
    contrast = float((np.percentile(gray,95)-np.percentile(gray,5))/255)
    resolution = float(min(1.,min(gray.shape)/40))
    integrity = max(0.,1-edge_ratio/.08)
    score = .35*resolution+.35*contrast+.30*integrity
    flags = []
    if edge_ratio > .015:
        flags.append("墨迹触边，可能截断")
    if contrast < .5:
        flags.append("低对比")
    if min(gray.shape) < 24:
        flags.append("分辨率偏低")
    return {"score":round(score,4),"resolution":resolution,"contrast":contrast,
            "edge_ink_fraction":edge_ratio,"flags":flags,"usable":score>=.25}


def family_ranking(fonts, distances, weights):
    """Select one face for all selected glyphs, then collapse weight variants."""
    if not len(fonts) or not len(weights) or np.sum(weights)<=0:
        return []
    means = np.average(distances,axis=1,weights=weights)
    families = {}
    for index,font in enumerate(fonts):
        family = font["family_group"]
        row = {**family,"distance":float(means[index]),"font_id":font["id"],
               "representative_face":font["postscript"]}
        if family["key"] not in families or row["distance"] < families[family["key"]]["distance"]:
            families[family["key"]] = row
    return sorted(families.values(),key=lambda a:a["distance"])


def confidence_details(ranking, fonts, distances, weights, quality):
    """Evidence index, NOT an estimated P(correct). No class-label inputs."""
    if not ranking:
        return [],["没有可比较的字形"]
    votes = [family_ranking(fonts,distances[:,i:i+1],[1])[0]["key"] for i in range(len(weights))]
    leave_one_out = []
    if len(weights)>1:
        for i in range(len(weights)):
            keep = np.arange(len(weights)) != i
            leave_one_out.append(family_ranking(fonts,distances[:,keep],np.asarray(weights)[keep])[0]["key"])
    top = []
    for rank,candidate in enumerate(ranking[:3]):
        support = float(np.average([v==candidate["key"] for v in votes],weights=weights))
        stability = float(np.mean([v==candidate["key"] for v in leave_one_out])) if leave_one_out else 0.
        gap = ranking[1]["distance"]-candidate["distance"] if rank==0 and len(ranking)>1 else 0.
        separation = float(np.clip(gap/.08,0,1))
        fit = float(np.clip(1-candidate["distance"]/.5,0,1))
        confidence = 100*fit*quality*(.4+.6*support)*(.5+.5*stability)*(.5+.5*separation)
        confidence = min(confidence,90 if len(weights)>=3 else 45)
        top.append({**candidate,"confidence":round(confidence,1),"confidence_kind":"uncalibrated_evidence_index",
                    "glyph_support":round(support,3),"leave_one_out_stability":round(stability,3),
                    "gap_to_runner_up":round(gap,5) if rank==0 else None})
    warnings = []
    if len(weights)<3: warnings.append("有效字不足3个")
    if ranking[0]["distance"]>.35: warnings.append("库内匹配不足，可能缺字体或输入受损")
    if quality<.6: warnings.append("裁片质量偏低")
    if top[0]["gap_to_runner_up"] is None or top[0]["gap_to_runner_up"]<.02:
        warnings.append("前两名过于接近")
    if top[0]["glyph_support"]<.67: warnings.append("多个字对候选支持不一致")
    if top[0]["leave_one_out_stability"]<.67: warnings.append("移除一个字后排名不稳定")
    if top[0]["confidence"]<40: warnings.append("参考置信度较低")
    return top,warnings


def load_samples():
    old = json.loads((ROOT/"data/input-ablation-v3/results.json").read_text())["questions"]
    samples = []
    for sample in old:
        glyphs = [{"character":g["character"],"path":f"data/input-ablation-v3/evidence/{g['id']}-refined.png",
                   "bbox":g["refined_bbox"]} for g in sample["glyphs"]]
        samples.append({"qid":sample["qid"],"expected":sample["expected"],"cohort":"开发9题","glyphs":glyphs})
    for sample in json.loads((ROOT/"data/holdout_samples_v1.json").read_text())["samples"]:
        glyphs = [{"character":g[2],"path":f"data/holdout-v1/crops/q{sample['qid']}-g{i}-input.png",
                   "bbox":g[0]} for i,g in enumerate(sample["glyphs"])]
        samples.append({"qid":sample["qid"],"expected":sample["group"],"cohort":"原留出12题（本轮复用）","glyphs":glyphs})
    return samples


def main():
    fonts = json.loads((OUTPUT/"templates/catalog.json").read_text())["fonts"]
    templates = {(f["id"],c):normalize(imread(OUTPUT/"templates"/p)) for f in fonts for c,p in f["templates"].items()}
    entries = {e["question_id"]:e for e in json.loads((ROOT/"data/manifest.json").read_text())["entries"]}
    results = []
    for sample in load_samples():
        qualities = [crop_quality(imread(ROOT/g["path"])) for g in sample["glyphs"]]
        selected = [i for i,q in sorted(enumerate(qualities),key=lambda p:-p[1]["score"]) if q["usable"]][:5]
        glyphs = [{**sample["glyphs"][i],"quality":qualities[i]} for i in selected]
        available = [f for f in fonts if all(g["character"] in f["templates"] for g in glyphs)]
        distances = np.zeros((len(available),len(glyphs)))
        for index,g in enumerate(glyphs):
            path = ROOT/g["path"]
            g["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            normalized = normalize(imread(path))
            for fi,font in enumerate(available):
                values,_ = score_pair(normalized,templates[font["id"],g["character"]])
                distances[fi,index] = values["hybrid"]
        weights = [g["quality"]["score"] for g in glyphs]
        ranking = family_ranking(available,distances,weights)
        quality = float(np.mean(weights)) if weights else 0.
        top,warnings = confidence_details(ranking,available,distances,weights,quality)
        for q in qualities:
            warnings.extend(q["flags"])
        modes = {}
        for mode,indices in (("full",list(range(len(available)))),("core",[i for i,f in enumerate(available) if f["family_group"]["key"] in {"shinmaru","ryumin","antique","ywheiti"}])):
            subset = [available[i] for i in indices]
            modes[mode] = {"weighted":family_ranking(subset,distances[indices],weights),
                           "unweighted":family_ranking(subset,distances[indices],np.ones(len(weights)))}
        record = {**sample,"glyphs":glyphs,"input_qualities":qualities,"top3":top,
                  "warnings":list(dict.fromkeys(warnings)),"status":"待复核" if warnings else "候选较一致，仍需复核",
                  "scope":"local_library_only","average_quality":quality,"comparisons":modes,
                  "excluded_fonts":[f["postscript"] for f in fonts if f not in available]}
        # Evidence grids show identical characters under each selected face.
        for row,candidate in enumerate(top):
            sheet = np.full((128,len(glyphs)*128,3),255,np.uint8)
            for col,g in enumerate(glyphs):
                ink = templates[candidate["font_id"],g["character"]]
                tile = cv2.resize(255-ink,(112,112),interpolation=cv2.INTER_NEAREST)
                sheet[8:120,col*128+8:col*128+120] = cv2.cvtColor(tile,cv2.COLOR_GRAY2BGR)
            imwrite(OUTPUT/"evidence"/f"q{sample['qid']}-candidate{row+1}.png",sheet)
        original = imread(ROOT/"data"/entries[sample["qid"]]["image_file"])
        for g in glyphs:
            x,y,w,h = g["bbox"]
            cv2.rectangle(original,(x,y),(x+w,y+h),(0,140,255),1)
        imwrite(OUTPUT/"evidence"/f"q{sample['qid']}-source.png",original)
        results.append(record)
    summary = {}
    for cohort in {r["cohort"] for r in results}:
        rows = [r for r in results if r["cohort"]==cohort]
        summary[cohort] = {"n":len(rows),"needs_review":sum(bool(r["warnings"]) for r in rows),
            "scores":{f"{scope}_{mode}":sum(bool(r["comparisons"][scope][mode]) and r["comparisons"][scope][mode][0]["answer"]==r["expected"] for r in rows)
            for scope in ("core","full") for mode in ("weighted","unweighted")}}
    output = {"confidence":"heuristic_evidence_not_probability","summary":summary,"results":results,
              "fonts":fonts,"rules":{"max_glyphs":5,"min_quality":.25,"minimum_for_confidence":3,
              "ranking":"same face per block, quality-weighted mean hybrid distance; best face per family"}}
    (OUTPUT/"results.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == "__main__":
    main()
