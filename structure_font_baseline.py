"""Fixed structural-score ablations on existing known-character inputs."""

import json

import cv2
import numpy as np

from match_font_baseline import match, normalize
from process_regions import ROOT, imread, imwrite


def structure_features(ink):
    """Spatial stroke orientations plus normalized horizontal/vertical profiles."""
    if ink.shape != (64,64):
        raise ValueError("Expected a normalized 64x64 glyph")
    gray = ink.astype(np.float32)/255
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    magnitude,angle = cv2.cartToPolar(gx,gy)
    bins = np.floor((angle % np.pi)*8/np.pi).astype(np.int32)%8
    cells = []
    for y in range(0,64,16):
        for x in range(0,64,16):
            hist = np.bincount(bins[y:y+16,x:x+16].ravel(),
                               weights=magnitude[y:y+16,x:x+16].ravel(),minlength=8)
            cells.append(hist/(hist.sum()+1e-8))
    profile = np.r_[gray.sum(axis=0),gray.sum(axis=1)]
    profile /= profile.sum()+1e-8
    return np.array(cells),profile


def structural_distance(a,b):
    orientations = float(np.abs(a[0]-b[0]).sum(axis=1).mean()/2)
    projection = float(np.abs(a[1]-b[1]).sum()/2)
    return .75*orientations+.25*projection


def score_pair(source,template):
    """Same limited transforms as v1; optimize each scoring branch separately."""
    baseline,aligned,transform = match(source,template)
    features = structure_features(source)
    structural = (float("inf"),None)
    for scale in (.96,1.,1.04):
        for dx in (-2,0,2):
            for dy in (-2,0,2):
                matrix = np.array([[scale,0,32*(1-scale)+dx],
                                   [0,scale,32*(1-scale)+dy]],np.float32)
                candidate = cv2.warpAffine(template,matrix,(64,64))
                loss = structural_distance(features,structure_features(candidate))
                if loss < structural[0]:
                    structural = (loss,candidate)
    at_baseline = structural_distance(features,structure_features(aligned))
    # Hybrid uses ONE alignment, not independently optimized incompatible parts.
    return {"binary":baseline,"structure":structural[0],
            "hybrid":.5*baseline+.5*at_baseline},structural[1]


def controlled_queries(catalog,templates,directory,output):
    """Known-library positive controls, not independent real-world evaluation."""
    records = []
    for known in catalog:
        totals = {f["id"]:{mode:[] for mode in ("binary","structure","hybrid")} for f in catalog}
        for char in ("今","万","ま"):
            original = imread(directory/known["templates"][char])
            ratio = 32/max(original.shape[:2])
            small = cv2.resize(original,(round(original.shape[1]*ratio),round(original.shape[0]*ratio)),
                               interpolation=cv2.INTER_AREA)
            blurred = cv2.GaussianBlur(small,(3,3),.45)
            ok,jpeg = cv2.imencode(".jpg",blurred,[cv2.IMWRITE_JPEG_QUALITY,65])
            if not ok:
                raise ValueError("JPEG control encoding failed")
            degraded = cv2.imdecode(jpeg,cv2.IMREAD_COLOR)
            imwrite(output/"controls"/f"{known['id']}-{ord(char):x}.png",degraded)
            source = normalize(degraded)
            for font in catalog:
                scores,_ = score_pair(source,templates[font["id"],char])
                for mode,value in scores.items():
                    totals[font["id"]][mode].append(value)
        winners = {}
        for mode in ("binary","structure","hybrid"):
            ranked = sorted(catalog,key=lambda f:np.mean(totals[f["id"]][mode]))
            winners[mode] = {"font":ranked[0]["postscript"],"group":ranked[0]["group"],
                             "exact_font":ranked[0]["id"]==known["id"]}
        records.append({"source_font":known["postscript"],"expected":known["group"],"winners":winners})
    return {"note":"same renderer and known library; positive control only; all 12 fonts, 3 chars each",
            "records":records,"group_correct":{mode:sum(r["winners"][mode]["group"]==r["expected"] for r in records)
            for mode in ("binary","structure","hybrid")}}


def main():
    directory = ROOT/"data/baseline-v1/templates"
    output = ROOT/"data/structure-baseline-v1"
    catalog = json.loads((directory/"catalog.json").read_text())["fonts"]
    inputs = json.loads((ROOT/"data/input-ablation-v3/results.json").read_text())["questions"]
    templates = {(f["id"],c):normalize(imread(directory/p)) for f in catalog for c,p in f["templates"].items()}
    results = []
    for sample in inputs:
        scores = {mode:{f["id"]:[] for f in catalog} for mode in ("binary","structure","hybrid")}
        glyphs = []
        for glyph in sample["glyphs"]:
            source_path = ROOT/"data/input-ablation-v3/evidence"/(glyph["id"]+"-refined.png")
            source = normalize(imread(source_path))
            details = []
            for font in catalog:
                values,_ = score_pair(source,templates[font["id"],glyph["character"]])
                for mode,value in values.items():
                    scores[mode][font["id"]].append(value)
                details.append({"font":font["postscript"],"group":font["group"],"scores":values})
            glyphs.append({"id":glyph["id"],"character":glyph["character"],"scores":details})
        variants = {}
        for mode,values in scores.items():
            ranking = sorted([{"font":f["postscript"],"group":f["group"],
                               "distance":float(np.mean(values[f["id"]]))} for f in catalog],key=lambda r:r["distance"])
            groups = []
            for entry in ranking:
                if entry["group"] not in [g["group"] for g in groups]:
                    groups.append(entry)
            variants[mode] = groups
        results.append({"qid":sample["qid"],"expected":sample["expected"],"variants":variants,"glyphs":glyphs})
    output.mkdir(parents=True,exist_ok=True)
    totals = {mode:sum(r["variants"][mode][0]["group"]==r["expected"] for r in results) for mode in scores}
    (output/"results.json").write_text(json.dumps({"status":"development_only","correct":totals,
        "config":{"orientation_bins":8,"cell_size":16,"structure_weights":[.75,.25],"hybrid_weights":[.5,.5]},
        "questions":results},ensure_ascii=False,indent=2)+"\n")
    print(totals)
    for r in results:
        print(r["qid"],{k:v[0]["group"] for k,v in r["variants"].items()})
    controls = controlled_queries(catalog,templates,directory,output)
    (output/"controls.json").write_text(json.dumps(controls,ensure_ascii=False,indent=2)+"\n")
    print("Known-library controls:",controls["group_correct"])


if __name__ == "__main__":
    main()
