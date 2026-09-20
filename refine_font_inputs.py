"""Paired crop/grayscale ablation, with no question-specific matching rules."""

import json

import cv2
import numpy as np

from match_font_baseline import match, normalize
from process_regions import ROOT, imread, imwrite


def refine_box(image, box):
    """Snap cell boundaries to nearby low-ink valleys in the original image."""
    x, y, w, h = box
    height, width = image.shape[:2]
    # Remove the known context margin, then search either side of each cut.
    left, top, right, bottom = x+2, y+2, x+w-2, y+h-2
    radius = max(2, min(5, round(min(w,h)*.09)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = (255-gray)/255.
    def snap(position, limit, profile, direction):
        # Expansion-only: an internal glyph gap must not erase existing ink.
        low, high = (position-radius,position) if direction < 0 else (position,position+radius)
        candidates = np.arange(max(0,low), min(limit-1,high)+1)
        costs = profile[candidates] + .025*abs(candidates-position)/radius
        return int(candidates[np.argmin(costs)])
    horizontal = ink[max(0,top):min(height,bottom)].mean(axis=0)
    vertical = ink[:,max(0,left):min(width,right)].mean(axis=1)
    left = snap(left,width,horizontal,-1)
    right = snap(right,width,horizontal,1)
    top = snap(top,height,vertical,-1)
    bottom = snap(bottom,height,vertical,1)
    if right <= left or bottom <= top:
        raise ValueError("Invalid refined crop")
    return [left,top,right-left,bottom-top]


def soft_normalize(image):
    """Keep grayscale coverage; threshold only establishes the bounding box."""
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ys,xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Blank glyph")
    ink = 255-gray[ys.min():ys.max()+1,xs.min():xs.max()+1]
    ratio = 52/max(ink.shape)
    w,h = max(1,round(ink.shape[1]*ratio)),max(1,round(ink.shape[0]*ratio))
    resized = cv2.resize(ink,(w,h),interpolation=cv2.INTER_AREA)
    result = np.zeros((64,64),np.uint8)
    x,y = (64-w)//2,(64-h)//2
    result[y:y+h,x:x+w] = resized
    return result


def soft_match(source, template):
    """Soft overlap under the same bounded alignments as the binary baseline."""
    a = source.astype(np.float32)/255
    best = (float("inf"),None,None)
    for scale in (.96,1.,1.04):
        for dx in (-2,0,2):
            for dy in (-2,0,2):
                matrix = np.array([[scale,0,32*(1-scale)+dx],
                                   [0,scale,32*(1-scale)+dy]],np.float32)
                aligned = cv2.warpAffine(template,matrix,(64,64))
                b = aligned.astype(np.float32)/255
                # L1 normalized by total ink; no background-dominated accuracy.
                loss = float(np.abs(a-b).sum()/max(float(a.sum()+b.sum()),1e-6))
                if loss < best[0]:
                    best = (loss,aligned,{"scale":scale,"dx":dx,"dy":dy})
    return best


def native_template(template, source):
    """Downsample template to observed ink resolution before normalizing."""
    def tight(image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ys,xs = np.nonzero(mask)
        return gray[ys.min():ys.max()+1,xs.min():xs.max()+1]
    ink = tight(template)
    target = tight(source)
    ratio = max(target.shape)/max(ink.shape)
    small = cv2.resize(ink,(max(1,round(ink.shape[1]*ratio)),max(1,round(ink.shape[0]*ratio))),
                       interpolation=cv2.INTER_AREA)
    return soft_normalize(cv2.copyMakeBorder(small,2,2,2,2,cv2.BORDER_CONSTANT,value=255))


def main():
    output = ROOT/"data/input-ablation-v3"
    source = ROOT/"data/processed/typographic-v3"
    template_dir = ROOT/"data/baseline-v1/templates"
    config = json.loads((ROOT/"data/baseline_samples.json").read_text())
    fonts = json.loads((template_dir/"catalog.json").read_text())["fonts"]
    regions = {r["question_id"]:r for r in json.loads((source/"results.json").read_text())}
    templates = {}
    for mode,normalizer in (("binary",normalize),("soft",soft_normalize)):
        templates[mode] = {(f["id"],char):normalizer(imread(template_dir/file))
                           for f in fonts for char,file in f["templates"].items()}
    results = []
    for sample in config["samples"]:
        qid = sample["qid"]
        original = imread(source/f"q{qid}-source.png")
        overlay = original.copy()
        records = []
        scores = {name:{f["id"]:[] for f in fonts} for name in
                  ("original_binary","refined_binary","original_soft","refined_soft","refined_native")}
        for col,index,char in sample["glyphs"]:
            region = regions[qid]["columns"][col]["glyphs"][index]
            box = refine_box(original,region["source_bbox"])
            x,y,w,h = box
            crops = {"original":imread(source/region["source_file"])[2:-2,2:-2],
                     "refined":original[y:y+h,x:x+w]}
            sid = f"q{qid}-c{col}-g{index}"
            record = {"id":sid,"character":char,"original_bbox":region["source_bbox"],
                      "refined_bbox":box,"rankings":{}}
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,150,255),1)
            for crop_kind,crop in crops.items():
                imwrite(output/"evidence"/f"{sid}-{crop_kind}.png",crop)
                modes = [("binary",normalize,match),("soft",soft_normalize,soft_match)]
                if crop_kind == "refined":
                    modes.append(("native",soft_normalize,soft_match))
                for mode,normalizer,matcher in modes:
                    key = f"{crop_kind}_{mode}"
                    normalized = normalizer(crop)
                    ranking = []
                    for font in fonts:
                        template = (native_template(imread(template_dir/font["templates"][char]),crop)
                                    if mode == "native" else templates[mode][font["id"],char])
                        loss,_,_ = matcher(normalized,template)
                        scores[key][font["id"]].append(loss)
                        ranking.append({"font":font["postscript"],"group":font["group"],"distance":loss})
                    record["rankings"][key] = sorted(ranking,key=lambda a:a["distance"])
            records.append(record)
        variants = {}
        for key,per_font in scores.items():
            ranking = sorted([{"font":f["postscript"],"group":f["group"],
                               "distance":float(np.mean(per_font[f["id"]]))} for f in fonts],
                             key=lambda a:a["distance"])
            groups = []
            for r in ranking:
                if r["group"] not in [g["group"] for g in groups]:
                    groups.append(r)
            variants[key] = groups
        imwrite(output/"evidence"/f"q{qid}-refined-boxes.png",overlay)
        results.append({"qid":qid,"expected":sample["group"],"variants":variants,"glyphs":records})
    totals = {key:sum(r["variants"][key][0]["group"]==r["expected"] for r in results)
              for key in results[0]["variants"]}
    (output/"results.json").write_text(json.dumps({"status":"development_ablation_not_blind_test",
        "note":"soft changes both coverage representation and metric, not an isolated grayscale-only experiment",
        "correct":totals,"questions":results},ensure_ascii=False,indent=2)+"\n")
    print(totals)
    for r in results:
        print(r["qid"],r["expected"],{k:v[0]["group"] for k,v in r["variants"].items()})


if __name__ == "__main__":
    main()
