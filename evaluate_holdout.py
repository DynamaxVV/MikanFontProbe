"""Prepare reviewed crops, freeze inputs, then evaluate unchanged scorers."""

import argparse
import hashlib
import json

import cv2
import numpy as np

from process_regions import ROOT, imread, imwrite
from match_font_baseline import normalize
from structure_font_baseline import score_pair

OUTPUT = ROOT/"data/holdout-v1"
CONFIG = ROOT/"data/holdout_samples_v1.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (OUTPUT/"results.json").exists():
        raise ValueError("Evaluation already run; do not revise this holdout")
    config = json.loads(CONFIG.read_text())
    entries = {e["question_id"]:e for e in json.loads((ROOT/"data/manifest.json").read_text())["entries"]}
    sheet = np.full((len(config["samples"])*110,420,3),255,np.uint8)
    provenance = []
    for row,sample in enumerate(config["samples"]):
        qid = sample["qid"]
        path = ROOT/"data"/entries[qid]["image_file"]
        image = imread(path)
        for index,(box,_,char) in enumerate(sample["glyphs"]):
            x,y,w,h = box
            if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                raise ValueError(f"Out of bounds q{qid}")
            crop = image[y:y+h,x:x+w]
            sid = f"q{qid}-g{index}"
            imwrite(OUTPUT/"crops"/f"{sid}-source.png",crop)
            prepared = 255-crop if sample["polarity"] == "light" else crop
            imwrite(OUTPUT/"crops"/f"{sid}-input.png",prepared)
            ratio = 80/max(crop.shape[:2])
            thumb = cv2.resize(crop,(round(w*ratio),round(h*ratio)),interpolation=cv2.INTER_NEAREST)
            tx,ty = index*140,row*110+25
            sheet[ty:ty+thumb.shape[0],tx:tx+thumb.shape[1]] = thumb
            cv2.putText(sheet,f"q{qid} g{index} U+{ord(char):04X}",(tx,ty-7),cv2.FONT_HERSHEY_SIMPLEX,.31,(0,0,0),1)
        provenance.append({"qid":qid,"image":str(path.relative_to(ROOT)),"sha256":digest(path)})
    imwrite(OUTPUT/"crop-review.png",sheet)
    tracked = [CONFIG,ROOT/"match_font_baseline.py",ROOT/"structure_font_baseline.py",ROOT/"build_font_templates.py",ROOT/"evaluate_holdout.py"]
    tracked += sorted((OUTPUT/"crops").glob("*.png"))
    freeze = {"input_files":{str(p.relative_to(ROOT)):digest(p) for p in tracked},"images":provenance}
    (OUTPUT/"freeze.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+"\n")


def evaluate():
    freeze = json.loads((OUTPUT/"freeze.json").read_text())
    for file,expected in freeze["input_files"].items():
        if digest(ROOT/file) != expected:
            raise ValueError(f"Frozen file changed: {file}")
    for image in freeze["images"]:
        if digest(ROOT/image["image"]) != image["sha256"]:
            raise ValueError("Frozen source image changed")
    config = json.loads(CONFIG.read_text())
    directory = OUTPUT/"templates"
    catalog = json.loads((directory/"catalog.json").read_text())["fonts"]
    for font in catalog:
        if digest(ROOT/font["path"]) != font["sha256"]:
            raise ValueError("Font file changed")
    templates = {(f["id"],c):normalize(imread(directory/p)) for f in catalog for c,p in f["templates"].items()}
    rows = []
    for sample in config["samples"]:
        scores = {mode:{f["id"]:[] for f in catalog} for mode in ("binary","structure","hybrid")}
        for index,(_,_,char) in enumerate(sample["glyphs"]):
            source = normalize(imread(OUTPUT/"crops"/f"q{sample['qid']}-g{index}-input.png"))
            for font in catalog:
                values,_ = score_pair(source,templates[font["id"],char])
                for mode,value in values.items():
                    scores[mode][font["id"]].append(value)
        variants = {}
        for mode,values in scores.items():
            ranking = sorted([{"font":f["postscript"],"group":f["group"],
                "distance":float(np.mean(values[f["id"]]))} for f in catalog],key=lambda r:r["distance"])
            groups = []
            for r in ranking:
                if r["group"] not in [g["group"] for g in groups]:
                    groups.append(r)
            variants[mode] = groups
        rows.append({"qid":sample["qid"],"expected":sample["group"],"variants":variants})
    totals = {mode:sum(r["variants"][mode][0]["group"]==r["expected"] for r in rows) for mode in scores}
    result = {"status":"first_question_level_holdout_not_source_disjoint", "freeze_sha256":digest(OUTPUT/"freeze.json"),
              "correct":totals,"questions":rows}
    (OUTPUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(totals)
    for r in rows:
        print(r["qid"],r["expected"],{k:v[0]["group"] for k,v in r["variants"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","evaluate"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else evaluate()
