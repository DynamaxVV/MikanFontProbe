"""Expand pre-reviewed column transcripts to immutable known-character pools."""

import hashlib
import json

import cv2
import numpy as np

from mikan_font_probe.process_regions import ROOT, imread, imwrite
from mikan_font_probe.rank_font_candidates import crop_quality
from mikan_font_probe.refine_font_inputs import refine_box

OUTPUT = ROOT/"data/active-selection-v1"


def main():
    if (OUTPUT/"results.json").exists():
        raise ValueError("Selection evaluated: use a new version before changing pools")
    config = json.loads((ROOT/"data/active_selection_pools_v1.json").read_text())
    source = ROOT/"data/processed"/config["source_version"]
    regions = {r["question_id"]:r for r in json.loads((source/"results.json").read_text())}
    rows = []
    for sample in config["samples"]:
        qid = sample["qid"]
        original = imread(source/f"q{qid}-source.png")
        glyphs = []
        for col,transcript in sample["columns"].items():
            slots = regions[qid]["columns"][int(col)]["glyphs"]
            if len(slots)!=len(transcript):
                raise ValueError(f"q{qid} column {col}: {len(slots)} boxes, {len(transcript)} characters")
            for index,(region,char) in enumerate(zip(slots,transcript)):
                if char == "_" or region["cut_crosses_ink"] or [int(col),index] in sample.get("exclude",[]):
                    continue
                x,y,w,h = refine_box(original,region["source_bbox"])
                crop = original[y:y+h,x:x+w]
                sid = f"q{qid}-c{col}-g{index}"
                path = OUTPUT/"crops"/f"{sid}.png"
                imwrite(path,crop)
                gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                _,ink = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                ys,xs = np.nonzero(ink)
                resolution = int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)) if len(xs) else 0
                glyphs.append({"id":sid,"character":char,"bbox":[x,y,w,h],
                    "path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                    "quality":crop_quality(crop),"resolution":resolution,
                    "script":"kana" if 0x3040<=ord(char)<=0x30ff else "kanji"})
        sheet = np.full((((len(glyphs)+7)//8)*104,8*100,3),255,np.uint8)
        for i,g in enumerate(glyphs):
            crop = imread(ROOT/g["path"])
            scale = 70/max(crop.shape[:2])
            thumb = cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)),interpolation=cv2.INTER_NEAREST)
            x,y = i%8*100,i//8*104+23
            sheet[y:y+thumb.shape[0],x:x+thumb.shape[1]] = thumb
            cv2.putText(sheet,f"{i} U+{ord(g['character']):04X}",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,0,0),1)
        imwrite(OUTPUT/"review"/f"q{qid}.png",sheet)
        rows.append({"qid":qid,"expected":sample["expected"],"glyphs":glyphs})
    document = {"status":"known-character development pools, manually reviewed","samples":rows}
    (OUTPUT/"pools.json").write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n")
    chars = sorted({g["character"] for r in rows for g in r["glyphs"]})
    (OUTPUT/"characters.txt").write_text("".join(chars)+"\n")
    print({"questions":len(rows),"glyphs":sum(len(r["glyphs"]) for r in rows),"characters":len(chars)})


if __name__ == "__main__":
    main()
