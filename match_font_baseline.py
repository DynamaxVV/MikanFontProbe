"""Known-character, aspect-preserving font matching development baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from process_regions import ROOT, imread, imwrite


def normalize(image, size=64):
    """Normalize dark ink isotropically; do not repair or thicken strokes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Blank glyph")
    crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    ratio = (size-12)/max(crop.shape)
    w, h = max(1, round(crop.shape[1]*ratio)), max(1, round(crop.shape[0]*ratio))
    resized = cv2.resize(crop, (w,h), interpolation=cv2.INTER_AREA)
    result = np.zeros((size,size), np.uint8)
    x, y = (size-w)//2, (size-h)//2
    result[y:y+h,x:x+w] = resized
    return result


def match(source, template):
    """Search bounded uniform scale and translation; return distance, not probability."""
    a = source >= 128
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    best = (float("inf"), None, None)
    for scale in (.96, 1., 1.04):
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                transform = np.array([[scale,0,32*(1-scale)+dx],
                                      [0,scale,32*(1-scale)+dy]], np.float32)
                aligned = cv2.warpAffine(template, transform, (64,64))
                b = aligned >= 128
                if not b.any():
                    continue
                db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
                dice_loss = 1-2*np.count_nonzero(a & b)/(a.sum()+b.sum())
                chamfer = (float(da[b].mean())+float(db[a].mean()))/2
                loss = .65*dice_loss + .35*min(chamfer/4, 1)
                if loss < best[0]:
                    best = (float(loss), aligned, {"scale": scale, "dx": dx, "dy": dy})
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/baseline-v1")
    args = parser.parse_args()
    output = args.output
    catalog = json.loads((output/"templates/catalog.json").read_text())
    fonts = catalog["fonts"]
    config = json.loads((ROOT/"data/baseline_samples.json").read_text())
    source = ROOT/"data/processed"/config["source_version"]
    regions = {a["question_id"]: a for a in json.loads((source/"results.json").read_text())}
    templates = {(font["id"], char): normalize(imread(output/"templates"/file))
                 for font in fonts for char,file in font["templates"].items()}
    report = []
    for sample in config["samples"]:
        qid = sample["qid"]
        per_font = {f["id"]: [] for f in fonts}
        glyph_results = []
        sheet = np.full((len(sample["glyphs"])*128, 920, 3), 255, np.uint8)
        for row, (col,index,char) in enumerate(sample["glyphs"]):
            region = regions[qid]["columns"][col]["glyphs"][index]
            if region["cut_crosses_ink"]:
                raise ValueError(f"Review risky sample before matching: q{qid} {col}/{index}")
            crop = imread(source/region["source_file"])
            # Stored cells have a two-pixel context margin. Keep original intact;
            # use its inner cell for the diagnostic matching mask.
            normalized = normalize(crop[2:-2,2:-2])
            sample_id = f"q{qid}-c{col}-g{index}"
            imwrite(output/"evidence"/f"{sample_id}-source.png", crop)
            imwrite(output/"evidence"/f"{sample_id}-normalized.png", 255-normalized)
            ranking = []
            aligned_images = {}
            for font in fonts:
                key = (font["id"],char)
                if key not in templates:
                    continue
                loss, aligned, transform = match(normalized, templates[key])
                per_font[font["id"]].append(loss)
                aligned_images[font["id"]] = aligned
                ranking.append({"font_id": font["id"], "font": font["postscript"],
                                "group": font["group"], "distance": round(loss,6),
                                "alignment": transform})
            ranking.sort(key=lambda a: a["distance"])
            glyph_results.append({"id": sample_id, "character": char,
                                  "script": "kana" if 0x3040 <= ord(char) <= 0x30ff else "kanji",
                                  "source_bbox": region["source_bbox"],
                                  "source_file": str((source/region["source_file"]).relative_to(ROOT)),
                                  "source_sha256": hashlib.sha256((source/region["source_file"]).read_bytes()).hexdigest(),
                                  "ranking": ranking})
            panels = [("source " + hex(ord(char)), normalized)]
            representatives = []
            for candidate in ranking:
                if candidate["group"] not in [r["group"] for r in representatives]:
                    representatives.append(candidate)
            panels += [(r["font"]+" %.3f" % r["distance"], aligned_images[r["font_id"]]) for r in representatives]
            for panel, (label, ink) in enumerate(panels):
                x, y = panel*230, row*128
                large = cv2.resize(255-ink, (96,96), interpolation=cv2.INTER_NEAREST)
                sheet[y+24:y+120,x+60:x+156] = cv2.cvtColor(large, cv2.COLOR_GRAY2BGR)
                cv2.putText(sheet, label[:37], (x+3,y+16), cv2.FONT_HERSHEY_SIMPLEX, .32, (0,0,0), 1)
        ranked_fonts = sorted([{"font_id": f["id"], "font": f["postscript"], "group": f["group"],
                                "mean_distance": round(float(np.mean(per_font[f["id"]])),6)}
                               for f in fonts if len(per_font[f["id"]]) == len(sample["glyphs"])],
                              key=lambda a:a["mean_distance"])
        groups = []
        for font in ranked_fonts:
            if font["group"] not in [g["group"] for g in groups]:
                groups.append(font)
        imwrite(output/"evidence"/f"q{qid}-comparison.png", sheet)
        report.append({"qid": qid, "expected_group": sample["group"], "group_ranking": groups,
                       "font_ranking": ranked_fonts, "glyphs": glyph_results})
    correct = sum(r["group_ranking"][0]["group"] == r["expected_group"] for r in report)
    result = {"status": "development_baseline_not_blind_accuracy", "probabilities": False,
              "opencv": cv2.__version__,
              "group_aggregation": "mean distance across characters for each font, then minimum per group",
              "selected_questions": len(report), "top1_correct": correct, "results": report}
    (output/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"questions":len(report),"top1_correct":correct,"output":str(output)}))
    for r in report:
        print(r["qid"], r["expected_group"], [(g["group"],g["font"],g["mean_distance"]) for g in r["group_ranking"]])


if __name__ == "__main__":
    main()
