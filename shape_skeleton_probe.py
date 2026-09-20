"""Width-neutral geometry probe using topology-preserving parallel thinning."""
import json
import cv2
import numpy as np

from match_font_baseline import normalize
from process_regions import ROOT, imread


def thin(ink):
    mask = (ink >= 128).astype(np.uint8)
    for _ in range(64):
        changed = False
        for phase in (0, 1):
            p = np.pad(mask, 1)
            n = [p[:-2,1:-1], p[:-2,2:], p[1:-1,2:], p[2:,2:],
                 p[2:,1:-1], p[2:,:-2], p[1:-1,:-2], p[:-2,:-2]]
            count = sum(n)
            transitions = sum((n[i] == 0) & (n[(i+1)%8] == 1) for i in range(8))
            if phase == 0:
                constraint = (n[0]*n[2]*n[4] == 0) & (n[2]*n[4]*n[6] == 0)
            else:
                constraint = (n[0]*n[2]*n[6] == 0) & (n[0]*n[4]*n[6] == 0)
            remove = (mask == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & constraint
            changed |= bool(remove.any())
            mask[remove] = 0
        if not changed:
            break
    return mask


def shape_loss(a, b):
    da = cv2.distanceTransform(1-a, cv2.DIST_L2, 3)
    db = cv2.distanceTransform(1-b, cv2.DIST_L2, 3)
    return (float(da[b > 0].mean())+float(db[a > 0].mean()))/8


def main():
    base = ROOT/"data/category-expansion-v3"
    fonts = json.loads((base/"templates/catalog.json").read_text())["fonts"]
    rows = json.loads((base/"pools.json").read_text())["samples"]
    output = []
    for row in rows:
        if row["expected"] != "轻吟体":
            continue
        sources = [thin(normalize(imread(ROOT/g["path"]))) for g in row["glyphs"]]
        ranked = []
        for font in fonts:
            losses = []
            for source, glyph in zip(sources, row["glyphs"]):
                target = thin(normalize(imread(base/"templates"/font["templates"][glyph["character"]])))
                losses.append(shape_loss(source, target))
            ranked.append({"font": font["postscript"], "mean": float(np.mean(losses))})
        ranked.sort(key=lambda row: row["mean"])
        output.append({"qid": row["qid"], "ranking": ranked})
        print(row["qid"], ranked[:5], flush=True)
    (ROOT/"data/humming-v4/skeleton-probe.json").write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
