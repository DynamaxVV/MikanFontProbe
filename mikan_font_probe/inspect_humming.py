"""Generate same-character source/template visual evidence, no rule changes."""
import json

import cv2
import numpy as np

from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import ROOT, imread, imwrite
from mikan_font_probe.paths import resolve_repo_path

OUT = ROOT/"data/humming-v4"
BASE = ROOT/"data/category-expansion-v3"


def main():
    rows = json.loads((BASE/"pools.json").read_text())["samples"]
    fonts = json.loads((BASE/"templates/catalog.json").read_text())["fonts"]
    families = ("humming", "folk", "skip")
    candidates = [f for key in families for f in fonts if f["family_group"]["key"] == key]
    for row in rows:
        if row["expected"] != "轻吟体":
            continue
        glyphs = row["glyphs"]
        sheet = np.full((len(glyphs)*110, (len(candidates)+1)*140, 3), 255, np.uint8)
        for n, glyph in enumerate(glyphs):
            images = [(f"Source U+{ord(glyph['character']):04X}", normalize(imread(resolve_repo_path(glyph["path"]))))]
            for font in candidates:
                image = imread(BASE/"templates"/font["templates"][glyph["character"]])
                images.append((font["postscript"], normalize(image)))
            for index, (label, ink) in enumerate(images):
                x, y = index*140, n*110
                image = cv2.resize(255-ink, (80,80), interpolation=cv2.INTER_NEAREST)
                sheet[y+24:y+104, x+20:x+100] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.putText(sheet,label,(x+2,y+15),cv2.FONT_HERSHEY_SIMPLEX,.29,(0,0,0),1)
        imwrite(OUT/"visual"/f"q{row['qid']}.png",sheet)


if __name__ == "__main__":
    main()
