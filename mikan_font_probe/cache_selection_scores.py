"""Cache fixed matcher outputs; query policy must not inspect unread columns."""

import hashlib
import json

import numpy as np

from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.structure_font_baseline import score_pair

OUTPUT = ROOT/"data/active-selection-v1"


def main():
    pools = json.loads((OUTPUT/"pools.json").read_text())
    directory = OUTPUT/"templates"
    fonts = json.loads((directory/"catalog.json").read_text())["fonts"]
    required = {g["character"] for s in pools["samples"] for g in s["glyphs"]}
    for font in fonts:
        if not required <= set(font["templates"]):
            raise ValueError(f"Missing glyphs: {font['postscript']}")
        if hashlib.sha256((ROOT/font["path"]).read_bytes()).hexdigest()!=font["sha256"]:
            raise ValueError("Font changed")
    templates = {(f["id"],c):normalize(imread(directory/f["templates"][c])) for f in fonts for c in required}
    matrices = {}
    for sample in pools["samples"]:
        matrix = np.zeros((len(fonts),len(sample["glyphs"])),np.float64)
        for index,glyph in enumerate(sample["glyphs"]):
            path = resolve_repo_path(glyph["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=glyph["sha256"]:
                raise ValueError("Crop changed")
            source = normalize(imread(path))
            for fi,font in enumerate(fonts):
                values,_ = score_pair(source,templates[font["id"],glyph["character"]])
                matrix[fi,index] = values["hybrid"]
        matrices[f"q{sample['qid']}"] = matrix
        print(f"q{sample['qid']}: {matrix.shape}",flush=True)
    np.savez_compressed(OUTPUT/"match_scores.npz",**matrices)
    sources = [OUTPUT/"pools.json",directory/"catalog.json",resolve_repo_path("match_font_baseline.py"),resolve_repo_path("structure_font_baseline.py")]
    freeze = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUTPUT/"match_freeze.json").write_text(json.dumps(freeze,indent=2)+"\n")


if __name__ == "__main__":
    main()
