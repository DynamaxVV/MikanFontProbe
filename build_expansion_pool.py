"""Build original-pixel character pools from bounded visual annotations."""
import argparse
import json
from pathlib import Path

import prepare_fresh_holdout as base


def build(path):
    document = json.loads(Path(path).read_text())
    output = Path(path).parent
    base.OUT = output
    if (output/"pools.json").exists():
        raise ValueError("Already built; preserve inputs after scoring")
    rows = []
    sources = {e["question_id"]: base.ROOT/"data"/e["image_file"]
               for e in json.loads((base.ROOT/"data/manifest.json").read_text())["entries"]}
    for sample in document["samples"]:
        qid = sample["qid"]
        source = sources[qid]
        original = base.read_image(source)
        polarity = sample.get("polarity", "dark")
        cells = list(sample.get("glyphs", []))
        for column in sample.get("columns", []):
            text, edges = column["text"], column["y_edges"]
            assert len(edges) == len(text)+1
            x0, x1 = column["x_range"]
            for index, char in enumerate(text):
                if char != "_":
                    cells.append({"character": char, "bbox": [x0, edges[index],
                                  x1-x0, edges[index+1]-edges[index]]})
        glyphs = []
        for index, cell in enumerate(cells):
            bbox, crop = base.glyph_crop(original, cell["bbox"], polarity)
            target = output/"crops"/f"q{qid}-g{index}.png"
            base.write_image(target, crop)
            quality, resolution = base.crop_quality(crop, polarity)
            char = cell["character"]
            glyphs.append({"id": f"q{qid}-g{index}", "character": char,
                "bbox": bbox, "cell_bbox": cell["bbox"],
                "path": str(target.relative_to(base.ROOT)),
                "sha256": base.digest(target.read_bytes()), "quality": quality,
                "resolution": resolution, "polarity": polarity,
                "script": "kana" if 0x3040 <= ord(char) <= 0x30ff else "kanji"})
        if not glyphs:
            raise ValueError(f"q{qid}: no reviewed glyph; report blocker explicitly")
        base.reviews(qid, original, glyphs)
        rows.append({**{k: v for k, v in sample.items() if k not in ("glyphs", "columns")},
                     "glyphs": glyphs, "source_path": str(source.relative_to(base.ROOT)),
                     "source_sha256": base.digest(source.read_bytes())})
    base.write_json(output/"pools.json", {"samples": rows, "status": "visually reviewed known characters"})
    chars = sorted({g["character"] for row in rows for g in row["glyphs"]})
    (output/"characters.txt").write_text("".join(chars)+"\n")
    print([(s["qid"], len(s["glyphs"])) for s in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    build(parser.parse_args().annotations)
