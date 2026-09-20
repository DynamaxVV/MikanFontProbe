"""Render known upright glyphs with existing Pillow; no global font install."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

from PIL import Image, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parent


def font_metadata(path, characters):
    """Read SFNT names and Unicode glyph presence without font fallback."""
    data = path.read_bytes()
    count = struct.unpack_from(">H", data, 4)[0]
    tables = {}
    for index in range(count):
        tag, _, offset, size = struct.unpack_from(">4sIII", data, 12+16*index)
        if offset + size > len(data):
            raise ValueError(f"Invalid SFNT table: {path}")
        tables[tag.decode()] = offset
    base = tables["name"]
    _, count, strings = struct.unpack_from(">HHH", data, base)
    names = {}
    for index in range(count):
        platform, _, language, name_id, length, offset = struct.unpack_from(">HHHHHH", data, base+6+12*index)
        if name_id not in (1, 2, 6) or platform not in (0, 3):
            continue
        value = data[base+strings+offset:base+strings+offset+length].decode("utf-16-be")
        if name_id not in names or language == 1033:
            names[name_id] = value
    base = tables["cmap"]
    present = set()
    for index in range(struct.unpack_from(">H", data, base+2)[0]):
        platform, encoding, relative = struct.unpack_from(">HHI", data, base+4+8*index)
        if not (platform == 0 or platform == 3 and encoding in (1, 10)):
            continue
        sub = base + relative
        fmt = struct.unpack_from(">H", data, sub)[0]
        for char in characters:
            cp = ord(char)
            if fmt == 12:
                for j in range(struct.unpack_from(">I", data, sub+12)[0]):
                    start, end, glyph = struct.unpack_from(">III", data, sub+16+12*j)
                    if start <= cp <= end and glyph+cp-start != 0:
                        present.add(char)
                        break
            elif fmt == 4 and cp < 65535:
                n = struct.unpack_from(">H", data, sub+6)[0]//2
                ends, starts = sub+14, sub+16+2*n
                deltas, offsets = starts+2*n, starts+4*n
                for j in range(n):
                    start = struct.unpack_from(">H", data, starts+2*j)[0]
                    end = struct.unpack_from(">H", data, ends+2*j)[0]
                    if not start <= cp <= end:
                        continue
                    delta = struct.unpack_from(">h", data, deltas+2*j)[0]
                    relative = struct.unpack_from(">H", data, offsets+2*j)[0]
                    glyph = (cp+delta) & 65535
                    if relative:
                        glyph = struct.unpack_from(">H", data, offsets+2*j+relative+2*(cp-start))[0]
                        glyph = (glyph+delta) & 65535 if glyph else 0
                    if glyph:
                        present.add(char)
                    break
    return {"family": names.get(1), "style": names.get(2), "postscript": names.get(6),
            "weight": struct.unpack_from(">H", data, tables["OS/2"]+4)[0],
            "sha256": hashlib.sha256(data).hexdigest(), "characters": sorted(present),
            "vertical_tables": [t for t in ("vhea", "vmtx", "GSUB") if t in tables]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=ROOT/"data/baseline_samples.json")
    args = parser.parse_args()
    config = json.loads(args.samples.read_text())
    characters = sorted({g[2] for s in config["samples"] for g in s["glyphs"]})
    catalog = []
    args.output.mkdir(parents=True, exist_ok=True)
    for group in config["policy"]["primary_labels"]:
        for path in sorted((ROOT/"font_sample"/group).iterdir()):
            if path.suffix.lower() not in (".ttf", ".otf"):
                continue
            meta = font_metadata(path, characters)
            meta.update(id=meta["sha256"][:16], group=group, path=str(path.relative_to(ROOT)))
            font = ImageFont.truetype(str(path), 128)
            meta["templates"] = {}
            for char in meta["characters"]:
                left, top, right, bottom = font.getbbox(char)
                image = Image.new("L", (right-left+16, bottom-top+16), 255)
                ImageDraw.Draw(image).text((8-left, 8-top), char, font=font, fill=0)
                filename = f"{meta['id']}-{ord(char):04x}.png"
                image.save(args.output/filename)
                meta["templates"][char] = filename
            catalog.append(meta)
    result = {"rendering": {"engine": "Pillow FreeType", "freetype": features.version("freetype2"),
              "em_pixels": 128, "layout": "isolated upright glyph; no vertical punctuation shaping"},
              "fonts": catalog}
    (args.output/"catalog.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"fonts": len(catalog), "unique_characters": len(characters),
                      "templates": sum(len(f["templates"]) for f in catalog)}))


if __name__ == "__main__":
    main()
