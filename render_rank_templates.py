"""Render all local candidate families without modifying frozen baselines."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

from build_font_templates import ROOT, font_metadata
from font_family_policy import family_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--characters-file",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"data/ranked-v1/templates")
    args = parser.parse_args()
    characters = set()
    for filename in ("baseline_samples.json", "holdout_samples_v1.json"):
        config = json.loads((ROOT/"data"/filename).read_text())
        characters.update(g[2] for sample in config["samples"] for g in sample["glyphs"])
    if args.characters_file:
        characters.update(c for c in args.characters_file.read_text() if not c.isspace())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    fonts = []
    for path in sorted((ROOT/"font_sample").rglob("*")):
        if path.suffix.lower() not in (".otf", ".ttf"):
            continue
        font = font_metadata(path, sorted(characters))
        font.update(id=font["sha256"][:16], path=str(path.relative_to(ROOT)),
                    family_group=family_policy(font["postscript"], path.parent.name), templates={})
        renderer = ImageFont.truetype(str(path), 128)
        for char in font["characters"]:
            left,top,right,bottom = renderer.getbbox(char)
            image = Image.new("L", (right-left+16,bottom-top+16), 255)
            ImageDraw.Draw(image).text((8-left,8-top),char,font=renderer,fill=0)
            name = f"{font['id']}-{ord(char):04x}.png"
            image.save(output/name)
            font["templates"][char] = name
        fonts.append(font)
    result = {"fonts": fonts, "rendering": {"engine": "Pillow/FreeType", "version": features.version("freetype2"), "em":128}}
    (output/"catalog.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print({"faces":len(fonts),"families":len({f['family_group']['key'] for f in fonts}),"characters":len(characters)})


if __name__ == "__main__":
    main()
