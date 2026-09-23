"""Build the current 26-face template gallery without changing frozen galleries."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

from mikan_font_probe.font_family_policy import family_policy
from mikan_font_probe.font_style_policy import font_style_policy
from mikan_font_probe.paths import ROOT


DEFAULT_ASSETS = ROOT / "data/font-assets-v2/manifest.json"
DEFAULT_CHARACTERS = ROOT / "data/default-gallery-v1/characters.txt"
DEFAULT_OUTPUT = ROOT / "data/default-gallery-v1/templates"


def family_group(font: dict, path: Path) -> dict:
    """Keep legacy answers where defined; use current policy for newer faces."""
    try:
        return family_policy(font["postscript"], path.parent.name)
    except ValueError:
        policy = font_style_policy(font["postscript"], path.parent.name, font)
        family = policy["source_family"]
        replacements = policy["replacement_candidates"]
        # The legacy field requires a display label. An unmapped replacement
        # uses the source-family name and must not be treated as a Chinese font.
        return {**family, "answer": replacements[0]["name"] if replacements else family["name"]}


def build(assets_path: Path = DEFAULT_ASSETS, characters_path: Path = DEFAULT_CHARACTERS,
          output: Path = DEFAULT_OUTPUT) -> dict:
    # Pillow is needed only to regenerate raster assets, not to inspect a catalog.
    from PIL import Image, ImageDraw, ImageFont, features
    from mikan_font_probe.build_font_templates import font_metadata

    if (output / "catalog.json").exists():
        raise FileExistsError(f"gallery already complete: {output}")
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    characters = sorted({char for char in characters_path.read_text(encoding="utf-8")
                         if not char.isspace()})
    if not characters:
        raise ValueError("empty gallery character list")
    rows = assets["assets"]
    if len(rows) != 26 or assets["summary"]["confirmed_faces"] != 26:
        raise ValueError("default gallery requires the confirmed 26-face manifest")
    output.mkdir(parents=True, exist_ok=True)
    fonts = []
    rendered = reused = 0
    for asset in rows:
        path = ROOT / asset["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"font hash differs from manifest: {path}")
        font = font_metadata(path, characters)
        if font["sha256"] != asset["sha256"] or font["postscript"] != asset["sfnt"]["postscript"]:
            raise ValueError(f"font identity differs from manifest: {path}")
        font.update(id=font["sha256"][:16], path=asset["path"],
                    family_group=family_group(font, path), templates={})
        renderer = ImageFont.truetype(str(path), 128)
        for char in font["characters"]:
            left, top, right, bottom = renderer.getbbox(char)
            image = Image.new("L", (right - left + 16, bottom - top + 16), 255)
            ImageDraw.Draw(image).text((8 - left, 8 - top), char, font=renderer, fill=0)
            name = f"{font['id']}-{ord(char):04x}.png"
            destination = output / name
            payload = BytesIO()
            image.save(payload, format="PNG")
            if destination.exists():
                if destination.read_bytes() != payload.getvalue():
                    raise ValueError(f"existing template differs: {destination}")
                reused += 1
            else:
                destination.write_bytes(payload.getvalue())
                rendered += 1
            font["templates"][char] = name
        fonts.append(font)
    catalog = {
        "schema": "mikan-default-gallery/v1",
        "font_assets_inventory_sha256": assets["inventory_sha256"],
        "characters": "".join(characters),
        "fonts": fonts,
        "rendering": {"engine": "Pillow/FreeType", "version": features.version("freetype2"), "em": 128},
    }
    (output / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"faces": len(fonts), "source_families": len({asset["family_mapping"]["family_id"] for asset in rows}),
            "characters": len(characters), "rendered": rendered, "reused": reused,
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--characters-file", type=Path, default=DEFAULT_CHARACTERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.assets, args.characters_file, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
