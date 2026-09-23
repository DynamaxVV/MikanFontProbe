"""Analyze one already-cropped manga text region with the existing matcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from mikan_font_probe.process_regions import ROOT, extract, imread, imwrite
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.rank_font_candidates import crop_quality
from mikan_font_probe.rank_font_guarded import rank_sample
from mikan_font_probe.segment_text import boxes_from_mask
from mikan_font_probe.typographic_regions import fit_grid


DEFAULT_TEMPLATES = ROOT / "data/default-gallery-v1/templates"


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    return "".join(character for character in text if not character.isspace())


def parse_exclude(value: str) -> list[int]:
    if not value.strip():
        return []
    result = sorted({int(part) for part in value.split(",")})
    if result and result[0] < 0:
        raise ValueError("Excluded candidate indices must be non-negative")
    return result


def next_output_directory(image_path: Path) -> Path:
    root = ROOT / "output/region-analysis"
    stem = image_path.stem
    index = 1
    while True:
        candidate = root / f"{stem}-{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def padded_box(box: list[int], shape: tuple[int, ...]) -> list[int]:
    x, y, width, height = box
    pad = max(2, round(min(width, height) * .08))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(shape[1], x + width + pad)
    y1 = min(shape[0], y + height + pad)
    return [x0, y0, x1 - x0, y1 - y0]


def candidate_boxes_from_mask(mask: np.ndarray, layout: str) -> list[list[int]]:
    """Split each detected text line with the existing soft pitch grid."""
    lines, _ = boxes_from_mask(mask, layout)
    boxes = []
    for x, y, width, height in lines:
        if layout == "vertical":
            strip = mask[:, x:x + width]
            cells, _ = fit_grid(strip, width)
            for top, bottom, _ in cells:
                ys, xs = np.nonzero(strip[top:bottom])
                if len(xs):
                    boxes.append([x + int(xs.min()), top + int(ys.min()),
                                  int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)])
        else:
            strip = mask[y:y + height, :]
            cells, _ = fit_grid(strip.T, height)
            for left, right, _ in cells:
                ys, xs = np.nonzero(strip[:, left:right])
                if len(xs):
                    boxes.append([left + int(xs.min()), y + int(ys.min()),
                                  int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)])
    return boxes


def ink_resolution(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(ink)
    return int(max(np.ptp(xs) + 1, np.ptp(ys) + 1)) if len(xs) else 0


def script_for(character: str) -> str:
    return "kana" if 0x3040 <= ord(character) <= 0x30ff else "kanji"


def stored_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def save_segmentation_preview(image: np.ndarray, boxes: list[list[int]], excluded: set[int],
                              output_path: Path) -> None:
    preview = image.copy()
    for index, (x, y, width, height) in enumerate(boxes):
        color = (110, 110, 110) if index in excluded else (0, 150, 255)
        cv2.rectangle(preview, (x, y), (x + width, y + height), color, 1)
        cv2.putText(preview, str(index), (x, max(11, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
    imwrite(output_path, preview)


def tile(image: np.ndarray, size: int = 72) -> np.ndarray:
    canvas = np.full((size, size, 3), 255, np.uint8)
    scale = min((size - 8) / image.shape[1], (size - 8) / image.shape[0])
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    x, y = (size - width) // 2, (size - height) // 2
    canvas[y:y + height, x:x + width] = resized
    return canvas


def save_match_preview(glyphs: list[dict], result: dict, template_directory: Path,
                       output_path: Path) -> None:
    selected = result.get("selected_indices", [])
    if not selected:
        return
    fonts = json.loads((template_directory / "catalog.json").read_text())["fonts"]
    by_face = {font["postscript"]: font for font in fonts}
    rows = []
    source_tiles = [tile(imread(resolve_repo_path(glyphs[index]["path"]))) for index in selected]
    source = np.concatenate(source_tiles, axis=1)
    rows.append(("matched input", source))
    for rank, candidate in enumerate(result.get("display_top3", []), 1):
        font = by_face.get(candidate.get("representative_face"))
        if not font:
            continue
        rendered = []
        for index in selected:
            character = glyphs[index]["character"]
            filename = font["templates"].get(character)
            if filename:
                rendered.append(tile(imread(template_directory / filename)))
        if len(rendered) == len(selected):
            rows.append((f"{rank} {font['postscript']}", np.concatenate(rendered, axis=1)))
    label_width = 210
    canvas = np.full((len(rows) * 92, label_width + source.shape[1], 3), 255, np.uint8)
    for row_index, (label, strip) in enumerate(rows):
        top = row_index * 92
        cv2.putText(canvas, label[:31], (6, top + 42), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (20, 20, 20), 1, cv2.LINE_AA)
        canvas[top:top + strip.shape[0], label_width:label_width + strip.shape[1]] = strip
    imwrite(output_path, canvas)


def analyze(image_path: Path, text: str | None = None, *, layout: str = "auto",
            exclude: tuple[int, ...] | list[int] = (), budget: int = 5,
            output_directory: Path | None = None,
            template_directory: Path = DEFAULT_TEMPLATES) -> dict:
    """Segment a cropped region, align known text, and run the existing matcher."""
    image_path = Path(image_path)
    template_directory = Path(template_directory)
    output_directory = Path(output_directory) if output_directory else next_output_directory(image_path)
    output_directory.mkdir(parents=True, exist_ok=False)
    image = imread(image_path)
    normalized_text = clean_text(text)
    chosen_layout = ("vertical" if image.shape[0] >= image.shape[1] else "horizontal") \
        if layout == "auto" else layout
    if chosen_layout not in {"vertical", "horizontal"}:
        raise ValueError("layout must be auto, vertical, or horizontal")
    if not 1 <= budget <= 5:
        raise ValueError("budget must be between 1 and 5")

    mask, mask_meta = extract(image)
    boxes = candidate_boxes_from_mask(mask, chosen_layout)
    excluded = set(exclude)
    if any(index < 0 or index >= len(boxes) for index in excluded):
        raise ValueError("Excluded candidate index is outside the detected range")
    active = [(index, box) for index, box in enumerate(boxes) if index not in excluded]
    imwrite(output_directory / "source.png", image)
    imwrite(output_directory / "mask.png", mask)
    save_segmentation_preview(image, boxes, excluded, output_directory / "segments.png")

    report = {
        "state": "text_required_for_matching" if normalized_text is None else "character_count_mismatch",
        "source": stored_path(image_path),
        "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "layout": chosen_layout,
        "original_polarity": "light" if float(np.median(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))) < 100 else "dark",
        "text": normalized_text,
        "text_character_count": len(normalized_text) if normalized_text is not None else None,
        "candidate_count": len(boxes),
        "active_candidate_count": len(active),
        "excluded_candidates": sorted(excluded),
        "candidates": [{"index": index, "bbox": box, "excluded": index in excluded}
                       for index, box in enumerate(boxes)],
        "mask": mask_meta,
        "artifacts": {"source": "source.png", "mask": "mask.png", "segments": "segments.png"},
        "scope": "known_or_user_corrected_text; local_font_library_only",
    }
    if normalized_text is None or len(normalized_text) != len(active):
        (output_directory / "analysis.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report

    for (candidate_index, _), character in zip(active, normalized_text):
        report["candidates"][candidate_index]["character"] = character

    fonts = json.loads((template_directory / "catalog.json").read_text())["fonts"]
    common_characters = set.intersection(*(set(font["templates"]) for font in fonts))
    light = report["original_polarity"] == "light"
    glyphs = []
    unsupported = []
    for (candidate_index, box), character in zip(active, normalized_text):
        x, y, width, height = padded_box(box, image.shape)
        source_crop = image[y:y + height, x:x + width]
        input_crop = 255 - source_crop if light else source_crop.copy()
        source_path = output_directory / "crops" / f"candidate-{candidate_index:02d}-source.png"
        input_path = output_directory / "crops" / f"candidate-{candidate_index:02d}-input.png"
        imwrite(source_path, source_crop)
        imwrite(input_path, input_crop)
        record = {
            "id": f"candidate-{candidate_index:02d}",
            "candidate_index": candidate_index,
            "character": character,
            "bbox": [x, y, width, height],
            "source_path": stored_path(source_path),
            "path": stored_path(input_path),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "quality": crop_quality(input_crop),
            "resolution": ink_resolution(input_crop),
            "script": script_for(character),
            "polarity": "dark",
            "input_transform": "inverted_from_light_text" if light else "none",
        }
        if character in common_characters:
            glyphs.append(record)
        else:
            unsupported.append({"candidate_index": candidate_index, "character": character})

    report["unsupported_characters"] = unsupported
    report["glyphs"] = glyphs
    if not glyphs:
        report["state"] = "no_supported_characters"
        (output_directory / "analysis.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report

    result = rank_sample({"glyphs": glyphs}, template_directory, budget=budget)
    report["state"] = "matched" if result.get("ranking") else "no_usable_glyphs"
    report["match"] = result
    report["selected_candidate_indices"] = [glyphs[index]["candidate_index"]
                                             for index in result.get("selected_indices", [])]
    if result.get("selected_indices"):
        report["artifacts"]["comparison"] = "comparison.png"
        save_match_preview(glyphs, result, template_directory, output_directory / "comparison.png")
    (output_directory / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one cropped manga text region")
    parser.add_argument("image", type=Path)
    parser.add_argument("--text", help="Known or corrected text; whitespace is ignored")
    parser.add_argument("--layout", choices=("auto", "vertical", "horizontal"), default="auto")
    parser.add_argument("--exclude", default="", help="Comma-separated candidate indices from segments.png")
    parser.add_argument("--budget", type=int, default=5, help="Use at most 1-5 distinct usable glyphs")
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or next_output_directory(args.image)
    result = analyze(args.image, args.text, layout=args.layout,
                     exclude=parse_exclude(args.exclude), budget=args.budget,
                     output_directory=output, template_directory=args.templates)
    print(json.dumps({"state": result["state"], "output": str(output),
                      "candidate_count": result["candidate_count"],
                      "active_candidate_count": result["active_candidate_count"],
                      "display_top3": result.get("match", {}).get("display_top3", [])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
