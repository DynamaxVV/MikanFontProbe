"""Conservative OpenCV text-mask prototype for already cropped manga questions.

The output is an evidence overlay and candidate mask, not a guaranteed glyph mask.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent


def imread(path: pathlib.Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")
    return image


def imwrite(path: pathlib.Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(str(path))


def extract(image: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, dict]:
    height, width = image.shape[:2]
    if roi:
        x, y, w, h = roi
        x, y = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
    else:
        x, y, x2, y2 = 0, 0, width, height
    crop = image[y:y2, x:x2]
    if not crop.size:
        raise ValueError("Empty ROI")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # White lettering on dark panels needs the opposite ink polarity.
    bright_ink = float(np.median(gray)) < 100
    threshold_kind = cv2.THRESH_BINARY if bright_ink else cv2.THRESH_BINARY_INV
    otsu_level, dark = cv2.threshold(blur, 0, 255, threshold_kind + cv2.THRESH_OTSU)
    if bright_ink:
        # In screened dark panels the grey halftone is much dimmer than letters.
        _, dark = cv2.threshold(blur, max(200, otsu_level), 255, cv2.THRESH_BINARY)
    # A second, local threshold helps uneven paper/backgrounds; intersecting
    # it with the global threshold avoids admitting most pale textures.
    block = min(31, min(crop.shape[:2]) | 1)
    block = max(3, block)
    local = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_kind, block, -8 if bright_ink else 8)
    ink = cv2.bitwise_and(dark, local)

    ch, cw = crop.shape[:2]
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, ch // 6))))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, cw // 6), 1)))
    line_mask = cv2.bitwise_or(vertical, horizontal)
    ink = cv2.subtract(ink, line_mask)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    kept = np.zeros_like(ink)
    boxes = []
    minimum_area = max(3, int(ch * cw * 0.00006))
    for component in range(1, count):
        left, top, box_w, box_h, area = map(int, stats[component])
        if area < minimum_area or area > ch * cw * 0.1:
            continue
        if box_w > cw * 0.8 or box_h > ch * 0.8:
            continue
        if max(box_w / max(1, box_h), box_h / max(1, box_w)) > 9 and area > minimum_area * 2:
            continue
        # Balloon outlines and page art commonly enter from a crop edge.
        # Components touching that edge are poor evidence for font matching.
        edge = 2
        if left <= edge or top <= edge or left + box_w >= cw - edge or top + box_h >= ch - edge:
            continue
        kept[labels == component] = 255
        boxes.append([x + left, y + top, box_w, box_h, area])

    full_mask = np.zeros((height, width), np.uint8)
    full_mask[y:y2, x:x2] = kept
    ys, xs = np.nonzero(full_mask)
    text_box = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] if len(xs) else None
    return full_mask, {"roi": [x, y, x2 - x, y2 - y], "text_box": text_box, "components": len(boxes), "ink_pixels": int(cv2.countNonZero(full_mask))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=pathlib.Path, nargs="*")
    parser.add_argument("--ids", type=int, nargs="*")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    by_id = {e["question_id"]: e for e in manifest["entries"]}
    annotations = {}
    for annotation_path in args.annotations or []:
        annotations.update({int(str(a["question_id"]).removeprefix("q")): a for a in json.loads(annotation_path.read_text())})
    ids = args.ids or list(by_id)
    output = ROOT / "data/processed" / args.version
    results = []
    for question_id in ids:
        entry = by_id[question_id]
        if not entry["image_file"]:
            continue
        image = imread(ROOT / "data" / entry["image_file"])
        annotation = annotations.get(question_id)
        line_rois = [tuple(line["bbox"]) for line in annotation.get("lines", [])] if annotation else []
        if line_rois:
            mask = np.zeros(image.shape[:2], np.uint8)
            subregions = []
            for line_index, line_roi in enumerate(line_rois):
                lx, ly, lw, lh = line_roi
                if str(annotation.get("layout", "")).startswith("horizontal"):
                    # Text-line estimates can truncate the final characters.
                    px, py = max(8, int(lw * 0.3)), 8
                else:
                    # Coarse LLM line boxes often clip final characters.
                    px, py = max(8, int(lw * 0.3)), max(8, int(lh * 0.3))
                padded_roi = (lx - px, ly - py, lw + px * 2, lh + py * 2)
                line_mask, line_meta = extract(image, padded_roi)
                # The padded ROI helps thresholding at a clipped glyph edge,
                # but pixels far outside the proposed text column are usually
                # balloon outlines, speed lines, or illustration. Keep a small
                # evidence margin around the original column estimate.
                guard = max(6, min(12, int(min(lw, lh) * 0.12)))
                clip = np.zeros_like(line_mask)
                x1, y1 = max(0, lx - guard), max(0, ly - guard)
                x2, y2 = min(image.shape[1], lx + lw + guard), min(image.shape[0], ly + lh + guard)
                clip[y1:y2, x1:x2] = 255
                line_mask = cv2.bitwise_and(line_mask, clip)
                ys, xs = np.nonzero(line_mask)
                line_meta["text_box"] = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] if len(xs) else None
                line_meta["ink_pixels"] = int(cv2.countNonZero(line_mask))
                line_meta["components"] = int(cv2.connectedComponents(line_mask, 8)[0] - 1)
                line_meta["mask_file"] = f"q{question_id}-line{line_index}-mask.png"
                imwrite(output / line_meta["mask_file"], line_mask)
                mask = cv2.bitwise_or(mask, line_mask)
                subregions.append(line_meta)
            ys, xs = np.nonzero(mask)
            text_box = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] if len(xs) else None
            meta = {"roi": None, "text_box": text_box, "components": sum(s["components"] for s in subregions), "ink_pixels": int(cv2.countNonZero(mask)), "subregions": subregions}
        else:
            roi = tuple(annotation["bbox"]) if annotation and annotation.get("bbox") else None
            mask, meta = extract(image, roi)
        overlay = image.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        if meta["text_box"] and not line_rois:
            x, y, w, h = meta["text_box"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for subregion in meta.get("subregions", []):
            if subregion["text_box"]:
                x, y, w, h = subregion["text_box"]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)
        name = f"q{question_id}"
        imwrite(output / f"{name}-mask.png", mask)
        imwrite(output / f"{name}-overlay.png", overlay)
        results.append({"question_id": question_id, "coarse_label": entry["coarse_label"], **meta})
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"processed": len(results), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
