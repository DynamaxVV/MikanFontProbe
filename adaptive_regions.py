"""Expand coarse text-column seeds along the ink flow before clipping masks.

The LLM bounding box is a seed, not a hard crop. The output contract matches
process_regions.py so the existing segment_text.py and overlays remain usable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from process_regions import ROOT, extract, imread, imwrite


def merged_spans(active: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    spans = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], int(end))
        else:
            merged.append((int(start), int(end)))
    return merged


def grow_line(image: np.ndarray, seed: tuple[int, int, int, int], horizontal: bool) -> tuple[np.ndarray, dict]:
    x, y, w, h = seed
    height, width = image.shape[:2]
    transverse = h if horizontal else w
    longitudinal = w if horizontal else h
    side = max(8, min(16, round(transverse * 0.15)))
    travel = max(2 * transverse, round(longitudinal * 0.4))
    if horizontal:
        expanded = (x - travel, y - side, w + travel * 2, h + side * 2)
        strip = (max(0, y - side), min(height, y + h + side))
    else:
        expanded = (x - side, y - travel, w + side * 2, h + travel * 2)
        strip = (max(0, x - side), min(width, x + w + side))

    candidate, meta = extract(image, expanded)
    if horizontal:
        candidate[: strip[0], :] = 0
        candidate[strip[1] :, :] = 0
        projection = np.count_nonzero(candidate[strip[0] : strip[1], :], axis=0)
        seed_start, seed_end = max(0, x), min(width, x + w)
    else:
        candidate[:, : strip[0]] = 0
        candidate[:, strip[1] :] = 0
        projection = np.count_nonzero(candidate[:, strip[0] : strip[1]], axis=1)
        seed_start, seed_end = max(0, y), min(height, y + h)

    # Ink beyond a coarse box can be its missing final glyph, but an object
    # displaced sideways (q72's phone, for example) is not a continuation.
    if horizontal:
        sy, sx = np.nonzero(candidate[strip[0] : strip[1], seed_start:seed_end])
        seed_center = strip[0] + float(np.median(sy)) if len(sy) else y + h / 2
        for pos in np.flatnonzero(projection):
            if seed_start <= pos < seed_end:
                continue
            transverse_positions = np.flatnonzero(candidate[strip[0] : strip[1], pos]) + strip[0]
            if abs(float(np.median(transverse_positions)) - seed_center) > max(10, transverse * 0.35):
                candidate[:, pos] = 0
                projection[pos] = 0
    else:
        sy, sx = np.nonzero(candidate[seed_start:seed_end, strip[0] : strip[1]])
        seed_center = strip[0] + float(np.median(sx)) if len(sx) else x + w / 2
        for pos in np.flatnonzero(projection):
            if seed_start <= pos < seed_end:
                continue
            transverse_positions = np.flatnonzero(candidate[pos, strip[0] : strip[1]]) + strip[0]
            if abs(float(np.median(transverse_positions)) - seed_center) > max(10, transverse * 0.35):
                candidate[pos, :] = 0
                projection[pos] = 0

    minimum_ink = max(2, round(transverse * 0.04))
    spans = merged_spans(projection >= minimum_ink, max(12, round(transverse * 0.55)))
    # Keep all seed-supported groups: punctuation and the final glyph may be
    # separated by a larger gap than the ordinary character pitch (q140 は).
    supported = [
        span for span in spans
        if projection[max(span[0], seed_start) : min(span[1], seed_end)].sum() > 0
    ]
    if supported:
        start, end = supported[0][0], supported[-1][1]
    else:
        start, end = seed_start, seed_end
    start, end = max(0, start - 2), min(len(projection), end + 2)
    if horizontal:
        candidate[:, :start] = 0
        candidate[:, end:] = 0
    else:
        candidate[:start, :] = 0
        candidate[end:, :] = 0

    ys, xs = np.nonzero(candidate)
    meta.update(
        seed_bbox=list(seed),
        grown_extent=[start, end],
        text_box=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] if len(xs) else None,
        ink_pixels=int(cv2.countNonZero(candidate)),
    )
    return candidate, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--version", default="adaptive-v1")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    entries = {entry["question_id"]: entry for entry in manifest["entries"]}
    annotations = {}
    for path in args.annotations:
        annotations.update({int(a["question_id"]): a for a in json.loads(path.read_text())})
    output = ROOT / "data/processed" / args.version
    results = []
    for qid in args.ids or sorted(annotations):
        entry = entries[qid]
        image = imread(ROOT / "data" / entry["image_file"])
        annotation = annotations[qid]
        horizontal = str(annotation.get("layout", "vertical")).startswith("horizontal")
        mask = np.zeros(image.shape[:2], np.uint8)
        subregions = []
        for index, line in enumerate(annotation["lines"]):
            line_mask, meta = grow_line(image, tuple(line["bbox"]), horizontal)
            meta["mask_file"] = f"q{qid}-line{index}-mask.png"
            imwrite(output / meta["mask_file"], line_mask)
            mask = cv2.bitwise_or(mask, line_mask)
            subregions.append(meta)
        ys, xs = np.nonzero(mask)
        text_box = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] if len(xs) else None
        overlay = image.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        for region in subregions:
            if region["text_box"]:
                bx, by, bw, bh = region["text_box"]
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
        imwrite(output / f"q{qid}-mask.png", mask)
        imwrite(output / f"q{qid}-overlay.png", overlay)
        results.append({"question_id": qid, "coarse_label": entry["coarse_label"], "text_box": text_box, "ink_pixels": int(cv2.countNonZero(mask)), "subregions": subregions})
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"processed": len(results), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
