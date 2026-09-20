"""Group candidate ink into text lines and conservative glyph candidates."""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

from process_regions import ROOT, imread, imwrite


def spans(values: np.ndarray, threshold: int = 1) -> list[tuple[int, int]]:
    active = values >= threshold
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    return list(zip(np.where(edges == 1)[0].tolist(), np.where(edges == -1)[0].tolist()))


def merge_near(ranges: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def boxes_from_mask(mask: np.ndarray, layout: str) -> tuple[list[list[int]], list[list[int]]]:
    vertical = not layout.startswith("horizontal")
    h, w = mask.shape
    line_projection = np.count_nonzero(mask, axis=0 if vertical else 1)
    line_spans = merge_near(spans(line_projection, max(1, int(max(h, w) * 0.002))), 3)
    lines: list[list[int]] = []
    chars: list[list[int]] = []
    for start, end in line_spans:
        strip = mask[:, start:end] if vertical else mask[start:end, :]
        ys, xs = np.nonzero(strip)
        if len(xs) < 12:
            continue
        if vertical:
            line = [start + int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        else:
            line = [int(xs.min()), start + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        if vertical and (line[2] < 8 or line[3] < 2 * line[2]):
            continue
        if not vertical and (line[3] < 8 or line[2] < 2 * line[3]):
            continue
        lines.append(line)
        cross = np.count_nonzero(strip, axis=1 if vertical else 0)
        # Merge tiny internal stroke gaps and detached dakuten with their base.
        expected = line[2] if vertical else line[3]
        for a, b in merge_near(spans(cross, 1), max(2, int(expected * 0.17))):
            sub = strip[a:b, :] if vertical else strip[:, a:b]
            cy, cx = np.nonzero(sub)
            if len(cx) < 10:
                continue
            if vertical:
                box = [start + int(cx.min()), a + int(cy.min()), int(cx.max() - cx.min() + 1), int(cy.max() - cy.min() + 1)]
            else:
                box = [a + int(cx.min()), start + int(cy.min()), int(cx.max() - cx.min() + 1), int(cy.max() - cy.min() + 1)]
            if box[2] >= 3 and box[3] >= 3:
                chars.append(box)
    if vertical:
        lines.sort(key=lambda b: -b[0])
        chars.sort(key=lambda b: (-b[0], b[1]))
    else:
        lines.sort(key=lambda b: b[1])
        chars.sort(key=lambda b: (b[1], b[0]))
    return lines, chars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2")
    parser.add_argument("--annotations", type=pathlib.Path, nargs="*")
    args = parser.parse_args()
    source = ROOT / "data/processed" / args.version
    processed = json.loads((source / "results.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    entries = {e["question_id"]: e for e in manifest["entries"]}
    annotations = {}
    for annotation_path in args.annotations or []:
        annotations.update({int(str(a["question_id"]).removeprefix("q")): a for a in json.loads(annotation_path.read_text())})
    report = []
    for item in processed:
        qid = item["question_id"]
        mask = imread(source / f"q{qid}-mask.png")[:, :, 0]
        original = imread(ROOT / "data" / entries[qid]["image_file"])
        annotation = annotations.get(qid, {})
        layout = annotation.get("layout") or ("vertical" if original.shape[0] >= original.shape[1] else "horizontal")
        submasks = [region.get("mask_file") for region in item.get("subregions", []) if region.get("mask_file")]
        if submasks:
            lines, chars = [], []
            for filename in submasks:
                local_lines, local_chars = boxes_from_mask(imread(source / filename)[:, :, 0], layout)
                lines.extend(local_lines)
                chars.extend(local_chars)
        else:
            lines, chars = boxes_from_mask(mask, layout)
        overlay = original.copy()
        for x, y, w, h in lines:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)
        for x, y, w, h in chars:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 180, 255), 1)
        imwrite(source / f"q{qid}-segments.png", overlay)
        report.append({"question_id": qid, "layout": layout, "line_boxes": lines, "character_candidates": chars})
    (source / "segments.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"segmented": len(report), "lines": sum(len(r["line_boxes"]) for r in report), "character_candidates": sum(len(r["character_candidates"]) for r in report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
