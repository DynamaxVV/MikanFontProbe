"""Render original / three extraction routes side by side for review."""

from __future__ import annotations

import json

import cv2
import numpy as np

from process_regions import ROOT, imread, imwrite


def main() -> None:
    priority = json.loads((ROOT / "data/priority.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    entries = {entry["question_id"]: entry for entry in manifest["entries"]}
    versions = ("priority-v7", "adaptive-v3", "ctd-raw-v2")
    output = ROOT / "data/ab-comparison"
    for group in priority["groups"]:
        for qid in group["sample_ids"]:
            original = imread(ROOT / "data" / entries[qid]["image_file"])
            images = [original] + [imread(ROOT / "data/processed" / version / f"q{qid}-overlay.png") for version in versions]
            scale = min(1.0, 720 / original.shape[0])
            w, h = round(original.shape[1] * scale), round(original.shape[0] * scale)
            gap, heading = 12, 38
            canvas = np.full((h + heading, 4 * w + 5 * gap, 3), 245, np.uint8)
            for index, (label, image) in enumerate(zip(("source",) + versions, images)):
                left = gap + index * (w + gap)
                canvas[heading : heading + h, left : left + w] = cv2.resize(image, (w, h))
                cv2.putText(canvas, f"q{qid} {label}", (left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (30, 30, 30), 1, cv2.LINE_AA)
            imwrite(output / f"q{qid}-comparison.png", canvas)
    print(json.dumps({"comparison_images": 18, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
