"""Run the comic-text-detector ONNX segmentation head with OpenCV DNN.

The downloaded upstream checkpoint stays outside this repository. This is a
raw detector baseline: no LLM box is used to rescue or restrict its mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mikan_font_probe.process_regions import ROOT, imread, imwrite


def predict(net: cv2.dnn.Net, image: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    scaled_width, scaled_height = round(width * ratio), round(height * ratio)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scaled = cv2.resize(rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        scaled, 0, size - scaled_height, 0, size - scaled_width,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    net.setInput(cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0))
    output_names = net.getUnconnectedOutLayersNames()
    outputs = dict(zip(output_names, net.forward(output_names)))
    # The published ONNX model exposes 'seg' for text-pixel probability and
    # 'det' for text-line probability; YOLO 'blk' is not needed for this test.
    text = outputs["seg"][0, 0, :scaled_height, :scaled_width]
    line = outputs["det"][0, 0, :scaled_height, :scaled_width]
    predicted = outputs["blk"][0]
    scores = predicted[:, 4] * predicted[:, 5:].max(axis=1)
    candidates = predicted[scores >= 0.25]
    boxes = []
    confidences = []
    for row, score in zip(candidates, scores[scores >= 0.25]):
        cx, cy, bw, bh = row[:4] / ratio
        x1, y1 = max(0, round(cx - bw / 2)), max(0, round(cy - bh / 2))
        x2, y2 = min(width, round(cx + bw / 2)), min(height, round(cy + bh / 2))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(score))
    selected = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.35)
    blocks = [{"bbox": boxes[int(i)], "score": round(confidences[int(i)], 3)} for i in np.ravel(selected)]
    return (
        cv2.resize(text, (width, height), interpolation=cv2.INTER_LINEAR),
        cv2.resize(line, (width, height), interpolation=cv2.INTER_LINEAR),
        blocks,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--version", default="ctd-raw-v1")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must lie between 0 and 1")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    entries = {entry["question_id"]: entry for entry in manifest["entries"]}
    priority = json.loads((ROOT / "data/priority.json").read_text())
    ids = args.ids or [qid for group in priority["groups"] for qid in group["sample_ids"]]
    net = cv2.dnn.readNetFromONNX(str(args.model))
    output = ROOT / "data/processed" / args.version
    results = []
    for qid in ids:
        entry = entries[qid]
        image = imread(ROOT / "data" / entry["image_file"])
        text_prob, line_prob, blocks = predict(net, image, args.size)
        mask = (text_prob >= args.threshold).astype(np.uint8) * 255
        overlay = image.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        imwrite(output / f"q{qid}-mask.png", mask)
        imwrite(output / f"q{qid}-overlay.png", overlay)
        block_overlay = image.copy()
        for block in blocks:
            x, y, w, h = block["bbox"]
            cv2.rectangle(block_overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
        imwrite(output / f"q{qid}-blocks.png", block_overlay)
        imwrite(output / f"q{qid}-probability.png", (text_prob * 255).clip(0, 255).astype(np.uint8))
        imwrite(output / f"q{qid}-line-probability.png", (line_prob * 255).clip(0, 255).astype(np.uint8))
        results.append({
            "question_id": qid, "coarse_label": entry["coarse_label"],
            "ink_pixels": int(cv2.countNonZero(mask)),
            "blocks": blocks,
            "model": args.model.name, "input_size": args.size, "threshold": args.threshold,
        })
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"processed": len(results), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
