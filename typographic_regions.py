"""Typography-aware candidates; source pixels remain the comparison evidence.

Two explicit routes: vertical print on a clean background, and dark print with
a pale outline. Routing is an experimental input, not an inferred font label.
"""

import argparse
import json

import cv2
import numpy as np

from process_regions import ROOT, imread, imwrite
from segment_text import merge_near, spans


def outlined_ink(image, scale):
    """Keep dark components isolated by pale surroundings, without eroding ink.

    Components are computed on the entire image before any ROI restriction:
    cutting a background line at an ROI boundary must not turn it into a glyph.
    Small detached marks are retained for the later grid, not called characters.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    selected = np.zeros_like(dark)
    evidence = []
    radius = max(2, round(scale * .06))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    height, width = gray.shape
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        if area < max(3, scale * scale * .001) or h > scale * 1.65 or w > scale * 1.5:
            continue
        # A long, narrow speed line is not a full-width digital glyph.
        if h > scale * .55 and w < scale * .20:
            continue
        x0, y0 = max(0, x-radius), max(0, y-radius)
        x1, y1 = min(width, x+w+radius), min(height, y+h+radius)
        component = (labels[y0:y1, x0:x1] == index).astype(np.uint8)
        # Skip the antialiased transition immediately next to the dark fill.
        inner = cv2.dilate(component, np.ones((3, 3), np.uint8))
        ring = (cv2.dilate(component, kernel) > 0) & (inner == 0)
        pale_support = float(np.mean(gray[y0:y1, x0:x1][ring] > 175)) if ring.any() else 0.
        if pale_support < .72:
            continue
        selected[y0:y1, x0:x1][component > 0] = 255
        evidence.append({"bbox": [x, y, w, h], "pale_support": round(pale_support, 3)})
    return selected, evidence


def column_bands(mask, nominal_width):
    """Find ink bands; reject thin isolated rules, retaining full source evidence."""
    projection = np.count_nonzero(mask, axis=0)
    bands = merge_near(spans(projection, max(2, int(mask.shape[0] * .006))),
                       max(2, round(nominal_width * .10)))
    return [(a, b) for a, b in bands if b-a >= nominal_width * .32]


def fit_grid(strip, scale):
    """Fit a soft full-width pitch prior to low-ink horizontal cut positions.

    Returns reviewable cells, not guaranteed single characters. No stroke is
    deleted when a boundary crosses ink; that cell is explicitly flagged.
    """
    rows = np.count_nonzero(strip, axis=1).astype(float)
    active = np.flatnonzero(rows)
    if not len(active):
        return [], {}
    start, stop = int(active[0]), int(active[-1]) + 1
    normalizer = max(1., float(np.percentile(rows[active], 80)))
    smoothed = np.convolve(rows / normalizer, np.ones(3)/3, mode="same")
    best = None
    for pitch in range(max(5, round(scale * .9)), max(6, round(scale * 1.45)) + 1):
        for phase in range(pitch):
            cuts = np.arange(phase, stop, pitch)
            cuts = cuts[(cuts > start+2) & (cuts < stop-2)]
            if len(cuts) == 0:
                continue
            cost = float(smoothed[cuts].mean()) + .10 * ((pitch/scale-1.12)/.3)**2
            if best is None or cost < best[0]:
                best = (cost, pitch, phase, cuts)
    if best is None:
        return [[start, stop, False]], {"status": "too_short"}
    cost, pitch, phase, cuts = best
    # Permit local rhythm variation instead of forcing a perfectly rigid grid.
    snapped = []
    radius = max(1, round(pitch*.08))
    for cut in cuts:
        candidates = np.arange(max(start+1,cut-radius), min(stop,cut+radius+1))
        costs = smoothed[candidates] + .02 * abs(candidates-cut)/radius
        snapped.append(int(candidates[np.argmin(costs)]))
    bounds = [start, *sorted(set(snapped)), stop]
    cells = []
    for a, b in zip(bounds, bounds[1:]):
        if rows[a:b].sum() < 3:
            continue
        crossing = (a > start and rows[a] > normalizer*.15) or (b < stop and rows[b] > normalizer*.15)
        cells.append([a, b, bool(crossing)])
    return cells, {"pitch": pitch, "phase": phase, "cost": round(cost, 4),
                   "status": "candidate_only", "scale": round(scale, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--outlined-ids", type=int, nargs="*", default=[])
    parser.add_argument("--version", default="typographic-v1")
    args = parser.parse_args()
    entries = {e["question_id"]: e for e in json.loads((ROOT/"data/manifest.json").read_text())["entries"]}
    annotations = {}
    for name in ("priority-a", "priority-b"):
        annotations.update({a["question_id"]: a for a in json.loads((ROOT/f"data/{name}.json").read_text())})
    source = ROOT/"data/processed/adaptive-v3"
    output = ROOT/"data/processed"/args.version
    report = []
    for qid in args.ids or sorted(annotations):
        annotation = annotations[qid]
        if str(annotation.get("layout", "vertical")).startswith("horizontal"):
            raise ValueError("This pilot supports vertical type only")
        image = imread(ROOT/"data"/entries[qid]["image_file"])
        scale = float(np.median([line["bbox"][2] for line in annotation["lines"]]))
        mask = imread(source/f"q{qid}-mask.png")[:, :, 0]
        components = []
        route = "vertical_print"
        if qid in args.outlined_ids:
            route = "dark_print_pale_outline"
            mask, components = outlined_ink(image, scale)
            # Coarse column seeds constrain lateral location but do not clip
            # longitudinally: q27's final か lies beyond the old coarse box.
            allowed = np.zeros_like(mask)
            for line in annotation["lines"]:
                x, y, w, h = line["bbox"]
                pad = round(scale*.12)
                allowed[:, max(0,x-pad):min(mask.shape[1],x+w+pad)] = 255
            mask = cv2.bitwise_and(mask, allowed)
        bands = column_bands(mask, scale)
        overlay = image.copy()
        columns = []
        for index, (left, right) in enumerate(reversed(bands)):
            strip = mask[:, left:right]
            # Shared robust main-type width: do not turn a narrow punctuation
            # column into its own implausibly small character pitch.
            widths = [b-a for a,b in bands]
            fitted_scale = float(np.median(widths)) if widths else scale
            cells, grid = fit_grid(strip, fitted_scale)
            glyphs = []
            for number, (top, bottom, crossing) in enumerate(cells):
                ys, xs = np.nonzero(strip[top:bottom])
                if not len(xs):
                    continue
                box = [left+int(xs.min()), top+int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
                # Keep the whole grid cell with context, not just binary ink.
                x0, x1 = max(0,left-2), min(image.shape[1],right+2)
                y0, y1 = max(0,top-2), min(image.shape[0],bottom+2)
                filename = f"q{qid}-c{index}-g{number}-source.png"
                imwrite(output/filename, image[y0:y1,x0:x1])
                color = (0,0,255) if crossing else (0,160,255)
                cv2.rectangle(overlay, (left,top), (right-1,bottom-1), color, 1)
                fragment = box[2] < fitted_scale*.35 or box[3] < fitted_scale*.35
                glyphs.append({"ink_bbox": box, "source_bbox": [x0,y0,x1-x0,y1-y0],
                               "source_file": filename, "cut_crosses_ink": crossing,
                               "possible_mark_or_noise": fragment})
            columns.append({"x_range": [left,right], "grid": grid, "glyphs": glyphs})
        imwrite(output/f"q{qid}-mask.png", mask)
        imwrite(output/f"q{qid}-segments.png", overlay)
        imwrite(output/f"q{qid}-source.png", image)
        tint = image.copy()
        tint[mask > 0] = (tint[mask > 0]*.5 + np.array([0,255,0])*.5).astype(np.uint8)
        imwrite(output/f"q{qid}-overlay.png", tint)
        report.append({"question_id": qid, "route": route, "columns": columns,
                       "outline_components": components, "status": "needs_visual_review"})
    (output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"processed": len(report), "output": str(output)}))


if __name__ == "__main__":
    main()
