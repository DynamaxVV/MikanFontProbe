"""Diagnostic continuous local-shape and axis-width evidence.

No source label is read. Scores are uncalibrated distances, not probabilities.
"""

import cv2
import numpy as np

from mikan_font_probe.glyph_geometry import thin
from mikan_font_probe.stroke_correspondence_v7 import correspondence, mapping


def _sample_patch(gray, center, radius, scale=(1., 1.), sigma=0.):
    image = cv2.GaussianBlur(gray, (0, 0), sigma) if sigma else gray
    axis = np.linspace(-radius, radius, 25, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    mx = (center[0] + xx / scale[0]).astype(np.float32)
    my = (center[1] + yy / scale[1]).astype(np.float32)
    return cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255).astype(np.float32) / 255


def _edge(patch):
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _distance(source, reference):
    # Ink and edges retain gray antialiasing instead of a hard square/round state.
    ink = np.abs(source - reference)
    edge = np.abs(_edge(source) - _edge(reference))
    focus = np.maximum(1 - source, 1 - reference)
    return float(.6 * np.mean(ink * (.3 + focus)) + .4 * np.mean(edge) / 4)


def local_evidence(source, a, b):
    """Compare corresponding terminal and box-corner patches, one per region."""
    pair, _ = correspondence(a, b)
    sa, _ = correspondence(source, a)
    sb, _ = correspondence(source, b)
    lookup_a = {j: i for i, j in sa}
    lookup_b = {j: i for i, j in sb}
    transform_a = mapping(source["gray"], a["gray"])
    transform_b = mapping(source["gray"], b["gray"])
    if transform_a is None or transform_b is None:
        return []
    rows = []
    used_boxes = set()
    for ia, ib in pair:
        fa, fb = a["features"][ia], b["features"][ib]
        if fa["kind"] not in ("terminal", "box_corner"):
            continue
        index = lookup_a.get(ia)
        if index is None or index != lookup_b.get(ib):
            continue
        fs = source["features"][index]
        if fa["kind"] == "box_corner":
            key = tuple(fs["box_bbox"])
            if key in used_boxes:
                continue  # Four corners of one hole are not independent samples.
            used_boxes.add(key)
        native = fs.get("native_width", 0)
        if native < 3 or source["pixels"] < 24:
            continue
        radius = float(np.clip(1.5 * fs["width"], 6, 17))
        observed = _sample_patch(source["gray"], fs["point"], radius)
        candidates = []
        for view, feature, transform in ((a, fa, transform_a), (b, fb, transform_b)):
            scores = [_distance(observed, _sample_patch(view["gray"], feature["point"],
                                                      radius, transform[0], blur))
                      for blur in (0., .6, 1.)]
            candidates.append(min(scores))
        difference = abs(candidates[0] - candidates[1])
        rows.append({"kind": fa["kind"], "source_index": index,
                     "character_region": fs.get("role", "free_end"),
                     "distance_a": candidates[0], "distance_b": candidates[1],
                     "margin": difference, "winner": "a" if candidates[0] < candidates[1] else "b",
                     "native_width": native,
                     "note": "diagnostic; shared local shape and blur can confound"})
    return rows


def axis_width(gray, pixels):
    """Robust straight-stroke vertical/horizontal width ratio; abstain if sparse."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (mask > 0).astype(np.uint8)
    skeleton = thin(mask).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    degree = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16))
    junction = cv2.dilate(((degree >= 5) & (skeleton > 0)).astype(np.uint8),
                          np.ones((7, 7), np.uint8)) > 0
    ys, xs = np.nonzero((skeleton > 0) & (degree == 3) & ~junction)
    widths = {"horizontal": [], "vertical": []}
    for x, y in zip(xs, ys):
        if x < 4 or y < 4 or x >= 124 or y >= 124:
            continue
        if 2 * distance[y, x] * pixels / 104 < 2.5:
            continue
        horizontal = skeleton[y, x-4:x+5].sum()
        vertical = skeleton[y-4:y+5, x].sum()
        if horizontal >= 7 and vertical <= 2:
            widths["horizontal"].append(float(2 * distance[y, x]))
        elif vertical >= 7 and horizontal <= 2:
            widths["vertical"].append(float(2 * distance[y, x]))
    counts = {key: len(value) for key, value in widths.items()}
    if min(counts.values()) < 8:
        return {"ratio": None, "counts": counts, "reason": "insufficient_straight_strokes"}
    medians = {key: float(np.median(value)) for key, value in widths.items()}
    return {"ratio": medians["vertical"] / medians["horizontal"],
            "widths": medians, "counts": counts, "reason": "measured"}
