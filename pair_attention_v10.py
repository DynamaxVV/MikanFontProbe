"""Pair-conditional, no-training local attention with explicit abstention.

Source regions are structural proposals. Template contrast is retained only
where it survives shared rendering perturbations. No answer label is read.
"""

import cv2
import numpy as np

from contrastive_patch_v9 import _align, _representation
from closed_box_features import _is_white_hole


POLICY = {
    "blur_sigmas": (0., .65, 1.25),
    "rectangularity": .86,
    "min_hole_side": 8,
    "min_native_pixels": 32,
    "min_native_terminal_width": 3.5,
    "min_template_contrast": .025,
    "min_margin": .055,
    "min_characters": 2,
    "min_agreement": .8,
}


def _holes(gray, threshold):
    mask = (gray < threshold).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    rows = []
    for index, contour in enumerate(contours):
        if not _is_white_hole(index, hierarchy[0]):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        rotated = cv2.minAreaRect(contour)
        denominator = rotated[1][0] * rotated[1][1]
        if min(w, h) < POLICY["min_hole_side"] or area < 25 or denominator <= 0:
            continue
        rows.append({"bbox": (x, y, x+w-1, y+h-1),
                     "center": (x+w/2, y+h/2), "area": area,
                     "rectangularity": float(area/denominator)})
    return rows


def rectangular_holes(gray):
    """Keep rectangular white holes stable over three gray thresholds."""
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    middle = float(np.clip(otsu, 96, 160))
    groups = [_holes(gray, t) for t in (middle-20, middle, middle+20)]
    accepted = []
    for center in groups[1]:
        peers = [min(group, key=lambda row: np.linalg.norm(
            np.subtract(row["center"], center["center"])), default=None) for group in groups]
        if any(row is None or np.linalg.norm(np.subtract(row["center"], center["center"])) > 3
               or row["rectangularity"] < POLICY["rectangularity"]
               or not .7 <= row["area"]/center["area"] <= 1.3 for row in peers):
            continue
        accepted.append({"bbox": center["bbox"], "rectangularity":
                         min(row["rectangularity"] for row in peers)})
    return accepted


def _regions(source):
    gray, pixels = source["gray"], source["pixels"]
    regions = []
    for hole in rectangular_holes(gray):
        x0, y0, x1, y1 = hole["bbox"]
        if min(x1-x0, y1-y0) * pixels/104 < 7:
            continue
        pad = int(np.clip(.18 * max(x1-x0, y1-y0), 5, 12))
        regions.append({"kind": "rect_box", "bbox": (x0-pad, y0-pad, x1+pad, y1+pad),
                        "hole_bbox": hole["bbox"],
                        "rectangularity": hole["rectangularity"]})
    terminals = [row for row in source["features"] if row["kind"] == "terminal"
                 and row["native_width"] >= POLICY["min_native_terminal_width"]]
    seen = []
    for row in terminals:
        x, y = row["point"]
        if any(np.hypot(x-other[0], y-other[1]) < max(6, row["width"]) for other in seen):
            continue
        seen.append((x, y))
        radius = int(np.clip(1.8 * row["width"], 8, 18))
        regions.append({"kind": "free_end", "bbox": (x-radius, y-radius,
                                                        x+radius, y+radius),
                        "point": [x, y], "native_width": row["native_width"]})
    return regions


def _blur(gray, sigma):
    return cv2.GaussianBlur(gray, (0, 0), sigma) if sigma else gray


def _prepare(source, left, right):
    variants = [(_blur(left, sigma), _blur(right, sigma))
                for sigma in POLICY["blur_sigmas"]]
    representations = {mode: [(_representation(a, mode), _representation(b, mode))
                              for a, b in variants] for mode in ("ink", "edge")}
    contrasts = [np.abs(a-b) for a, b in representations["ink"]]
    stable = np.minimum.reduce(contrasts)
    nuisance = np.maximum(np.abs(representations["ink"][0][0] - representations["ink"][-1][0]),
                          np.abs(representations["ink"][0][1] - representations["ink"][-1][1]))
    saliency = np.maximum(stable - .5*nuisance, 0)
    return representations, saliency, {mode: _representation(source, mode)
                                       for mode in ("ink", "edge")}


def _patch_score(prepared, region):
    representations, saliency, observations = prepared
    window = 17
    potential = cv2.blur(saliency, (window, window))
    x0, y0, x1, y1 = region["bbox"]
    shape = observations["ink"].shape
    allowed = np.zeros(shape, bool)
    allowed[max(window, y0):min(shape[0]-window, y1+1),
            max(window, x0):min(shape[1]-window, x1+1)] = True
    potential[~allowed] = 0
    y, x = np.unravel_index(np.argmax(potential), potential.shape)
    if potential[y, x] < POLICY["min_template_contrast"]:
        return {"vote": 0, "reason": "template_difference_unstable_or_shared",
                "contrast": float(potential[y, x])}
    span = slice(y-8, y+9), slice(x-8, x+9)
    decisions = {}
    for mode, pairs in representations.items():
        observed = observations[mode][span]
        margins = []
        for a, b in pairs:
            weight = saliency[span]
            denominator = max(float(weight.sum()), 1e-6)
            da = float(np.sum(weight * np.abs(observed-a[span])) / denominator)
            db = float(np.sum(weight * np.abs(observed-b[span])) / denominator)
            margins.append(db-da)
        median = float(np.median(margins))
        consistent = all(value * median > 0 for value in margins)
        decisions[mode] = {"margins": margins, "median": median,
                           "stable": consistent and abs(median) >= POLICY["min_margin"]}
    ink, edge = decisions["ink"], decisions["edge"]
    vote = 1 if ink["stable"] and edge["stable"] and ink["median"] > 0 and edge["median"] > 0 else (
        -1 if ink["stable"] and edge["stable"] and ink["median"] < 0 and edge["median"] < 0 else 0)
    return {"vote": vote, "reason": "stable_agreement" if vote else "unstable_or_conflicting",
            "center": [int(x), int(y)], "contrast": float(potential[y, x]),
            "ink": ink, "edge": edge}


def compare(source, a, b):
    """Compare one known character against two exact-face templates."""
    if source["pixels"] < POLICY["min_native_pixels"]:
        return {"vote": 0, "reason": "insufficient_native_resolution", "regions": []}
    left, right = _align(source["gray"], a["gray"]), _align(source["gray"], b["gray"])
    if left is None or right is None:
        return {"vote": 0, "reason": "aspect_alignment_failed", "regions": []}
    prepared = _prepare(source["gray"], left, right)
    regions = []
    for region in _regions(source):
        regions.append({**region, **_patch_score(prepared, region)})
    votes = [row["vote"] for row in regions if row["vote"]]
    if not votes:
        reason, vote = "no_stable_part", 0
    elif all(value == votes[0] for value in votes):
        reason, vote = "stable_part_agreement", votes[0]
    else:
        reason, vote = "conflicting_parts", 0
    return {"vote": vote, "reason": reason, "regions": regions}


def decide(characters):
    """Conservative pair-level abstention; values are not probabilities."""
    votes = [row["vote"] for row in characters if row["vote"]]
    if len(votes) < POLICY["min_characters"]:
        return {"vote": 0, "reason": "insufficient_distinct_characters"}
    winner = 1 if sum(value > 0 for value in votes)/len(votes) >= POLICY["min_agreement"] else (
        -1 if sum(value < 0 for value in votes)/len(votes) >= POLICY["min_agreement"] else 0)
    return {"vote": winner, "reason": "stable_character_agreement" if winner else "conflicting_characters",
            "supporting_characters": len(votes)}
