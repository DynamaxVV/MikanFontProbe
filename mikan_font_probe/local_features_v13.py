"""Conservative endpoint/corner evidence for the v13 reranker.

The feature extractor is intentionally deterministic and abstains when the
crop is too small or has no stable local geometry.  It is a tie-breaker for
the global embedding, not a replacement for it.
"""

from __future__ import annotations

import cv2
import numpy as np

from mikan_font_probe.glyph_geometry import thin
from mikan_font_probe.stroke_correspondence_v7 import correspondence
from mikan_font_probe.stroke_feature_matcher import gray_canvas
from mikan_font_probe.stroke_terminal_features import extract_terminals


STABLE_STATES = {"square", "rounded"}


def effective_state(feature: dict) -> str:
    """Use threshold-stable geometry even when the generic policy rejects
    the feature solely because the manga crop is below its absolute width
    floor.  A majority across threshold perturbations is still required."""
    if feature.get("state") in STABLE_STATES:
        return feature["state"]
    states = [state for state in feature.get("metrics", {}).get("threshold_states", [])
              if state in STABLE_STATES]
    if len(states) >= 2 and states.count(max(set(states), key=states.count)) >= 2:
        return max(set(states), key=states.count)
    return "unknown"


def local_reference(gray: np.ndarray, native_pixels: int) -> dict:
    """Normalize a tight glyph and extract bounded terminal/corner records.

    The existing full corner fitter is intentionally stricter and more
    expensive because it is used for pairwise proof evidence.  A reranker
    needs a cheaper candidate descriptor, so corners here come from
    ``goodFeaturesToTrack`` while free caps still use the threshold-stable
    terminal fitter.
    """
    canvas, pixels = gray_canvas(np.asarray(gray, dtype=np.uint8), native_pixels)
    terminals = extract_terminals(canvas, pixels)
    mask = cv2.threshold(canvas, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    harris = cv2.cornerHarris(np.float32(mask) / 255., 3, 3, .04)
    scale = max(float(np.max(np.abs(harris))), 1e-6)
    points = cv2.goodFeaturesToTrack(cv2.GaussianBlur(mask, (3, 3), .7),
                                     maxCorners=16, qualityLevel=.04,
                                     minDistance=5, blockSize=5)
    corners = []
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if points is not None:
        for point in points[:, 0, :]:
            x, y = (int(round(float(value))) for value in point)
            if not (0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]):
                continue
            width = max(2., 2. * float(distance[y, x]))
            corners.append({"kind": "corner", "point": [x, y],
                            "width": width, "native_width": width * pixels / 104.,
                            "state": "unknown", "square_score": 0.,
                            "metrics": {"corner_response": float(abs(harris[y, x]) / scale)},
                            "reason": "fast_corner_candidate"})
    return {"gray": canvas, "pixels": pixels,
            "features": terminals[:16] + corners[:16]}


def _feature_shape_similarity(left: dict, right: dict) -> float:
    """Compare measured local shape values after correspondence."""
    values = []
    if left["kind"] == "terminal":
        for key, scale in (("flat_residual", 0.12), ("round_residual", 0.12),
                           ("shaft_width_variation", 0.25)):
            if key in left.get("metrics", {}) and key in right.get("metrics", {}):
                delta = abs(float(left["metrics"][key]) - float(right["metrics"][key]))
                values.append(np.clip(delta / scale, 0., 1.))
    elif left["kind"] == "corner":
        for key, scale in (("corner_response", .45),
                           ("turn_concentration", 0.45),
                           ("intersection_gap", 0.35),
                           ("circle_residual", 0.12),
                           ("line_residual", 0.12)):
            if key in left.get("metrics", {}) and key in right.get("metrics", {}):
                delta = abs(float(left["metrics"][key]) - float(right["metrics"][key]))
                values.append(np.clip(delta / scale, 0., 1.))
    if "width" in left and "width" in right:
        ratio = (float(left["width"]) + 1e-3) / (float(right["width"]) + 1e-3)
        values.append(np.clip(abs(np.log(ratio)) / .55, 0., 1.))
    return float(1. - np.mean(values)) if values else .5


def compare_local(source: dict, target: dict) -> dict:
    """Return a bounded local similarity and explicit abstention evidence."""
    pairs, reason = correspondence(source, target)
    if not pairs:
        return {"score": .5, "reliable": False, "reason": reason,
                "matched": 0, "stable_pairs": 0, "kinds": {}}

    weighted, total_weight = [], 0.
    stable_pairs = 0
    kinds = {"terminal": 0, "corner": 0}
    for left_index, right_index in pairs:
        left, right = source["features"][left_index], target["features"][right_index]
        kind_weight = 2. if left["kind"] == "corner" else 1.
        kinds[left["kind"]] = kinds.get(left["kind"], 0) + 1
        left_state, right_state = effective_state(left), effective_state(right)
        state_score = .5
        if left_state in STABLE_STATES and right_state in STABLE_STATES:
            stable_pairs += 1
            state_score = 1. if left_state == right_state else 0.
        shape_score = _feature_shape_similarity(left, right)
        weighted.append(kind_weight * (.62 * state_score + .38 * shape_score))
        total_weight += kind_weight

    matched = len(pairs)
    source_count = len(source["features"])
    target_count = len(target["features"])
    coverage = matched / max(1, min(source_count, target_count))
    score = float(np.clip(.78 * sum(weighted) / max(total_weight, 1e-6)
                          + .22 * min(1., coverage), 0., 1.))
    reliable = stable_pairs >= 2 and matched >= 2
    return {"score": score if reliable else .5 + .25 * (score - .5),
            "reliable": reliable, "reason": "stable_local_correspondence"
            if reliable else "insufficient_stable_local_features",
            "matched": matched, "stable_pairs": stable_pairs, "kinds": kinds}


def local_similarity(source_gray: np.ndarray, target_gray: np.ndarray,
                     source_native: int, target_native: int) -> dict:
    """Build references and compare endpoints/corners for one source/template."""
    return compare_local(local_reference(source_gray, source_native),
                         local_reference(target_gray, target_native))
