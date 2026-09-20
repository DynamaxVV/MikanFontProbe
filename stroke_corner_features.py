"""Conservative outer-corner geometry, independent of handwriting order.

Corners describe two adjacent contour edges on a connected stroke. They do
not establish a pen trajectory, stroke segmentation, or writing order. All
distances in metrics are divided by adjacent stroke width unless named pixels.
No font, glyph identity, or corpus-specific rules are used.
"""

import cv2
import numpy as np

from glyph_geometry import thin


CORNER_POLICY = {
    "max_features": 16,
    "threshold_delta": 20,
    "min_native_width": 6.0,
    "min_width": 5.0,
    "min_turn_degrees": 50.0,
    "max_turn_degrees": 130.0,
    "max_line_residual": 0.045,
    "max_flank_turn_degrees": 12.0,
    "max_flank_turn_fraction": 0.12,
    "min_concentration": 0.32,
    "square_concentration": 0.70,
    "square_gap": 0.12,
    "rounded_gap": 0.18,
    "max_circle_residual": 0.035,
    "continuous_arc_residual": 0.025,
    "rounded_concentration": 0.65,
    "min_circle_radius": 0.5,
    "max_circle_radius": 2.5,
    "exclusion_radius": 1.6,
    "merge_radius": 1.5,
    "score_residual_scale": 0.09,
    "max_point_shift": 0.25,
    "max_width_variation": 0.20,
}


def _angle(a, b):
    cross = a[0] * b[1] - a[1] * b[0]
    return float(np.degrees(np.arctan2(cross, np.dot(a, b))))


def _line(points):
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    direction = vectors[0]
    if np.dot(direction, points[-1] - points[0]) < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    residual = float(np.sqrt(np.mean(((points - center) @ normal) ** 2)))
    return center, direction, residual


def _resample(contour):
    points = contour[:, 0, :].astype(float)
    points = np.vstack([points, points[0]])
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.arange(0.0, lengths[-1], 1.0)
    return np.column_stack([np.interp(samples, lengths, points[:, axis])
                            for axis in (0, 1)])


def _smooth(points, width):
    sigma = max(0.8, 0.065 * width)
    offsets = np.arange(-int(np.ceil(3 * sigma)), int(np.ceil(3 * sigma)) + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    return sum(weight * np.roll(points, int(offset), axis=0)
               for offset, weight in zip(offsets, weights))


def _topology(mask):
    skeleton = thin(mask * 255)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(skeleton)
    points = np.column_stack([xs, ys]).astype(float)
    padded = np.pad(skeleton, 1)
    neighbors = [padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
                 padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
                 padded[1:-1, :-2], padded[:-2, :-2]]
    degree = sum(neighbors)
    runs = sum((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1)
               for i in range(8))
    # Neighbor runs avoid treating ordinary staircase triangles as branches.
    bad = (degree[ys, xs] <= 1) | (runs[ys, xs] >= 3)
    return points, 2.0 * distance[ys, xs], points[bad]


def _adjacent_width(point, skeleton, widths, bad_points):
    distances = np.linalg.norm(skeleton - point, axis=1)
    initial = float(widths[np.argmin(distances)])
    nearby = (distances >= 0.7 * initial) & (distances <= 2.0 * initial)
    if nearby.sum() < 6:
        return None
    width = float(np.median(widths[nearby]))
    if width < CORNER_POLICY["min_width"]:
        return None
    if (len(bad_points) and np.linalg.norm(bad_points - point, axis=1).min()
            < CORNER_POLICY["exclusion_radius"] * width):
        return None
    shell = (distances >= 1.5 * width) & (distances <= 2.2 * width)
    image = np.zeros((128, 128), np.uint8)
    coordinates = skeleton[shell].astype(int)
    image[coordinates[:, 1], coordinates[:, 0]] = 1
    count, _, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    arms = sum(stats[i, cv2.CC_STAT_AREA] >= max(2, 0.25 * width)
               for i in range(1, count))
    if arms != 2:
        return None
    variation = float(np.std(widths[nearby]) / max(width, 1.0))
    if variation > CORNER_POLICY["max_width_variation"]:
        return None
    return width, variation


def _circle_residual(points):
    centered = points - points.mean(axis=0)
    design = np.column_stack([2 * centered, np.ones(len(points))])
    solution = np.linalg.lstsq(design, np.sum(centered ** 2, axis=1), rcond=None)[0]
    distances = np.linalg.norm(centered - solution[:2], axis=1)
    radius = float(np.mean(distances))
    return float(np.sqrt(np.mean((distances - radius) ** 2))), radius


def _measure(points, index, width, orientation):
    offsets = np.linspace(-2.0, 2.0, 81) * width
    positions = (index + offsets) % len(points)
    base = np.floor(positions).astype(int)
    fraction = (positions - base)[:, None]
    patch = points[base] * (1 - fraction) + points[(base + 1) % len(points)] * fraction
    patch = (patch - patch[40]) / width
    left, right = _line(patch[:21]), _line(patch[60:])
    turn = float(orientation * _angle(left[1], right[1]))
    if not CORNER_POLICY["min_turn_degrees"] <= turn <= CORNER_POLICY["max_turn_degrees"]:
        return None
    residual = max(left[2], right[2])
    flank_turn = max(abs(_angle(_line(patch[:11])[1], _line(patch[10:21])[1])),
                     abs(_angle(_line(patch[60:71])[1], _line(patch[70:])[1])))
    if residual > CORNER_POLICY["max_line_residual"] or flank_turn > CORNER_POLICY["max_flank_turn_degrees"]:
        return None
    if flank_turn > CORNER_POLICY["max_flank_turn_fraction"] * turn:
        return None  # Continuous curvature must not masquerade as two edges.
    # Tangents on either side of a width-scaled central arc measure how much
    # of the overall turn is localized here, rather than spread along a curve.
    central_turn = orientation * _angle(_line(patch[30:35])[1], _line(patch[46:51])[1])
    concentration = float(np.clip(central_turn / turn, 0.0, 1.0))
    if concentration < CORNER_POLICY["min_concentration"]:
        return None
    broad_arc_residual, _ = _circle_residual(patch)
    if broad_arc_residual < CORNER_POLICY["continuous_arc_residual"]:
        return None  # A continuous arc is not evidence of a connection corner.
    matrix = np.column_stack([left[1], -right[1]])
    intersection = left[0] + np.linalg.solve(matrix, right[0] - left[0])[0] * left[1]
    gap = float(np.linalg.norm(patch[20:61] - intersection, axis=1).min())
    center_gap = float(np.linalg.norm(intersection))
    circle_residual, radius = _circle_residual(patch[28:53])
    return {"turn_degrees": turn, "line_residual": residual,
            "flank_turn_degrees": flank_turn, "turn_concentration": concentration,
            "intersection_gap": gap, "center_gap": center_gap,
            "broad_arc_residual": broad_arc_residual,
            "circle_residual": circle_residual,
            "circle_radius": radius}


def _state(metrics):
    if (metrics["turn_concentration"] >= CORNER_POLICY["square_concentration"]
            and metrics["intersection_gap"] <= CORNER_POLICY["square_gap"]):
        return "square"
    # Rounded requires its own positive arc evidence; failure of square is
    # never sufficient. The gap also demands visible departure from an apex.
    if (metrics["turn_concentration"] < CORNER_POLICY["rounded_concentration"]
            and metrics["intersection_gap"] >= CORNER_POLICY["rounded_gap"]
            and metrics["circle_residual"] <= CORNER_POLICY["max_circle_residual"]
            and CORNER_POLICY["min_circle_radius"] <= metrics["circle_radius"]
            <= CORNER_POLICY["max_circle_radius"]):
        return "rounded"
    return "unknown"


def _candidates(gray, threshold):
    mask = (gray < threshold).astype(np.uint8)
    skeleton, widths, bad_points = _topology(mask)
    if not len(skeleton):
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        if len(contour) < 24:
            continue
        raw = _resample(contour)
        orientation = np.sign(cv2.contourArea(contour, oriented=True))
        for index in range(0, len(raw), 2):
            local = _adjacent_width(raw[index], skeleton, widths, bad_points)
            if local is None:
                continue
            width, variation = local
            if len(raw) < 6 * width:
                continue
            metrics = _measure(_smooth(raw, width), index, width, orientation)
            if metrics is None:
                continue
            metrics["width_variation"] = variation
            candidates.append({"point": raw[index], "width": width,
                               "state": _state(metrics), "metrics": metrics})
    # Prefer the apex/arc center; merge nearby samples of that one feature.
    candidates.sort(key=lambda row: (-round(row["metrics"]["turn_concentration"], 1),
                                     row["metrics"]["center_gap"]))
    selected = []
    for row in candidates:
        if any(np.linalg.norm(row["point"] - other["point"])
               < CORNER_POLICY["merge_radius"] * max(row["width"], other["width"])
               for other in selected):
            continue
        selected.append(row)
    return selected[:CORNER_POLICY["max_features"]]


def extract_corners(gray, pixels):
    """Extract at most 16 JSON-serializable outer connection corner records.

    Args:
        gray: 128x128 grayscale array in [0, 255], black ink on white; ink
            maximum extent is normalized to 104. Supply the source image or
            a template at the same effective resolution, not a sharpened mask.
        pixels: Positive effective native maximum ink extent, before upscaling.

    Returns:
        List of corner records. Width is measured in normalized image pixels;
        native_width equals width * pixels / 104. Unknown retains ambiguous
        candidates; an empty list means no supported candidate, not roundness.
        square_score is evidence strength, not a calibrated probability, and
        is zero unless all resolution and stability gates approve square.
    """
    gray = np.asarray(gray)
    if (gray.shape != (128, 128) or not np.issubdtype(gray.dtype, np.number)
            or not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 255):
        raise ValueError("gray must be a finite 128x128 grayscale array in [0, 255]")
    if not np.isfinite(pixels) or pixels <= 0:
        raise ValueError("pixels must be a positive finite native ink extent")
    gray = gray.astype(np.uint8)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold = float(np.clip(otsu, 96, 160))
    delta = CORNER_POLICY["threshold_delta"]
    thresholds = [threshold - delta, threshold, threshold + delta]
    candidates = [_candidates(gray, value) for value in thresholds]
    features = []
    for row in candidates[1]:
        matches = []
        for rows in candidates:
            match = min(rows, key=lambda other: np.linalg.norm(other["point"] - row["point"]), default=None)
            if match is not None and np.linalg.norm(match["point"] - row["point"]) > CORNER_POLICY["max_point_shift"] * row["width"]:
                match = None
            matches.append(match)
        states = [match["state"] if match else "missing" for match in matches]
        variation = (float(np.ptp([match["width"] for match in matches]) / row["width"])
                     if all(match is not None for match in matches) else None)
        stable = len(set(states)) == 1 and variation is not None and variation <= CORNER_POLICY["max_width_variation"]
        native = float(row["width"] * pixels / 104)
        state = row["state"] if stable else "unknown"
        reason = "stable_outer_corner_fit" if state != "unknown" else "ambiguous_outer_corner_geometry"
        if not stable:
            reason = "threshold_unstable"
        if native < CORNER_POLICY["min_native_width"]:
            state, reason = "unknown", "native_stroke_too_thin"
        metrics = {**row["metrics"], "thresholds": thresholds,
                   "threshold_states": states, "threshold_width_variation": variation}
        score = 0.0
        if state == "square":
            score = float(np.clip(row["metrics"]["turn_concentration"]
                                  * (1 - row["metrics"]["line_residual"]
                                     / CORNER_POLICY["score_residual_scale"]), 0, 1))
        features.append({"kind": "corner", "point": row["point"].tolist(),
                         "width": float(row["width"]), "native_width": native,
                         "state": state, "square_score": score,
                         "metrics": metrics, "reason": reason})
    return features
