"""Conservative corner evidence from enclosed white regions in dark strokes.

The four records of a hole share its center and bounding box. A circular hole
is not a box: straight side support is required before corner classification.
No glyph identity or reference answer is used here.
"""

import cv2
import numpy as np


BOX_POLICY = {
    "threshold_delta": 20,
    "min_hole_area": 25,
    "min_hole_side": 10,
    "min_native_width": 4.0,
    "min_native_side": 8.0,
    "min_native_flank": 4.0,
    "min_native_arc_radius": 2.5,
    "max_side_residual": 0.12,
    "square_gap": 0.20,
    "rounded_gap": 0.27,
    "max_arc_residual": 0.10,
    "max_point_shift": 0.35,
    "max_width_variation": 0.30,
}


def _frame(contour):
    corners = cv2.boxPoints(cv2.minAreaRect(contour)).astype(float)
    edges = [corners[1] - corners[0], corners[2] - corners[1]]
    horizontal = max(edges, key=lambda edge: abs(edge[0]) / max(np.linalg.norm(edge), 1))
    u = horizontal / np.linalg.norm(horizontal)
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    points = contour[:, 0, :].astype(float)
    projected = np.column_stack((points @ u, points @ v))
    low, high = projected.min(axis=0), projected.max(axis=0)
    return points, projected - low, high - low, low, u, v


def _side_support(local, sizes, width):
    results = {}
    for name, axis, extreme in (("top", 0, 0), ("bottom", 0, 1),
                                ("left", 1, 0), ("right", 1, 1)):
        length, other = sizes[axis], sizes[1 - axis]
        along, across = local[:, axis], local[:, 1 - axis]
        middle = (along >= 0.23 * length) & (along <= 0.77 * length)
        target = other if extreme else 0.0
        near = middle & (np.abs(across - target) <= min(1.5, 0.10 * other))
        if near.sum() < 4:
            return None
        span = float(np.ptp(along[near]))
        if span < max(3.0, 0.7 * width):
            return None
        fit = np.polyfit(along[near], across[near], 1)
        residual = float(np.sqrt(np.mean((np.polyval(fit, along[near]) - across[near]) ** 2)))
        if residual > BOX_POLICY["max_side_residual"] * width:
            return None
        results[name] = (span, residual)
    return results


def _stroke_width(mask, points, local, sizes, u, v):
    widths = []
    for axis, extreme, normal in ((0, 0, -v), (0, 1, v),
                                  (1, 0, -u), (1, 1, u)):
        length, other = sizes[axis], sizes[1 - axis]
        along, across = local[:, axis], local[:, 1 - axis]
        central = np.abs(along - length / 2) <= 0.12 * length
        target = other if extreme else 0.0
        indices = np.flatnonzero(central)
        if not len(indices):
            continue
        index = indices[np.argmin(np.abs(across[indices] - target))]
        start = points[index]
        samples = np.rint(start + np.arange(1, 33)[:, None] * normal).astype(int)
        valid = ((samples >= 0) & (samples < 128)).all(axis=1)
        inside = np.zeros(len(samples), bool)
        inside[valid] = mask[samples[valid, 1], samples[valid, 0]] != 0
        if not inside[0]:
            continue
        end = np.flatnonzero(~inside)
        if len(end):
            widths.append(float(end[0]))
    return float(np.median(widths)) if len(widths) >= 2 else None


def _fallback_width(mask, contour):
    region = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(region, [contour], -1, 1, -1)
    near = cv2.dilate(region, np.ones((17, 17), np.uint8)) & mask
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 5)[near != 0]
    return max(1.0, float(2 * np.percentile(distances, 80))) if len(distances) else 1.0


def _is_white_hole(index, hierarchy):
    depth = 0
    parent = hierarchy[index, 3]
    while parent >= 0:
        depth += 1
        parent = hierarchy[parent, 3]
    return depth % 2 == 1


def _circle_fit(points):
    centered = points - points.mean(axis=0)
    design = np.column_stack((2 * centered, np.ones(len(points))))
    solution = np.linalg.lstsq(design, np.sum(centered ** 2, axis=1), rcond=None)[0]
    center = solution[:2] + points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = float(np.mean(distances))
    residual = float(np.sqrt(np.mean((distances - radius) ** 2)))
    angles = np.unwrap(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    return center, radius, residual, float(np.degrees(np.ptp(angles)))


def _corner(local, sizes, width, role):
    x = local[:, 0] if role in ("tl", "bl") else sizes[0] - local[:, 0]
    y = local[:, 1] if role in ("tl", "tr") else sizes[1] - local[:, 1]
    reach = min(0.48 * min(sizes), 3.0 * width)
    nearby = (x <= reach) & (y <= reach)
    metrics = {"apex_gap": None, "arc_radius": None,
               "arc_residual": None, "arc_degrees": None}
    if nearby.sum() < 4:
        return "unknown", "corner_boundary_missing", metrics
    gap = float(np.min(np.hypot(x[nearby], y[nearby])) / width)
    metrics["apex_gap"] = gap
    if gap <= BOX_POLICY["square_gap"]:
        return "square", "stable_box_apex", metrics
    if gap < BOX_POLICY["rounded_gap"]:
        return "unknown", "ambiguous_corner_gap", metrics
    arc = nearby & (x > 0.10 * width) & (y > 0.10 * width)
    if arc.sum() < 6:
        return "unknown", "arc_not_supported", metrics
    center, radius, residual, degrees = _circle_fit(np.column_stack((x[arc], y[arc])))
    metrics.update(arc_radius=radius, arc_residual=residual, arc_degrees=degrees)
    if (0.45 * width <= radius <= 2.5 * width
            and residual <= BOX_POLICY["max_arc_residual"] * width
            and np.linalg.norm(center - radius) <= 0.30 * width
            and 45 <= degrees <= 135):
        return "rounded", "stable_box_arc", metrics
    return "unknown", "arc_not_supported", metrics


def _holes(gray, threshold):
    mask = (gray < threshold).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    holes = []
    tree = hierarchy[0]
    for index, contour in enumerate(contours):
        if not _is_white_hole(index, tree) or cv2.contourArea(contour) < BOX_POLICY["min_hole_area"]:
            continue
        points, local, sizes, origin, u, v = _frame(contour)
        if min(sizes) < BOX_POLICY["min_hole_side"]:
            continue
        measured_width = _stroke_width(mask, points, local, sizes, u, v)
        width = measured_width if measured_width is not None else _fallback_width(mask, contour)
        sides = _side_support(local, sizes, width)
        bbox = cv2.boundingRect(contour)
        center = np.mean(points, axis=0)
        records = []
        for role, corner, adjacent in (
                ("tl", (0, 0), ("top", "left")),
                ("tr", (sizes[0], 0), ("top", "right")),
                ("br", (sizes[0], sizes[1]), ("bottom", "right")),
                ("bl", (0, sizes[1]), ("bottom", "left"))):
            if sides is None or measured_width is None:
                state, reason = "unknown", "straight_side_insufficient"
                metrics = {"apex_gap": None, "arc_radius": None,
                           "arc_residual": None, "arc_degrees": None,
                           "side_residual": None, "flank_length": 0.0}
            else:
                state, reason, metrics = _corner(local, sizes, width, role)
                metrics.update(side_residual=max(sides[name][1] for name in adjacent) / width,
                               flank_length=min(sides[name][0] for name in adjacent))
            point = (origin + np.asarray(corner) @ np.stack((u, v))).tolist()
            records.append({"kind": "box_corner", "point": point,
                            "role": role, "box_center": center.tolist(),
                            "box_bbox": [int(bbox[0]), int(bbox[1]),
                                         int(bbox[0] + bbox[2] - 1), int(bbox[1] + bbox[3] - 1)],
                            "width": width, "state": state, "reason": reason,
                            "metrics": metrics})
        holes.append({"center": center, "sizes": sizes, "records": records})
    return holes


def extract_boxes(gray, pixels):
    """Return four JSON-serializable corner records per supported closed box.

    Args:
        gray: 128x128 gray image, black ink on white, maximum ink extent 104.
        pixels: Positive native effective maximum ink extent before resizing.

    Unknown means the hole was located but shape evidence was insufficient.
    An empty list means no supported closed box was found.
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
    thresholds = [threshold - BOX_POLICY["threshold_delta"], threshold,
                  threshold + BOX_POLICY["threshold_delta"]]
    groups = [_holes(gray, value) for value in thresholds]
    features = []
    for hole in groups[1]:
        matched = []
        for group in groups:
            nearest = min(group, key=lambda item: np.linalg.norm(item["center"] - hole["center"]),
                          default=None)
            if (nearest is not None and
                    np.linalg.norm(nearest["center"] - hole["center"]) >
                    max(2.0, 0.15 * min(hole["sizes"]))):
                nearest = None
            matched.append(nearest)
        for row in hole["records"]:
            peers = [next((item for item in match["records"] if item["role"] == row["role"]), None)
                     if match is not None else None for match in matched]
            states = [peer["state"] if peer else "missing" for peer in peers]
            widths = [peer["width"] for peer in peers if peer]
            variation = float(np.ptp(widths) / row["width"]) if len(widths) == 3 else None
            shift = (max(np.linalg.norm(np.asarray(peer["point"]) - row["point"])
                         for peer in peers) if all(peer is not None for peer in peers) else None)
            stable = (len(set(states)) == 1 and variation is not None
                      and variation <= BOX_POLICY["max_width_variation"]
                      and shift <= BOX_POLICY["max_point_shift"] * row["width"])
            native_width = row["width"] * pixels / 104
            native_side = min(hole["sizes"]) * pixels / 104
            native_flank = row["metrics"]["flank_length"] * pixels / 104
            native_arc = (row["metrics"]["arc_radius"] or 0.0) * pixels / 104
            state, reason = row["state"], row["reason"]
            if not stable:
                state, reason = "unknown", "threshold_unstable"
            if state != "unknown" and (native_width < BOX_POLICY["min_native_width"]
                    or native_side < BOX_POLICY["min_native_side"]
                    or native_flank < BOX_POLICY["min_native_flank"]
                    or (state == "rounded" and native_arc < BOX_POLICY["min_native_arc_radius"])):
                state, reason = "unknown", "native_information_insufficient"
            metrics = {**row["metrics"], "thresholds": thresholds,
                       "threshold_states": states, "threshold_width_variation": variation,
                       "threshold_point_shift": shift, "native_hole_side": native_side,
                       "native_flank_length": native_flank, "native_arc_radius": native_arc}
            score = (float(np.clip(1 - metrics["apex_gap"] / BOX_POLICY["square_gap"], 0, 1))
                     if state == "square" else 0.0)
            features.append({**row, "native_width": float(native_width),
                             "state": state, "reason": reason,
                             "square_score": score, "metrics": metrics})
    return features
