"""Free-cap geometry relative to the adjacent shaft, not handwriting order."""
import cv2
import numpy as np

from mikan_font_probe.glyph_geometry import thin

TERMINAL_POLICY = {"min_native_width": 5., "flat_residual": .07,
                   "round_residual": .10, "model_margin": .045,
                   "threshold_delta": 20, "max_features": 16}


def _trace(skeleton, start, limit):
    visited, path = {start}, [start]
    for _ in range(limit):
        x, y = path[-1]
        neighbors = [(x+dx, y+dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if (dx or dy) and 0 <= x+dx < skeleton.shape[1]
                     and 0 <= y+dy < skeleton.shape[0] and skeleton[y+dy, x+dx]
                     and (x+dx, y+dy) not in visited]
        if len(neighbors) != 1:
            break  # Branches, loops and uncertain connections are not caps.
        visited.add(neighbors[0])
        path.append(neighbors[0])
    return np.asarray(path, dtype=float)


def _crossing(profile, positions, threshold, last=True):
    ids = np.flatnonzero(profile < threshold)
    if not len(ids):
        return None
    j = int(ids[-1] if last else ids[0])
    k = j+1 if last else j-1
    if not 0 <= k < len(profile):
        return None
    denominator = float(profile[k])-float(profile[j])
    fraction = (threshold-float(profile[j]))/denominator if denominator else 0.
    return float(positions[j]+fraction*(positions[k]-positions[j]))


def _cap_profile(gray, point, outward, width, threshold):
    radius = width/2
    side = np.array([-outward[1], outward[0]])
    xs = np.linspace(-3*radius, 3*radius, 121)
    ys = np.linspace(-1.5*radius, 1.5*radius, 81)
    grid = point+xs[None, :, None]*outward+ys[:, None, None]*side
    patch = cv2.remap(gray, grid[:, :, 0].astype(np.float32),
                      grid[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    spans, centers = [], []
    for x in (-1.8*radius, -1.3*radius, -.8*radius):
        column = patch[:, np.argmin(abs(xs-x))]
        lo, hi = _crossing(column, ys, threshold, False), _crossing(column, ys, threshold)
        if lo is None or hi is None or hi-lo < radius:
            return None
        spans.append(hi-lo)
        centers.append((hi+lo)/2)
    if np.std(spans)/np.mean(spans) > .15 or np.ptp(centers)/radius > .2:
        return None  # Shaft is bent, tapered, touching or poorly oriented.
    actual_radius, center = np.median(spans)/2, np.median(centers)
    fractions = np.linspace(-.8, .8, 9)
    fronts = []
    for fraction in fractions:
        y = center+actual_radius*fraction
        row = patch[np.argmin(abs(ys-y))]
        front = _crossing(row, xs, threshold)
        if front is None:
            return None
        fronts.append(front/actual_radius)
    front = np.asarray(fronts)
    design = np.column_stack([np.ones(9), fractions])
    flat_fit = design@np.linalg.lstsq(design, front, rcond=None)[0]
    circular = np.sqrt(1-fractions**2)
    round_fit = design@np.linalg.lstsq(design, front-circular, rcond=None)[0]+circular
    flat = float(np.sqrt(np.mean((front-flat_fit)**2)))
    rounded = float(np.sqrt(np.mean((front-round_fit)**2)))
    return {"flat_residual": flat, "round_residual": rounded,
            "crown": float(front[4]-(front[0]+front[-1])/2),
            "shaft_width": float(2*actual_radius),
            "shaft_width_variation": float(np.std(spans)/np.mean(spans))}


def _state(metrics):
    if metrics is None:
        return "unknown"
    flat, rounded = metrics["flat_residual"], metrics["round_residual"]
    if flat < TERMINAL_POLICY["flat_residual"] and rounded-flat > TERMINAL_POLICY["model_margin"]:
        return "square"
    if rounded < TERMINAL_POLICY["round_residual"] and flat-rounded > TERMINAL_POLICY["model_margin"]:
        return "rounded"
    return "unknown"


def extract_terminals(gray, pixels):
    """Return stable planar-cap evidence; absent square evidence stays unknown.

    Input is 128-square grayscale, dark ink, max ink extent 104. `pixels`
    is the effective native extent, not the displayed/upscaled dimensions.
    """
    otsu, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    skeleton = thin(mask)
    degree = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16))
    ys, xs = np.nonzero((skeleton == 1) & (degree == 2))
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    features = []
    for x, y in zip(xs, ys):
        initial = max(2., 2*float(dt[y, x]))
        path = _trace(skeleton, (int(x), int(y)), int(3*initial)+2)
        if len(path) < max(5, 1.5*initial):
            continue
        shaft = path[int(.5*initial):int(2*initial)+1]
        width = float(2*np.median(dt[shaft[:, 1].astype(int), shaft[:, 0].astype(int)]))
        direction = path[0]-path[min(len(path)-1, int(1.5*width))]
        direction /= max(1e-6, np.linalg.norm(direction))
        # Otsu can be 0 for hard binary input; use the contrast midpoint there.
        threshold = float(np.clip(otsu, 96, 160))
        delta = TERMINAL_POLICY["threshold_delta"]
        fits = [_cap_profile(gray, path[0], direction, width, t)
                for t in (threshold-delta, threshold, threshold+delta)]
        mid = fits[1]
        measured_width = mid["shaft_width"] if mid else width
        native = measured_width*pixels/104
        states = [_state(f) for f in fits]
        state = states[1] if len(set(states)) == 1 else "unknown"
        reason = "stable_cap_fit" if state != "unknown" else "unstable_or_non_cap_geometry"
        if native < TERMINAL_POLICY["min_native_width"]:
            state, reason = "unknown", "native_shaft_too_thin"
        features.append({"kind": "terminal", "point": [int(x), int(y)],
                         "direction": direction.tolist(), "width": measured_width,
                         "native_width": float(native), "state": state,
                         "square_score": float(np.clip(1-mid["flat_residual"]/.15, 0, 1)) if state == "square" else 0.,
                         "metrics": {**(mid or {}), "threshold_states": states}, "reason": reason})
    return features[:TERMINAL_POLICY["max_features"]]
