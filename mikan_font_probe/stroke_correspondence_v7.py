"""Structure-aware correspondence for free caps and closed box corners.

The aspect transform moves landmark coordinates only. It never deforms
the measured local cap/corner shape or infers a handwriting stroke order.
"""
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from mikan_font_probe.process_regions import imread
from mikan_font_probe.stroke_feature_matcher import gray_canvas
from mikan_font_probe.stroke_terminal_features import extract_terminals
from mikan_font_probe.stroke_corner_features import extract_corners
from mikan_font_probe.glyph_geometry import thin

CORRESPONDENCE_POLICY = {"max_aspect_scale": 1.35, "min_aspect_scale": .74,
                         "max_point_distance": 12., "max_box_center_distance": 14.,
                         "max_segment_center_distance": 18.,
                         "min_direction_dot": .8, "min_distinct_votes": 3,
                         "agreement": .8, "square_weight": 2.,
                         "round_weight": 1., "requires_square_anchor": True}


def extract(gray, pixels):
    from mikan_font_probe.closed_box_features import extract_boxes
    terminals = extract_terminals(gray, pixels)
    _segment_context(gray, terminals)
    return terminals+extract_corners(gray, pixels)+extract_boxes(gray, pixels)


def _segment_context(gray, terminals):
    """Attach isolated skeleton-component location as a stroke proxy."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    skeleton = thin(mask).astype(np.uint8)
    count, labels = cv2.connectedComponents(skeleton, connectivity=8)
    degree = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16))
    for terminal in terminals:
        x, y = terminal["point"]
        component = labels[y, x]
        if component == 0:
            continue
        region = labels == component
        if (int(np.count_nonzero(region & (degree == 2))) != 2
                or np.any(region & (degree >= 5))):
            continue
        ys, xs = np.nonzero(region)
        terminal["segment_center"] = [float((xs.min()+xs.max())/2),
                                      float((ys.min()+ys.max())/2)]
        terminal["segment_span"] = float(max(np.ptp(xs), np.ptp(ys)))


def ink_box(gray):
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    y, x = np.nonzero(ink)
    if not len(x):
        raise ValueError("Blank image")
    return np.array([x.min(), y.min(), x.max(), y.max()], float)


def mapping(source_gray, target_gray):
    src, ref = ink_box(source_gray), ink_box(target_gray)
    denominator = np.maximum(ref[2:]-ref[:2], 1)
    scale = (src[2:]-src[:2])/denominator
    if np.any(scale < CORRESPONDENCE_POLICY["min_aspect_scale"]) or np.any(
            scale > CORRESPONDENCE_POLICY["max_aspect_scale"]):
        return None
    return scale, src[:2]-scale*ref[:2]


def mapped(features, transform):
    scale, shift = transform
    rows = copy.deepcopy(features)
    for row in rows:
        row["point"] = (np.asarray(row["point"])*scale+shift).tolist()
        if "box_center" in row:
            row["box_center"] = (np.asarray(row["box_center"])*scale+shift).tolist()
        if "segment_center" in row:
            row["segment_center"] = (np.asarray(row["segment_center"])*scale+shift).tolist()
            row["segment_span"] *= float(max(scale))
        if "direction" in row:
            vector = np.asarray(row["direction"])*scale
            row["direction"] = (vector/max(np.linalg.norm(vector), 1e-8)).tolist()
    return rows


def correspondence(source, target):
    transform = mapping(source["gray"], target["gray"])
    if transform is None:
        return [], "aspect_outside_bounded_range"
    aligned = mapped(target["features"], transform)
    distances = np.full((len(source["features"]), len(aligned)), np.inf)
    for i, left in enumerate(source["features"]):
        for j, right in enumerate(aligned):
            if left["kind"] != right["kind"]:
                continue
            if left["kind"] == "terminal":
                dot = float(np.dot(left["direction"], right["direction"]))
                if dot < CORRESPONDENCE_POLICY["min_direction_dot"]:
                    continue
                if "segment_center" in left and "segment_center" in right:
                    segment_distance = np.linalg.norm(np.asarray(left["segment_center"])-right["segment_center"])
                    limit = max(CORRESPONDENCE_POLICY["max_segment_center_distance"],
                                .35*max(left["segment_span"], right["segment_span"]))
                    if segment_distance > limit:
                        continue
            if left["kind"] == "box_corner":
                if left["role"] != right["role"]:
                    continue
                center_distance = np.linalg.norm(np.asarray(left["box_center"])-right["box_center"])
                if center_distance > CORRESPONDENCE_POLICY["max_box_center_distance"]:
                    continue
            point_distance = np.linalg.norm(np.asarray(left["point"])-right["point"])
            if point_distance <= CORRESPONDENCE_POLICY["max_point_distance"]:
                distances[i, j] = point_distance
    pairs = []
    for i in range(len(source["features"])):
        if len(aligned) == 0:
            break
        j = int(np.argmin(distances[i]))
        if np.isfinite(distances[i, j]) and int(np.argmin(distances[:, j])) == i:
            pairs.append((i, j))
    return pairs, "matched" if pairs else "no_unique_local_correspondence"


def pair_features(a, b):
    matches, reason = correspondence(a, b)
    rows = []
    for i, j in matches:
        left, right = a["features"][i], b["features"][j]
        rows.append({"index_a": i, "index_b": j, "kind": left["kind"],
                     "role": left.get("role"), "state_a": left["state"],
                     "state_b": right["state"],
                     "distinct": {left["state"], right["state"]} == {"square", "rounded"}})
    return {"pairs": rows, "utility": sum(r["distinct"] for r in rows), "reason": reason}


def _opening(feature):
    x0, y0, x1, y1 = feature["box_bbox"]
    width, height = x1-x0+1, y1-y0+1
    stroke = feature["width"]
    return [float(width/(width+2*stroke)), float(height/(height+2*stroke))]


def compare_features(source, a, b, pair=None):
    if pair is None:
        pair = pair_features(a, b)
    la, reason_a = correspondence(source, a)
    lb, reason_b = correspondence(source, b)
    lookup_a = {j: i for i, j in la}
    lookup_b = {j: i for i, j in lb}
    evidence = []
    boxes = []
    for row in pair["pairs"]:
        if row["kind"] == "box_corner" and row["role"] == "tl":
            ia, ib = lookup_a.get(row["index_a"]), lookup_b.get(row["index_b"])
            if ia is not None and ia == ib:
                observed_box = source["features"][ia]
                reference_a = a["features"][row["index_a"]]
                reference_b = b["features"][row["index_b"]]
                boxes.append({"source_index": ia, "source_opening": _opening(observed_box),
                              "opening_a": _opening(reference_a), "opening_b": _opening(reference_b),
                              "source_corner_state": observed_box["state"],
                              "reason": "shape_diagnostic_only"})
        if not row["distinct"]:
            continue
        i, j = lookup_a.get(row["index_a"]), lookup_b.get(row["index_b"])
        observed = source["features"][i] if i is not None and i == j else None
        state = observed["state"] if observed else "unknown"
        support = None
        if state in ("square", "rounded"):
            support = "a" if state == row["state_a"] else "b"
        evidence.append({**row, "source_index": i if observed else None,
                         "source_state": state, "support": support,
                         "native_width": observed["native_width"] if observed else None,
                         "reason": observed["reason"] if observed else "no_shared_source_correspondence"})
    votes = {"square_support_a": sum(r["support"] == "a" and r["source_state"] == "square" for r in evidence),
             "square_support_b": sum(r["support"] == "b" and r["source_state"] == "square" for r in evidence),
             "round_support_a": sum(r["support"] == "a" and r["source_state"] == "rounded" for r in evidence),
             "round_support_b": sum(r["support"] == "b" and r["source_state"] == "rounded" for r in evidence)}
    sa = CORRESPONDENCE_POLICY["square_weight"]*votes["square_support_a"]+votes["round_support_a"]
    sb = CORRESPONDENCE_POLICY["square_weight"]*votes["square_support_b"]+votes["round_support_b"]
    vote = 0
    if sa+sb and max(sa, sb)/(sa+sb) >= CORRESPONDENCE_POLICY["agreement"]:
        vote = 1 if sa > sb else -1
    anchor = votes["square_support_a"] > 0 if vote > 0 else votes["square_support_b"] > 0 if vote < 0 else False
    return {"vote": vote, "square_anchor": anchor, "features": evidence,
            "box_geometry": boxes,
            "template_utility": pair["utility"], "alignment_a": reason_a,
            "alignment_b": reason_b, **votes}


def decide(rows):
    effective = [r for r in rows if r["vote"]]
    if len(effective) < CORRESPONDENCE_POLICY["min_distinct_votes"]:
        return 0, "insufficient_distinct_characters"
    positives = sum(r["vote"] > 0 for r in effective)/len(effective)
    winner = 1 if positives >= CORRESPONDENCE_POLICY["agreement"] else -1 if (
        1-positives >= CORRESPONDENCE_POLICY["agreement"]) else 0
    if not winner:
        return 0, "conflicting_features"
    if CORRESPONDENCE_POLICY["requires_square_anchor"] and not any(
            r["square_anchor"] and r["vote"] == winner for r in effective):
        return 0, "round_only_no_square_anchor"
    return winner, "separated_features_agree"


class FeatureReferences:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.fonts = json.loads((self.directory/"catalog.json").read_text())["fonts"]
        self.cache = {}

    def reference(self, font, char, pixels):
        key = (font["id"], char, pixels)
        if key not in self.cache:
            gray, _ = gray_canvas(imread(self.directory/font["templates"][char]), pixels)
            self.cache[key] = {"gray": gray, "features": extract(gray, pixels)}
        return self.cache[key]
