"""v6 separated, shaft-relative terminal/corner evidence and correspondence."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from glyph_geometry import effective_resolution
from match_font_baseline import match, normalize
from process_regions import imread
from stroke_terminal_features import extract_terminals

FEATURE_POLICY = {"max_point_distance": 12., "min_direction_dot": .8,
                  "square_weight": 2., "round_weight": 1.,
                  "min_distinct_votes": 3, "agreement": .8,
                  "requires_square_anchor": True}


def gray_canvas(image, pixels=None):
    """Preserve grayscale; font templates first rasterize at source resolution."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if pixels is None:
        model = effective_resolution(image)
        pixels = model["effective"]
        if model["repeat_scale"] > 1:
            scale = model["repeat_scale"]
            gray = cv2.resize(gray, (max(1, gray.shape[1]//scale), max(1, gray.shape[0]//scale)),
                              interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Blank glyph")
    tight = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    ratio = min(1., pixels/max(tight.shape))
    if ratio < 1:
        tight = cv2.resize(tight, (max(1, round(tight.shape[1]*ratio)),
                                  max(1, round(tight.shape[0]*ratio))), interpolation=cv2.INTER_AREA)
    scale = 104/max(tight.shape)
    resized = cv2.resize(tight, (max(1, round(tight.shape[1]*scale)),
                                max(1, round(tight.shape[0]*scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    result = np.full((128, 128), 255, np.uint8)
    h, w = resized.shape
    result[(128-h)//2:(128-h)//2+h, (128-w)//2:(128-w)//2+w] = resized
    return result, int(pixels)


def extract(gray, pixels):
    from stroke_corner_features import extract_corners
    return extract_terminals(gray, pixels)+extract_corners(gray, pixels)


def transform_features(features, transform):
    output = copy.deepcopy(features)
    for feature in output:
        point = np.asarray(feature["point"], float)
        feature["point"] = ((point-64)*transform["scale"]+64
                            +2*np.array([transform["dx"], transform["dy"]])).tolist()
    return output


def correspond(a, b):
    """Mutual nearest same-kind matches; never select by desired shape class."""
    if not a or not b:
        return []
    distance = np.full((len(a), len(b)), np.inf)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            if left["kind"] != right["kind"]:
                continue
            if left["kind"] == "terminal" and np.dot(left["direction"], right["direction"]) < FEATURE_POLICY["min_direction_dot"]:
                continue
            distance[i, j] = np.linalg.norm(np.asarray(left["point"])-right["point"])
    pairs = []
    for i in range(len(a)):
        j = int(np.argmin(distance[i]))
        if distance[i, j] <= FEATURE_POLICY["max_point_distance"] and np.argmin(distance[:, j]) == i:
            pairs.append((i, j))
    return pairs


def pair_features(a, b):
    _, _, transform = match(normalize(a["gray"]), normalize(b["gray"]))
    aligned = transform_features(b["features"], transform)
    pairs = []
    for i, j in correspond(a["features"], aligned):
        left, right = a["features"][i], b["features"][j]
        distinct = {left["state"], right["state"]} == {"square", "rounded"}
        pairs.append({"index_a": i, "index_b": j, "kind": left["kind"],
                      "state_a": left["state"], "state_b": right["state"],
                      "distinct": distinct})
    return {"pairs": pairs, "utility": sum(p["distinct"] for p in pairs)}


def compare_features(source, a, b, pair=None):
    pair = pair or pair_features(a, b)
    alignments = []
    for reference in (a, b):
        _, _, transform = match(normalize(source["gray"]), normalize(reference["gray"]))
        aligned = transform_features(reference["features"], transform)
        alignments.append(dict((j, i) for i, j in correspond(source["features"], aligned)))
    evidence = []
    for row in pair["pairs"]:
        if not row["distinct"]:
            continue
        i = alignments[0].get(row["index_a"])
        j = alignments[1].get(row["index_b"])
        observed = source["features"][i] if i is not None and i == j else None
        state = observed["state"] if observed else "unknown"
        support = None
        if state in ("square", "rounded"):
            support = "a" if state == row["state_a"] else "b"
        evidence.append({**row, "source_index": i if observed else None,
                         "source_state": state, "support": support,
                         "native_width": observed["native_width"] if observed else None,
                         "reason": observed["reason"] if observed else "no_shared_source_correspondence"})
    square_a = sum(r["support"] == "a" and r["source_state"] == "square" for r in evidence)
    square_b = sum(r["support"] == "b" and r["source_state"] == "square" for r in evidence)
    rounded_a = sum(r["support"] == "a" and r["source_state"] == "rounded" for r in evidence)
    rounded_b = sum(r["support"] == "b" and r["source_state"] == "rounded" for r in evidence)
    da = FEATURE_POLICY["square_weight"]*square_a+FEATURE_POLICY["round_weight"]*rounded_a
    db = FEATURE_POLICY["square_weight"]*square_b+FEATURE_POLICY["round_weight"]*rounded_b
    vote = 0
    if da+db and max(da, db)/(da+db) >= FEATURE_POLICY["agreement"]:
        vote = 1 if da > db else -1
    anchor = square_a > 0 if vote > 0 else square_b > 0 if vote < 0 else False
    return {"vote": vote, "square_anchor": anchor, "features": evidence,
            "square_support_a": square_a, "square_support_b": square_b,
            "round_support_a": rounded_a, "round_support_b": rounded_b,
            "template_utility": pair["utility"]}


def decide(rows):
    effective = [r for r in rows if r["vote"]]
    if len(effective) < FEATURE_POLICY["min_distinct_votes"]:
        return 0, "insufficient_distinct_characters"
    positives = sum(r["vote"] > 0 for r in effective)/len(effective)
    agreement = FEATURE_POLICY["agreement"]
    winner = 1 if positives >= agreement else -1 if 1-positives >= agreement else 0
    if not winner:
        return 0, "conflicting_features"
    if FEATURE_POLICY["requires_square_anchor"] and not any(
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
