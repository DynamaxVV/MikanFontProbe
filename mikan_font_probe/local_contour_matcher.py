"""Experimental v5 local outline evidence; no answer labels or font bonuses."""
import cv2
import numpy as np

from mikan_font_probe.glyph_geometry import thin
from mikan_font_probe.match_font_baseline import match

POLICY = {"min_pixels": 32, "patch_radius": 5, "sdf_clip": 3.,
          "width_offset_limit": .75, "noise_multiplier": 2.,
          "min_separation": .04, "min_vote_margin": .02,
          "min_votes": 3, "agreement": .8}


def landmarks(ink):
    """Skeleton endpoints plus image corner candidates, not semantic corners."""
    skeleton = thin(ink)
    degree = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16))
    ys, xs = np.nonzero((skeleton == 1) & (degree == 2))
    points = [(int(x), int(y)) for x, y in zip(xs, ys)]
    corners = cv2.goodFeaturesToTrack(cv2.GaussianBlur(ink, (3, 3), .7),
                                     maxCorners=24, qualityLevel=.08,
                                     minDistance=5, blockSize=5)
    if corners is not None:
        points.extend(tuple(map(int, p)) for p in corners[:, 0])
    return points


def region_mask(*inks):
    mask = np.zeros(inks[0].shape, np.uint8)
    for ink in inks:
        for point in landmarks(ink):
            cv2.circle(mask, point, POLICY["patch_radius"], 1, -1)
    return mask.astype(bool)


def sdf(ink):
    mask = (ink >= 128).astype(np.uint8)
    return np.clip(cv2.distanceTransform(mask, cv2.DIST_L2, 3)
                   - cv2.distanceTransform(1-mask, cv2.DIST_L2, 3),
                   -POLICY["sdf_clip"], POLICY["sdf_clip"])


def local_loss(a, b, mask):
    if not mask.any():
        return 1.
    delta = (sdf(a)-sdf(b))[mask]
    offset = np.clip(np.median(delta), -POLICY["width_offset_limit"],
                     POLICY["width_offset_limit"])
    return float(np.mean(np.abs(delta-offset))/(2*POLICY["sdf_clip"]))


def nuisance_floor(ink, mask):
    # Deliberately small subpixel raster perturbations; not a real-scan model.
    values = []
    for dx, dy in ((.35, 0), (-.35, 0), (0, .35), (0, -.35)):
        shifted = cv2.warpAffine(ink, np.float32([[1, 0, dx], [0, 1, dy]]),
                                 (64, 64), flags=cv2.INTER_LINEAR)
        values.append(local_loss(ink, (shifted >= 160).astype(np.uint8)*255, mask))
    return max(values)


def template_evidence(a, b, pixels):
    """The offline table and live evidence use this exact reference comparison."""
    _, aligned, _ = match(a, b)
    mask = region_mask(a, aligned)
    separation = local_loss(a, aligned, mask)
    noise = max(nuisance_floor(a, mask), nuisance_floor(aligned, mask))
    threshold = max(POLICY["min_separation"], POLICY["noise_multiplier"]*noise)
    reliable = pixels >= POLICY["min_pixels"] and separation > threshold
    return {"separation": separation, "noise_floor": noise, "threshold": threshold,
            "reliable": reliable, "reason": "informative" if reliable else
            ("low_resolution" if pixels < POLICY["min_pixels"] else "near_shared_or_raster_unstable"),
            "mask_fraction": float(mask.mean())}


def compare(source, a, b, pixels):
    evidence = template_evidence(a, b, pixels)
    _, aa, _ = match(source, a)
    _, bb, _ = match(source, b)
    # Both candidates are evaluated on exactly the SAME union of local regions.
    mask = region_mask(aa, bb)
    da, db = local_loss(source, aa, mask), local_loss(source, bb, mask)
    margin = db-da
    vote = 0
    if evidence["reliable"] and abs(margin) > max(POLICY["min_vote_margin"], evidence["noise_floor"]):
        vote = 1 if margin > 0 else -1
    return {**evidence, "distance_a": da, "distance_b": db,
            "margin": margin, "vote": vote}, (aa, bb, mask)


def decide(votes):
    nonzero = [v for v in votes if v]
    if len(nonzero) < POLICY["min_votes"]:
        return 0
    positive = sum(v > 0 for v in nonzero)/len(nonzero)
    if positive >= POLICY["agreement"]:
        return 1
    if 1-positive >= POLICY["agreement"]:
        return -1
    return 0
