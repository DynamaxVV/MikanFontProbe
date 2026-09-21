"""Bounded stroke-profile nuisance models, shared across a text region."""
import cv2
import numpy as np

from mikan_font_probe.match_font_baseline import match
from mikan_font_probe.structure_font_baseline import structure_features, structural_distance


def hybrid_distance(source, template):
    """Exact existing hybrid branch, without computing unused structure search."""
    distance, aligned, _ = match(source, template)
    return .5*distance + .5*structural_distance(structure_features(source),
                                              structure_features(aligned))


def axis_stroke(ink, radius, axis):
    if radius == 0:
        return ink.copy()
    extent = int(np.ceil(abs(radius)))
    shape = (1, 2*extent+1) if axis == "x" else (2*extent+1, 1)
    kernel = np.ones(shape, np.uint8)
    operation = cv2.dilate if radius > 0 else cv2.erode
    outer = operation(ink, kernel)
    inner_extent = extent-1
    inner_shape = (1, 2*inner_extent+1) if axis == "x" else (2*inner_extent+1, 1)
    inner = operation(ink, np.ones(inner_shape, np.uint8))
    fraction = abs(radius)-inner_extent
    return np.rint(inner*(1-fraction)+outer*fraction).astype(np.uint8)


def stroke_template(ink, x_radius, y_radius):
    """Modify reference ink only, never the original screenshot evidence."""
    return axis_stroke(axis_stroke(ink, x_radius, "x"), y_radius, "y")
