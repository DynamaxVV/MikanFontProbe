"""Candidate-pair-driven image comparison without landmark-state gating.

This module is diagnostic. It does not change the production font ranking.
"""

import cv2
import numpy as np

from mikan_font_probe.glyph_geometry import thin
from mikan_font_probe.stroke_correspondence_v7 import mapping


def _align(source, target):
    transform = mapping(source, target)
    if transform is None:
        return None
    scale, shift = transform
    yy, xx = np.indices(source.shape, dtype=np.float32)
    mx = ((xx - shift[0]) / scale[0]).astype(np.float32)
    my = ((yy - shift[1]) / scale[1]).astype(np.float32)
    return cv2.remap(target, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _representation(gray, mode):
    if mode == "ink":
        return (255 - gray.astype(np.float32)) / 255
    if mode == "edge":
        ink = (255 - gray.astype(np.float32)) / 255
        x = cv2.Sobel(ink, cv2.CV_32F, 1, 0, ksize=3)
        y = cv2.Sobel(ink, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.GaussianBlur(cv2.magnitude(x, y), (0, 0), 1.2)
    if mode == "skeleton":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = thin(mask).astype(np.float32)
        return cv2.GaussianBlur(skeleton, (0, 0), 1.5)
    raise ValueError(mode)


def compare(source, a, b, pixels, mode="skeleton", region=None):
    """Return most discriminative nonoverlapping template patches and direction."""
    aligned_a, aligned_b = _align(source, a), _align(source, b)
    if aligned_a is None or aligned_b is None or pixels < 24:
        return {"winner": None, "reason": "unsupported_resolution_or_alignment", "patches": []}
    src, left, right = (_representation(image, mode) for image in (source, aligned_a, aligned_b))
    contrast = np.abs(left - right)
    window = int(np.clip(round(16 * 64 / pixels), 12, 25))
    kernel = np.ones((window, window), np.float32) / window**2
    potential = cv2.filter2D(contrast, cv2.CV_32F, kernel)
    potential[:window, :] = potential[-window:, :] = 0
    potential[:, :window] = potential[:, -window:] = 0
    if region is not None:
        x0, y0, x1, y1 = region
        allowed = np.zeros(source.shape, bool)
        allowed[max(0, y0):min(source.shape[0], y1+1),
                max(0, x0):min(source.shape[1], x1+1)] = True
        potential[~allowed] = 0
    patches = []
    for _ in range(3):
        y, x = np.unravel_index(np.argmax(potential), potential.shape)
        if potential[y, x] < .01:
            break
        half = window // 2
        ys, xs = slice(y-half, y+half+1), slice(x-half, x+half+1)
        weight = contrast[ys, xs]
        denominator = float(weight.sum())
        if denominator <= 0:
            break
        da = float((weight * np.abs(src[ys, xs] - left[ys, xs])).sum() / denominator)
        db = float((weight * np.abs(src[ys, xs] - right[ys, xs])).sum() / denominator)
        patches.append({"center": [int(x), int(y)], "size": window,
                        "template_contrast": float(potential[y, x]),
                        "distance_a": da, "distance_b": db,
                        "winner": "a" if da < db else "b" if db < da else None})
        potential[max(0, y-window):y+window+1, max(0, x-window):x+window+1] = 0
    if not patches:
        return {"winner": None, "reason": "templates_near_shared", "patches": []}
    margin = float(np.mean([row["distance_b"]-row["distance_a"] for row in patches]))
    return {"winner": "a" if margin > 0 else "b" if margin < 0 else None,
            "margin": margin, "reason": "diagnostic_uncalibrated", "patches": patches}
