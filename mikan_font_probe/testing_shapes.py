"""Synthetic shapes shared by geometric tests and diagnostic renderers."""

import cv2
import numpy as np


def shaft(rounded=False, width=14, angle=0):
    gray = np.full((128, 128), 255, np.uint8)
    if rounded:
        cv2.line(gray, (28, 64), (100, 64), 0, width)
    else:
        cv2.rectangle(gray, (21, 64-width//2), (107, 64+width//2), 0, -1)
    if angle:
        gray = cv2.warpAffine(gray, cv2.getRotationMatrix2D((64, 64), angle, 1),
                              (128, 128), borderValue=255)
    return gray


def bend(width=12, radius=0, angle=0):
    """Supersampled L with flat caps and an optional circular connection."""
    scale = 4
    image = np.full((128 * scale, 128 * scale), 255, np.uint8)
    half = width / 2
    center = np.array([90.0 - radius, 90.0 - radius])
    if radius:
        theta = np.linspace(0, np.pi / 2, 101)
        outer = center + (radius + half) * np.column_stack([np.cos(theta), np.sin(theta)])
        inner = center + (radius - half) * np.column_stack([np.cos(theta[::-1]), np.sin(theta[::-1])])
        polygon = np.vstack([[90 + half, 20], outer, [20, 90 + half],
                             [20, 90 - half], inner, [90 - half, 20]])
    else:
        polygon = np.array([[90 + half, 20], [90 + half, 90 + half],
                            [20, 90 + half], [20, 90 - half],
                            [90 - half, 90 - half], [90 - half, 20]])
    cv2.fillPoly(image, [np.rint(polygon * scale).astype(np.int32)], 0)
    if angle:
        transform = cv2.getRotationMatrix2D((64 * scale, 64 * scale), angle, 1)
        image = cv2.warpAffine(image, transform, image.shape[::-1], borderValue=255)
    return cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
