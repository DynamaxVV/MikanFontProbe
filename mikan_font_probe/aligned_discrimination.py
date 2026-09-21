"""Font-only discrimination through the exact screenshot matching pipeline."""

import json
from pathlib import Path

import cv2
import numpy as np

from mikan_font_probe.match_font_baseline import normalize
from mikan_font_probe.process_regions import imread
from mikan_font_probe.structure_font_baseline import score_pair


def synthetic_query(image, pixels):
    """Resize tight grayscale ink to observed ink extent, then normalize normally."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Blank font glyph")
    tight = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    ratio = pixels / max(tight.shape)
    small = cv2.resize(tight, (max(1, round(tight.shape[1]*ratio)),
                              max(1, round(tight.shape[0]*ratio))),
                       interpolation=cv2.INTER_AREA)
    return normalize(cv2.copyMakeBorder(small, 4, 4, 4, 4,
                                       cv2.BORDER_CONSTANT, value=255))


class AlignedSeparation:
    """Conservative excess cross-font loss over same-font rasterization loss.

    Both directions and every weight pairing must remain distinguishable.
    Units are hybrid matcher distance; 0.015 is the existing observed-margin
    floor, not the unrelated v1 soft-IoU threshold or a fitted probability.
    """

    near_shared = .015

    def __init__(self, directory, cache_path):
        self.directory = Path(directory)
        self.cache_path = Path(cache_path)
        self.fonts = json.loads((self.directory / "catalog.json").read_text())["fonts"]
        self.cache = {}
        self.templates = {}
        self.queries = {}
        self.losses = {}

    def template(self, font, char):
        key = (font["id"], char)
        if key not in self.templates:
            self.templates[key] = normalize(imread(self.directory/font["templates"][char]))
        return self.templates[key]

    def loss(self, source, target, char, pixels):
        key = (source["id"], target["id"], char, pixels)
        if key not in self.losses:
            query_key = (source["id"], char, pixels)
            if query_key not in self.queries:
                self.queries[query_key] = synthetic_query(
                    imread(self.directory/source["templates"][char]), pixels)
            values, _ = score_pair(self.queries[query_key], self.template(target, char))
            self.losses[key] = values["hybrid"]
        return self.losses[key]

    def __call__(self, a, b, char, pixels):
        if pixels < 16:
            return None
        # Same conservative resolution floor as v1; no upsampled evidence.
        band = max(r for r in (16, 24, 32, 48, 64) if r <= pixels)
        a, b = sorted((a, b))
        key = json.dumps([a, b, char, band], ensure_ascii=False)
        if key not in self.cache:
            groups = [[f for f in self.fonts if f["family_group"]["key"] == family
                       and char in f["templates"]] for family in (a, b)]
            margins = []
            for left in groups[0]:
                for right in groups[1]:
                    for source, target in ((left, right), (right, left)):
                        margins.append(self.loss(source, target, char, band)
                                       - self.loss(source, source, char, band))
            self.cache[key] = max(0., min(margins)) if margins else None
        return self.cache[key]

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps({"metric": "minimum_bidirectional_excess_hybrid_loss",
            "near_shared": self.near_shared, "pairs": self.cache}, ensure_ascii=False, indent=2)+"\n")
