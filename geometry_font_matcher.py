"""Same-resolution reference matching with retained appearance and geometry."""
import json
from pathlib import Path

import cv2
import numpy as np

from aligned_discrimination import synthetic_query
from glyph_geometry import thin, shape_loss
from match_font_baseline import match
from process_regions import imread
from structure_font_baseline import structure_features, structural_distance


def prepare(ink):
    return {"ink": ink, "skeleton": thin(ink), "structure": structure_features(ink)}


def score(source, target):
    binary, aligned, transform = match(source["ink"], target["ink"])
    appearance = .5*binary+.5*structural_distance(source["structure"], structure_features(aligned))
    scale, dx, dy = (transform[k] for k in ("scale", "dx", "dy"))
    matrix = np.array([[scale,0,32*(1-scale)+dx],[0,scale,32*(1-scale)+dy]], np.float32)
    aligned_skeleton = cv2.warpAffine(target["skeleton"], matrix, (64,64), flags=cv2.INTER_NEAREST)
    geometry = shape_loss(source["skeleton"], aligned_skeleton)
    return .5*appearance+.5*geometry


class GeometryReferences:
    """Real matching and discrimination lookup share reference construction."""
    near_shared = .015

    def __init__(self, directory):
        self.directory = Path(directory)
        self.fonts = json.loads((self.directory/"catalog.json").read_text())["fonts"]
        self.references = {}
        self.distances = {}

    def reference(self, font, char, pixels):
        key = (font["id"], char, int(pixels))
        if key not in self.references:
            image = imread(self.directory/font["templates"][char])
            self.references[key] = prepare(synthetic_query(image, max(8, int(pixels))))
        return self.references[key]

    def __call__(self, a, b, char, pixels):
        if pixels < 16:
            return None
        a, b = sorted((a, b))
        key = (a, b, char, int(pixels))
        if key not in self.distances:
            groups = [[f for f in self.fonts if f["family_group"]["key"] == name
                       and char in f["templates"]] for name in (a,b)]
            values = []
            for left in groups[0]:
                for right in groups[1]:
                    x = self.reference(left, char, pixels)
                    y = self.reference(right, char, pixels)
                    values.extend((score(x,y), score(y,x)))
            self.distances[key] = min(values) if values else None
        return self.distances[key]
