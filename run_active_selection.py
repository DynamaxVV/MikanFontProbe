"""Connect the Luna-built font table to candidate-driven querying."""

import numpy as np

from build_glyph_discrimination import load_index
from evaluate_active_selection import evaluate_with_table
from informative_selection import NEAR_SHARED
from process_regions import ROOT


class SeparationLookup:
    def __init__(self, directory=ROOT/"data/glyph-discrimination-v1"):
        self.index = load_index(directory)
        # Materialize once: NPZ member access otherwise decompresses repeatedly.
        self.distances = np.asarray(self.index._arrays["family_distance"])
        if float(self.index.metadata["low_distance_threshold"])!=NEAR_SHARED:
            raise ValueError("Selection and table shared-distance bands differ")

    def __call__(self,a,b,char,pixels):
        if pixels < min(self.index.resolutions):
            return None  # Do not invent finer evidence than the measured image.
        if char not in self.index.character_index:
            return None
        resolution = max(r for r in self.index.resolutions if r<=pixels)
        value = self.distances[self.index.family_index[a],self.index.family_index[b],
                               self.index.resolution_index[resolution],self.index.character_index[char]]
        return float(value) if np.isfinite(value) else None


def main():
    directory = ROOT/"data/glyph-discrimination-v1"
    lookup = SeparationLookup(directory)
    evaluate_with_table(lookup,[directory/"index.json",directory/"glyphs.npz",ROOT/"build_glyph_discrimination.py",ROOT/"run_active_selection.py"])


if __name__ == "__main__":
    main()
