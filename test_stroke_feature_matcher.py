import unittest

import numpy as np

from stroke_feature_matcher import compare_features, correspond, decide, pair_features
from test_stroke_terminal_features import shaft


def item(state, x=30, kind="terminal"):
    return {"kind": kind, "point": [x, 64], "state": state, "direction": [-1, 0],
            "native_width": 12, "reason": "test"}


class FeatureTests(unittest.TestCase):
    def test_no_cross_kind_match(self):
        self.assertEqual(correspond([item("square")], [item("square", kind="corner")]), [])

    def test_no_absence_as_round(self):
        a = {"gray": shaft(), "features": [item("square")]}
        b = {"gray": shaft(), "features": [item("unknown")]}
        self.assertEqual(pair_features(a, b)["utility"], 0)

    def test_real_square_anchor_required(self):
        soft = [{"vote": 1, "square_anchor": False} for _ in range(3)]
        self.assertEqual(decide(soft), (0, "round_only_no_square_anchor"))
        soft[0]["square_anchor"] = True
        self.assertEqual(decide(soft)[0], 1)

    def test_matching_source_required(self):
        a = {"gray": shaft(), "features": [item("square")]}
        b = {"gray": shaft(), "features": [item("rounded")]}
        source = {"gray": shaft(), "features": [item("unknown")]}
        self.assertEqual(compare_features(source, a, b)["vote"], 0)
        source["features"][0]["state"] = "square"
        row = compare_features(source, a, b)
        self.assertEqual(row["vote"], 1)
        self.assertTrue(row["square_anchor"])

    def test_nearest_match_not_state_cherry_pick(self):
        self.assertEqual(correspond([item("square")], [item("rounded", 31), item("square", 38)]), [(0, 0)])

    def test_opposing_square_cannot_anchor_round_only_winner(self):
        positions = range(15, 105, 10)
        left = [item("rounded", x) for x in positions]
        right = [item("square", x) for x in positions]
        observed = [item("rounded", x) for x in positions]
        observed[-1]["state"] = "square"
        a = {"gray": shaft(), "features": left}
        b = {"gray": shaft(), "features": right}
        source = {"gray": shaft(), "features": observed}
        result = compare_features(source, a, b)
        self.assertEqual(result["vote"], 1)
        self.assertFalse(result["square_anchor"])
        self.assertEqual(decide([result]*3)[0], 0)


if __name__ == "__main__":
    unittest.main()
