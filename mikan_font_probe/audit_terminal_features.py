"""Bounded synthetic cap audit; coverage and wrong hard decisions both matter."""
import argparse
import json
from pathlib import Path

from mikan_font_probe.stroke_feature_matcher import gray_canvas
from mikan_font_probe.stroke_terminal_features import extract_terminals
from mikan_font_probe.testing_shapes import shaft


def audit():
    rows = []
    for rounded in (False, True):
        expected = "rounded" if rounded else "square"
        for width in (10, 14, 20):
            for angle in (0, 15, 30, 45, 60, 90):
                for pixels in (24, 32, 48, 64, 96):
                    gray, pixels = gray_canvas(shaft(rounded, width, angle), pixels)
                    features = extract_terminals(gray, pixels)
                    rows.append({"expected": expected, "width": width, "angle": angle, "pixels": pixels,
                                 "states": [f["state"] for f in features],
                                 "native_widths": [f["native_width"] for f in features],
                                 "wrong": sum(f["state"] not in ("unknown", expected) for f in features)})
    return {"status": "synthetic only; not proof on manga rasterization", "cases": rows,
            "wrong_hard_decisions": sum(r["wrong"] for r in rows),
            "known_features": sum(s != "unknown" for r in rows for s in r["states"]),
            "detected_features": sum(len(r["states"]) for r in rows),
            "expected_free_caps": 2*len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous synthetic audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print({k: v for k, v in result.items() if k != "cases"})
