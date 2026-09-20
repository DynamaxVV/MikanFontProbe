"""Evaluate the font-only local encoder on synthetic and real manga crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluate_active_selection import quality_order
from local_encoder_v11 import LocalShapeEncoder
from process_regions import ROOT, imread
from train_local_encoder_v11 import CATALOG_PATH, TEMPLATE_DIR, tight_gray, augment

CATEGORY_ANSWERS = {
    "少女体": "少女",
    "方圆体": "方圆",
    "海报体": "海报",
    "雅士黑": "润圆/雅士黑",
    "雷盖体": "雷盖/雷鬼体",
}


def load_model(checkpoint_path: Path, device: torch.device) -> LocalShapeEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LocalShapeEncoder(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def device_for_run(requested: str) -> torch.device:
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor_view(crop: np.ndarray, native: int, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    return augment(crop, rng, int(np.clip(native, 16, 96)), deterministic=True)


@torch.no_grad()
def embed(model: LocalShapeEncoder, views: list[tuple[np.ndarray, int]],
          device: torch.device) -> np.ndarray:
    images = torch.from_numpy(np.stack([row[0][None] for row in views])).to(device)
    pixels = torch.tensor([row[1] for row in views], dtype=torch.float32, device=device)
    return model(images, pixels).cpu().numpy()


def catalog_context() -> tuple[dict, list[dict]]:
    catalog = json.loads(CATALOG_PATH.read_text())
    fonts = catalog["fonts"]
    for font in fonts:
        font["category"] = Path(font["path"]).parent.name
    return catalog, fonts


def face_templates(fonts: list[dict], character: str, native: int,
                   seed: int) -> tuple[list[np.ndarray], list[dict]]:
    views, metadata = [], []
    for index, font in enumerate(fonts):
        filename = font["templates"].get(character)
        if filename is None:
            continue
        crop = tight_gray(TEMPLATE_DIR / filename)
        views.append(tensor_view(crop, native, seed + index * 1009))
        metadata.append({"face": font["postscript"], "category": font["category"],
                         "weight": font.get("weight")})
    return views, metadata


def rank_glyph(model: LocalShapeEncoder, glyph: dict, fonts: list[dict],
               device: torch.device) -> dict:
    path = ROOT / glyph["path"]
    source = tight_gray(path)
    native = int(glyph.get("resolution") or max(source.shape))
    source_view = tensor_view(source, native, 1777 + ord(glyph["character"]))
    template_views, metadata = face_templates(fonts, glyph["character"], native, 3001)
    vectors = embed(model, [source_view] + template_views, device)
    similarities = vectors[1:] @ vectors[0]
    faces = [{**meta, "similarity": float(score)}
             for meta, score in zip(metadata, similarities)]
    by_category = defaultdict(list)
    for row in faces:
        by_category[row["category"]].append(row)
    categories = [{"category": category,
                   "answer": CATEGORY_ANSWERS.get(category, category),
                   "similarity": max(row["similarity"] for row in rows),
                   "face": max(rows, key=lambda row: row["similarity"])["face"]}
                  for category, rows in by_category.items()]
    categories.sort(key=lambda row: row["similarity"], reverse=True)
    return {"character": glyph["character"], "native_pixels": native,
            "categories": categories, "faces": sorted(faces, key=lambda row: row["similarity"], reverse=True)}


def rank_question(model: LocalShapeEncoder, sample: dict, fonts: list[dict],
                  device: torch.device, budget: int = 7) -> dict:
    if sample.get("polarity") == "light" or any(g.get("polarity") == "light" for g in sample["glyphs"]):
        return {"qid": sample["qid"], "expected": sample["expected"], "state": "unsupported_light_input",
                "selected": [], "ranking": []}
    indices = quality_order(sample["glyphs"])[:budget]
    glyph_results = [rank_glyph(model, sample["glyphs"][index], fonts, device)
                     for index in indices]
    scores = defaultdict(list)
    for glyph in glyph_results:
        for row in glyph["categories"]:
            scores[row["category"]].append(row["similarity"])
    ranking = [{"category": category,
                "answer": CATEGORY_ANSWERS.get(category, category),
                "similarity": float(np.mean(values)),
                "support": len(values)} for category, values in scores.items()]
    ranking.sort(key=lambda row: row["similarity"], reverse=True)
    return {"qid": sample["qid"], "expected": sample["expected"], "state": "evaluated",
            "selected": [{"index": index, "character": sample["glyphs"][index]["character"]}
                         for index in indices], "ranking": ranking, "glyphs": glyph_results}


def load_samples(path: Path) -> list[dict]:
    return json.loads(path.read_text())["samples"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "data/local-encoder-v11/final/checkpoint.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/local-encoder-v11/final/evaluation.json")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    device = device_for_run(args.device)
    model = load_model(args.checkpoint, device)
    _, fonts = catalog_context()
    sample_sets = []
    for group, relative in (("fresh_holdout", "data/fresh-holdout-v2/pools.json"),
                            ("category_expansion", "data/category-expansion-v3/pools.json"),
                            ("folk_controls", "data/humming-v4/folk-controls/pools.json")):
        for sample in load_samples(ROOT / relative):
            sample_sets.append((group, sample))
    rows = []
    for group, sample in sample_sets:
        result = rank_question(model, sample, fonts, device)
        rows.append({"group": group, **result})
        top = result["ranking"][0]["answer"] if result["ranking"] else None
        print(sample["qid"], sample["expected"], top, result["state"], flush=True)
    summary = {}
    for group in {row["group"] for row in rows}:
        group_rows = [row for row in rows if row["group"] == group]
        eligible = [row for row in group_rows if row["state"] == "evaluated"]
        summary[group] = {"questions": len(group_rows), "eligible": len(eligible),
                          "top1": sum(row["ranking"][0]["answer"] == row["expected"] for row in eligible),
                          "top3": sum(row["expected"] in [x["answer"] for x in row["ranking"][:3]] for row in eligible)}
        summary[group]["top1_rate"] = summary[group]["top1"] / max(1, len(eligible))
        summary[group]["top3_rate"] = summary[group]["top3"] / max(1, len(eligible))
    provenance = {"checkpoint": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  "catalog": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
                  "fresh_pools": hashlib.sha256((ROOT/"data/fresh-holdout-v2/pools.json").read_bytes()).hexdigest(),
                  "category_pools": hashlib.sha256((ROOT/"data/category-expansion-v3/pools.json").read_bytes()).hexdigest(),
                  "controls_pools": hashlib.sha256((ROOT/"data/humming-v4/folk-controls/pools.json").read_bytes()).hexdigest()}
    output = {"status": "font_only_encoder_real_crop_evaluation", "device": str(device),
              "provenance": provenance, "summary": summary, "samples": rows,
              "notes": ["Templates and real crops share the encoder; no real crop was used for training.",
                        "Category score is mean over selected glyphs of the best face within category.",
                        "The font identity in a manga crop is not independently labeled; expected is the question-bank Chinese mapping."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
