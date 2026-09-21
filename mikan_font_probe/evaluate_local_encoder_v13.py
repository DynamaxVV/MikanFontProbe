"""Isolated v13 evaluation for aggregation, training, and local reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from mikan_font_probe.evaluate_active_selection import quality_order
from mikan_font_probe.local_encoder_v11 import LocalShapeEncoder
from mikan_font_probe.local_features_v13 import compare_local, local_reference
from mikan_font_probe.process_regions import ROOT, imread
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.train_local_encoder_v11 import CATALOG_PATH, TEMPLATE_DIR, augment, tight_gray


CATEGORY_ANSWERS = {
    "少女体": "少女", "方圆体": "方圆", "海报体": "海报",
    "雅士黑": "润圆/雅士黑", "雷盖体": "雷盖/雷鬼体",
}
AGGREGATIONS = ("max", "top2", "top3", "mean", "prototype")
LOCAL_FOCUS = {"轻吟体", "方新书", "雅士黑"}
LOCAL_TEMPLATE_CACHE: dict[tuple[str, str, int], dict] = {}


def load_model(checkpoint_path: Path, device: torch.device) -> LocalShapeEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LocalShapeEncoder(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def device_for_run(requested: str) -> torch.device:
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def source_gray(path: Path, polarity: str | None) -> np.ndarray:
    image = imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if polarity == "light":
        gray = 255 - gray
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"blank glyph crop: {path}")
    return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def tensor_view(crop: np.ndarray, native: int, seed: int) -> tuple[np.ndarray, int]:
    return augment(crop, np.random.default_rng(seed), int(np.clip(native, 16, 96)),
                   deterministic=True)


@torch.no_grad()
def embed(model: LocalShapeEncoder, views: list[tuple[np.ndarray, int]],
          device: torch.device) -> np.ndarray:
    images = torch.from_numpy(np.stack([row[0][None] for row in views])).to(device)
    pixels = torch.tensor([row[1] for row in views], dtype=torch.float32, device=device)
    return model(images, pixels).cpu().numpy()


def catalog_context() -> list[dict]:
    catalog = json.loads(CATALOG_PATH.read_text())
    fonts = catalog["fonts"]
    for font in fonts:
        font["category"] = Path(font["path"]).parent.name
    return fonts


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
                         "weight": font.get("weight"), "gray": crop})
    return views, metadata


def aggregate(values: list[float], method: str) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float32))[::-1]
    if method == "max":
        return float(ordered[0])
    if method == "top2":
        return float(np.mean(ordered[:min(2, len(ordered))]))
    if method == "top3":
        return float(np.mean(ordered[:min(3, len(ordered))]))
    if method == "mean":
        return float(np.mean(ordered))
    raise ValueError(f"unsupported scalar aggregation: {method}")


def category_rows(vectors: np.ndarray, metadata: list[dict],
                  source_native: int, source_gray_crop: np.ndarray,
                  character: str, include_local: bool) -> list[dict]:
    source_vector = vectors[0]
    source_local = local_reference(source_gray_crop, source_native) if include_local else None
    by_category: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    similarities = vectors[1:] @ source_vector
    for index, (meta, similarity) in enumerate(zip(metadata, similarities)):
        row = {key: value for key, value in meta.items() if key != "gray"}
        row["_gray"] = meta["gray"]
        row["similarity"] = float(similarity)
        if include_local:
            key = (meta["face"], character, source_native)
            target_local = LOCAL_TEMPLATE_CACHE.get(key)
            if target_local is None:
                target_local = local_reference(meta["gray"], source_native)
                LOCAL_TEMPLATE_CACHE[key] = target_local
            local = compare_local(source_local, target_local)
            row["local"] = local
        by_category[row["category"]].append((index, row))

    output = []
    for category, indexed_rows in by_category.items():
        rows = [row for _, row in indexed_rows]
        category_vectors = np.asarray([vectors[index + 1] for index, _ in indexed_rows])
        prototype = category_vectors.mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
        result = {
            "category": category,
            "answer": CATEGORY_ANSWERS.get(category, category),
            "similarity": aggregate([row["similarity"] for row in rows], "max"),
            "prototype_similarity": float(source_vector @ prototype),
            "face_count": len(rows),
            "face": max(rows, key=lambda row: row["similarity"])["face"],
            "faces": sorted(rows, key=lambda row: row["similarity"], reverse=True),
        }
        if include_local:
            local_rows = [row["local"] for row in rows]
            reliable = [row["score"] for row in local_rows if row["reliable"]]
            result["local_similarity"] = aggregate(reliable, "top2") if reliable else .5
            result["local_reliable_faces"] = len(reliable)
        for row in result["faces"]:
            row.pop("_gray", None)
        output.append(result)
    return output


def rank_glyph(model: LocalShapeEncoder, glyph: dict, fonts: list[dict],
               device: torch.device, include_local: bool) -> dict:
    crop = source_gray(resolve_repo_path(glyph["path"]), glyph.get("polarity"))
    native = int(glyph.get("resolution") or max(crop.shape))
    source_view = tensor_view(crop, native, 1777 + ord(glyph["character"]))
    template_views, metadata = face_templates(fonts, glyph["character"], native, 3001)
    vectors = embed(model, [source_view] + template_views, device)
    return {
        "character": glyph["character"], "native_pixels": native,
        "quality": float(glyph.get("quality", {}).get("score", 1.)),
        "categories": category_rows(vectors, metadata, native, crop,
                                     glyph["character"], include_local),
    }


def rank_question(model: LocalShapeEncoder, sample: dict, fonts: list[dict],
                  device: torch.device, aggregation: str, include_local: bool,
                  local_weight: float, budget: int = 7) -> dict:
    indices = quality_order(sample["glyphs"])[:budget]
    glyph_results = [rank_glyph(model, sample["glyphs"][index], fonts, device,
                                include_local) for index in indices]
    global_scores = defaultdict(list)
    for result in glyph_results:
        for row in result["categories"]:
            if aggregation == "prototype":
                global_score = row["prototype_similarity"]
            else:
                global_score = aggregate(
                    [face["similarity"] for face in row["faces"]], aggregation)
            global_scores[row["category"]].append(float(global_score))
    base_ranking = sorted(
        ((category, float(np.mean(values))) for category, values in global_scores.items()),
        key=lambda row: row[1], reverse=True)
    close_focus_pair = (
        include_local and len(base_ranking) >= 2
        and {base_ranking[0][0], base_ranking[1][0]}.issubset(LOCAL_FOCUS)
        and base_ranking[0][1] - base_ranking[1][1] <= .06
    )
    scores = defaultdict(list)
    for result in glyph_results:
        for row in result["categories"]:
            if aggregation == "prototype":
                global_score = row["prototype_similarity"]
            else:
                global_score = aggregate(
                    [face["similarity"] for face in row["faces"]], aggregation)
            if close_focus_pair and row["category"] in LOCAL_FOCUS:
                global_score += local_weight * (row["local_similarity"] - .5)
            scores[row["category"]].append(float(global_score))
    ranking = [{"category": category,
                "answer": CATEGORY_ANSWERS.get(category, category),
                # The face aggregation happens inside each glyph.  Across
                # selected glyphs we retain the established mean evidence
                # rule, otherwise `max` would silently become a question-level
                # max and let one lucky character hijack the answer.
                "similarity": float(np.mean(values)),
                "support": len(values)}
               for category, values in scores.items()]
    ranking.sort(key=lambda row: row["similarity"], reverse=True)
    margin = ranking[0]["similarity"] - ranking[1]["similarity"] if len(ranking) > 1 else 0.
    for row in ranking:
        row["top_margin"] = float(margin)
        row["confidence_kind"] = "uncalibrated_similarity_margin"
    return {"qid": sample["qid"], "expected": sample["expected"],
            "state": "evaluated", "selected": [
                {"index": index, "character": sample["glyphs"][index]["character"]}
                for index in indices], "ranking": ranking, "glyphs": glyph_results,
            "local_gate": "enabled_close_focus_pair" if close_focus_pair else "abstained"}


def load_samples(path: Path) -> list[dict]:
    return json.loads(path.read_text())["samples"]


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for group in sorted({row["group"] for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        eligible = [row for row in group_rows if row["state"] == "evaluated"]
        summary[group] = {
            "questions": len(group_rows), "eligible": len(eligible),
            "top1": sum(row["ranking"] and row["ranking"][0]["answer"] == row["expected"]
                         for row in eligible),
            "top3": sum(row["expected"] in [item["answer"] for item in row["ranking"][:3]]
                         for row in eligible),
        }
        summary[group]["top1_rate"] = summary[group]["top1"] / max(1, len(eligible))
        summary[group]["top3_rate"] = summary[group]["top3"] / max(1, len(eligible))
    return summary


def evaluate_variant(model: LocalShapeEncoder, samples: list[tuple[str, dict]],
                     fonts: list[dict], device: torch.device, aggregation: str,
                     include_local: bool, local_weight: float) -> list[dict]:
    rows = []
    for group, sample in samples:
        result = rank_question(model, sample, fonts, device, aggregation,
                               include_local, local_weight)
        rows.append({"group": group, **result})
        top = result["ranking"][0]["answer"] if result["ranking"] else None
        print(aggregation, sample["qid"], sample["expected"], top, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "data/local-encoder-v11/robust/checkpoint.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/local-encoder-v13/step2-aggregation/evaluation.json")
    parser.add_argument("--aggregation", choices=(*AGGREGATIONS, "all"), default="all")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--local-weight", type=float, default=.12)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    device = device_for_run(args.device)
    model = load_model(args.checkpoint, device)
    fonts = catalog_context()
    samples = []
    for group, relative in (("fresh_holdout", "data/fresh-holdout-v2/pools.json"),
                            ("category_expansion", "data/category-expansion-v3/pools.json"),
                            ("folk_controls", "data/humming-v4/folk-controls/pools.json")):
        samples.extend((group, sample) for sample in load_samples(ROOT / relative))
    variants = AGGREGATIONS if args.aggregation == "all" else (args.aggregation,)
    all_rows = {}
    for variant in variants:
        rows = evaluate_variant(model, samples, fonts, device, variant,
                                args.local, args.local_weight)
        all_rows[variant] = {"summary": summarize(rows), "samples": rows}
    provenance = {
        "checkpoint": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "catalog": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        "fresh_pools": hashlib.sha256((ROOT / "data/fresh-holdout-v2/pools.json").read_bytes()).hexdigest(),
        "category_pools": hashlib.sha256((ROOT / "data/category-expansion-v3/pools.json").read_bytes()).hexdigest(),
        "controls_pools": hashlib.sha256((ROOT / "data/humming-v4/folk-controls/pools.json").read_bytes()).hexdigest(),
    }
    output = {"status": "v13_isolated_evaluation", "device": str(device),
              "checkpoint": str(args.checkpoint), "aggregation": args.aggregation,
              "local": args.local, "local_weight": args.local_weight,
              "provenance": provenance, "variants": all_rows,
              "notes": [
                  "v11 robust encoder is kept fixed for aggregation ablation.",
                  "Question selection is the original quality_order policy.",
                  "prototype is the normalized mean of all face embeddings in one category.",
                  "local evidence is an abstaining endpoint/corner tie-breaker, not a probability.",
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value["summary"] for key, value in all_rows.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
