"""Train the first shared local Japanese glyph-shape encoder.

The training corpus is rendered from the repository's Japanese font files and
their frozen template catalog. Real manga pixels are never used for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from local_encoder_v11 import LocalShapeEncoder, parameter_count
from process_regions import ROOT, imread


TEMPLATE_DIR = ROOT / "data/category-expansion-v3/templates"
CATALOG_PATH = TEMPLATE_DIR / "catalog.json"
SEED = 20260921


def family_key(postscript: str, category: str) -> str:
    """Keep known weight variants together while preserving explicit font families."""
    prefixes = (
        "PShinMGoPr6N", "PRyuminPr6N", "PAntiqueANpProN", "YWHeiTi", "SkipStd",
    )
    for prefix in prefixes:
        if postscript.startswith(prefix):
            return prefix
    # The user treats 少女/方圆/海报 as separate families even though the
    # Japanese source name shares the RodinHappyPro prefix.
    if category in {"少女体", "方圆体", "海报体"}:
        return category
    return postscript


def stable_split(character: str) -> str:
    """Split by character identity, not by augmented view or rendered file."""
    return "val" if (ord(character) * 2654435761) % 10 < 2 else "train"


def build_records() -> tuple[list[dict], dict]:
    catalog = json.loads(CATALOG_PATH.read_text())
    records = []
    families, faces, weights = set(), set(), set()
    for font in catalog["fonts"]:
        source = ROOT / font["path"]
        category = Path(font["path"]).parent.name
        family = family_key(font["postscript"], category)
        face = font["postscript"]
        weight = str(font.get("weight", "unknown"))
        families.add(family)
        faces.add(face)
        weights.add(weight)
        for character, filename in sorted(font["templates"].items()):
            records.append({
                "font_id": font["id"], "face": face, "family": family,
                "category": category, "weight": weight, "character": character,
                "template": str((TEMPLATE_DIR / filename).relative_to(ROOT)),
                "source_font": str(source.relative_to(ROOT)),
                "split": stable_split(character),
            })
    family_list = sorted(families)
    face_list = sorted(faces)
    weight_list = sorted(weights, key=lambda value: (value == "unknown", int(value) if value.isdigit() else 0))
    indexes = {"family": {value: index for index, value in enumerate(family_list)},
               "face": {value: index for index, value in enumerate(face_list)},
               "weight": {value: index for index, value in enumerate(weight_list)}}
    for row in records:
        row["family_index"] = indexes["family"][row["family"]]
        row["face_index"] = indexes["face"][row["face"]]
        row["weight_index"] = indexes["weight"][row["weight"]]
    summary = {"families": family_list, "faces": face_list, "weights": weight_list,
               "characters": sorted({row["character"] for row in records}),
               "record_count": len(records), "split_counts": {
                   split: sum(row["split"] == split for row in records)
                   for split in ("train", "val")}}
    return records, {"indexes": indexes, "summary": summary}


def tight_gray(path: Path) -> np.ndarray:
    image = imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"blank template: {path}")
    return gray[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()


def _place(crop: np.ndarray, rng: np.random.Generator, native: int,
           local: bool) -> np.ndarray:
    target_extent = int(rng.integers(44, 57))
    ratio = target_extent / max(crop.shape)
    target = (max(2, round(crop.shape[1] * ratio)), max(2, round(crop.shape[0] * ratio)))
    low_extent = int(np.clip(round(native * rng.uniform(.62, .82)), 16, target_extent))
    low_ratio = low_extent / max(crop.shape)
    low_size = (max(2, round(crop.shape[1] * low_ratio)), max(2, round(crop.shape[0] * low_ratio)))
    low = cv2.resize(crop, low_size, interpolation=cv2.INTER_AREA)
    resized = cv2.resize(low, target, interpolation=cv2.INTER_LINEAR)
    canvas = np.full((80, 80), 255, np.uint8)
    jitter = rng.integers(-3, 4, size=2)
    x = (80 - resized.shape[1]) // 2 + int(jitter[0])
    y = (80 - resized.shape[0]) // 2 + int(jitter[1])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(80, x + resized.shape[1]), min(80, y + resized.shape[0])
    canvas[y0:y1, x0:x1] = resized[y0-y:y1-y, x0-x:x1-x]
    if local:
        _, mask = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.nonzero(mask)
        center = (int(rng.choice(xs)), int(rng.choice(ys))) if len(xs) else (40, 40)
        left = int(np.clip(center[0] - 32 + rng.integers(-6, 7), 0, 16))
        top = int(np.clip(center[1] - 32 + rng.integers(-6, 7), 0, 16))
    else:
        left = top = 8
    return canvas[top:top+64, left:left+64]


def _white_outline(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Render a dark-background, white-outlined manga-text variant."""
    ink = cv2.threshold(255 - image, 18, 255, cv2.THRESH_BINARY)[1]
    radius = int(rng.choice((1, 1, 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(ink, kernel)
    background = int(rng.integers(35, 145))
    outline = int(rng.integers(220, 256))
    fill = int(rng.integers(0, 70))
    result = np.full_like(image, background)
    result[dilated > 0] = outline
    result[ink > 0] = fill
    return result


def augment(crop: np.ndarray, rng: np.random.Generator,
            native: int | None = None, deterministic: bool = False,
            preserve_weight: bool = False) -> tuple[np.ndarray, int]:
    """Create one 64-square grayscale view and retain its native resolution."""
    if native is None:
        native = int(rng.integers(24, 81))
    image = _place(crop, rng, native, local=not deterministic and rng.random() < .38)
    if not deterministic:
        # Keep the explicit Rodin L/B/UB answer distinction from being
        # destroyed by the augmentation itself.  The auxiliary exact-face
        # and weight losses remain active for every family.
        if rng.random() < (.16 if preserve_weight else .32):
            kernel = np.ones((3, 3), np.uint8)
            ink = 255 - image
            image = 255 - (cv2.dilate(ink, kernel) if rng.random() < .5
                            else cv2.erode(ink, kernel))
        if rng.random() < (.12 if preserve_weight else .20):
            image = _white_outline(image, rng)
        if rng.random() < .75:
            sigma = float(rng.uniform(0, .9 if preserve_weight else 1.35))
            if sigma:
                image = cv2.GaussianBlur(image, (0, 0), sigma)
        if rng.random() < .35:
            background = int(rng.integers(220, 256))
            image = np.clip(image.astype(np.float32) * (background / 255)
                            + (255 - background), 0, 255).astype(np.uint8)
        if rng.random() < .22:
            noise = rng.normal(0, rng.uniform(.4, 2.0), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if rng.random() < .30:
            quality = int(rng.integers(45, 93))
            encoded = cv2.imencode(".jpg", image,
                                   [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return (1. - image.astype(np.float32) / 255.).astype(np.float32), native


class GlyphDataset(Dataset):
    """Two stochastic views per frozen font/character record."""

    def __init__(self, records: list[dict], split: str, seed: int):
        self.records = [row for row in records if row["split"] == split]
        self.seed = seed
        self.epoch = 0
        self.images = {row["template"]: tight_gray(ROOT / row["template"])
                       for row in self.records}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        row = self.records[index]
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + index * 7919)
        base_native = int(rng.integers(24, 81))
        preserve_weight = row["category"] in {"少女体", "方圆体", "海报体"}
        first, native_a = augment(self.images[row["template"]], rng, base_native,
                                  preserve_weight=preserve_weight)
        second, native_b = augment(self.images[row["template"]], rng,
                                   int(np.clip(base_native + rng.integers(-3, 4), 24, 81)),
                                   preserve_weight=preserve_weight)
        return {"a": torch.from_numpy(first[None]), "b": torch.from_numpy(second[None]),
                "native_a": torch.tensor(native_a, dtype=torch.float32),
                "native_b": torch.tensor(native_b, dtype=torch.float32),
                "family": row["family_index"], "face": row["face_index"],
                "weight": row["weight_index"], "character": ord(row["character"])}


class CharacterBatchSampler(Sampler[list[int]]):
    """Make same-character hard negatives common without duplicating records."""

    def __init__(self, records: list[dict], batch_size: int, seed: int):
        self.groups = defaultdict(list)
        for index, row in enumerate(records):
            if row["split"] == "train":
                self.groups[row["character"]].append(index)
        self.characters = sorted(self.groups)
        self.batch_size = batch_size
        self.seed = seed
        self.count = max(1, math.ceil(sum(map(len, self.groups.values())) / batch_size))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        per_character = max(2, self.batch_size // min(16, len(self.characters)))
        for _ in range(self.count):
            chosen = rng.sample(self.characters, min(16, len(self.characters)))
            indices = []
            for character in chosen:
                group = self.groups[character]
                if len(group) <= per_character:
                    indices.extend(group)
                else:
                    indices.extend(rng.sample(group, per_character))
            rng.shuffle(indices)
            yield indices[:self.batch_size]


def device_for_run(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_loss(model: LocalShapeEncoder, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    a = model(batch["a"].to(device), batch["native_a"].to(device), heads=True)
    b = model(batch["b"].to(device), batch["native_b"].to(device), heads=True)
    family = batch["family"].to(device)
    face = batch["face"].to(device)
    weight = batch["weight"].to(device)
    ce = nn.CrossEntropyLoss()
    family_loss = (ce(a["family"], family) + ce(b["family"], family)) / 2
    face_loss = (ce(a["face"], face) + ce(b["face"], face)) / 2
    weight_loss = (ce(a["weight"], weight) + ce(b["weight"], weight)) / 2
    similarity = a["embedding"] @ b["embedding"].T
    targets = torch.arange(similarity.shape[0], device=device)
    nce = (ce(similarity / .08, targets) + ce(similarity.T / .08, targets)) / 2
    positive = torch.diagonal(similarity)
    characters = batch["character"].to(device)
    family_ids = family
    same_character = characters[:, None] == characters[None, :]
    different_family = family_ids[:, None] != family_ids[None, :]
    negative_mask = same_character & different_family
    hard = torch.relu(.18 + similarity - positive[:, None])
    hard = hard[negative_mask].mean() if negative_mask.any() else similarity.new_zeros(())
    loss = .22 * family_loss + .18 * face_loss + .10 * weight_loss + .35 * nce + .15 * hard
    metrics = {"loss": float(loss.detach().cpu()), "family": float(family_loss.detach().cpu()),
               "face": float(face_loss.detach().cpu()), "weight": float(weight_loss.detach().cpu()),
               "nce": float(nce.detach().cpu()), "hard": float(hard.detach().cpu())}
    return loss, metrics


@torch.no_grad()
def evaluate(model: LocalShapeEncoder, dataset: GlyphDataset, device: torch.device) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    totals = defaultdict(int)
    correct = defaultdict(int)
    for batch in loader:
        outputs = model(batch["a"].to(device), batch["native_a"].to(device), heads=True)
        for name, key in (("family", "family"), ("face", "face"), ("weight", "weight")):
            prediction = outputs[name].argmax(dim=1).cpu()
            target = batch[key]
            totals[name] += len(target)
            correct[name] += int((prediction == target).sum())
    model.train()
    return {f"{name}_accuracy": correct[name] / max(1, totals[name]) for name in totals}


def manifest(records: list[dict], labels: dict) -> dict:
    files = {str(path): hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
             for path in {row["source_font"] for row in records} | {
                 str(CATALOG_PATH.relative_to(ROOT))}}
    return {"schema": "local_encoder_v11", "seed": SEED,
            "catalog": str(CATALOG_PATH.relative_to(ROOT)), "files": files,
            "labels": labels, "records": records,
            "protocol": {"font_files_only_training": True,
                         "character_split": "ord(character)*2654435761 mod 10 < 2 -> val",
                         "positive": "two independent rendering views of same face and character",
                         "hard_negative": "same character from different source families in batch",
                         "weight": "exact face and explicit weight heads; no weight erasure",
                         "render_augmentation": "scale, low-resolution rasterization, blur, morphology, white outline, noise, JPEG compression",
                         "weight_sensitive_augmentation": "少女体/方圆体/海报体 use reduced morphology and blur",
                         "real_pixels_in_training": False}}


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    records, labels = build_records()
    train_set = GlyphDataset(records, "train", SEED)
    val_set = GlyphDataset(records, "val", SEED + 13)
    sampler = CharacterBatchSampler(train_set.records, args.batch_size, SEED)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
    device = device_for_run(args.device)
    model = LocalShapeEncoder(embedding_dim=args.embedding_dim,
                              family_count=len(labels["indexes"]["family"]),
                              face_count=len(labels["indexes"]["face"]),
                              weight_count=len(labels["indexes"]["weight"])).to(device)
    count = parameter_count(model)
    if not 300_000 <= count <= 1_000_000:
        raise ValueError(f"unexpected model size: {count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history = []
    best = (float("inf"), None)
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()
        sums = defaultdict(float)
        steps = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = batch_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            steps += 1
            for key, value in metrics.items():
                sums[key] += value
        scheduler.step()
        train_metrics = {f"train_{key}": value / max(1, steps) for key, value in sums.items()}
        val_metrics = evaluate(model, val_set, device)
        row = {"epoch": epoch + 1, "learning_rate": scheduler.get_last_lr()[0],
               **train_metrics, **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        score = row.get("val_face_accuracy", 0.) + row.get("val_family_accuracy", 0.)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if best[1] is None or score > best[0]:
            best = (score, {key: value.detach().cpu() for key, value in model.state_dict().items()})
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    frozen_manifest = manifest(records, labels)
    (output / "manifest.json").write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2) + "\n")
    checkpoint = {"model": best[1], "config": {"embedding_dim": args.embedding_dim,
                 "family_count": len(labels["indexes"]["family"]),
                 "face_count": len(labels["indexes"]["face"]),
                 "weight_count": len(labels["indexes"]["weight"])},
                  "parameter_count": count, "best_score": best[0], "seed": SEED,
                  "device": str(device)}
    torch.save(checkpoint, output / "checkpoint.pt")
    result = {"status": "trained_font_only_development_model", "parameter_count": count,
              "embedding_dim": args.embedding_dim, "device": str(device),
              "records": len(records), "train_records": len(train_set),
              "val_records": len(val_set), "history": history,
              "best_score": best[0], "output": str(output)}
    (output / "training.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/local-encoder-v11/final")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"preserve existing output: {args.output}")
    train(args)


if __name__ == "__main__":
    main()
