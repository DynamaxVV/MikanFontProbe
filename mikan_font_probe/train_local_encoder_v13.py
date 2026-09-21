"""Train v13 isolated variants with weaker morphology and category prototypes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from mikan_font_probe.local_encoder_v11 import LocalShapeEncoder, parameter_count
from mikan_font_probe.local_encoder_v13 import CategoryPrototypeBank, category_prototype_loss
from mikan_font_probe.process_regions import ROOT
from mikan_font_probe.train_local_encoder_v11 import (
    CATALOG_PATH, SEED, CharacterBatchSampler, _place, _white_outline,
    build_records, device_for_run, evaluate, manifest, tight_gray,
)


def augment_weak(crop: np.ndarray, rng: np.random.Generator,
                 native: int | None = None, deterministic: bool = False,
                 preserve_weight: bool = False) -> tuple[np.ndarray, int]:
    """v11-style rasterization with morphology reduced but domain noise kept."""
    if native is None:
        native = int(rng.integers(24, 81))
    image = _place(crop, rng, native, local=not deterministic and rng.random() < .38)
    if not deterministic:
        morphology_probability = .08 if preserve_weight else .18
        if rng.random() < morphology_probability:
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
        if rng.random() < (.08 if preserve_weight else .14):
            image = 255 - image
    return (1. - image.astype(np.float32) / 255.).astype(np.float32), native


class GlyphDatasetV13(Dataset):
    def __init__(self, records: list[dict], split: str, seed: int,
                 category_indexes: dict[str, int]):
        self.records = [row for row in records if row["split"] == split]
        self.seed = seed
        self.epoch = 0
        self.category_indexes = category_indexes
        self.images = {row["template"]: tight_gray(ROOT / row["template"])
                       for row in self.records}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + index * 7919)
        base_native = int(rng.integers(24, 81))
        preserve_weight = row["category"] in {"少女体", "方圆体", "海报体"}
        first, native_a = augment_weak(self.images[row["template"]], rng, base_native,
                                       preserve_weight=preserve_weight)
        second, native_b = augment_weak(
            self.images[row["template"]], rng,
            int(np.clip(base_native + rng.integers(-3, 4), 24, 81)),
            preserve_weight=preserve_weight)
        return {"a": torch.from_numpy(first[None]), "b": torch.from_numpy(second[None]),
                "native_a": torch.tensor(native_a, dtype=torch.float32),
                "native_b": torch.tensor(native_b, dtype=torch.float32),
                "family": row["family_index"], "face": row["face_index"],
                "weight": row["weight_index"], "category": self.category_indexes[row["category"]],
                "character": ord(row["character"])}


def records_with_categories() -> tuple[list[dict], dict, dict[str, int]]:
    records, labels = build_records()
    categories = sorted({row["category"] for row in records})
    indexes = {category: index for index, category in enumerate(categories)}
    return records, labels, indexes


def batch_loss_v13(model: LocalShapeEncoder, bank: CategoryPrototypeBank,
                   batch: dict, device: torch.device,
                   use_category_prototype: bool) -> tuple[torch.Tensor, dict]:
    a = model(batch["a"].to(device), batch["native_a"].to(device), heads=True)
    b = model(batch["b"].to(device), batch["native_b"].to(device), heads=True)
    family, face, weight = (batch[key].to(device) for key in ("family", "face", "weight"))
    ce = nn.CrossEntropyLoss()
    family_loss = (ce(a["family"], family) + ce(b["family"], family)) / 2
    face_loss = (ce(a["face"], face) + ce(b["face"], face)) / 2
    weight_loss = (ce(a["weight"], weight) + ce(b["weight"], weight)) / 2
    similarity = a["embedding"] @ b["embedding"].T
    targets = torch.arange(similarity.shape[0], device=device)
    nce = (ce(similarity / .08, targets) + ce(similarity.T / .08, targets)) / 2
    positive = torch.diagonal(similarity)
    characters = batch["character"].to(device)
    negative_mask = (characters[:, None] == characters[None, :]) & (family[:, None] != family[None, :])
    hard = torch.relu(.18 + similarity - positive[:, None])
    hard = hard[negative_mask].mean() if negative_mask.any() else similarity.new_zeros(())
    prototype = similarity.new_zeros(())
    if use_category_prototype:
        category = batch["category"].to(device)
        prototype = (category_prototype_loss(bank, a["embedding"], category)
                     + category_prototype_loss(bank, b["embedding"], category)) / 2
    if use_category_prototype:
        # Keep the total scale stable while reserving 0.12 for the new
        # category-center objective.
        loss = (.20 * family_loss + .16 * face_loss + .08 * weight_loss
                + .30 * nce + .14 * hard + .12 * prototype)
    else:
        # Exact v11 objective: this stage isolates augmentation only.
        loss = (.22 * family_loss + .18 * face_loss + .10 * weight_loss
                + .35 * nce + .15 * hard)
    metrics = {"loss": float(loss.detach().cpu()),
               "family": float(family_loss.detach().cpu()),
               "face": float(face_loss.detach().cpu()),
               "weight": float(weight_loss.detach().cpu()),
               "nce": float(nce.detach().cpu()), "hard": float(hard.detach().cpu()),
               "prototype": float(prototype.detach().cpu())}
    return loss, metrics


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    records, labels, category_indexes = records_with_categories()
    train_set = GlyphDatasetV13(records, "train", SEED, category_indexes)
    val_set = GlyphDatasetV13(records, "val", SEED + 13, category_indexes)
    sampler = CharacterBatchSampler(train_set.records, args.batch_size, SEED)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
    device = device_for_run(args.device)
    model = LocalShapeEncoder(
        embedding_dim=args.embedding_dim,
        family_count=len(labels["indexes"]["family"]),
        face_count=len(labels["indexes"]["face"]),
        weight_count=len(labels["indexes"]["weight"]),
    ).to(device)
    bank = CategoryPrototypeBank(len(category_indexes), args.embedding_dim).to(device)
    count = parameter_count(model)
    if not 300_000 <= count <= 1_000_000:
        raise ValueError(f"unexpected model size: {count}")
    parameters = list(model.parameters()) + (list(bank.parameters()) if args.prototype else [])
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history = []
    best = (float("-inf"), None)
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()
        sums = defaultdict(float)
        steps = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = batch_loss_v13(model, bank, batch, device, args.prototype)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2.)
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
    frozen_manifest["schema"] = "local_encoder_v13"
    frozen_manifest["protocol"]["morphology"] = "reduced: 0.08 weight-sensitive, 0.18 other"
    frozen_manifest["protocol"]["category_prototype"] = bool(args.prototype)
    frozen_manifest["protocol"]["real_pixels_in_training"] = False
    (output / "manifest.json").write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2) + "\n")
    config = {"embedding_dim": args.embedding_dim,
              "family_count": len(labels["indexes"]["family"]),
              "face_count": len(labels["indexes"]["face"]),
              "weight_count": len(labels["indexes"]["weight"])}
    torch.save({"model": best[1], "config": config, "parameter_count": count,
                "best_score": best[0], "seed": SEED, "device": str(device)},
               output / "checkpoint.pt")
    result = {"status": "trained_font_only_v13_development_model",
              "stage": args.stage, "parameter_count": count,
              "embedding_dim": args.embedding_dim, "device": str(device),
              "records": len(records), "train_records": len(train_set),
              "val_records": len(val_set), "history": history,
              "best_score": best[0], "output": str(output),
              "prototype": args.prototype, "category_count": len(category_indexes)}
    (output / "training.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("weak-augmentation", "category-prototype"),
                        required=True)
    parser.add_argument("--prototype", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"preserve existing output: {args.output}")
    if args.stage == "category-prototype" and not args.prototype:
        raise ValueError("category-prototype stage requires --prototype")
    train(args)


if __name__ == "__main__":
    main()
