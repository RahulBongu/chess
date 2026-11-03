#!/usr/bin/env python3
"""Train a clean-room 13-class RGB chess-square classifier.

Data:
  * FENiT v1 (Mendeley DOI 10.17632/wpp9xnbcp6.1, CC BY 4.0). FENiT square
    labels are derived from the published FEN placement and already-rectified
    boards whose orientation was resolved against that FEN.
  * Zenodo 6656212 (Campos & Berlanga, UC3M, CC BY 4.0), physical Sets 1-3
    for training and Set 4 as a cross-set holdout.

The output is a mobile-sized ONNX classifier (13 classes: empty + six white +
six black piece types). This script never overwrites the production HOG files.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance

PIECE_TYPES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
CLASSES = ["empty"] + [f"white_{p}" for p in PIECE_TYPES] + [f"black_{p}" for p in PIECE_TYPES]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PIECE_TYPE_TO_IDX = {p: i for i, p in enumerate(PIECE_TYPES)}

LOGICAL_SIZE = 72
INPUT_SIZE = 64
SEED = 20261103


@dataclass
class Sample:
    path: str
    label: int
    group: str
    source: str


def class_from_char(ch: str) -> str:
    if not ch:
        return "empty"
    name = {
        "p": "pawn",
        "n": "knight",
        "b": "bishop",
        "r": "rook",
        "q": "queen",
        "k": "king",
    }.get(ch.lower())
    if name is None:
        raise ValueError(f"unknown piece char {ch!r}")
    return ("white_" if ch.isupper() else "black_") + name


def stable_bucket(text: str, modulo: int = 7) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def load_fenit(records_path: Path) -> Tuple[List[Sample], List[Sample]]:
    records = [json.loads(line) for line in records_path.open(encoding="utf-8") if line.strip()]
    scene_counts = collections.Counter(r.get("scene_id") for r in records)
    train: List[Sample] = []
    val: List[Sample] = []
    skipped = collections.Counter()

    for record in records:
        if record.get("status") != "PROCESSED":
            skipped["not_processed"] += 1
            continue
        scene_id = record.get("scene_id")
        if not scene_id or scene_counts[scene_id] > 1:
            skipped["scene_id_collision"] += 1
            continue
        crops = record.get("crops") or []
        if len(crops) != 64 or any(not Path(c["crop_path"]).exists() for c in crops):
            skipped["missing_or_incomplete_crops"] += 1
            continue
        placement = record.get("ground_truth_placement") or record.get("file")
        # Keep every scene of one position on the same side of the split.
        split = "val" if stable_bucket(placement, 7) == 0 else "train"
        destination = val if split == "val" else train
        for crop in crops:
            label_name = class_from_char(crop.get("piece_char") or "")
            destination.append(
                Sample(
                    path=crop["crop_path"],
                    label=CLASS_TO_IDX[label_name],
                    group=placement,
                    source=f"fenit:{split}",
                )
            )
    print(
        f"FENiT: train={len(train)} val={len(val)} skipped={dict(skipped)}",
        flush=True,
    )
    return train, val


def load_zenodo(root: Path) -> Tuple[List[Sample], List[Sample]]:
    train: List[Sample] = []
    val: List[Sample] = []
    for split, destination in (("train", train), ("valid", val)):
        split_dir = root / split
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_name = class_dir.name
            if class_name not in CLASS_TO_IDX:
                raise ValueError(f"unexpected Zenodo class {class_name}")
            for path in sorted(class_dir.glob("*.png")):
                # Zenodo's valid/ directory is physical Set 4. Filenames inside
                # it still carry "_1_", so the split has to override the ID.
                if split == "valid":
                    set_id = "4"
                else:
                    match = re.search(r"_(\d+)_[A-Ha-h][1-8]\.png$", path.name)
                    set_id = match.group(1) if match else "empty"
                group = f"zenodo:{split}:{set_id}"
                destination.append(
                    Sample(
                        path=str(path),
                        label=CLASS_TO_IDX[class_name],
                        group=group,
                        source=f"zenodo:{split}:set{set_id}",
                    )
                )
    print(f"Zenodo: train={len(train)} valid(Set4)={len(val)}", flush=True)
    return train, val


def load_digital(manifest_path: Path) -> List[Sample]:
    if not manifest_path.exists():
        print(f"Digital manifest not found: {manifest_path}", flush=True)
        return []
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = [
        Sample(
            path=r["path"],
            label=r["label"],
            group=r["group"],
            source=r["source"],
        )
        for r in records
    ]
    print(f"Digital: train={len(samples)}", flush=True)
    return samples


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def build_cache(samples: Sequence[Sample], name: str, cache_dir: Path) -> np.ndarray:
    cache_path = cache_dir / f"{name}_{LOGICAL_SIZE}.npy"
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")
    cache_dir.mkdir(parents=True, exist_ok=True)
    array = np.empty((len(samples), LOGICAL_SIZE, LOGICAL_SIZE, 3), dtype=np.uint8)
    for index, sample in enumerate(samples):
        image = Image.fromarray(load_image(sample.path), mode="RGB")
        image = image.resize((LOGICAL_SIZE, LOGICAL_SIZE), Image.Resampling.BILINEAR)
        array[index] = np.asarray(image, dtype=np.uint8)
        if (index + 1) % 5000 == 0:
            print(f"  cached {index + 1}/{len(samples)} {name}", flush=True)
    np.save(cache_path, array)
    return array


def augment_batch(batch: np.ndarray, rng: np.random.Generator, strong: bool = False) -> np.ndarray:
    """Cheap per-sample augmentation on cached 72x72 RGB uint8 crops."""
    count = batch.shape[0]
    crop = LOGICAL_SIZE - INPUT_SIZE
    offsets_y = rng.integers(0, crop + 1, size=count)
    offsets_x = rng.integers(0, crop + 1, size=count)
    out = np.empty((count, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    for i in range(count):
        y = int(offsets_y[i])
        x = int(offsets_x[i])
        tile = batch[i, y : y + INPUT_SIZE, x : x + INPUT_SIZE].copy()
        if rng.random() < 0.5:
            tile = tile[:, ::-1]
        # Never rotate a square by 90/180 degrees or flip it vertically.
        # Chess pieces are upright within an 8x8 board grid regardless of how
        # the phone is held, so upside-down training squares teach invariance
        # to a cue that cannot occur at inference and make king/queen/pawn
        # discrimination needlessly harder.
        # Brightness and channel gain changes model exposure and board colour.
        tile = tile.astype(np.float32)
        if strong:
            gain = rng.uniform(0.68, 1.38)
            channel_gain = rng.uniform(0.84, 1.16, size=3).astype(np.float32)
            bias = rng.uniform(-22.0, 22.0)
        else:
            gain = rng.uniform(0.78, 1.26)
            channel_gain = rng.uniform(0.94, 1.06, size=3).astype(np.float32)
            bias = rng.uniform(-14.0, 14.0)
        tile = tile * gain * channel_gain + bias
        if strong and rng.random() < 0.08:
            # Occasionally remove colour entirely so shape must carry the class.
            gray = tile.mean(axis=2, keepdims=True)
            tile = np.repeat(gray, 3, axis=2)
        elif strong and rng.random() < 0.25:
            gray = tile.mean(axis=2, keepdims=True)
            mix = rng.uniform(0.2, 0.6)
            tile = tile * (1.0 - mix) + gray * mix
        if strong and rng.random() < 0.35:
            gamma = rng.uniform(0.72, 1.45)
            tile = 255.0 * np.power(np.clip(tile, 0.0, 255.0) / 255.0, gamma)
        # Mild blur (2x2 box) approximates camera focus variation.
        if rng.random() < 0.16:
            h, w, _ = tile.shape
            tile = (
                tile[: h // 2 * 2, : w // 2 * 2]
                .reshape(h // 2, 2, w // 2, 2, 3)
                .mean(axis=(1, 3))
            )
            tile = np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)
        # Random occlusion: force the network to use visible structure.
        if rng.random() < 0.18:
            h = rng.integers(8, 22)
            w = rng.integers(8, 22)
            y0 = rng.integers(0, INPUT_SIZE - h)
            x0 = rng.integers(0, INPUT_SIZE - w)
            fill = tile.reshape(-1, 3).mean(axis=0)
            tile[y0 : y0 + h, x0 : x0 + w] = fill
        out[i] = np.clip(tile, 0, 255).astype(np.uint8)
    return out


def to_tensor(batch: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0)
    return (tensor - 0.5) / 0.5


class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu6(self.bn1(self.depthwise(x)))
        y = F.relu6(self.bn2(self.pointwise(y)))
        if self.residual:
            y = y + x
        return y


class SquareClassifier(nn.Module):
    def __init__(self, num_classes: int = 13, width: float = 1.0) -> None:
        super().__init__()
        c1 = max(8, int(16 * width))
        c2 = max(16, int(32 * width))
        c3 = max(32, int(64 * width))
        c4 = max(48, int(96 * width))
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseBlock(c1, c2),
            DepthwiseBlock(c2, c3, stride=2),
            DepthwiseBlock(c3, c3),
            DepthwiseBlock(c3, c4, stride=2),
            DepthwiseBlock(c4, c4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, object]:
    model.eval()
    logits_all = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        # Training samples a 64x64 window from the 72x72 cache. Evaluation must
        # use the same window scale, otherwise pieces appear ~12% smaller and
        # every held-out number is measured under a different preprocessing.
        if batch.shape[1] > INPUT_SIZE and batch.shape[2] > INPUT_SIZE:
            offset_y = (batch.shape[1] - INPUT_SIZE) // 2
            offset_x = (batch.shape[2] - INPUT_SIZE) // 2
            batch = batch[
                :,
                offset_y : offset_y + INPUT_SIZE,
                offset_x : offset_x + INPUT_SIZE,
            ]
        logits_all.append(model(to_tensor(batch)).cpu())
    logits = torch.cat(logits_all)
    probs = logits.softmax(dim=1).numpy()
    predictions = probs.argmax(axis=1)
    top2 = np.argsort(probs, axis=1)[:, -2:][:, ::-1]
    correct = predictions == labels
    top2_correct = (top2 == labels[:, None]).any(axis=1)

    result: Dict[str, object] = {
        "n": int(len(labels)),
        "top1": float(correct.mean()),
        "top2": float(top2_correct.mean()),
        "per_source": {},
        "per_class": {},
    }
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        result["per_source"][source] = {
            "n": int(mask.sum()),
            "top1": float(correct[mask].mean()),
            "top2": float(top2_correct[mask].mean()),
        }
    for class_index, class_name in enumerate(CLASSES):
        mask = labels == class_index
        if not mask.any():
            continue
        result["per_class"][class_name] = {
            "n": int(mask.sum()),
            "recall": float(correct[mask].mean()),
        }
    recalls = [entry["recall"] for entry in result["per_class"].values()]
    result["macro_recall"] = float(np.mean(recalls)) if recalls else 0.0

    # FENiT samples are contiguous in blocks of 64, one block per scene. Zenodo
    # crops are stored by class, not by board, so no full-board claim is made
    # for that source. Compute these from the FENiT rows even when the caller
    # passes a mixed domain evaluation: otherwise a mixed run reports a
    # misleading 0% occupied accuracy and best-epoch selection can never move
    # past epoch 1.
    fenit_mask = sources == "fenit:val"
    if fenit_mask.any() and int(fenit_mask.sum()) % 64 == 0:
        fenit_correct = correct[fenit_mask].reshape(-1, 64)
        fenit_labels = labels[fenit_mask].reshape(-1, 64)
        occupied = fenit_labels != 0
        board_correct = fenit_correct.all(axis=1)
        result["board_count"] = int(len(board_correct))
        result["board_exact"] = float(board_correct.mean())
        result["occupied_top1"] = float(fenit_correct[occupied].mean())
        result["empty_top1"] = float(fenit_correct[~occupied].mean())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fenit-manifest",
        default=r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl",
    )
    parser.add_argument(
        "--zenodo-dir",
        default=r"C:\dev\chess-capture-workspace\external\zenodo\extracted",
    )
    parser.add_argument(
        "--cache-dir",
        default=r"C:\dev\chess-capture-workspace\training-cache",
    )
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--learning-rate", type=float, default=2e-3, help="AdamW learning rate."
    )
    parser.add_argument(
        "--width", type=float, default=1.0, help="Channel-width multiplier for the CNN."
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional PyTorch state dict to fine-tune from.",
    )
    parser.add_argument(
        "--mode",
        choices=("mixed", "balanced", "zenodo", "fenit"),
        default="mixed",
        help=(
            "mixed: natural FENiT + Zenodo mix; balanced: equal total sampling "
            "weight per source domain; zenodo: physical Sets 1-3 only; fenit: "
            "FENiT only (shape pretraining)."
        ),
    )
    parser.add_argument(
        "--strong-augment",
        action="store_true",
        help="Wider colour/exposure/rotation augmentation for cross-domain robustness.",
    )
    parser.add_argument(
        "--class-balanced",
        action="store_true",
        help=(
            "Sample every class (and, in balanced mode, every domain/class pair) "
            "with equal expected mass. Prevents empty squares from dominating."
        ),
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help=(
            "Cap the number of training samples drawn per epoch (0 = use every "
            "sample once). Class-balanced sampling draws with replacement, so a "
            "cap keeps large mixed datasets within a practical runtime."
        ),
    )
    parser.add_argument(
        "--select-metric",
        choices=("legacy", "fenit_occupied", "fenit_macro", "fenit_piece"),
        default="legacy",
        help=(
            "Best-epoch criterion. 'legacy' is the historical weighted Top-1 and "
            "uses the Zenodo Set 4 holdout, which biases that holdout. "
            "'fenit_occupied' / 'fenit_macro' select on real FENiT validation only, "
            "keeping Set 4 untouched; 'fenit_piece' averages the two."
        ),
    )
    parser.add_argument(
        "--include-digital",
        action="store_true",
        help="Include clean-room digital piece crops for laptop-screen support.",
    )
    parser.add_argument(
        "--digital-manifest",
        default=r"C:\dev\chess\dist\cache\digital_crops\manifest.json",
    )
    parser.add_argument(
        "--piece-loss-weighting",
        action="store_true",
        help="Boost loss weights for weak piece classes (kings, knights, bishops, rooks).",
    )
    parser.add_argument("--suffix", default="", help="Optional tag for cache and output names.")
    parser.add_argument(
        "--output-onnx",
        default=None,
        help="Defaults to models/rgb_square_classifier_v1_<mode>[_strong].onnx",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Defaults to dist/rgb-square-model-report-<mode>[_strong].json",
    )
    args = parser.parse_args()

    run_tag = args.mode + ("_strong" if args.strong_augment else "")
    if args.suffix:
        run_tag += f"_{args.suffix}"
    output_onnx = Path(
        args.output_onnx or rf"C:\dev\chess\models\rgb_square_classifier_v1_{run_tag}.onnx"
    )
    output_report = Path(
        args.output_report or rf"C:\dev\chess\dist\rgb-square-model-report-{run_tag}.json"
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    fenit_train, fenit_val = load_fenit(Path(args.fenit_manifest))
    zenodo_train, zenodo_val = load_zenodo(Path(args.zenodo_dir))
    digital_train = load_digital(Path(args.digital_manifest)) if args.include_digital else []
    if args.mode == "zenodo":
        train_samples = list(zenodo_train)
    elif args.mode == "fenit":
        train_samples = list(fenit_train)
    else:
        train_samples = fenit_train + zenodo_train + digital_train
    # Evaluate both domains for every mode; never pool them into one claim.
    val_samples = fenit_val + zenodo_val
    if not train_samples:
        raise SystemExit("No training samples for this mode.")

    print("Caching images...", flush=True)
    cache_dir = Path(args.cache_dir)
    # Strong augmentation does not change cached source pixels, so mixed and
    # balanced share the same cache; only the sampling policy differs.
    digital_tag = "_digital" if args.include_digital else ""
    train_cache_name = f"train_rgb_{args.mode}{digital_tag}"
    train_images = build_cache(train_samples, train_cache_name, cache_dir)
    val_images = build_cache(val_samples, "val_rgb", cache_dir)
    train_labels = np.array([s.label for s in train_samples], dtype=np.int64)
    val_labels = np.array([s.label for s in val_samples], dtype=np.int64)
    val_sources = np.array([s.source for s in val_samples])

    domains = np.array([
        "fenit" if s.source.startswith("fenit:")
        else ("digital" if s.source.startswith("digital:") else "zenodo")
        for s in train_samples
    ])
    domain_counts = collections.Counter(domains.tolist())
    if args.mode == "balanced":
        # Every source domain contributes equal expected mass per epoch. Zenodo
        # has ~8x fewer crops than FENiT, so replacement is unavoidable here.
        probabilities = np.array(
            [1.0 / domain_counts[domain] for domain in domains], dtype=np.float64
        )
        probabilities /= probabilities.sum()
        sampling = "domain-balanced (equal source mass, with replacement)"
        replace = True
    else:
        probabilities = np.full(len(train_samples), 1.0 / len(train_samples), dtype=np.float64)
        sampling = (
            "Zenodo Sets 1-3 only"
            if args.mode == "zenodo"
            else "FENiT only"
            if args.mode == "fenit"
            else "natural FENiT + Zenodo mix"
        )
        replace = False

    if args.class_balanced:
        # Each (domain, class) pair gets equal expected mass. Empty squares
        # otherwise outnumber every piece class by an order of magnitude.
        pair_counts = collections.Counter(zip(domains.tolist(), train_labels.tolist()))
        if args.mode == "balanced":
            probabilities = np.array(
                [1.0 / pair_counts[(domain, int(label))] for domain, label in zip(domains, train_labels)],
                dtype=np.float64,
            )
        else:
            class_counts = collections.Counter(train_labels.tolist())
            probabilities = np.array(
                [1.0 / class_counts[int(label)] for label in train_labels],
                dtype=np.float64,
            )
        probabilities /= probabilities.sum()
        sampling += " + class-balanced"
        replace = True

    raw_counts = np.bincount(train_labels, minlength=len(CLASSES))
    effective_counts = np.zeros(len(CLASSES), dtype=np.float64)
    for label, probability in zip(train_labels, probabilities):
        effective_counts[label] += probability * len(train_samples)
    counts = np.maximum(np.round(effective_counts), 1).astype(np.int64)
    # Square-root inverse frequency keeps rare pieces visible without allowing
    # the queen class to dominate the loss.
    weights = np.sqrt(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)))
    weights = weights / weights.mean()
    if args.piece_loss_weighting:
        # Heavily penalise collapsing complex pieces (knights, bishops, rooks, kings) into pawns.
        piece_boost = {
            "empty": 1.0,
            "white_pawn": 1.0,
            "black_pawn": 1.0,
            "white_knight": 1.6,
            "black_knight": 1.6,
            "white_bishop": 1.6,
            "black_bishop": 1.6,
            "white_rook": 1.4,
            "black_rook": 1.4,
            "white_queen": 1.4,
            "black_queen": 1.4,
            "white_king": 2.2,
            "black_king": 2.2,
        }
        boost_factors = np.array([piece_boost[c] for c in CLASSES], dtype=np.float32)
        weights = weights * boost_factors
        weights = weights / weights.mean()
        print("Piece loss weighting enabled: kings=2.2x, minor pieces=1.6x, rooks/queens=1.4x", flush=True)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print("Domain samples:", dict(domain_counts), flush=True)
    print("Raw class counts:", dict(zip(CLASSES, raw_counts.tolist())), flush=True)
    print("Effective class counts:", dict(zip(CLASSES, np.round(effective_counts, 1).tolist())), flush=True)
    print("Class weights:", dict(zip(CLASSES, np.round(weights, 3).tolist())), flush=True)
    print(f"Sampling: {sampling}; strong_augment={bool(args.strong_augment)}", flush=True)

    model = SquareClassifier(len(CLASSES), width=args.width)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(state)
        print(f"Initialised from {args.init_checkpoint}", flush=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {parameter_count:,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    rng = np.random.default_rng(SEED)

    best_score = -1.0
    best_state = None
    best_epoch = 0
    history = []
    checkpoint_path = output_onnx.parent / f"{output_onnx.stem}_best.pt"
    # The evidence gate is expressed against the held-out physical set, so its
    # weight rises as FENiT is removed from training.
    fenit_weight, zenodo_weight = {
        "mixed": (0.5, 0.5),
        "balanced": (0.35, 0.65),
        "zenodo": (0.0, 1.0),
        "fenit": (1.0, 0.0),
    }[args.mode]
    for epoch in range(args.epochs):
        model.train()
        epoch_size = (
            args.samples_per_epoch
            if args.samples_per_epoch and args.samples_per_epoch > 0
            else len(train_samples)
        )
        if replace:
            order = rng.choice(len(train_samples), size=epoch_size, replace=True, p=probabilities)
        else:
            order = np.arange(len(train_samples))
            rng.shuffle(order)
            order = order[:epoch_size]
        epoch_start = time.time()
        total_loss = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch = augment_batch(
                np.asarray(train_images[indices]), rng, strong=args.strong_augment
            )
            inputs = to_tensor(batch)
            targets = torch.from_numpy(train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)
        scheduler.step()
        evaluation = evaluate(model, np.asarray(val_images), val_labels, val_sources)
        fenit = evaluation["per_source"].get("fenit:val", {})
        zenodo = evaluation["per_source"].get("zenodo:valid:set4", {})
        # FENiT is 80% empty, so raw Top-1 can look respectable while the model
        # cannot identify a single piece. Track the metrics that matter for a
        # physical board separately.
        fenit_occupied = float(evaluation.get("occupied_top1", 0.0))
        fenit_board_exact = float(evaluation.get("board_exact", 0.0))
        fenit_macro = float(evaluation.get("macro_recall", 0.0))
        legacy_score = fenit_weight * fenit.get("top1", 0.0) + zenodo_weight * zenodo.get(
            "top1", 0.0
        )
        if args.select_metric == "fenit_occupied":
            score = fenit_occupied
        elif args.select_metric == "fenit_macro":
            score = fenit_macro
        elif args.select_metric == "fenit_piece":
            score = 0.5 * fenit_occupied + 0.5 * fenit_macro
        else:
            score = legacy_score
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / max(seen, 1),
                "fenit_top1": fenit.get("top1", 0.0),
                "fenit_occupied_top1": fenit_occupied,
                "fenit_macro_recall": fenit_macro,
                "fenit_board_exact": fenit_board_exact,
                "zenodo_set4_top1": zenodo.get("top1", 0.0),
                "score": score,
                "seconds": time.time() - epoch_start,
            }
        )
        print(
            f"epoch {epoch + 1:02d}/{args.epochs} loss={total_loss / max(seen, 1):.4f} "
            f"fenit={fenit.get('top1', 0.0) * 100:.2f}% "
            f"occ={fenit_occupied * 100:.2f}% "
            f"macro={fenit_macro * 100:.2f}% "
            f"exact={fenit_board_exact * 100:.1f}% "
            f"zenodo4={zenodo.get('top1', 0.0) * 100:.2f}% "
            f"score={score * 100:.2f}% ({time.time() - epoch_start:.1f}s)",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            output_onnx.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    torch.save(best_state if best_state is not None else model.state_dict(), checkpoint_path)
    print(f"Checkpoint: {checkpoint_path}", flush=True)

    final = evaluate(model, np.asarray(val_images), val_labels, val_sources)
    # Full breakdown for the two validation domains.
    for source in ("fenit:val", "zenodo:valid:set4"):
        mask = val_sources == source
        if not mask.any():
            continue
        domain = evaluate(model, np.asarray(val_images[mask]), val_labels[mask], val_sources[mask])
        final["per_source"][source].update(domain)

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(output_onnx),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    # Verify the exported graph against PyTorch. A file that merely exists is
    # not evidence that the Android session will receive the same logits.
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(output_onnx))
    parity_batch = np.ascontiguousarray(
        np.asarray(val_images[:1])[
            :, (LOGICAL_SIZE - INPUT_SIZE) // 2 : (LOGICAL_SIZE + INPUT_SIZE) // 2,
            (LOGICAL_SIZE - INPUT_SIZE) // 2 : (LOGICAL_SIZE + INPUT_SIZE) // 2,
        ]
    )
    with torch.no_grad():
        torch_logits = model(to_tensor(parity_batch)).cpu().numpy()
    ort_session = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
    ort_input = (parity_batch.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    ort_logits = ort_session.run(None, {"input": ort_input})[0]
    parity_max_abs_diff = float(np.abs(torch_logits - ort_logits).max())
    batch_logits = ort_session.run(None, {"input": np.repeat(ort_input, 64, axis=0)})[0]
    batch_shape_ok = tuple(batch_logits.shape) == (64, len(CLASSES))
    if not batch_shape_ok:
        raise SystemExit(f"ONNX batch shape check failed: got {batch_logits.shape}")
    if parity_max_abs_diff > 1e-4:
        raise SystemExit(
            f"ONNX parity check failed: max |torch - onnx| = {parity_max_abs_diff:.6f}"
        )
    print(
        f"ONNX parity max|diff|={parity_max_abs_diff:.2e}; batch64 shape={batch_logits.shape}",
        flush=True,
    )

    set4_top1 = (
        final.get("per_source", {}).get("zenodo:valid:set4", {}).get("top1", 0.0)
    )
    set4_top2 = (
        final.get("per_source", {}).get("zenodo:valid:set4", {}).get("top2", 0.0)
    )
    gate = {
        "required_set4_top1": 0.70,
        "required_set4_top2": 0.85,
        "actual_set4_top1": set4_top1,
        "actual_set4_top2": set4_top2,
        "passed": bool(set4_top1 >= 0.70 and set4_top2 >= 0.85),
    }
    fenit_final = final.get("per_source", {}).get("fenit:val", {})
    real_board_evidence = {
        "fenit_top1": fenit_final.get("top1", 0.0),
        "fenit_occupied_top1": fenit_final.get("occupied_top1", 0.0),
        "fenit_empty_top1": fenit_final.get("empty_top1", 0.0),
        "fenit_board_exact": fenit_final.get("board_exact", 0.0),
        "fenit_board_count": fenit_final.get("board_count", 0),
        "fenit_macro_recall": fenit_final.get("macro_recall", 0.0),
        "empty_class_share": (
            fenit_final.get("per_class", {}).get("empty", {}).get("n", 0)
            / max(fenit_final.get("n", 1), 1)
        ),
        "note": (
            "FENiT validation is 80% empty squares, so top1 alone is not evidence "
            "of piece recognition. occupied_top1 counts only non-empty squares."
        ),
    }

    report = {
        "model": "rgb_square_classifier_v1",
        "run_tag": run_tag,
        "mode": args.mode,
        "strong_augment": bool(args.strong_augment),
        "class_balanced_sampling": bool(args.class_balanced),
        "sampling": sampling,
        "select_metric": args.select_metric,
        "score_weighting": {"fenit": fenit_weight, "zenodo_set4": zenodo_weight},
        "classes": CLASSES,
        "input_size": INPUT_SIZE,
        "normalization": {"mean": 0.5, "std": 0.5, "channel_order": "RGB"},
        "parameter_count": parameter_count,
        "width": args.width,
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch or len(train_samples),
        "learning_rate": args.learning_rate,
        "initial_checkpoint": args.init_checkpoint,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
        "validation": final,
        "real_board_evidence": real_board_evidence,
        "promotion_gate": gate,
        "onnx_parity_max_abs_diff": parity_max_abs_diff,
        "checkpoint": str(checkpoint_path),
        "dataset": {
            "fenit": {
                "source": "Mendeley DOI 10.17632/wpp9xnbcp6.1",
                "license": "CC BY 4.0",
                "train_samples": len(fenit_train),
                "val_samples": len(fenit_val),
                "used_for_training": args.mode != "zenodo",
            },
            "zenodo": {
                "source": "Zenodo 6656212, UC3M",
                "license": "CC BY 4.0",
                "train_samples": len(zenodo_train),
                "val_samples": len(zenodo_val),
                "split_policy": "valid/ is physical Set 4; never trained on",
            },
            "domain_train_samples": dict(domain_counts),
            "effective_class_counts": dict(
                zip(CLASSES, np.round(effective_counts, 1).tolist())
            ),
        },
        "onnx_sha256": hashlib.sha256(output_onnx.read_bytes()).hexdigest(),
    }
    report_path = output_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)
    print(
        f"Promotion gate: set4_top1={set4_top1 * 100:.2f}% "
        f"set4_top2={set4_top2 * 100:.2f}% passed={gate['passed']}",
        flush=True,
    )
    print(f"ONNX: {output_onnx} ({output_onnx.stat().st_size / 1024:.1f} KiB)", flush=True)
    print(f"Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
