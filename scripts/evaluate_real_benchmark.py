#!/usr/bin/env python3
"""
evaluate_real_benchmark.py — Offline Real-World Piece Recognition Benchmark

Milestone 4.0 — Phase 4.3D
AI Chess Navigator (Offline Mobile Application)

Objective:
Evaluates the deployed synthetic-trained HOG piece classifier (from
src/vision/pieceModelWeights.ts) against real-world physical chessboard
piece images without modifying model weights.

Dataset Provenance:
- Source: Zenodo Record 6656212 (Universidad Carlos III de Madrid)
- DOI: 10.5281/zenodo.6656212
- Authors: Pablo Tomas Campos & Antonio Berlanga (UC3M)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Verified Checksum: md5:4583133a23383c05c982dd4ddd549068
- Characteristics: 4 distinct physical chess sets captured by static overhead
  camera across all 64 squares of a chessboard (181x181 PNG crops).

Output:
- Full evaluation metrics (Precision, Recall, F1, Top-1, Top-2, Confusion Matrix,
  Abstention Rate, Latency, Failure Breakdown) comparing Synthetic vs Real.
"""

import argparse
import glob
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

PIECE_CLASSES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PIECE_CLASSES)}

# Exact HOG Parameters matching classifyPiece.ts
HOG_PATCH_SIZE = 64
HOG_CELL_SIZE = 16
HOG_NUM_BINS = 8
HOG_CELLS_PER_AXIS = 4
HOG_BLOCKS_PER_AXIS = 3
HOG_BLOCK_SIZE = 2
HOG_FEATURE_DIMENSION = 288

# Production Thresholds matching classifyPiece.ts
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_MARGIN_THRESHOLD = 0.20


def extract_hog_288(image_patch: np.ndarray) -> np.ndarray:
    """
    Extracts deterministic 288-bin HOG features matching classifyPiece.ts exactly.
    Input: (64, 64) float32/uint8 2D grayscale image array.
    Output: (288,) float32 normalized feature vector.
    """
    if image_patch.shape != (HOG_PATCH_SIZE, HOG_PATCH_SIZE):
        raise ValueError(
            f"Expected patch shape ({HOG_PATCH_SIZE}, {HOG_PATCH_SIZE}), got {image_patch.shape}."
        )

    img = image_patch.astype(np.float32)
    cell_hist = np.zeros((HOG_CELLS_PER_AXIS, HOG_CELLS_PER_AXIS, HOG_NUM_BINS), dtype=np.float32)

    for py in range(1, HOG_PATCH_SIZE - 1):
        cell_y = min(HOG_CELLS_PER_AXIS - 1, py // HOG_CELL_SIZE)
        for px in range(1, HOG_PATCH_SIZE - 1):
            cell_x = min(HOG_CELLS_PER_AXIS - 1, px // HOG_CELL_SIZE)

            # Central difference gradients matching classifyPiece.ts
            gx = float(img[py, px + 1]) - float(img[py, px - 1])
            gy = float(img[py + 1, px]) - float(img[py - 1, px])

            mag = math.sqrt(gx * gx + gy * gy)
            if mag < 1.0:
                continue

            # Unsigned angle in degrees [0, 180)
            angle = (math.atan2(gy, gx) * 180.0) / math.pi
            if angle < 0.0:
                angle += 180.0
            if angle >= 180.0:
                angle = 0.0

            bin_idx = min(HOG_NUM_BINS - 1, int(angle / 22.5))
            cell_hist[cell_y, cell_x, bin_idx] += mag

    # 3x3 overlapping blocks of 2x2 cells, L2-norm normalized
    blocks: List[float] = []
    eps = 1e-4

    for by in range(HOG_BLOCKS_PER_AXIS):
        for bx in range(HOG_BLOCKS_PER_AXIS):
            block_vec: List[float] = []
            for cy in range(by, by + HOG_BLOCK_SIZE):
                for cx in range(bx, bx + HOG_BLOCK_SIZE):
                    block_vec.extend(cell_hist[cy, cx, :])

            b_arr = np.array(block_vec, dtype=np.float32)
            norm = math.sqrt(float(np.sum(b_arr * b_arr)) + eps * eps)
            b_arr /= norm
            blocks.extend(b_arr)

    return np.array(blocks, dtype=np.float32)


def load_deployed_weights_from_ts(ts_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Parses weights, biases, and temperature directly from src/vision/pieceModelWeights.ts.
    """
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"Weights file not found at: {ts_path}")

    with open(ts_path, "r", encoding="utf-8") as f:
        content = f.read()

    weights_match = re.search(r"weights:\s*\[(.*?)\]\s*,\s*biases:", content, re.DOTALL)
    if not weights_match:
        raise ValueError("Could not locate 'weights' in pieceModelWeights.ts")

    rows = re.findall(r"\[(.*?)\]", weights_match.group(1), re.DOTALL)
    weights = []
    for r in rows:
        weights.append([float(x.strip()) for x in r.split(",") if x.strip()])

    biases_match = re.search(r"biases:\s*\[(.*?)\]", content)
    if not biases_match:
        raise ValueError("Could not locate 'biases' in pieceModelWeights.ts")

    biases = [float(x.strip()) for x in biases_match.group(1).split(",") if x.strip()]

    temp_match = re.search(r"temperature:\s*([0-9.]+)", content)
    temperature = float(temp_match.group(1)) if temp_match else 1.0

    W = np.array(weights, dtype=np.float32)
    b = np.array(biases, dtype=np.float32)

    if W.shape != (6, HOG_FEATURE_DIMENSION):
        raise ValueError(f"Expected W shape (6, {HOG_FEATURE_DIMENSION}), got {W.shape}")
    if b.shape != (6,):
        raise ValueError(f"Expected b shape (6,), got {b.shape}")

    return W, b, temperature


def predict_piece(
    features: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[int, np.ndarray, float, float]:
    """
    Evaluates linear classifier with temperature-scaled softmax.
    Returns: (predicted_class_idx, probabilities, top_confidence, margin)
    """
    logits = np.dot(W, features) + b
    scaled_logits = logits / max(temperature, 0.01)
    exp_z = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_z / np.sum(exp_z)

    pred_idx = int(np.argmax(probs))
    sorted_probs = np.sort(probs)
    conf = float(sorted_probs[-1])
    margin = float(sorted_probs[-1] - sorted_probs[-2])

    return pred_idx, probs, conf, margin


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    conf_thresh: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_thresh: float = DEFAULT_MARGIN_THRESHOLD,
) -> Dict:
    """
    Computes precision, recall, F1, top-1, top-2, confusion matrix, and abstention rate.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return {}

    n_classes = len(PIECE_CLASSES)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    top1_correct = int(np.sum(y_true == y_pred))
    top1_acc = float(top1_correct / n_samples)

    top2_correct = 0
    abstain_count = 0

    for i in range(n_samples):
        # Top-2 check
        top2_classes = np.argsort(y_prob[i])[-2:]
        if y_true[i] in top2_classes:
            top2_correct += 1

        # Abstention check
        s_prob = np.sort(y_prob[i])
        conf = float(s_prob[-1])
        margin = float(s_prob[-1] - s_prob[-2])
        if conf < conf_thresh or margin < margin_thresh:
            abstain_count += 1

    top2_acc = float(top2_correct / n_samples)
    abstain_rate = float(abstain_count / n_samples)

    # Per-class metrics
    per_class = {}
    for c_idx, c_name in enumerate(PIECE_CLASSES):
        tp = int(cm[c_idx, c_idx])
        fp = int(np.sum(cm[:, c_idx]) - tp)
        fn = int(np.sum(cm[c_idx, :]) - tp)
        support = int(np.sum(cm[c_idx, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        per_class[c_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    return {
        "n_samples": n_samples,
        "top1_correct": top1_correct,
        "top1_accuracy": round(top1_acc, 4),
        "top2_correct": top2_correct,
        "top2_accuracy": round(top2_acc, 4),
        "abstain_count": abstain_count,
        "abstain_rate": round(abstain_rate, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def evaluate_zenodo_split(
    data_dir: str,
    split_name: str,
    W: np.ndarray,
    b: np.ndarray,
    temperature: float = 1.0,
    preprocessing: str = "direct_resize",
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluates real chess piece images from the specified Zenodo split folder.
    """
    split_path = os.path.join(data_dir, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split directory not found: {split_path}")

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    records: List[Dict] = []

    # Latency tracking
    extraction_times_ns: List[int] = []
    inference_times_ns: List[int] = []

    for cls_dir in sorted(os.listdir(split_path)):
        if cls_dir == "empty":
            continue

        parts = cls_dir.split("_", 1)
        if len(parts) != 2:
            continue
        color, piece_type = parts[0], parts[1]

        if piece_type not in CLASS_TO_IDX:
            continue
        true_idx = CLASS_TO_IDX[piece_type]

        img_files = glob.glob(os.path.join(split_path, cls_dir, "*.png"))
        for fpath in img_files:
            fname = os.path.basename(fpath)
            # Parse filename e.g. white_knight_1_E4.png
            m = re.match(r"([a-z]+)_([a-z]+)_([0-9]+)_([A-H][1-8])\.png", fname)
            square_coord = m.group(4) if m else "unknown"

            # Image loading and preprocessing
            raw_im = Image.open(fpath).convert("L")
            if preprocessing == "direct_resize":
                proc_im = raw_im.resize((HOG_PATCH_SIZE, HOG_PATCH_SIZE), Image.Resampling.BILINEAR)
            elif preprocessing == "center_crop":
                # Resize square to 100x100, then crop central 64x64 matching mobile app
                im100 = raw_im.resize((100, 100), Image.Resampling.BILINEAR)
                left = (100 - HOG_PATCH_SIZE) // 2
                top = (100 - HOG_PATCH_SIZE) // 2
                proc_im = im100.crop((left, top, left + HOG_PATCH_SIZE, top + HOG_PATCH_SIZE))
            else:
                proc_im = raw_im.resize((HOG_PATCH_SIZE, HOG_PATCH_SIZE), Image.Resampling.BILINEAR)

            patch_arr = np.array(proc_im, dtype=np.float32)

            t0 = time.perf_counter_ns()
            feat = extract_hog_288(patch_arr)
            t1 = time.perf_counter_ns()
            pred_idx, probs, conf, margin = predict_piece(feat, W, b, temperature)
            t2 = time.perf_counter_ns()

            extraction_times_ns.append(t1 - t0)
            inference_times_ns.append(t2 - t1)

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_prob.append(probs)

            records.append({
                "file": fname,
                "color": color,
                "piece_type": piece_type,
                "square": square_coord,
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "pred_type": PIECE_CLASSES[pred_idx],
                "confidence": round(conf, 4),
                "margin": round(margin, 4),
                "is_correct": bool(true_idx == pred_idx),
            })

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))

    # Add latency statistics (convert ns to ms)
    avg_extract_ms = float(np.mean(extraction_times_ns) / 1e6)
    avg_infer_ms = float(np.mean(inference_times_ns) / 1e6)
    total_patch_ms = avg_extract_ms + avg_infer_ms
    board_64_ms = total_patch_ms * 64

    metrics["latency"] = {
        "avg_extraction_ms_per_patch": round(avg_extract_ms, 4),
        "avg_inference_ms_per_patch": round(avg_infer_ms, 4),
        "total_ms_per_patch": round(total_patch_ms, 4),
        "projected_64_square_board_ms": round(board_64_ms, 2),
    }

    return metrics, records


def print_metrics_table(title: str, metrics: Dict):
    """Prints a structured ASCII report of evaluation metrics."""
    print(f"\n========================================================================================")
    print(f" {title.upper()}")
    print(f"========================================================================================")
    print(f"Total Evaluated Samples: {metrics['n_samples']}")
    print(f"Top-1 Accuracy:          {metrics['top1_accuracy'] * 100:.2f}% ({metrics['top1_correct']}/{metrics['n_samples']})")
    print(f"Top-2 Accuracy:          {metrics['top2_accuracy'] * 100:.2f}% ({metrics['top2_correct']}/{metrics['n_samples']})")
    print(f"Abstention Rate:         {metrics['abstain_rate'] * 100:.2f}% ({metrics['abstain_count']}/{metrics['n_samples']}) [conf < 0.60 or margin < 0.20]")

    lat = metrics.get("latency", {})
    if lat:
        print(f"HOG Extraction Latency:  {lat['avg_extraction_ms_per_patch']:.3f} ms / patch")
        print(f"Linear Inference Latency:{lat['avg_inference_ms_per_patch']:.4f} ms / patch")
        print(f"Projected Full Board:    {lat['projected_64_square_board_ms']:.1f} ms / 64 squares")

    print("\n--- Per-Class Classification Report ---")
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 52)
    for c_name in PIECE_CLASSES:
        pc = metrics["per_class"][c_name]
        print(f"{c_name:<10} {pc['precision']:>10.4f} {pc['recall']:>10.4f} {pc['f1']:>10.4f} {pc['support']:>10d}")

    print("\n--- Confusion Matrix (Rows = Ground Truth, Columns = Predicted) ---")
    header = "          " + " ".join(f"{c[:4]:>6}" for c in PIECE_CLASSES)
    print(header)
    for r_idx, row in enumerate(metrics["confusion_matrix"]):
        print(f"{PIECE_CLASSES[r_idx]:<8}  " + " ".join(f"{v:6d}" for v in row))


def analyze_breakdowns(records: List[Dict]):
    """Analyzes failure modes across color and board position."""
    print("\n--- Failure Mode Breakdown ---")

    # 1. Color Breakdown
    white_recs = [r for r in records if r["color"] == "white"]
    black_recs = [r for r in records if r["color"] == "black"]

    white_acc = sum(r["is_correct"] for r in white_recs) / len(white_recs) if white_recs else 0.0
    black_acc = sum(r["is_correct"] for r in black_recs) / len(black_recs) if black_recs else 0.0

    print(f"White Pieces Accuracy:   {white_acc * 100:.2f}% ({sum(r['is_correct'] for r in white_recs)}/{len(white_recs)})")
    print(f"Black Pieces Accuracy:   {black_acc * 100:.2f}% ({sum(r['is_correct'] for r in black_recs)}/{len(black_recs)})")

    # 2. Board Position Breakdown (Center vs Corners)
    center_squares = {"D4", "D5", "E4", "E5"}
    corner_squares = {"A1", "A8", "H1", "H8"}

    center_recs = [r for r in records if r["square"] in center_squares]
    corner_recs = [r for r in records if r["square"] in corner_squares]
    other_recs = [r for r in records if r["square"] not in center_squares and r["square"] not in corner_squares]

    center_acc = sum(r["is_correct"] for r in center_recs) / len(center_recs) if center_recs else 0.0
    corner_acc = sum(r["is_correct"] for r in corner_recs) / len(corner_recs) if corner_recs else 0.0
    other_acc = sum(r["is_correct"] for r in other_recs) / len(other_recs) if other_recs else 0.0

    print(f"Center Squares (D4,D5,E4,E5): {center_acc * 100:.2f}% ({sum(r['is_correct'] for r in center_recs)}/{len(center_recs)})")
    print(f"Corner Squares (A1,A8,H1,H8): {corner_acc * 100:.2f}% ({sum(r['is_correct'] for r in corner_recs)}/{len(corner_recs)})")
    print(f"Remaining 56 Squares:         {other_acc * 100:.2f}% ({sum(r['is_correct'] for r in other_recs)}/{len(other_recs)})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deployed HOG piece classifier on real images.")
    parser.add_argument(
        "--data-dir",
        default=r"C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\extracted",
        help="Path to extracted Zenodo 6656212 dataset directory",
    )
    parser.add_argument(
        "--weights-file",
        default=r"C:\dev\chess\src\vision\pieceModelWeights.ts",
        help="Path to deployed TypeScript weights file",
    )
    args = parser.parse_args()

    print("========================================================================================")
    print("AI CHESS NAVIGATOR — OFFLINE PIECE RECOGNITION REAL-WORLD BENCHMARK")
    print("========================================================================================")
    print(f"Dataset Provenance: Zenodo Record 6656212 (Campos & Berlanga, UC3M, CC BY 4.0)")
    print(f"Model Deployed:     {args.weights_file}")

    W, b, temp = load_deployed_weights_from_ts(args.weights_file)
    print(f"Model Matrix Shape: W={W.shape}, b={b.shape}, temperature={temp}")

    # Evaluation on Held-Out Set 4 (valid/ folder)
    print("\n>>> Evaluating Held-Out Real Chess Set 4 (valid/) [Zero-Leakage Test Split]...")
    valid_metrics_resize, valid_records_resize = evaluate_zenodo_split(
        args.data_dir, "valid", W, b, temp, preprocessing="direct_resize"
    )
    print_metrics_table("Held-Out Real Set 4 (valid/ split) — Direct Resize", valid_metrics_resize)
    analyze_breakdowns(valid_records_resize)

    # Evaluation on Sets 1-3 (train/ folder)
    print("\n>>> Evaluating Real Sets 1, 2, 3 (train/) [In-Domain Physical Variations]...")
    train_metrics, train_records = evaluate_zenodo_split(
        args.data_dir, "train", W, b, temp, preprocessing="direct_resize"
    )
    print_metrics_table("Real Sets 1, 2, 3 (train/ split) — Cross-Set Analysis", train_metrics)

    print("\nPhase 4.3D real-world baseline benchmark completed successfully.")


if __name__ == "__main__":
    main()
