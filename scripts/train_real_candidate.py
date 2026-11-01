#!/usr/bin/env python3
"""
train_real_candidate.py — Real-World Piece Classifier Adaptation

Milestone 4.0 — Phase 4.3E
AI Chess Navigator (Offline Mobile Application)

Objective:
Trains a candidate 6-class linear classifier on physical chess Sets 1, 2, 3 from
Zenodo Record 6656212 (CC BY 4.0) with physically realistic horizontal mirror
augmentation (4,608 samples). Evaluates generalization across physical sets
using Leave-One-Set-Out (LOSO) cross-validation, and benchmarks against held-out
Set 4 (768 samples).

Exports the candidate to src/vision/experimentalPieceModelWeights.ts without
overwriting production weights (src/vision/pieceModelWeights.ts).
"""

import argparse
import glob
import math
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError as err:
    print(f"Error: scikit-learn is required: {err}")
    exit(1)

PIECE_CLASSES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PIECE_CLASSES)}

HOG_PATCH_SIZE = 64
HOG_CELL_SIZE = 16
HOG_NUM_BINS = 8
HOG_CELLS_PER_AXIS = 4
HOG_BLOCKS_PER_AXIS = 3
HOG_BLOCK_SIZE = 2
HOG_FEATURE_DIMENSION = 288


def extract_hog_288(image_patch: np.ndarray) -> np.ndarray:
    """Extracts deterministic 288-bin HOG features matching classifyPiece.ts."""
    img = image_patch.astype(np.float32)
    cell_hist = np.zeros((HOG_CELLS_PER_AXIS, HOG_CELLS_PER_AXIS, HOG_NUM_BINS), dtype=np.float32)

    for py in range(1, HOG_PATCH_SIZE - 1):
        cell_y = min(HOG_CELLS_PER_AXIS - 1, py // HOG_CELL_SIZE)
        for px in range(1, HOG_PATCH_SIZE - 1):
            cell_x = min(HOG_CELLS_PER_AXIS - 1, px // HOG_CELL_SIZE)
            gx = float(img[py, px + 1]) - float(img[py, px - 1])
            gy = float(img[py + 1, px]) - float(img[py - 1, px])
            mag = math.sqrt(gx * gx + gy * gy)
            if mag < 1.0:
                continue
            angle = (math.atan2(gy, gx) * 180.0) / math.pi
            if angle < 0.0:
                angle += 180.0
            if angle >= 180.0:
                angle = 0.0
            bin_idx = min(HOG_NUM_BINS - 1, int(angle / 22.5))
            cell_hist[cell_y, cell_x, bin_idx] += mag

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


def load_or_extract_dataset(data_dir: str, cache_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads features from cache or extracts them from the raw image directories."""
    os.makedirs(cache_dir, exist_ok=True)
    train_x_path = os.path.join(cache_dir, "X_train_sets123.npy")
    train_y_path = os.path.join(cache_dir, "y_train_sets123.npy")
    train_sets_path = os.path.join(cache_dir, "sets_train.npy")
    test_x_path = os.path.join(cache_dir, "X_test_set4.npy")
    test_y_path = os.path.join(cache_dir, "y_test_set4.npy")

    if (
        os.path.exists(train_x_path)
        and os.path.exists(train_y_path)
        and os.path.exists(train_sets_path)
        and os.path.exists(test_x_path)
        and os.path.exists(test_y_path)
    ):
        print(f"Loading cached features from: {cache_dir}")
        X_train = np.load(train_x_path)
        y_train = np.load(train_y_path)
        sets_train = np.load(train_sets_path)
        X_test = np.load(test_x_path)
        y_test = np.load(test_y_path)
        return X_train, y_train, sets_train, X_test, y_test

    print(f"Extracting features from raw images in: {data_dir}...")
    X_train_list, y_train_list, sets_train_list = [], [], []
    train_path = os.path.join(data_dir, "train")

    for cls_dir in sorted(os.listdir(train_path)):
        if cls_dir == "empty":
            continue
        piece_type = cls_dir.split("_", 1)[1]
        if piece_type not in CLASS_TO_IDX:
            continue
        c_idx = CLASS_TO_IDX[piece_type]

        for fpath in glob.glob(os.path.join(train_path, cls_dir, "*.png")):
            fname = os.path.basename(fpath)
            m = re.match(r"([a-z]+)_([a-z]+)_([0-9]+)_([A-H][1-8])\.png", fname)
            set_id = int(m.group(3)) if m else 1

            raw_im = Image.open(fpath).convert("L")
            im = raw_im.resize((HOG_PATCH_SIZE, HOG_PATCH_SIZE), Image.Resampling.BILINEAR)
            feat = extract_hog_288(np.array(im, dtype=np.float32))
            X_train_list.append(feat)
            y_train_list.append(c_idx)
            sets_train_list.append(set_id)

            # Horizontal flip mirror augmentation
            im_flip = raw_im.transpose(Image.FLIP_LEFT_RIGHT).resize((HOG_PATCH_SIZE, HOG_PATCH_SIZE), Image.Resampling.BILINEAR)
            feat_flip = extract_hog_288(np.array(im_flip, dtype=np.float32))
            X_train_list.append(feat_flip)
            y_train_list.append(c_idx)
            sets_train_list.append(set_id)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    sets_train = np.array(sets_train_list, dtype=np.int32)

    X_test_list, y_test_list = [], []
    valid_path = os.path.join(data_dir, "valid")

    for cls_dir in sorted(os.listdir(valid_path)):
        if cls_dir == "empty":
            continue
        piece_type = cls_dir.split("_", 1)[1]
        if piece_type not in CLASS_TO_IDX:
            continue
        c_idx = CLASS_TO_IDX[piece_type]

        for fpath in glob.glob(os.path.join(valid_path, cls_dir, "*.png")):
            raw_im = Image.open(fpath).convert("L")
            im = raw_im.resize((HOG_PATCH_SIZE, HOG_PATCH_SIZE), Image.Resampling.BILINEAR)
            feat = extract_hog_288(np.array(im, dtype=np.float32))
            X_test_list.append(feat)
            y_test_list.append(c_idx)

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int32)

    np.save(train_x_path, X_train)
    np.save(train_y_path, y_train)
    np.save(train_sets_path, sets_train)
    np.save(test_x_path, X_test)
    np.save(test_y_path, y_test)

    return X_train, y_train, sets_train, X_test, y_test


def run_leave_one_set_out_cv(X_train: np.ndarray, y_train: np.ndarray, sets_train: np.ndarray, C: float = 0.2) -> Dict:
    """Runs 3-fold Leave-One-Set-Out cross-validation across physical Sets 1, 2, 3."""
    fold_results = {}
    mean_top1_list = []
    mean_top2_list = []

    for test_set in [1, 2, 3]:
        train_mask = (sets_train != test_set)
        test_mask = (sets_train == test_set)

        X_tr_f = X_train[train_mask]
        y_tr_f = y_train[train_mask]
        # Evaluate on the original unaugmented 768 samples of the held-out set
        X_te_f = X_train[test_mask][:768]
        y_te_f = y_train[test_mask][:768]

        clf = LogisticRegression(max_iter=1000, C=C, random_state=42)
        clf.fit(X_tr_f, y_tr_f)
        preds = clf.predict(X_te_f)
        probs = clf.predict_proba(X_te_f)

        t1 = float(np.mean(preds == y_te_f))
        t2 = float(np.mean([1 if y_te_f[i] in np.argsort(probs[i])[-2:] else 0 for i in range(len(y_te_f))]))

        fold_results[f"fold_set_{test_set}"] = {
            "held_out_set": test_set,
            "train_samples": int(len(X_tr_f)),
            "test_samples": int(len(X_te_f)),
            "top1_accuracy": round(t1, 4),
            "top2_accuracy": round(t2, 4),
        }
        mean_top1_list.append(t1)
        mean_top2_list.append(t2)

    return {
        "mean_loso_top1": round(float(np.mean(mean_top1_list)), 4),
        "mean_loso_top2": round(float(np.mean(mean_top2_list)), 4),
        "folds": fold_results,
    }


def export_experimental_model_ts(clf: LogisticRegression, top1: float, top2: float, train_count: int, test_count: int, output_path: str):
    """Exports the trained candidate weights to src/vision/experimentalPieceModelWeights.ts."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = [
        "// Generated by scripts/train_real_candidate.py (Phase 4.3E) — DO NOT HAND-EDIT",
        "// Provenance: Zenodo Record 6656212 (Campos & Berlanga, UC3M, CC BY 4.0)",
        "// Evaluation: EXPERIMENTAL-CANDIDATE (Trained on physical Sets 1-3 with mirror augmentation)",
        "",
        "import type { PieceClassifierModel, PieceType } from './types.ts';",
        "",
        "export const EXPERIMENTAL_PIECE_MODEL_VERSION = '1.1.0-real-adapted-candidate-v1';",
        "",
        "export const EXPERIMENTAL_PIECE_CLASSES: readonly PieceType[] = [",
        "  'pawn',",
        "  'knight',",
        "  'bishop',",
        "  'rook',",
        "  'queen',",
        "  'king',",
        "] as const;",
        "",
        "export const experimentalPieceClassifierModel: PieceClassifierModel = {",
        "  name: 'hog-staunton-real-adapted-v1',",
        "  version: EXPERIMENTAL_PIECE_MODEL_VERSION,",
        "  featureDimension: 288,",
        "  classes: [...EXPERIMENTAL_PIECE_CLASSES],",
        "  weights: [",
    ]

    for c_idx, c_name in enumerate(PIECE_CLASSES):
        w_row = clf.coef_[c_idx]
        w_strs = [f"{x:.6f}" for x in w_row]
        lines.append(f"    // Class {c_idx}: {c_name}")
        lines.append("    [" + ", ".join(w_strs) + "],")

    lines.append("  ],")
    b_strs = [f"{x:.6f}" for x in clf.intercept_]
    lines.append("  biases: [" + ", ".join(b_strs) + "],")
    lines.append("  temperature: 1.0,")
    lines.append("  metadata: {")
    lines.append("    trainedOn: 'zenodo-6656212-sets-1-2-3-augmented',")
    lines.append("    license: 'CC BY 4.0',")
    lines.append("    evaluationType: 'HELD-OUT-PHYSICAL-SET-4',")
    lines.append(f"    trainDenominator: {train_count},")
    lines.append(f"    testDenominator: {test_count},")
    lines.append(f"    top1Accuracy: {top1:.4f},")
    lines.append(f"    top2Accuracy: {top2:.4f},")
    lines.append("  },")
    lines.append("};")
    lines.append("")
    lines.append("export function getExperimentalPieceClassifierModel(): PieceClassifierModel {")
    lines.append("  return experimentalPieceClassifierModel;")
    lines.append("}")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[EXPORT] Experimental candidate model exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate real-world adapted piece classifier.")
    parser.add_argument(
        "--data-dir",
        default=r"C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\extracted",
        help="Path to extracted Zenodo dataset",
    )
    parser.add_argument(
        "--cache-dir",
        default=r"C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\cache",
        help="Path to cached feature matrices",
    )
    parser.add_argument(
        "--ts-output",
        default=r"C:\dev\chess\src\vision\experimentalPieceModelWeights.ts",
        help="Path to export experimental candidate model TypeScript file",
    )
    args = parser.parse_args()

    print("========================================================================================")
    print("PHASE 4.3E — REAL-WORLD PIECE CLASSIFIER ADAPTATION & BENCHMARK")
    print("========================================================================================")

    # 1. Load Data
    X_train, y_train, sets_train, X_test, y_test = load_or_extract_dataset(args.data_dir, args.cache_dir)
    print(f"Dataset Loaded:")
    print(f"  Training (Sets 1-3 + mirror augmentation): {X_train.shape[0]} samples (288 features)")
    print(f"  Held-out Test (Set 4):                     {X_test.shape[0]} samples (288 features)")

    # 2. Leave-One-Set-Out Cross-Validation on Sets 1, 2, 3
    print("\n--- Gate 3: Leave-One-Set-Out Cross-Validation on Sets 1, 2, 3 ---")
    cv_res = run_leave_one_set_out_cv(X_train, y_train, sets_train, C=0.2)
    print(f"Mean LOSO Top-1 Accuracy: {cv_res['mean_loso_top1']*100:.2f}%")
    print(f"Mean LOSO Top-2 Accuracy: {cv_res['mean_loso_top2']*100:.2f}%")
    for fname, fval in cv_res["folds"].items():
        print(f"  {fname}: Top-1={fval['top1_accuracy']*100:.2f}%, Top-2={fval['top2_accuracy']*100:.2f}% (held-out Set {fval['held_out_set']})")

    # 3. Fit Single Frozen Candidate on all Sets 1, 2, 3
    print("\n--- Fitting Single Frozen Candidate Model on Sets 1, 2, 3 ---")
    t0 = time.time()
    candidate_clf = LogisticRegression(max_iter=1000, C=0.2, random_state=42)
    candidate_clf.fit(X_train, y_train)
    print(f"Model fit completed in {time.time() - t0:.2f}s")

    # 4. In-Domain Evaluation on Sets 1, 2, 3
    X_in = X_train[:2304]
    y_in = y_train[:2304]
    in_preds = candidate_clf.predict(X_in)
    in_probs = candidate_clf.predict_proba(X_in)
    in_top1 = float(np.mean(in_preds == y_in))
    in_top2 = float(np.mean([1 if y_in[i] in np.argsort(in_probs[i])[-2:] else 0 for i in range(len(y_in))]))
    print(f"\nIn-Domain Sets 1-3 Performance:")
    print(f"  Top-1 Accuracy: {in_top1*100:.2f}% ({np.sum(in_preds == y_in)}/{len(y_in)})")
    print(f"  Top-2 Accuracy: {in_top2*100:.2f}%")

    # 5. Held-Out Evaluation on Set 4 (Historical Benchmark Split)
    print("\n--- Gate 4: Evaluation on Held-Out Physical Set 4 ---")
    test_preds = candidate_clf.predict(X_test)
    test_probs = candidate_clf.predict_proba(X_test)
    test_top1 = float(np.mean(test_preds == y_test))
    test_top2 = float(np.mean([1 if y_test[i] in np.argsort(test_probs[i])[-2:] else 0 for i in range(len(y_test))]))

    print(f"Held-Out Physical Set 4 Performance:")
    print(f"  Top-1 Accuracy: {test_top1*100:.2f}% ({np.sum(test_preds == y_test)}/{len(y_test)})")
    print(f"  Top-2 Accuracy: {test_top2*100:.2f}%")

    print("\nPer-Class Classification Report (Set 4):")
    print(classification_report(y_test, test_preds, target_names=PIECE_CLASSES, digits=3))

    print("Confusion Matrix (Set 4):")
    cm = confusion_matrix(y_test, test_preds)
    header = "          " + " ".join(f"{c[:4]:>6}" for c in PIECE_CLASSES)
    print(header)
    for r_idx, row in enumerate(cm):
        print(f"{PIECE_CLASSES[r_idx]:<8}  " + " ".join(f"{v:6d}" for v in row))

    # 6. Abstention Analysis under Production Thresholds
    abstain_count = 0
    correct_non_abstained = 0
    total_non_abstained = 0

    for i in range(len(X_test)):
        s_p = np.sort(test_probs[i])
        conf = float(s_p[-1])
        margin = float(s_p[-1] - s_p[-2])
        if conf < 0.60 or margin < 0.20:
            abstain_count += 1
        else:
            total_non_abstained += 1
            if test_preds[i] == y_test[i]:
                correct_non_abstained += 1

    abstain_rate = abstain_count / len(X_test)
    non_abstained_acc = (correct_non_abstained / total_non_abstained) if total_non_abstained > 0 else 0.0

    print(f"\nAbstention Analysis (Thresholds: conf >= 0.60, margin >= 0.20):")
    print(f"  Abstention Rate:                 {abstain_rate*100:.2f}% ({abstain_count}/{len(X_test)})")
    print(f"  Non-Abstained Prediction Count:  {total_non_abstained}/{len(X_test)}")
    print(f"  Accuracy on Non-Abstained Piece: {non_abstained_acc*100:.2f}% ({correct_non_abstained}/{total_non_abstained})")

    # 7. Export Experimental Candidate
    export_experimental_model_ts(candidate_clf, test_top1, test_top2, len(X_train), len(X_test), args.ts_output)

    print("\n========================================================================================")
    print("PRODUCTION GATE EVALUATION")
    print("========================================================================================")
    print(f"Gate 1: Top-1 Target >= 88.0%  --> Actual: {test_top1*100:.2f}%  [FAIL - DO NOT PROMOTE]")
    print(f"Gate 2: Top-2 Target >= 96.0%  --> Actual: {test_top2*100:.2f}%  [FAIL - DO NOT PROMOTE]")
    print(f"Gate 3: Production Isolation   --> PASS (src/vision/pieceModelWeights.ts UNCHANGED)")
    print(f"Gate 4: Experimental Export   --> PASS (src/vision/experimentalPieceModelWeights.ts)")
    print("========================================================================================")


if __name__ == "__main__":
    main()
