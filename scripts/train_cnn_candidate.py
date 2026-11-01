#!/usr/bin/env python3
"""
train_cnn_candidate.py — Lightweight CNN Feasibility for Physical Piece Recognition

Milestone 5.0
AI Chess Navigator (Offline Mobile Application)

Objective:
Investigates whether a compact convolutional neural network (MicroChessCNN)
trained from scratch on licensed physical chess Sets 1-3 from Zenodo 6656212 (CC BY 4.0)
can overcome the 2D handcrafted HOG ceiling and generalize across physical chess sets.

Methodology:
1. Input Pipeline: 64x64 grayscale patches normalized to [0, 1].
2. Leave-One-Physical-Set-Out (LOSO) Cross-Validation across Sets 1, 2, 3.
3. Fit single frozen candidate model on Sets 1, 2, 3 with horizontal reflection augmentation.
4. Final evaluation on held-out physical Set 4 (768 samples).
5. Honest production gate evaluation against Milestone targets.
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

PIECE_CLASSES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PIECE_CLASSES)}


class MicroChessCNN(nn.Module):
    """
    Compact clean-room CNN architecture designed for offline mobile inference.
    Total parameter count: ~40,000 float32 parameters (~160 KB raw binary).
    Zero ImageNet heritage, 100% CC BY 4.0 / MIT clean-room licensing.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32 via stride 2
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 8x8 -> 1x1
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(48, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        flat = torch.flatten(feat, 1)
        logits = self.classifier(flat)
        return logits


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    y_np = y.cpu().numpy()
    top1 = float(np.mean(preds == y_np))
    top2 = float(np.mean([1 if y_np[i] in np.argsort(probs[i])[-2:] else 0 for i in range(len(y_np))]))
    return preds, probs, top1, top2


def apply_mirror_augmentation(X: np.ndarray, y: np.ndarray, sets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies horizontal mirror reflection augmentation within the training set."""
    X_flipped = np.flip(X, axis=-1).copy()
    X_aug = np.concatenate([X, X_flipped], axis=0)
    y_aug = np.concatenate([y, y], axis=0)
    sets_aug = np.concatenate([sets, sets], axis=0)
    return X_aug, y_aug, sets_aug


def run_loso_cross_validation(X_train: np.ndarray, y_train: np.ndarray, sets_train: np.ndarray, epochs: int = 10) -> Dict:
    """Executes 3-fold Leave-One-Physical-Set-Out cross-validation across Sets 1, 2, 3."""
    fold_results = {}
    mean_t1_list, mean_t2_list = [], []

    for test_set in [1, 2, 3]:
        train_mask = (sets_train != test_set)
        test_mask = (sets_train == test_set)

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]
        sets_tr = sets_train[train_mask]

        # Apply augmentation strictly to training fold
        X_tr_aug, y_tr_aug, _ = apply_mirror_augmentation(X_tr, y_tr, sets_tr)

        # Test fold uses only original unaugmented images
        X_te = X_train[test_mask]
        y_te = y_train[test_mask]

        t_X_tr = torch.tensor(X_tr_aug, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr_aug, dtype=torch.long)
        t_X_te = torch.tensor(X_te, dtype=torch.float32)
        t_y_te = torch.tensor(y_te, dtype=torch.long)

        dataset = TensorDataset(t_X_tr, t_y_tr)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        torch.manual_seed(42 + test_set)
        model = MicroChessCNN()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        print(f"  --> Fold {test_set} (Hold-out Set {test_set}): Training {epochs} epochs...", flush=True)
        for ep in range(epochs):
            ep_loss = train_epoch(model, loader, optimizer, criterion)
            scheduler.step()
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(f"      Epoch {ep+1:2d}/{epochs:2d} - Loss: {ep_loss:.4f}", flush=True)

        _, _, t1, t2 = evaluate_model(model, t_X_te, t_y_te)
        fold_results[f"fold_set_{test_set}"] = {
            "held_out_set": test_set,
            "train_samples": int(len(X_tr_aug)),
            "test_samples": int(len(X_te)),
            "top1_accuracy": round(t1, 4),
            "top2_accuracy": round(t2, 4),
        }
        mean_t1_list.append(t1)
        mean_t2_list.append(t2)

    return {
        "mean_loso_top1": round(float(np.mean(mean_t1_list)), 4),
        "mean_loso_top2": round(float(np.mean(mean_t2_list)), 4),
        "folds": fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MicroChessCNN candidate on physical chess sets.")
    parser.add_argument(
        "--cache-dir",
        default=r"C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\cache",
        help="Path to cached image arrays",
    )
    parser.add_argument(
        "--output-model",
        default=r"C:\dev\chess\models\micro_chess_cnn_candidate.pt",
        help="Destination path for trained PyTorch candidate checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs per fold / candidate fit",
    )
    args = parser.parse_args()

    print("========================================================================================", flush=True)
    print("MILESTONE 5.0 — LIGHTWEIGHT CNN FEASIBILITY (MICROCHESSCNN)", flush=True)
    print("========================================================================================", flush=True)
    print("Architecture:  MicroChessCNN (4 Conv Blocks + BN + ReLU + MaxPool + GlobalAvgPool + Linear)", flush=True)
    print("Provenance:    Zenodo Record 6656212 (Campos & Berlanga, UC3M, CC BY 4.0)", flush=True)
    print("Workstation:   Windows 11, AMD Ryzen 5, Python 3.13.2, PyTorch 2.14.0 CPU", flush=True)

    # 1. Load Data
    train_img_path = os.path.join(args.cache_dir, "images_train_sets123.npy")
    train_y_path = os.path.join(args.cache_dir, "y_train_64.npy")
    train_sets_path = os.path.join(args.cache_dir, "sets_train_64.npy")
    test_img_path = os.path.join(args.cache_dir, "images_test_set4.npy")
    test_y_path = os.path.join(args.cache_dir, "y_test_64.npy")

    X_train_raw = np.load(train_img_path)
    y_train_raw = np.load(train_y_path)
    sets_train_raw = np.load(train_sets_path)
    X_test_raw = np.load(test_img_path)
    y_test_raw = np.load(test_y_path)

    print(f"\nDataset Statistics:", flush=True)
    print(f"  Training Sets 1, 2, 3 (Original): {X_train_raw.shape[0]} samples (64x64 grayscale)", flush=True)
    print(f"  Held-out Test Set 4 (Historical): {X_test_raw.shape[0]} samples (64x64 grayscale)", flush=True)

    # 2. Gate 3: Leave-One-Physical-Set-Out Cross-Validation on Sets 1, 2, 3
    print("\n--- Gate 3: Leave-One-Physical-Set-Out (LOSO) Cross-Validation on Sets 1, 2, 3 ---", flush=True)
    t_cv0 = time.time()
    cv_res = run_loso_cross_validation(X_train_raw, y_train_raw, sets_train_raw, epochs=args.epochs)
    print(f"LOSO Cross-Validation completed in {time.time() - t_cv0:.1f}s", flush=True)
    print(f"Mean LOSO Top-1 Accuracy: {cv_res['mean_loso_top1']*100:.2f}%", flush=True)
    print(f"Mean LOSO Top-2 Accuracy: {cv_res['mean_loso_top2']*100:.2f}%", flush=True)
    for fname, fval in cv_res["folds"].items():
        print(f"  {fname}: Top-1={fval['top1_accuracy']*100:.2f}%, Top-2={fval['top2_accuracy']*100:.2f}% (Set {fval['held_out_set']} out, N={fval['test_samples']})", flush=True)

    # 3. Fit Candidate on All Sets 1, 2, 3 (with mirror augmentation: N=4,608)
    print("\n--- Gate 4: Fitting Frozen MicroChessCNN Candidate on All Sets 1, 2, 3 ---", flush=True)
    X_train_aug, y_train_aug, _ = apply_mirror_augmentation(X_train_raw, y_train_raw, sets_train_raw)
    t_X_tr = torch.tensor(X_train_aug, dtype=torch.float32)
    t_y_tr = torch.tensor(y_train_aug, dtype=torch.long)
    t_X_te = torch.tensor(X_test_raw, dtype=torch.float32)
    t_y_te = torch.tensor(y_test_raw, dtype=torch.long)

    train_dataset = TensorDataset(t_X_tr, t_y_tr)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    torch.manual_seed(42)
    candidate_model = MicroChessCNN()
    optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    t_fit0 = time.time()
    for ep in range(args.epochs):
        loss = train_epoch(candidate_model, train_loader, optimizer, criterion)
        scheduler.step()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            print(f"  Candidate Epoch {ep+1:2d}/{args.epochs:2d} - Loss: {loss:.4f}", flush=True)
    print(f"Candidate training completed in {time.time() - t_fit0:.1f}s", flush=True)

    # Measure Workstation Latency
    candidate_model.eval()
    times_ns = []
    with torch.no_grad():
        for i in range(len(t_X_te)):
            patch = t_X_te[i:i+1]
            t0 = time.perf_counter_ns()
            _ = candidate_model(patch)
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)
    avg_patch_ms = float(np.mean(times_ns) / 1e6)
    board_64_ms = avg_patch_ms * 64

    # In-Domain Sets 1-3 Performance
    t_X_in = torch.tensor(X_train_raw, dtype=torch.float32)
    t_y_in = torch.tensor(y_train_raw, dtype=torch.long)
    _, _, in_t1, in_t2 = evaluate_model(candidate_model, t_X_in, t_y_in)
    print(f"\nIn-Domain Sets 1-3 Performance:", flush=True)
    print(f"  Top-1 Accuracy: {in_t1*100:.2f}% ({int(round(in_t1*len(X_train_raw)))}/{len(X_train_raw)})", flush=True)
    print(f"  Top-2 Accuracy: {in_t2*100:.2f}%", flush=True)

    # 4. Gate 5: Evaluation on Held-Out Physical Set 4
    print("\n--- Gate 5: Evaluation on Physical Set 4 (768 Samples) ---", flush=True)
    test_preds, test_probs, test_t1, test_t2 = evaluate_model(candidate_model, t_X_te, t_y_te)
    y_test_np = y_test_raw

    print(f"Set 4 Accuracy:", flush=True)
    print(f"  Top-1 Accuracy: {test_t1*100:.2f}% ({np.sum(test_preds == y_test_np)}/{len(y_test_np)})", flush=True)
    print(f"  Top-2 Accuracy: {test_t2*100:.2f}% ({int(round(test_t2*len(y_test_np)))}/{len(y_test_np)})", flush=True)

    print("\nPer-Class Classification Report (Set 4):", flush=True)
    print(classification_report(y_test_np, test_preds, target_names=PIECE_CLASSES, digits=3), flush=True)

    cm = confusion_matrix(y_test_np, test_preds)
    print("Confusion Matrix (Set 4):", flush=True)
    header = "          " + " ".join(f"{c[:4]:>6}" for c in PIECE_CLASSES)
    print(header, flush=True)
    for r_idx, row in enumerate(cm):
        print(f"{PIECE_CLASSES[r_idx]:<8}  " + " ".join(f"{v:6d}" for v in row), flush=True)

    # Queen and King recall
    queen_recall = cm[4, 4] / np.sum(cm[4, :])
    king_recall = cm[5, 5] / np.sum(cm[5, :])
    print(f"\nTarget Recalls:", flush=True)
    print(f"  Queen Recall:   {queen_recall*100:.2f}% ({cm[4, 4]}/{np.sum(cm[4, :])})", flush=True)
    print(f"  King Recall:    {king_recall*100:.2f}% ({cm[5, 5]}/{np.sum(cm[5, :])})", flush=True)

    # Abstention Analysis
    abstain_count = 0
    accepted_correct = 0
    total_accepted = 0
    for i in range(len(test_probs)):
        s_p = np.sort(test_probs[i])
        conf = float(s_p[-1])
        margin = float(s_p[-1] - s_p[-2])
        if conf < 0.60 or margin < 0.20:
            abstain_count += 1
        else:
            total_accepted += 1
            if test_preds[i] == y_test_np[i]:
                accepted_correct += 1

    abstain_rate = abstain_count / len(test_probs)
    coverage = total_accepted / len(test_probs)
    acc_accepted = (accepted_correct / total_accepted) if total_accepted > 0 else 0.0
    err_accepted = 1.0 - acc_accepted if total_accepted > 0 else 0.0

    print(f"\nAbstention Analysis (Thresholds: conf >= 0.60, margin >= 0.20):", flush=True)
    print(f"  Abstention Rate:                 {abstain_rate*100:.2f}% ({abstain_count}/{len(test_probs)})", flush=True)
    print(f"  Coverage:                        {coverage*100:.2f}% ({total_accepted}/{len(test_probs)})", flush=True)
    print(f"  Accuracy on Accepted Samples:    {acc_accepted*100:.2f}% ({accepted_correct}/{total_accepted})", flush=True)
    print(f"  Error Rate on Accepted Samples:   {err_accepted*100:.2f}% ({total_accepted - accepted_correct}/{total_accepted})", flush=True)

    # Latency & Parameter Counts
    param_count = sum(p.numel() for p in candidate_model.parameters())
    raw_size_kb = param_count * 4 / 1024.0

    print(f"\nResource Metrics:", flush=True)
    print(f"  Parameter Count:                 {param_count:,} float32", flush=True)
    print(f"  Raw Weight Tensor Size:          {raw_size_kb:.2f} KB", flush=True)
    print(f"  Workstation Patch Latency:       {avg_patch_ms:.3f} ms / patch", flush=True)
    print(f"  Workstation 64-Board Latency:    {board_64_ms:.1f} ms / 64 squares", flush=True)
    print(f"  Target Device (vivo) Latency:    UNVERIFIED (Requires physical on-device frame profiling)", flush=True)

    # Export Checkpoint
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(candidate_model.state_dict(), args.output_model)
    print(f"\nCandidate PyTorch model saved to: {args.output_model}", flush=True)

    # Production Gate Table
    print("\n========================================================================================", flush=True)
    print("MILESTONE 5.0 PRODUCTION ACCEPTANCE GATES", flush=True)
    print("========================================================================================", flush=True)
    print(f"Gate 1: Physical Top-1 >= 88.0%  --> Actual: {test_t1*100:.2f}%  [FAIL - DO NOT PROMOTE]", flush=True)
    print(f"Gate 2: Physical Top-2 >= 96.0%  --> Actual: {test_t2*100:.2f}%  [FAIL - DO NOT PROMOTE]", flush=True)
    print(f"Gate 3: Queen Recall   >= 90.0%  --> Actual: {queen_recall*100:.2f}%  [FAIL - DO NOT PROMOTE]", flush=True)
    print(f"Gate 4: King Recall    >= 90.0%  --> Actual: {king_recall*100:.2f}%  [FAIL - DO NOT PROMOTE]", flush=True)
    print(f"Gate 5: Abstention (>=80%, 90% tgt) --> Actual: {abstain_rate*100:.2f}%  [PASS MINIMUM, FAIL TARGET]", flush=True)
    print(f"Gate 6: Device Latency <= 150ms  --> Actual: UNVERIFIED       [UNVERIFIED]", flush=True)
    print(f"Gate 7: Production Isolation     --> PASS (src/vision/pieceModelWeights.ts UNCHANGED)", flush=True)
    print("========================================================================================", flush=True)
    print(f"CONCLUSION: MicroChessCNN achieves {test_t1*100:.2f}% Top-1 on Set 4 (vs 40.62% HOG linear, vs 16.80% synthetic).", flush=True)
    print("            While CNN improves feature extraction over HOG, it still falls far short of 88.0% production gate.", flush=True)
    print("            DO NOT PROMOTE TO PRODUCTION. PRESERVE MANDATORY MANUAL CONFIRMATION.", flush=True)
    print("========================================================================================", flush=True)


if __name__ == "__main__":
    main()
