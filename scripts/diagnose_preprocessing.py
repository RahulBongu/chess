#!/usr/bin/env python3
"""
diagnose_preprocessing.py — Milestone 5.1 / 5.1A Minimal Controlled Diagnosis

Evaluates the exact effect of verified mobile-matched input transforms vs. direct resize
on MicroChessCNN cross-set generalization, including the standard ITU-R BT.601 BGR luminance fix.

Controlled variables:
- Exact same network architecture (MicroChessCNN, 40,038 parameters)
- Exact same optimization (AdamW, lr=1e-3, weight_decay=1e-4, CosineAnnealingLR, 8 epochs)
- Exact same random seeds (42 + test_set for folds, 42 for candidate)
- Exact same LOSO fold splits on Sets 1-3 and evaluation on Set 4
- Isolated single factor: Preprocessing transform (Direct Resize vs Mobile-Matched vs Cross-Transform Robustness)

Disclosure: Set 4 (Zenodo valid/ folder, N=768) is a historical holdout previously evaluated
in Milestone 4.3D and Milestone 5.0. It is NOT a new blind test and is not used for model tuning.
"""

import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

from train_cnn_candidate import (
    MicroChessCNN,
    train_epoch,
    evaluate_model,
    apply_mirror_augmentation,
    run_loso_cross_validation,
    PIECE_CLASSES,
)


def run_experiment(
    train_img_path: str,
    test_img_path: str,
    cache_dir: str,
    experiment_name: str,
    epochs: int = 8,
) -> Dict:
    print("=" * 80, flush=True)
    print(f"EXPERIMENT: {experiment_name}", flush=True)
    print("=" * 80, flush=True)

    X_train_raw = np.load(train_img_path)
    y_train_raw = np.load(os.path.join(cache_dir, "y_train_64.npy"))
    sets_train_raw = np.load(os.path.join(cache_dir, "sets_train_64.npy"))

    X_test_raw = np.load(test_img_path)
    y_test_raw = np.load(os.path.join(cache_dir, "y_test_64.npy"))

    # 1. Leave-One-Physical-Set-Out Cross-Validation
    print("--- 1. Leave-One-Physical-Set-Out (LOSO) Cross-Validation (Sets 1-3) ---", flush=True)
    t0 = time.time()
    cv_res = run_loso_cross_validation(X_train_raw, y_train_raw, sets_train_raw, epochs=epochs)
    print(f"LOSO Cross-Validation completed in {time.time() - t0:.1f}s", flush=True)
    print(f"  Mean LOSO Top-1: {cv_res['mean_loso_top1']*100:.2f}%", flush=True)
    print(f"  Mean LOSO Top-2: {cv_res['mean_loso_top2']*100:.2f}%", flush=True)
    for fname, fval in cv_res["folds"].items():
        print(
            f"    {fname} (Hold-out Set {fval['held_out_set']}): "
            f"Top-1 = {fval['top1_accuracy']*100:.2f}%, "
            f"Top-2 = {fval['top2_accuracy']*100:.2f}%",
            flush=True,
        )

    # 2. Candidate Fit on Sets 1-3
    print("\n--- 2. Fitting Candidate Model on Sets 1-3 (Augmented N=4,608) ---", flush=True)
    X_train_aug, y_train_aug, _ = apply_mirror_augmentation(X_train_raw, y_train_raw, sets_train_raw)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_aug, dtype=torch.float32),
            torch.tensor(y_train_aug, dtype=torch.long),
        ),
        batch_size=128,
        shuffle=True,
    )

    torch.manual_seed(42)
    candidate_model = MicroChessCNN()
    optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        loss = train_epoch(candidate_model, train_loader, optimizer, criterion)
        scheduler.step()
        if (ep + 1) % 4 == 0 or ep == epochs - 1:
            print(f"  Candidate Epoch {ep+1:2d}/{epochs:2d} - Loss: {loss:.4f}", flush=True)

    # 3. In-Domain Accuracy
    t_X_in = torch.tensor(X_train_raw, dtype=torch.float32)
    t_y_in = torch.tensor(y_train_raw, dtype=torch.long)
    _, _, in_t1, in_t2 = evaluate_model(candidate_model, t_X_in, t_y_in)
    print(f"\nIn-Domain Sets 1-3 Performance:", flush=True)
    print(f"  Top-1: {in_t1*100:.2f}% ({int(round(in_t1*len(X_train_raw)))}/{len(X_train_raw)})", flush=True)
    print(f"  Top-2: {in_t2*100:.2f}%", flush=True)

    # 4. Evaluation on Held-Out Set 4 (Historical Holdout)
    print("\n--- 3. Evaluation on Held-Out Physical Set 4 (Historical Holdout, N=768) ---", flush=True)
    t_X_te = torch.tensor(X_test_raw, dtype=torch.float32)
    t_y_te = torch.tensor(y_test_raw, dtype=torch.long)
    test_preds, test_probs, test_t1, test_t2 = evaluate_model(candidate_model, t_X_te, t_y_te)

    print(f"Set 4 Accuracy:", flush=True)
    print(f"  Top-1: {test_t1*100:.2f}% ({np.sum(test_preds == y_test_raw)}/{len(y_test_raw)})", flush=True)
    print(f"  Top-2: {test_t2*100:.2f}% ({int(round(test_t2*len(y_test_raw)))}/{len(y_test_raw)})", flush=True)

    print("\nPer-Class Classification Report (Set 4):", flush=True)
    print(classification_report(y_test_raw, test_preds, target_names=PIECE_CLASSES, digits=3), flush=True)

    cm = confusion_matrix(y_test_raw, test_preds)
    print("Confusion Matrix (Set 4):", flush=True)
    header = "          " + " ".join(f"{c[:4]:>6}" for c in PIECE_CLASSES)
    print(header, flush=True)
    for r_idx, row in enumerate(cm):
        print(f"{PIECE_CLASSES[r_idx]:<8}  " + " ".join(f"{v:6d}" for v in row), flush=True)

    # Recalls
    recalls = {PIECE_CLASSES[i]: cm[i, i] / np.sum(cm[i, :]) for i in range(6)}

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
            if test_preds[i] == y_test_raw[i]:
                accepted_correct += 1

    abstain_rate = abstain_count / len(test_probs)
    coverage = total_accepted / len(test_probs)
    acc_accepted = (accepted_correct / total_accepted) if total_accepted > 0 else 0.0

    return {
        "experiment_name": experiment_name,
        "mean_loso_top1": cv_res["mean_loso_top1"],
        "mean_loso_top2": cv_res["mean_loso_top2"],
        "folds": cv_res["folds"],
        "in_domain_top1": in_t1,
        "in_domain_top2": in_t2,
        "set4_top1": test_t1,
        "set4_top2": test_t2,
        "recalls": recalls,
        "abstain_rate": abstain_rate,
        "coverage": coverage,
        "acc_accepted": acc_accepted,
        "candidate_model": candidate_model,
    }


def main():
    cache_dir = r"C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\cache"

    # Preprocessing 1: Direct Resize (Baseline from Milestone 5.0)
    exp1_train = os.path.join(cache_dir, "images_train_sets123.npy")
    exp1_test = os.path.join(cache_dir, "images_test_set4.npy")

    # Preprocessing 2: Mobile-Matched Preprocessing (100x100 -> step 1.5625 nearest neighbor, standard ITU-R BT.601 BGR luminance)
    exp2_train = os.path.join(cache_dir, "images_train_mobile_matched.npy")
    exp2_test = os.path.join(cache_dir, "images_test_mobile_matched.npy")

    print("\nStarting Controlled Experiment 2: Mobile-Matched Preprocessing (Standard Luminance)...", flush=True)
    res_exp2 = run_experiment(
        exp2_train, exp2_test, cache_dir, "Experiment 2: Mobile-Matched Preprocessing (Standard Luminance)", epochs=8
    )

    # 3. Cross-Transform Robustness Test (Direct-Resize Checkpoint evaluated on Mobile-Matched Set 4)
    model_m50_path = r"C:\dev\chess\models\micro_chess_cnn_candidate.pt"
    if os.path.exists(model_m50_path):
        print("\n--- 4. Cross-Transform Robustness Test ---", flush=True)
        m50_model = MicroChessCNN()
        m50_model.load_state_dict(torch.load(model_m50_path, weights_only=True))
        m50_model.eval()

        t_X_mm_te = torch.tensor(np.load(exp2_test), dtype=torch.float32)
        y_test_raw = np.load(os.path.join(cache_dir, "y_test_64.npy"))
        t_y_te = torch.tensor(y_test_raw, dtype=torch.long)

        _, _, rob_t1, rob_t2 = evaluate_model(m50_model, t_X_mm_te, t_y_te)
        print(f"M5.0 Direct-Resize Model evaluated on Mobile-Matched Set 4 (Robustness): Top-1 = {rob_t1*100:.2f}%, Top-2 = {rob_t2*100:.2f}%")
    else:
        rob_t1, rob_t2 = None, None

    # Benchmark comparison values
    base_loso_mean = 0.2040
    base_loso_f1 = 0.2487
    base_loso_f2 = 0.1706
    base_loso_f3 = 0.1927
    base_in_domain = 0.6337
    base_set4_t1 = 0.2708
    base_set4_t2 = 0.5117

    mm_loso_mean = res_exp2["mean_loso_top1"]
    mm_loso_f1 = res_exp2["folds"]["fold_set_1"]["top1_accuracy"]
    mm_loso_f2 = res_exp2["folds"]["fold_set_2"]["top1_accuracy"]
    mm_loso_f3 = res_exp2["folds"]["fold_set_3"]["top1_accuracy"]
    mm_in_domain = res_exp2["in_domain_top1"]
    mm_set4_t1 = res_exp2["set4_top1"]
    mm_set4_t2 = res_exp2["set4_top2"]

    print("\n" + "=" * 90, flush=True)
    print("CONTROLLED DIAGNOSIS SUMMARY (MILESTONE 5.1 / 5.1A)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Metric':<30} | {'Direct Resize (M5.0)':<22} | {'Mobile-Matched (M5.1A)':<22} | {'Fold Difference':<12}", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Mean LOSO Top-1':<30} | {base_loso_mean*100:.2f}%{'':<16} | {mm_loso_mean*100:.2f}%{'':<16} | {(mm_loso_mean - base_loso_mean)*100:+.2f} pp", flush=True)
    print(f"{'  Fold 1 (Set 1 out)':<30} | {base_loso_f1*100:.2f}%{'':<16} | {mm_loso_f1*100:.2f}%{'':<16} | {(mm_loso_f1 - base_loso_f1)*100:+.2f} pp", flush=True)
    print(f"{'  Fold 2 (Set 2 out)':<30} | {base_loso_f2*100:.2f}%{'':<16} | {mm_loso_f2*100:.2f}%{'':<16} | {(mm_loso_f2 - base_loso_f2)*100:+.2f} pp", flush=True)
    print(f"{'  Fold 3 (Set 3 out)':<30} | {base_loso_f3*100:.2f}%{'':<16} | {mm_loso_f3*100:.2f}%{'':<16} | {(mm_loso_f3 - base_loso_f3)*100:+.2f} pp", flush=True)
    print(f"{'In-Domain Sets 1-3 Top-1':<30} | {base_in_domain*100:.2f}%{'':<16} | {mm_in_domain*100:.2f}%{'':<16} | {(mm_in_domain - base_in_domain)*100:+.2f} pp", flush=True)
    print(f"{'Set 4 Top-1 (Holdout)':<30} | {base_set4_t1*100:.2f}%{'':<16} | {mm_set4_t1*100:.2f}%{'':<16} | {(mm_set4_t1 - base_set4_t1)*100:+.2f} pp", flush=True)
    print(f"{'Set 4 Top-2 (Holdout)':<30} | {base_set4_t2*100:.2f}%{'':<16} | {mm_set4_t2*100:.2f}%{'':<16} | {(mm_set4_t2 - base_set4_t2)*100:+.2f} pp", flush=True)
    print(f"{'Pawn Recall':<30} | {'30.47%':<22} | {res_exp2['recalls']['pawn']*100:.2f}%", flush=True)
    print(f"{'Knight Recall':<30} | {'45.31%':<22} | {res_exp2['recalls']['knight']*100:.2f}%", flush=True)
    print(f"{'Bishop Recall':<30} | {'0.78%':<22} | {res_exp2['recalls']['bishop']*100:.2f}%", flush=True)
    print(f"{'Rook Recall':<30} | {'69.53%':<22} | {res_exp2['recalls']['rook']*100:.2f}%", flush=True)
    print(f"{'Queen Recall':<30} | {'14.84%':<22} | {res_exp2['recalls']['queen']*100:.2f}%", flush=True)
    print(f"{'King Recall':<30} | {'1.56%':<22} | {res_exp2['recalls']['king']*100:.2f}%", flush=True)
    print(f"{'Abstention Rate':<30} | {'95.18%':<22} | {res_exp2['abstain_rate']*100:.2f}%", flush=True)
    print(f"{'Coverage':<30} | {'4.82%':<22} | {res_exp2['coverage']*100:.2f}%", flush=True)
    print(f"{'Acc on Accepted':<30} | {'67.57%':<22} | {res_exp2['acc_accepted']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
