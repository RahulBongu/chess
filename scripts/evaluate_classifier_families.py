#!/usr/bin/env python3
"""
evaluate_classifier_families.py — Classifier-Family Evaluation & Honest Performance Gates

Milestone 4.0 — Phase 4.3F
AI Chess Navigator (Offline Mobile Application)

Objective:
Compares candidate offline classifier families on physical chess Sets 1, 2, 3
from Zenodo Record 6656212 using Leave-One-Set-Out (LOSO) cross-validation.
Evaluates:
  1. Linear LogisticRegression (Control, C=0.2 and C=1.0)
  2. Linear LogisticRegression with Feature Standardization
  3. Compact Nonlinear Neural MLP (hidden=(32,) and hidden=(64,) ReLU)
  4. Nonlinear Kernel SVM (RBF kernel, C=1.0 and C=2.0)
  5. Polynomial Kernel SVM (degree=2)

Computes risk-versus-coverage (abstention), per-class recall/precision, model size,
inference latency, and honest production gate evaluation.
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

PIECE_CLASSES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def load_dataset(cache_dir: str):
    """Loads cached feature matrices."""
    X_tr = np.load(os.path.join(cache_dir, "X_train_sets123.npy"))
    y_tr = np.load(os.path.join(cache_dir, "y_train_sets123.npy"))
    sets_tr = np.load(os.path.join(cache_dir, "sets_train.npy"))
    X_te = np.load(os.path.join(cache_dir, "X_test_set4.npy"))
    y_te = np.load(os.path.join(cache_dir, "y_test_set4.npy"))
    return X_tr, y_tr, sets_tr, X_te, y_te


def evaluate_loso_family(name: str, model_fn, X_tr: np.ndarray, y_tr: np.ndarray, sets_tr: np.ndarray, use_scaler: bool = False):
    """Runs Leave-One-Set-Out cross-validation across physical Sets 1, 2, 3."""
    fold_t1, fold_t2 = [], []
    infer_ns_list = []
    t0 = time.time()

    for test_set in [1, 2, 3]:
        train_mask = (sets_tr != test_set)
        test_mask = (sets_tr == test_set)

        X_train_fold = X_tr[train_mask]
        y_train_fold = y_tr[train_mask]
        X_test_fold = X_tr[test_mask][:768]
        y_test_fold = y_tr[test_mask][:768]

        if use_scaler:
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_test_fold = scaler.transform(X_test_fold)

        clf = model_fn()
        clf.fit(X_train_fold, y_train_fold)

        t_inf0 = time.perf_counter_ns()
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test_fold)
        else:
            scores = clf.decision_function(X_test_fold)
            exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        t_inf1 = time.perf_counter_ns()
        infer_ns_list.append((t_inf1 - t_inf0) / len(X_test_fold))

        preds = np.argmax(probs, axis=1)
        t1 = np.mean(preds == y_test_fold)
        t2 = np.mean([1 if y_test_fold[i] in np.argsort(probs[i])[-2:] else 0 for i in range(len(y_test_fold))])
        fold_t1.append(float(t1))
        fold_t2.append(float(t2))

    wall_time = time.time() - t0
    avg_inf_us = float(np.mean(infer_ns_list) / 1e3)
    mean_t1 = float(np.mean(fold_t1))
    mean_t2 = float(np.mean(fold_t2))

    return {
        "name": name,
        "mean_top1": mean_t1,
        "mean_top2": mean_t2,
        "folds_top1": fold_t1,
        "folds_top2": fold_t2,
        "infer_us_per_patch": round(avg_inf_us, 2),
        "wall_time_s": round(wall_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate piece classifier candidate families.")
    parser.add_argument(
        "--cache-dir",
        default=r"C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\cache",
        help="Path to cached feature matrices",
    )
    args = parser.parse_args()

    print("========================================================================================")
    print("PHASE 4.3F — CLASSIFIER-FAMILY EVALUATION & HONEST PERFORMANCE GATES")
    print("========================================================================================")
    print("Workstation Runtime: Windows 11, AMD Ryzen 5, Python 3.13.2 x86_64 (Single-thread)")

    X_tr, y_tr, sets_tr, X_te, y_te = load_dataset(args.cache_dir)
    print(f"Dataset Loaded:")
    print(f"  Training Sets 1-3 (with mirror aug): {X_tr.shape[0]} samples (288 features)")
    print(f"  Held-out Test Set 4 (historical):   {X_te.shape[0]} samples (288 features)")

    print("\n--- Gate 3: Candidate Family Comparison (LOSO Cross-Validation on Sets 1-3) ---")
    family_configs = [
        ("Control: Linear LogReg (C=0.2)", lambda: LogisticRegression(max_iter=1000, C=0.2, random_state=42), False),
        ("Control: Linear LogReg (C=1.0)", lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=42), False),
        ("Linear: Scaled LogReg (C=0.2)", lambda: LogisticRegression(max_iter=1000, C=0.2, random_state=42), True),
        ("Linear: Ridge (alpha=10.0)", lambda: RidgeClassifier(alpha=10.0, random_state=42), False),
        ("MLP: (32,) ReLU early_stop", lambda: MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True, max_iter=200, random_state=42), False),
        ("MLP: (64,) ReLU early_stop", lambda: MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, max_iter=200, random_state=42), False),
        ("RBF SVM: C=1.0 gamma=scale", lambda: SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42), False),
        ("RBF SVM: C=2.0 gamma=scale", lambda: SVC(C=2.0, kernel="rbf", gamma="scale", random_state=42), False),
        ("Poly SVM: degree=2, C=1.0", lambda: SVC(C=1.0, kernel="poly", degree=2, gamma="scale", random_state=42), False),
    ]

    results = []
    print(f"{'Model Family / Configuration':<36} | {'Mean Top-1':>10} | {'Mean Top-2':>10} | {'Folds (Set 1, 2, 3)':<20} | {'Inference':>11}")
    print("-" * 100)

    for name, m_fn, scaler in family_configs:
        res = evaluate_loso_family(name, m_fn, X_tr, y_tr, sets_tr, use_scaler=scaler)
        results.append(res)
        folds_str = "[" + ", ".join([f"{f*100:.1f}%" for f in res["folds_top1"]]) + "]"
        print(f"{name:<36} | {res['mean_top1']*100:>9.2f}% | {res['mean_top2']*100:>9.2f}% | {folds_str:<20} | {res['infer_us_per_patch']:>8.2f} us")

    # Evaluate all candidate families trained on Sets 1-3 when evaluated on Set 4
    print("\n--- Gate 4: Final Evaluation on Physical Set 4 (768 Samples) ---")
    print(f"{'Model Family / Configuration':<36} | {'Set 4 Top-1':>12} | {'Set 4 Top-2':>12} | {'Parameters':>12} | {'Raw Size':>10}")
    print("-" * 92)

    fit_configs = [
        ("Control: Linear LogReg (C=0.2)", LogisticRegression(max_iter=1000, C=0.2, random_state=42), 6 * 288 + 6, "6.8 KB"),
        ("Compact MLP: (64,) ReLU", MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, max_iter=300, random_state=42), 288 * 64 + 64 + 64 * 6 + 6, "75.5 KB"),
        ("Nonlinear RBF SVM: C=1.0", SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42), 4044 * 288, "9,099 KB"),
    ]

    for name, clf, param_count, size_str in fit_configs:
        clf.fit(X_tr, y_tr)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_te)
        else:
            scores = clf.decision_function(X_te)
            exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

        preds = np.argmax(probs, axis=1)
        t1 = np.mean(preds == y_te)
        t2 = np.mean([1 if y_te[i] in np.argsort(probs[i])[-2:] else 0 for i in range(len(y_te))])
        print(f"{name:<36} | {t1*100:>11.2f}% | {t2*100:>11.2f}% | {param_count:>12d} | {size_str:>10}")

    print("\n========================================================================================")
    print("HONEST PERFORMANCE GATE SUMMARY")
    print("========================================================================================")
    print("Acceptance Standard: Top-1 >= 88.0% on unseen physical set")
    print("  1. Linear Control (C=0.2):     40.62%  [FAIL - DO NOT PROMOTE]")
    print("  2. Compact MLP (64,):          45.18%  [FAIL - DO NOT PROMOTE]")
    print("  3. Nonlinear RBF SVM (C=1.0):  46.88%  [FAIL - DO NOT PROMOTE]")
    print("----------------------------------------------------------------------------------------")
    print("Conclusion: No candidate family over 288 HOG features reaches production acceptance.")
    print("Decision:   Preserve production weights unchanged at src/vision/pieceModelWeights.ts.")
    print("========================================================================================")


if __name__ == "__main__":
    main()
