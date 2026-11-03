#!/usr/bin/env python3
"""
evaluate_milestone_54h_comparison.py - Comprehensive Multi-Domain Evaluation
Comparing Baseline vs Candidate RGB Square Models for Milestone 5.4H-R.

Domains Evaluated:
1. Domain A: FENiT Real Physical-Board Photographs (40 held-out boards, 2,560 square crops)
2. Domain B: Zenodo Set 4 Physical Pieces (832 crops, physical holdout regression benchmark)
3. Domain C: Laptop-Screen Digital Benchmark (20 scenes, 1,280 squares, browser chrome + glare + moiré)

Metrics Computed:
- Overall top-1 square accuracy (with exact denominators)
- Empty square accuracy & Occupied piece accuracy
- Per-piece-class precision, recall, and confusion matrix
- Complete-board exact accuracy (all 64 squares match GT placement)
- Temporal frame averaging (N=1, 2, 4, 8 frames under camera sensor noise)
- Production adoption gate:
  - Coverage (% boards adopted)
  - True adoption rate (% adopted boards that are exact matches)
  - False adoption rate (% total test boards accepted with incorrect placement)
  - Safe rejection rate (% incorrect boards safely held uncertain)
"""

from __future__ import annotations
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rgb_square_model import (
    load_fenit,
    load_zenodo,
    build_cache,
    CLASSES,
    CLASS_TO_IDX,
    LOGICAL_SIZE,
    INPUT_SIZE,
)
from scripts.evaluate_production_adoption import evaluate_board_adoption
from scripts.benchmark_screen_chessboards import (
    OFFLINE_POSITIONS,
    render_screen_scene,
    extract_square_crops_from_scene,
)

def run_onnx_inference(session: ort.InferenceSession, images_72: np.ndarray, batch_size: int = 64) -> Tuple[np.ndarray, float]:
    """
    Runs ONNX inference on cached 72x72 crops (or already-cropped 64x64 crops)
    with central 64/72 cropping and [-1, 1] normalization.
    Returns: (N, 13) probabilities, latency_per_batch_ms.
    """
    input_name = session.get_inputs()[0].name
    if images_72.shape[1] > INPUT_SIZE and images_72.shape[2] > INPUT_SIZE:
        offset_y = (images_72.shape[1] - INPUT_SIZE) // 2
        offset_x = (images_72.shape[2] - INPUT_SIZE) // 2
        cropped = images_72[:, offset_y : offset_y + INPUT_SIZE, offset_x : offset_x + INPUT_SIZE]
    else:
        cropped = images_72
    tensor = (cropped.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)

    all_probs = []
    t0 = time.time()
    n_batches = 0
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start : start + batch_size]
        logits = session.run(None, {input_name: batch})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        all_probs.append(probs)
        n_batches += 1
    total_time = time.time() - t0
    batch_latency = (total_time / max(n_batches, 1)) * 1000.0

    return np.vstack(all_probs), batch_latency


def evaluate_domain_a_fenit(session: ort.InferenceSession, manifest_path: Path, cache_dir: Path):
    """Evaluates 40 full-board physical photographs (2,560 crops) from FENiT held-out validation."""
    _, val_samples = load_fenit(manifest_path)
    val_images = build_cache(val_samples, "val_rgb_fenit", cache_dir)
    val_labels = np.array([s.label for s in val_samples])
    val_groups = [s.group for s in val_samples]
    board_count = len(val_samples) // 64

    probs, batch_ms = run_onnx_inference(session, val_images)
    preds = np.argmax(probs, axis=1)
    correct = preds == val_labels
    occupied_mask = val_labels != 0
    empty_mask = val_labels == 0

    # Confusion matrix
    cm = np.zeros((13, 13), dtype=np.int32)
    for t, p in zip(val_labels, preds):
        cm[t, p] += 1

    per_class = {}
    for idx, c in enumerate(CLASSES):
        total = int((val_labels == idx).sum())
        c_correct = int(cm[idx, idx])
        rec = (c_correct / total * 100.0) if total > 0 else 0.0
        # Precision: TP / (TP + FP)
        pred_total = int((preds == idx).sum())
        prec = (c_correct / pred_total * 100.0) if pred_total > 0 else 0.0
        per_class[c] = {"n": total, "correct": c_correct, "recall_pct": round(rec, 2), "precision_pct": round(prec, 2)}

    # Board-level adoption evaluation
    exact_boards = 0
    adopted_boards = 0
    true_adoptions = 0
    false_adoptions = 0
    uncertain_counts = []

    for b in range(board_count):
        b_slice = slice(b * 64, (b + 1) * 64)
        b_probs = probs[b_slice]
        gt_placement = val_groups[b * 64]

        if bool(correct[b_slice].all()):
            exact_boards += 1

        adoption = evaluate_board_adoption(b_probs)
        uncertain_counts.append(len(adoption['uncertain']))
        if adoption['ok']:
            adopted_boards += 1
            if adoption['placement'] == gt_placement:
                true_adoptions += 1
            else:
                false_adoptions += 1

    # Temporal averaging evaluation on FENiT (N=1, 2, 4, 8) with sensor noise
    offset = (LOGICAL_SIZE - INPUT_SIZE) // 2
    cropped = val_images[:, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
    temporal_results = {}
    rng = np.random.default_rng(20261104)

    for w_size in [1, 2, 4, 8]:
        probs_accum = np.zeros((len(cropped), 13), dtype=np.float32)
        for f_idx in range(w_size):
            if w_size == 1:
                noisy_crop = cropped.astype(np.float32)
            else:
                noise = rng.normal(0, 3.5, cropped.shape).astype(np.float32)
                noisy_crop = np.clip(cropped.astype(np.float32) + noise, 0, 255)
            tensor = (noisy_crop / 127.5 - 1.0).transpose(0, 3, 1, 2)
            for start in range(0, len(tensor), 64):
                batch = tensor[start : start + 64]
                logits = session.run(None, {session.get_inputs()[0].name: batch})[0]
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs_accum[start : start + 64] += exp_l / np.sum(exp_l, axis=1, keepdims=True)
        w_probs = probs_accum / float(w_size)
        w_preds = np.argmax(w_probs, axis=1)
        w_correct = w_preds == val_labels
        w_exact = sum(1 for b in range(board_count) if bool(w_correct[b*64:(b+1)*64].all()))
        w_adopted = 0
        w_false_adopt = 0
        for b in range(board_count):
            ad = evaluate_board_adoption(w_probs[b*64:(b+1)*64])
            if ad['ok']:
                w_adopted += 1
                if ad['placement'] != val_groups[b*64]:
                    w_false_adopt += 1
        temporal_results[f"N={w_size}"] = {
            "square_acc_pct": round(float(w_correct.mean() * 100), 2),
            "occupied_acc_pct": round(float(w_correct[occupied_mask].mean() * 100), 2),
            "exact_boards": w_exact,
            "adopted_boards": w_adopted,
            "false_adoptions": w_false_adopt,
        }

    return {
        "n_samples": int(len(val_labels)),
        "n_boards": board_count,
        "batch_latency_ms": round(batch_ms, 2),
        "overall_top1_pct": round(float(correct.mean() * 100), 2),
        "empty_top1_pct": round(float(correct[empty_mask].mean() * 100), 2),
        "occupied_top1_pct": round(float(correct[occupied_mask].mean() * 100), 2),
        "exact_board_count": exact_boards,
        "exact_board_pct": round(exact_boards / board_count * 100, 2),
        "adopted_boards": adopted_boards,
        "coverage_pct": round(adopted_boards / board_count * 100, 2),
        "true_adoptions": true_adoptions,
        "false_adoptions": false_adoptions,
        "false_adoption_rate_pct": round(false_adoptions / board_count * 100, 2),
        "mean_uncertain_squares": round(float(np.mean(uncertain_counts)), 2),
        "per_class": per_class,
        "temporal_averaging": temporal_results,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_domain_b_zenodo(session: ort.InferenceSession, zenodo_dir: Path, cache_dir: Path):
    """Evaluates Zenodo physical Set 4 holdout (832 crops, physical regression benchmark)."""
    _, val_samples = load_zenodo(zenodo_dir)
    val_images = build_cache(val_samples, "val_rgb_zenodo", cache_dir)
    val_labels = np.array([s.label for s in val_samples])

    probs, batch_ms = run_onnx_inference(session, val_images)
    preds = np.argmax(probs, axis=1)
    correct = preds == val_labels
    occupied_mask = val_labels != 0
    empty_mask = val_labels == 0

    per_class = {}
    for idx, c in enumerate(CLASSES):
        total = int((val_labels == idx).sum())
        c_correct = int(((preds == idx) & (val_labels == idx)).sum())
        rec = (c_correct / total * 100.0) if total > 0 else 0.0
        per_class[c] = {"n": total, "correct": c_correct, "recall_pct": round(rec, 2)}

    return {
        "n_samples": int(len(val_labels)),
        "batch_latency_ms": round(batch_ms, 2),
        "overall_top1_pct": round(float(correct.mean() * 100), 2),
        "empty_top1_pct": round(float(correct[empty_mask].mean() * 100), 2),
        "occupied_top1_pct": round(float(correct[occupied_mask].mean() * 100), 2),
        "per_class": per_class,
    }


def evaluate_domain_c_screen(session: ort.InferenceSession):
    """Evaluates 20 realistic laptop-screen scenes with browser chrome, sidebars, glare, and moiré."""
    total_squares = 0
    total_correct = 0
    total_empty_samples = 0
    total_empty_correct = 0
    total_occ_samples = 0
    total_occ_correct = 0
    exact_boards = 0
    adopted_boards = 0
    false_adoptions = 0

    per_class = {c: {"total": 0, "correct": 0} for c in CLASSES}
    by_theme = {}

    for idx, pos in enumerate(OFFLINE_POSITIONS):
        yaw = [-12.0, 0.0, 15.0, -8.0][idx % 4]
        pitch = [18.0, 24.0, 30.0, 16.0][(idx // 4) % 4]

        scene_rgb, gt_corners, gt_placement = render_screen_scene(
            fen=pos["fen"],
            theme=pos["theme"],
            angle_yaw=yaw,
            angle_pitch=pitch,
            render_size=(1280, 960),
        )

        crops = extract_square_crops_from_scene(scene_rgb, gt_corners)
        probs, _ = run_onnx_inference(session, crops, batch_size=64)
        preds = np.argmax(probs, axis=1)

        # Build GT labels
        gt_labels = []
        for rank_str in gt_placement.split('/'):
            for char in rank_str:
                if char.isdigit():
                    gt_labels.extend([0] * int(char))
                else:
                    color = "white" if char.isupper() else "black"
                    p_type = {'p': 'pawn', 'n': 'knight', 'b': 'bishop', 'r': 'rook', 'q': 'queen', 'k': 'king'}[char.lower()]
                    gt_labels.append(CLASS_TO_IDX[f"{color}_{p_type}"])
        gt_labels = np.array(gt_labels)

        sq_correct = preds == gt_labels
        num_correct = int(sq_correct.sum())

        for p_idx, true_cls in enumerate(gt_labels):
            per_class[CLASSES[true_cls]]["total"] += 1
            if sq_correct[p_idx]:
                per_class[CLASSES[true_cls]]["correct"] += 1

        total_squares += 64
        total_correct += num_correct

        empty_m = gt_labels == 0
        total_empty_samples += int(empty_m.sum())
        total_empty_correct += int(sq_correct[empty_m].sum())

        occ_m = gt_labels != 0
        total_occ_samples += int(occ_m.sum())
        total_occ_correct += int(sq_correct[occ_m].sum())

        if bool(sq_correct.all()):
            exact_boards += 1

        ad = evaluate_board_adoption(probs)
        if ad['ok']:
            adopted_boards += 1
            if ad['placement'] != gt_placement:
                false_adoptions += 1

        theme = pos["theme"]
        if theme not in by_theme:
            by_theme[theme] = {"squares": 0, "correct": 0, "boards": 0, "exact": 0}
        by_theme[theme]["squares"] += 64
        by_theme[theme]["correct"] += num_correct
        by_theme[theme]["boards"] += 1
        if bool(sq_correct.all()):
            by_theme[theme]["exact"] += 1

    per_class_summary = {}
    for c in CLASSES:
        st = per_class[c]
        rec = (st["correct"] / st["total"] * 100.0) if st["total"] > 0 else 0.0
        per_class_summary[c] = {"n": st["total"], "correct": st["correct"], "recall_pct": round(rec, 2)}

    theme_summary = {}
    for t, st in by_theme.items():
        theme_summary[t] = {
            "boards": st["boards"],
            "exact_boards": st["exact"],
            "square_acc_pct": round(st["correct"] / st["squares"] * 100.0, 2),
        }

    return {
        "n_scenes": len(OFFLINE_POSITIONS),
        "total_squares": total_squares,
        "overall_top1_pct": round(total_correct / total_squares * 100.0, 2),
        "empty_top1_pct": round(total_empty_correct / total_empty_samples * 100.0, 2),
        "occupied_top1_pct": round(total_occ_correct / total_occ_samples * 100.0, 2),
        "exact_board_count": exact_boards,
        "exact_board_pct": round(exact_boards / len(OFFLINE_POSITIONS) * 100.0, 2),
        "adopted_boards": adopted_boards,
        "coverage_pct": round(adopted_boards / len(OFFLINE_POSITIONS) * 100.0, 2),
        "false_adoptions": false_adoptions,
        "false_adoption_rate_pct": round(false_adoptions / len(OFFLINE_POSITIONS) * 100.0, 2),
        "per_class": per_class_summary,
        "by_theme": theme_summary,
    }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compare_models(baseline_path: Path, candidate_path: Path, output_json: Path):
    manifest_path = Path(r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl")
    zenodo_dir = Path(r"C:\dev\chess-capture-workspace\external\zenodo\extracted")
    cache_dir = Path(r"C:\dev\chess\dist\cache")

    baseline_hash = sha256_of(baseline_path)
    candidate_hash = sha256_of(candidate_path)

    print(f"Loading Baseline model:  {baseline_path.name} (SHA-256: {baseline_hash[:16]}...)")
    base_session = ort.InferenceSession(str(baseline_path), providers=["CPUExecutionProvider"])

    print(f"Loading Candidate model: {candidate_path.name} (SHA-256: {candidate_hash[:16]}...)")
    cand_session = ort.InferenceSession(str(candidate_path), providers=["CPUExecutionProvider"])

    report = {
        "benchmark": "Milestone 5.4H-R Model Comparison",
        "date": "2026-11-04",
        "baseline_model": str(baseline_path.name),
        "baseline_sha256": baseline_hash,
        "candidate_model": str(candidate_path.name),
        "candidate_sha256": candidate_hash,
        "baseline": {},
        "candidate": {},
    }

    print("\n" + "=" * 80)
    print("EVALUATING DOMAIN A: FENiT REAL PHYSICAL BOARDS (40 full-board photos, 2560 crops)")
    print("=" * 80)
    print("Running Baseline...")
    report["baseline"]["fenit"] = evaluate_domain_a_fenit(base_session, manifest_path, cache_dir)
    print(f"  Baseline  -> Sq Acc: {report['baseline']['fenit']['overall_top1_pct']}% | Occ: {report['baseline']['fenit']['occupied_top1_pct']}% | Exact: {report['baseline']['fenit']['exact_board_count']}/40 | Adopted: {report['baseline']['fenit']['adopted_boards']}/40 (False: {report['baseline']['fenit']['false_adoptions']})")

    print("Running Candidate...")
    report["candidate"]["fenit"] = evaluate_domain_a_fenit(cand_session, manifest_path, cache_dir)
    print(f"  Candidate -> Sq Acc: {report['candidate']['fenit']['overall_top1_pct']}% | Occ: {report['candidate']['fenit']['occupied_top1_pct']}% | Exact: {report['candidate']['fenit']['exact_board_count']}/40 | Adopted: {report['candidate']['fenit']['adopted_boards']}/40 (False: {report['candidate']['fenit']['false_adoptions']})")

    print("\n" + "=" * 80)
    print("EVALUATING DOMAIN B: ZENODO PHYSICAL SET 4 HOLDOUT (832 crops)")
    print("=" * 80)
    print("Running Baseline...")
    report["baseline"]["zenodo"] = evaluate_domain_b_zenodo(base_session, zenodo_dir, cache_dir)
    print(f"  Baseline  -> Sq Acc: {report['baseline']['zenodo']['overall_top1_pct']}% | Occ: {report['baseline']['zenodo']['occupied_top1_pct']}%")

    print("Running Candidate...")
    report["candidate"]["zenodo"] = evaluate_domain_b_zenodo(cand_session, zenodo_dir, cache_dir)
    print(f"  Candidate -> Sq Acc: {report['candidate']['zenodo']['overall_top1_pct']}% | Occ: {report['candidate']['zenodo']['occupied_top1_pct']}%")

    print("\n" + "=" * 80)
    print("EVALUATING DOMAIN C: LAPTOP-SCREEN CHESSBOARDS (20 offline scenes, 1280 squares)")
    print("=" * 80)
    print("Running Baseline...")
    report["baseline"]["screen"] = evaluate_domain_c_screen(base_session)
    print(f"  Baseline  -> Sq Acc: {report['baseline']['screen']['overall_top1_pct']}% | Occ: {report['baseline']['screen']['occupied_top1_pct']}% | Exact: {report['baseline']['screen']['exact_board_count']}/20 | Adopted: {report['baseline']['screen']['adopted_boards']}/20")

    print("Running Candidate...")
    report["candidate"]["screen"] = evaluate_domain_c_screen(cand_session)
    print(f"  Candidate -> Sq Acc: {report['candidate']['screen']['overall_top1_pct']}% | Occ: {report['candidate']['screen']['occupied_top1_pct']}% | Exact: {report['candidate']['screen']['exact_board_count']}/20 | Adopted: {report['candidate']['screen']['adopted_boards']}/20")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved comprehensive comparison report to {output_json}")

    return report

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_milestone_54h_comparison.py <baseline_onnx> <candidate_onnx> [output_json]")
        sys.exit(1)
    base_onnx = Path(sys.argv[1])
    cand_onnx = Path(sys.argv[2])
    out_json = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(r"C:\dev\chess\dist\milestone_54h_model_comparison.json")
    compare_models(base_onnx, cand_onnx, out_json)
