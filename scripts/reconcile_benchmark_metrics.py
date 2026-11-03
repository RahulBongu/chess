#!/usr/bin/env python3
"""
reconcile_benchmark_metrics.py - Milestone 5.4I Metric Reconciliation & Final Benchmark

Evaluates and reconciles adoption metrics across physical and laptop-screen domains
with unambiguous field definitions:
- raw_single_frame_exact_boards: Complete 64-square argmax exact match on a single frame.
- averaged_exact_boards: Complete 64-square exact match after 4-frame temporal averaging.
- gate_accepted_boards: Total boards that pass the production adoption gate.
- correctly_accepted_boards: Gate-accepted boards whose FEN placement exactly matches ground truth.
- incorrectly_accepted_boards: Gate-accepted boards whose FEN placement differs from ground truth.
- rejected_boards: Boards held in UNVERIFIED/UNCERTAIN state by the gate.
- incorrect_adoption_rate_among_accepted: Incorrectly accepted boards / Gate-accepted boards (False Discovery Rate).
- incorrect_adoption_rate_across_all_boards: Incorrectly accepted boards / Total evaluated boards (False Adoption Rate).
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
from scripts.test_starting_position_verification import verify_starting_position_from_probs

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_inference_on_crops(session: ort.InferenceSession, crops_64: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Runs ONNX model on (N, 64, 64, 3) crops."""
    input_name = session.get_inputs()[0].name
    tensor = (crops_64.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    all_probs = []
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start : start + batch_size]
        logits = session.run(None, {input_name: batch})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        all_probs.append(exp_l / np.sum(exp_l, axis=1, keepdims=True))
    return np.vstack(all_probs)

def evaluate_reconciled_physical(session: ort.InferenceSession, manifest_path: Path, cache_dir: Path):
    """Evaluates 40 full-board physical photographs from FENiT."""
    _, val_samples = load_fenit(manifest_path)
    val_images = build_cache(val_samples, "val_rgb_fenit", cache_dir)
    val_labels = np.array([s.label for s in val_samples])
    val_groups = [s.group for s in val_samples]
    board_count = len(val_samples) // 64

    offset = (LOGICAL_SIZE - INPUT_SIZE) // 2
    cropped = val_images[:, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]

    # Single-frame inference
    single_probs = run_inference_on_crops(session, cropped)
    single_preds = np.argmax(single_probs, axis=1)
    single_correct = single_preds == val_labels

    # 4-frame averaged inference with camera sensor noise
    rng = np.random.default_rng(20261104)
    avg_probs_accum = np.zeros_like(single_probs)
    for _ in range(4):
        noise = rng.normal(0, 3.5, cropped.shape).astype(np.float32)
        n_crops = np.clip(cropped.astype(np.float32) + noise, 0, 255)
        avg_probs_accum += run_inference_on_crops(session, n_crops)
    avg_probs = avg_probs_accum / 4.0
    avg_preds = np.argmax(avg_probs, axis=1)
    avg_correct = avg_preds == val_labels

    # Metric evaluation
    raw_single_exact = sum(1 for b in range(board_count) if bool(single_correct[b*64:(b+1)*64].all()))
    avg_exact = sum(1 for b in range(board_count) if bool(avg_correct[b*64:(b+1)*64].all()))

    # Adoption gate evaluation under N=4 (production window, strict max_uncertain=0)
    gate_accepted = 0
    correctly_accepted = 0
    incorrectly_accepted = 0

    for b in range(board_count):
        ad = evaluate_board_adoption(avg_probs[b*64:(b+1)*64], max_uncertain=0)
        gt = val_groups[b*64]
        if ad['ok']:
            gate_accepted += 1
            if ad['placement'] == gt:
                correctly_accepted += 1
            else:
                incorrectly_accepted += 1

    rejected_boards = board_count - gate_accepted
    fdr = (incorrectly_accepted / gate_accepted * 100.0) if gate_accepted > 0 else 0.0
    far = (incorrectly_accepted / board_count * 100.0)

    occupied_mask = val_labels != 0
    empty_mask = val_labels == 0

    return {
        "domain": "Domain A: Physical Board Photography (FENiT held-out validation)",
        "total_evaluated_boards": board_count,
        "total_evaluated_squares": len(val_labels),
        "overall_square_accuracy_top1_pct": round(float(avg_correct.mean() * 100.0), 2),
        "empty_square_accuracy_top1_pct": round(float(avg_correct[empty_mask].mean() * 100.0), 2),
        "occupied_piece_accuracy_top1_pct": round(float(avg_correct[occupied_mask].mean() * 100.0), 2),
        "raw_single_frame_exact_boards": raw_single_exact,
        "raw_single_frame_exact_pct": round(raw_single_exact / board_count * 100.0, 2),
        "averaged_exact_boards": avg_exact,
        "averaged_exact_pct": round(avg_exact / board_count * 100.0, 2),
        "gate_accepted_boards": gate_accepted,
        "adoption_coverage_pct": round(gate_accepted / board_count * 100.0, 2),
        "correctly_accepted_boards": correctly_accepted,
        "correctly_accepted_pct_of_total": round(correctly_accepted / board_count * 100.0, 2),
        "incorrectly_accepted_boards": incorrectly_accepted,
        "rejected_boards": rejected_boards,
        "safe_rejection_coverage_pct": round(rejected_boards / board_count * 100.0, 2),
        "incorrect_adoption_rate_among_accepted_pct": round(fdr, 2),
        "incorrect_adoption_rate_across_all_boards_pct": round(far, 2),
    }

def evaluate_reconciled_screen(session: ort.InferenceSession):
    """Evaluates 20 laptop-screen scenes."""
    scene_count = len(OFFLINE_POSITIONS)
    raw_single_exact = 0
    avg_exact = 0

    gate_accepted = 0
    correctly_accepted = 0
    incorrectly_accepted = 0

    total_squares = scene_count * 64
    all_single_correct = 0
    all_avg_correct = 0
    all_empty_correct = 0
    all_empty_total = 0
    all_occ_correct = 0
    all_occ_total = 0

    rng = np.random.default_rng(20261104)

    for idx, pos in enumerate(OFFLINE_POSITIONS):
        yaw = [-12.0, 0.0, 15.0, -8.0][idx % 4]
        pitch = [18.0, 24.0, 30.0, 16.0][(idx // 4) % 4]

        scene_rgb, gt_corners, gt_placement = render_screen_scene(pos["fen"], pos["theme"], yaw, pitch)
        crops = extract_square_crops_from_scene(scene_rgb, gt_corners)

        # Single frame inference
        s_probs = run_inference_on_crops(session, crops)
        s_preds = np.argmax(s_probs, axis=1)

        # 4-frame averaged inference
        a_probs_accum = np.zeros_like(s_probs)
        for _ in range(4):
            noise = rng.normal(0, 2.0, crops.shape).astype(np.float32)
            n_crops = np.clip(crops.astype(np.float32) + noise, 0, 255)
            a_probs_accum += run_inference_on_crops(session, n_crops)
        a_probs = a_probs_accum / 4.0
        a_preds = np.argmax(a_probs, axis=1)

        # Ground truth labels
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

        s_correct = (s_preds == gt_labels)
        a_correct = (a_preds == gt_labels)

        all_single_correct += int(s_correct.sum())
        all_avg_correct += int(a_correct.sum())

        empty_m = (gt_labels == 0)
        occ_m = (gt_labels != 0)
        all_empty_total += int(empty_m.sum())
        all_empty_correct += int(a_correct[empty_m].sum())
        all_occ_total += int(occ_m.sum())
        all_occ_correct += int(a_correct[occ_m].sum())

        if bool(s_correct.all()):
            raw_single_exact += 1
        if bool(a_correct.all()):
            avg_exact += 1

        # Adoption gate evaluation under N=4 (production window, strict max_uncertain=0)
        ad = evaluate_board_adoption(a_probs, max_uncertain=0)
        if ad['ok']:
            gate_accepted += 1
            if ad['placement'] == gt_placement:
                correctly_accepted += 1
            else:
                incorrectly_accepted += 1

    rejected_boards = scene_count - gate_accepted
    fdr = (incorrectly_accepted / gate_accepted * 100.0) if gate_accepted > 0 else 0.0
    far = (incorrectly_accepted / scene_count * 100.0)

    return {
        "domain": "Domain C: Laptop-Screen Chessboards (20 offline scenes)",
        "total_evaluated_boards": scene_count,
        "total_evaluated_squares": total_squares,
        "overall_square_accuracy_top1_pct": round(all_avg_correct / total_squares * 100.0, 2),
        "empty_square_accuracy_top1_pct": round(all_empty_correct / all_empty_total * 100.0, 2),
        "occupied_piece_accuracy_top1_pct": round(all_occ_correct / all_occ_total * 100.0, 2),
        "raw_single_frame_exact_boards": raw_single_exact,
        "raw_single_frame_exact_pct": round(raw_single_frame_exact_boards := raw_single_exact / scene_count * 100.0, 2),
        "averaged_exact_boards": avg_exact,
        "averaged_exact_pct": round(avg_exact / scene_count * 100.0, 2),
        "gate_accepted_boards": gate_accepted,
        "adoption_coverage_pct": round(gate_accepted / scene_count * 100.0, 2),
        "correctly_accepted_boards": correctly_accepted,
        "correctly_accepted_pct_of_total": round(correctly_accepted / scene_count * 100.0, 2),
        "incorrectly_accepted_boards": incorrectly_accepted,
        "rejected_boards": rejected_boards,
        "safe_rejection_coverage_pct": round(rejected_boards / scene_count * 100.0, 2),
        "incorrect_adoption_rate_among_accepted_pct": round(fdr, 2),
        "incorrect_adoption_rate_across_all_boards_pct": round(far, 2),
    }

def main():
    model_path = Path("modules/live-camera/android/src/main/assets/rgb_square_classifier_v1.onnx")
    manifest_path = Path(r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl")
    cache_dir = Path(r"C:\dev\chess\dist\cache")
    out_json = Path(r"C:\dev\chess\dist\milestone_54i_reconciled_benchmark.json")

    model_hash = sha256_of(model_path)
    print("=" * 85)
    print("RECONCILING BENCHMARK METRICS (MILESTONE 5.4I)")
    print(f"Model: {model_path.name} (SHA-256: {model_hash})")
    print("=" * 85)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    print("\n1. Reconciling Domain A (Physical FENiT 40 Boards)...")
    res_physical = evaluate_reconciled_physical(session, manifest_path, cache_dir)
    print(f"   Raw single-frame exact boards:           {res_physical['raw_single_frame_exact_boards']}/{res_physical['total_evaluated_boards']} ({res_physical['raw_single_frame_exact_pct']}%)")
    print(f"   Averaged exact boards (N=4):             {res_physical['averaged_exact_boards']}/{res_physical['total_evaluated_boards']} ({res_physical['averaged_exact_pct']}%)")
    print(f"   Gate-accepted boards:                    {res_physical['gate_accepted_boards']}/{res_physical['total_evaluated_boards']} (Coverage: {res_physical['adoption_coverage_pct']}%)")
    print(f"   Correctly accepted boards:               {res_physical['correctly_accepted_boards']}/{res_physical['total_evaluated_boards']}")
    print(f"   Incorrectly accepted boards:             {res_physical['incorrectly_accepted_boards']}/{res_physical['total_evaluated_boards']}")
    print(f"   Rejected boards (Safe uncertainty):      {res_physical['rejected_boards']}/{res_physical['total_evaluated_boards']} ({res_physical['safe_rejection_coverage_pct']}%)")
    print(f"   Incorrect adoption rate among accepted:  {res_physical['incorrect_adoption_rate_among_accepted_pct']}%")
    print(f"   Incorrect adoption rate across all:      {res_physical['incorrect_adoption_rate_across_all_boards_pct']}%")

    print("\n2. Reconciling Domain C (Laptop-Screen 20 Scenes)...")
    res_screen = evaluate_reconciled_screen(session)
    print(f"   Raw single-frame exact boards:           {res_screen['raw_single_frame_exact_boards']}/{res_screen['total_evaluated_boards']} ({res_screen['raw_single_frame_exact_pct']}%)")
    print(f"   Averaged exact boards (N=4):             {res_screen['averaged_exact_boards']}/{res_screen['total_evaluated_boards']} ({res_screen['averaged_exact_pct']}%)")
    print(f"   Gate-accepted boards:                    {res_screen['gate_accepted_boards']}/{res_screen['total_evaluated_boards']} (Coverage: {res_screen['adoption_coverage_pct']}%)")
    print(f"   Correctly accepted boards:               {res_screen['correctly_accepted_boards']}/{res_screen['total_evaluated_boards']} ({res_screen['correctly_accepted_pct_of_total']}%)")
    print(f"   Incorrectly accepted boards:             {res_screen['incorrectly_accepted_boards']}/{res_screen['total_evaluated_boards']}")
    print(f"   Rejected boards (Safe uncertainty):      {res_screen['rejected_boards']}/{res_screen['total_evaluated_boards']} ({res_screen['safe_rejection_coverage_pct']}%)")
    print(f"   Incorrect adoption rate among accepted:  {res_screen['incorrect_adoption_rate_among_accepted_pct']}%")
    print(f"   Incorrect adoption rate across all:      {res_screen['incorrect_adoption_rate_across_all_boards_pct']}%")

    final_report = {
        "benchmark": "Milestone 5.4I Reconciled Benchmark",
        "date": "2026-11-04",
        "production_model": model_path.name,
        "production_model_sha256": model_hash,
        "physical_domain_a": res_physical,
        "laptop_screen_domain_c": res_screen,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved reconciled report to {out_json}")

if __name__ == "__main__":
    main()
