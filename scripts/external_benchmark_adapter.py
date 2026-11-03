#!/usr/bin/env python3
"""
external_benchmark_adapter.py - Milestone 5.2I External Real-Image Benchmark Adapter

Adapts external public full-board chess datasets (FENiT / Mendeley wpp9xnbcp6 v1)
for offline vision pipeline smoke testing without contaminating in-house physical
capture partitions or fabricating physical piece/set identities.

Key Invariants:
1. External records are OBSERVATION-ONLY. They never emit SETxx physical IDs,
   physical tag IDs, or physical-set identities.
2. External outputs live in C:\\dev\\chess-capture-workspace\\external\\..., never in
   in-house training, validation, or blind-holdout partitions.
3. Every step (decode, corner validation, rectification, square crop slicing,
   placement matching, classifier evaluation) is explicitly validated and recorded.
4. Unsupported images (timestamp-named, single-piece, missing board corners,
   corrupt geometry) are recorded with explicit skip reasons.
5. Deployed model weights (src/vision/pieceModelWeights.ts) are evaluated read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Adjust sys.path to access local scripts
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_real_benchmark import (  # noqa: E402
    CLASS_TO_IDX,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    HOG_PATCH_SIZE,
    PIECE_CLASSES,
    extract_hog_288,
    load_deployed_weights_from_ts,
    predict_piece,
)
from validate_capture_manifest import validate_placement_only  # noqa: E402

CANONICAL_BOARD_SIZE = 800
CANONICAL_SQUARE_SIZE = 100
GRID_DIM = 8

CLASS_NAMES = {
    0: "Board",
    1: "White-King",
    2: "White-Pawn",
    3: "Black-Pawn",
    4: "Black-King",
    5: "White-Queen",
    6: "White-Bishop",
    7: "White-Knight",
    8: "White-Rook",
    9: "Black-Bishop",
    10: "Black-Rook",
    11: "Black-Knight",
    12: "Black-Queen",
}

FENIT_TO_PIECE_INFO = {
    1: {"type": "king", "color": "white", "char": "K"},
    2: {"type": "pawn", "color": "white", "char": "P"},
    3: {"type": "pawn", "color": "black", "char": "p"},
    4: {"type": "king", "color": "black", "char": "k"},
    5: {"type": "queen", "color": "white", "char": "Q"},
    6: {"type": "bishop", "color": "white", "char": "B"},
    7: {"type": "knight", "color": "white", "char": "N"},
    8: {"type": "rook", "color": "white", "char": "R"},
    9: {"type": "bishop", "color": "black", "char": "b"},
    10: {"type": "rook", "color": "black", "char": "r"},
    11: {"type": "knight", "color": "black", "char": "n"},
    12: {"type": "queen", "color": "black", "char": "q"},
}

CANONICAL_DST = np.array(
    [
        [0.0, 0.0],
        [CANONICAL_BOARD_SIZE, 0.0],
        [CANONICAL_BOARD_SIZE, CANONICAL_BOARD_SIZE],
        [0.0, CANONICAL_BOARD_SIZE],
    ],
    dtype=np.float32,
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_fenit_stem(stem: str) -> str:
    s = stem.strip()
    s = re.sub(r"\s*\((?:full board|board)\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    s = re.sub(r"\s+[wb]$", "", s)
    return s.strip()


def parse_fenit_placement(stem: str) -> Optional[str]:
    candidate = normalize_fenit_stem(stem)
    for sep in ("_", "-"):
        parts = candidate.split(sep)
        if len(parts) != 8:
            continue
        if not all(re.fullmatch(r"[1-8pnbrqkPNBRQK]+", part) for part in parts):
            continue
        if any(re.search(r"\d{2}", part) for part in parts):
            continue
        if not all(sum(int(ch) if ch.isdigit() else 1 for ch in part) == 8 for part in parts):
            continue
        placement = "/".join(parts)
        valid, _ = validate_placement_only(placement)
        if valid:
            return placement
    return None


def fen_to_grid(placement: str) -> List[List[str]]:
    rows = []
    for rank_str in placement.split("/"):
        row = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend("." * int(ch))
            else:
                row.append(ch)
        rows.append(row)
    return rows


def parse_fenit_annotation(ann_path: Path) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    """
    Parses a FENiT YOLO pose annotation file.
    Returns: (board_corners_4x2, list_of_pieces)
    """
    if not ann_path.exists():
        return None, []

    lines = [
        line.strip()
        for line in ann_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    board_corners = None
    pieces = []

    for line in lines:
        tokens = line.split()
        if len(tokens) < 5:
            continue
        cls = int(tokens[0])
        cx, cy, w, h = (float(v) for v in tokens[1:5])
        kpts = [float(v) for v in tokens[5:]]

        if cls == 0:
            # Board: 4 keypoints (x, y, visibility)
            if len(kpts) >= 12:
                board_corners = np.array(
                    [[kpts[i * 3], kpts[i * 3 + 1]] for i in range(4)], dtype=np.float32
                )
        elif cls in FENIT_TO_PIECE_INFO:
            pieces.append(
                {
                    "class_id": cls,
                    "info": FENIT_TO_PIECE_INFO[cls],
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "kpts": kpts,
                }
            )

    return board_corners, pieces


def is_strictly_convex(corners: np.ndarray) -> bool:
    """Verifies that 4 points form a non-self-intersecting strictly convex quadrilateral."""
    if len(corners) != 4:
        return False
    # Check cross products of consecutive edge vectors
    signs = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        p2 = corners[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-6:
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def get_canonical_screen_corners(corners_norm: np.ndarray, w_img: int, h_img: int) -> np.ndarray:
    """Orders 4 corners clockwise starting at top-left in screen coordinates."""
    corners_px = corners_norm.copy()
    corners_px[:, 0] *= w_img
    corners_px[:, 1] *= h_img
    center = corners_px.mean(axis=0)
    angles = np.arctan2(corners_px[:, 1] - center[1], corners_px[:, 0] - center[0])
    order = np.argsort(angles)
    pts = corners_px[order]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def resolve_board_orientation(
    corners: np.ndarray,
    pieces: List[Dict[str, Any]],
    grid: List[List[str]],
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, str, float, int, bool, List[int]]:
    """
    Tests 4 cyclic shifts of canonical screen corners against ground-truth FEN occupancy
    to resolve physical orientation (White-bottom vs Flank-right vs Black-bottom vs Flank-left).
    Returns: (resolved_corners, orientation_name, match_rate, matches, is_tied, optimal_shifts)
    """
    h_img, w_img = image_shape
    C_screen = get_canonical_screen_corners(corners, w_img, h_img)

    orientation_names = [
        "white_perspective",
        "flank_right",
        "black_perspective",
        "flank_left",
    ]

    shift_matches = []
    for shift in range(4):
        shifted = np.roll(C_screen, -shift, axis=0)
        H = cv2.getPerspectiveTransform(shifted, CANONICAL_DST)
        matches = 0
        for p in pieces:
            px = p["cx"] * w_img
            py = p["cy"] * h_img
            den = H[2, 0] * px + H[2, 1] * py + H[2, 2]
            if abs(den) < 1e-12:
                continue
            u = (H[0, 0] * px + H[0, 1] * py + H[0, 2]) / den
            v = (H[1, 0] * px + H[1, 1] * py + H[1, 2]) / den
            if 0.0 <= u < CANONICAL_BOARD_SIZE and 0.0 <= v < CANONICAL_BOARD_SIZE:
                col = min(7, max(0, int(u / CANONICAL_SQUARE_SIZE)))
                row = min(7, max(0, int(v / CANONICAL_SQUARE_SIZE)))
                if grid[row][col] == p["info"]["char"]:
                    matches += 1
        shift_matches.append(matches)

    max_matches = max(shift_matches)
    best_shifts = [i for i, m in enumerate(shift_matches) if m == max_matches]
    is_tied = len(best_shifts) > 1

    # Deterministic default: if shift 0 is among optimal, prefer 0; otherwise first optimal
    best_shift = 0 if 0 in best_shifts else best_shifts[0]

    total_pieces = len(pieces)
    match_rate = (max_matches / total_pieces) if total_pieces > 0 else 0.0
    resolved_corners = np.roll(C_screen, -best_shift, axis=0)
    resolved_name = orientation_names[best_shift]

    return resolved_corners, resolved_name, match_rate, max_matches, is_tied, best_shifts


def resolve_image_driven_orientation(
    corners: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Orders 4 board corners purely from image plane geometry:
    top-left, top-right, bottom-right, bottom-left in screen coordinates.
    Never uses ground-truth FEN or piece labels.
    """
    h_img, w_img = image_shape
    return get_canonical_screen_corners(corners, w_img, h_img)


def compute_selective_accuracy_accounting(
    samples: List[Dict[str, Any]],
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> Dict[str, Any]:
    """
    Computes rigorous selective-accuracy accounting with independent numerators:
    - Overall accuracy: overall_correct / total_samples
    - Coverage: accepted / total_samples
    - Abstention rate: abstained / total_samples
    - Selective accuracy: accepted_and_correct / accepted
    - Error rate among accepted: accepted_and_incorrect / accepted
    - Diagnostic abstained accuracy: abstained_and_correct / abstained
    """
    total = len(samples)
    overall_correct = 0
    accepted = 0
    abstained = 0
    accepted_and_correct = 0
    accepted_and_incorrect = 0
    abstained_and_correct = 0
    abstained_and_incorrect = 0

    for s in samples:
        conf = float(s.get("confidence", 0.0))
        margin = float(s.get("margin", 0.0))
        is_correct = bool(s.get("is_correct", False))

        if is_correct:
            overall_correct += 1

        is_accepted = bool(conf >= conf_threshold and margin >= margin_threshold)
        if is_accepted:
            accepted += 1
            if is_correct:
                accepted_and_correct += 1
            else:
                accepted_and_incorrect += 1
        else:
            abstained += 1
            if is_correct:
                abstained_and_correct += 1
            else:
                abstained_and_incorrect += 1

    return {
        "total_samples": total,
        "overall_correct": overall_correct,
        "overall_accuracy": round((overall_correct / total), 4) if total > 0 else 0.0,
        "accepted_count": accepted,
        "coverage": round((accepted / total), 4) if total > 0 else 0.0,
        "abstention_count": abstained,
        "abstention_rate": round((abstained / total), 4) if total > 0 else 0.0,
        "accepted_and_correct": accepted_and_correct,
        "accepted_and_incorrect": accepted_and_incorrect,
        "selective_accuracy": round((accepted_and_correct / accepted), 4) if accepted > 0 else 0.0,
        "error_rate_among_accepted": round((accepted_and_incorrect / accepted), 4) if accepted > 0 else 0.0,
        "abstained_and_correct": abstained_and_correct,
        "abstained_and_incorrect": abstained_and_incorrect,
        "abstained_accuracy_diagnostic": round((abstained_and_correct / abstained), 4) if abstained > 0 else 0.0,
    }


def extract_observation_crops(
    rectified_board: np.ndarray,
    placement: str,
    out_crop_dir: Path,
    scene_id: str,
    W_model: Optional[np.ndarray] = None,
    b_model: Optional[np.ndarray] = None,
    temp_model: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Slices canonical 800x800 board into 64 square crops (a1-h8) and evaluates deployed
    HOG classifier on each occupied square.
    """
    grid = fen_to_grid(placement)
    crops: List[Dict[str, Any]] = []
    out_crop_dir.mkdir(parents=True, exist_ok=True)

    correct_piece_predictions = 0
    total_occupied_evaluated = 0
    accepted_count = 0
    abstain_count = 0
    accepted_and_correct = 0
    accepted_and_incorrect = 0
    abstained_and_correct = 0
    abstained_and_incorrect = 0

    for r_idx in range(8):
        rank_num = 8 - r_idx
        for f_idx in range(8):
            file_char = chr(97 + f_idx)
            square = f"{file_char}{rank_num}"
            ground_truth_char = grid[r_idx][f_idx]
            is_occupied = ground_truth_char != "."

            # Square bounds in rectified 800x800
            x0 = f_idx * CANONICAL_SQUARE_SIZE
            y0 = r_idx * CANONICAL_SQUARE_SIZE
            x1 = x0 + CANONICAL_SQUARE_SIZE
            y1 = y0 + CANONICAL_SQUARE_SIZE

            square_crop = rectified_board[y0:y1, x0:x1]
            crop_filename = f"{scene_id}_{square}.png"
            crop_path = out_crop_dir / crop_filename
            cv2.imwrite(str(crop_path), square_crop)

            piece_type = "empty"
            piece_color = "none"
            pred_type = None
            pred_conf = None
            pred_margin = None
            is_correct = None
            is_accepted = None

            if is_occupied:
                piece_color = "white" if ground_truth_char.isupper() else "black"
                char_lower = ground_truth_char.lower()
                type_map = {
                    "p": "pawn",
                    "n": "knight",
                    "b": "bishop",
                    "r": "rook",
                    "q": "queen",
                    "k": "king",
                }
                piece_type = type_map.get(char_lower, "unknown")

                # Deployed HOG inference if model is loaded
                if W_model is not None and b_model is not None:
                    total_occupied_evaluated += 1
                    # Convert to grayscale 64x64 patch
                    gray = cv2.cvtColor(square_crop, cv2.COLOR_BGR2GRAY)
                    patch = cv2.resize(gray, (HOG_PATCH_SIZE, HOG_PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
                    features = extract_hog_288(patch.astype(np.float32))
                    pred_idx, probs, conf, margin = predict_piece(features, W_model, b_model, temp_model)
                    pred_type = PIECE_CLASSES[pred_idx]
                    pred_conf = round(float(conf), 4)
                    pred_margin = round(float(margin), 4)
                    is_correct = bool(pred_type == piece_type)
                    if is_correct:
                        correct_piece_predictions += 1

                    is_accepted = bool(conf >= DEFAULT_CONFIDENCE_THRESHOLD and margin >= DEFAULT_MARGIN_THRESHOLD)
                    if is_accepted:
                        accepted_count += 1
                        if is_correct:
                            accepted_and_correct += 1
                        else:
                            accepted_and_incorrect += 1
                    else:
                        abstain_count += 1
                        if is_correct:
                            abstained_and_correct += 1
                        else:
                            abstained_and_incorrect += 1

            crops.append(
                {
                    "square": square,
                    "crop_path": str(crop_path),
                    "crop_type": "external_observation",
                    "is_occupied": is_occupied,
                    "piece_char": None if not is_occupied else ground_truth_char,
                    "piece_type": piece_type,
                    "piece_color": piece_color,
                    "predicted_type": pred_type,
                    "confidence": pred_conf,
                    "margin": pred_margin,
                    "is_accepted": is_accepted,
                    "is_prediction_correct": is_correct,
                }
            )

    eval_stats = {
        "occupied_squares": total_occupied_evaluated,
        "correct_predictions": correct_piece_predictions,
        "overall_accuracy": round(
            (correct_piece_predictions / total_occupied_evaluated) if total_occupied_evaluated > 0 else 0.0,
            4,
        ),
        "accepted_count": accepted_count,
        "abstain_count": abstain_count,
        "accepted_and_correct": accepted_and_correct,
        "accepted_and_incorrect": accepted_and_incorrect,
        "abstained_and_correct": abstained_and_correct,
        "abstained_and_incorrect": abstained_and_incorrect,
        "selective_accuracy": round(
            (accepted_and_correct / accepted_count) if accepted_count > 0 else 0.0,
            4,
        ),
        "error_rate_among_accepted": round(
            (accepted_and_incorrect / accepted_count) if accepted_count > 0 else 0.0,
            4,
        ),
    }

    return crops, eval_stats


def process_external_image(
    image_path: Path,
    ann_path: Path,
    out_dir: Path,
    W_model: Optional[np.ndarray],
    b_model: Optional[np.ndarray],
    temp_model: float,
) -> Dict[str, Any]:
    """Processes a single candidate external source image through the smoke-test pipeline."""
    stem = image_path.stem
    scene_id = f"FENIT_{stem[:40]}"
    source_sha256 = sha256_file(image_path)

    # 1. Image decoding & input-dimension validation
    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "status": "SKIPPED",
            "reason": "decode_failed",
            "file": image_path.name,
            "scene_id": scene_id,
        }
    h_img, w_img, channels = img.shape
    if channels != 3 or h_img < 500 or w_img < 500:
        return {
            "status": "SKIPPED",
            "reason": f"invalid_dimensions_{w_img}x{h_img}x{channels}",
            "file": image_path.name,
            "scene_id": scene_id,
        }

    # 2. Annotation & Ground Truth parsing
    placement = parse_fenit_placement(stem)
    if not placement:
        # Check if single piece or timestamp
        if re.fullmatch(r"\d{8}_\d{6}", stem):
            reason = "timestamp_named_no_ground_truth_fen"
        elif len(stem) <= 6 and stem in {"B", "K", "N", "P", "Q", "R"}:
            reason = "isolated_piece_photo_no_board_grid"
        else:
            reason = "unparseable_placement_ground_truth"
        return {
            "status": "SKIPPED",
            "reason": reason,
            "file": image_path.name,
            "scene_id": scene_id,
            "image_dimensions": [w_img, h_img],
        }

    board_corners, pieces = parse_fenit_annotation(ann_path)
    if board_corners is None:
        return {
            "status": "SKIPPED",
            "reason": "missing_board_corner_annotations",
            "file": image_path.name,
            "scene_id": scene_id,
            "placement": placement,
            "image_dimensions": [w_img, h_img],
        }

    if not is_strictly_convex(board_corners):
        return {
            "status": "FAILED",
            "reason": "non_convex_board_corners",
            "file": image_path.name,
            "scene_id": scene_id,
        }

    # 3. Perspective rectification & orientation resolution (Mode 1: Oracle)
    grid = fen_to_grid(placement)
    resolved_corners, orientation, match_rate, matches, is_tied, best_shifts = resolve_board_orientation(
        board_corners, pieces, grid, (h_img, w_img)
    )

    # 3b. Perspective rectification & orientation resolution (Mode 2: Image-driven)
    image_driven_corners = resolve_image_driven_orientation(board_corners, (h_img, w_img))
    H_img_driven = cv2.getPerspectiveTransform(image_driven_corners, CANONICAL_DST)
    img_driven_matches = 0
    for p in pieces:
        px = p["cx"] * w_img
        py = p["cy"] * h_img
        den = H_img_driven[2, 0] * px + H_img_driven[2, 1] * py + H_img_driven[2, 2]
        if abs(den) < 1e-12:
            continue
        u = (H_img_driven[0, 0] * px + H_img_driven[0, 1] * py + H_img_driven[0, 2]) / den
        v = (H_img_driven[1, 0] * px + H_img_driven[1, 1] * py + H_img_driven[1, 2]) / den
        if 0.0 <= u < CANONICAL_BOARD_SIZE and 0.0 <= v < CANONICAL_BOARD_SIZE:
            col = min(7, max(0, int(u / CANONICAL_SQUARE_SIZE)))
            row = min(7, max(0, int(v / CANONICAL_SQUARE_SIZE)))
            if grid[row][col] == p["info"]["char"]:
                img_driven_matches += 1
    total_pieces = len(pieces)
    img_driven_match_rate = (img_driven_matches / total_pieces) if total_pieces > 0 else 0.0
    agrees_with_oracle = bool(0 in best_shifts)
    is_unique_optimal = bool(best_shifts == [0])

    H = cv2.getPerspectiveTransform(resolved_corners, CANONICAL_DST)
    rectified = cv2.warpPerspective(img, H, (CANONICAL_BOARD_SIZE, CANONICAL_BOARD_SIZE))

    rect_dir = out_dir / "rectified"
    rect_dir.mkdir(parents=True, exist_ok=True)
    rectified_path = rect_dir / f"{scene_id}_rectified_800x800.png"
    cv2.imwrite(str(rectified_path), rectified)

    # 4. Square crop extraction & deployed piece classifier evaluation
    crop_dir = out_dir / "crops" / scene_id
    crops, eval_stats = extract_observation_crops(
        rectified,
        placement,
        crop_dir,
        scene_id,
        W_model=W_model,
        b_model=b_model,
        temp_model=temp_model,
    )

    return {
        "status": "PROCESSED",
        "file": image_path.name,
        "source_sha256": source_sha256,
        "scene_id": scene_id,
        "dataset": "FENiT_v1",
        "dataset_doi": "10.17632/wpp9xnbcp6.1",
        "license": "CC BY 4.0",
        "image_dimensions": [w_img, h_img],
        "ground_truth_placement": placement,
        "annotated_piece_count": len(pieces),
        "board_corners": board_corners.tolist(),
        "resolved_orientation": orientation,
        "placement_match_score": round(match_rate, 4),
        "placement_matched_pieces": matches,
        "rectified_board_path": str(rectified_path),
        "derived_crops_count": len(crops),
        "deployed_classifier_eval": eval_stats,
        "evaluation_modes": {
            "mode1_ground_truth_assisted": {
                "description": "Orientation resolved using FEN oracle to select optimal 4-way rotation",
                "resolved_orientation": orientation,
                "placement_match_score": round(match_rate, 4),
                "placement_matched_pieces": matches,
                "is_oracle_assisted": True,
                "is_tied": is_tied,
                "optimal_shifts": best_shifts,
            },
            "mode2_image_driven": {
                "description": "Orientation determined by image plane screen coordinates without FEN oracle",
                "resolved_orientation": "canonical_screen_space",
                "placement_match_score": round(img_driven_match_rate, 4),
                "placement_matched_pieces": img_driven_matches,
                "orientation_agrees_with_oracle": agrees_with_oracle,
                "is_unique_optimal": is_unique_optimal,
                "is_tied": is_tied,
                "is_oracle_assisted": False,
            },
        },
        "crops": crops,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="External real-image chess benchmark adapter.")
    parser.add_argument(
        "--raw-dir",
        default=r"C:\dev\chess-capture-workspace\external\fenit\raw",
        help="Path to downloaded FENiT raw directory",
    )
    parser.add_argument(
        "--out-dir",
        default=r"C:\dev\chess-capture-workspace\external\fenit\benchmark_output",
        help="Path to external benchmark output directory (outside Git)",
    )
    parser.add_argument(
        "--weights-file",
        default=r"C:\dev\chess\src\vision\pieceModelWeights.ts",
        help="Path to deployed TypeScript weights file",
    )
    parser.add_argument(
        "--manifest-out",
        default=r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl",
        help="Path to write external benchmark JSONL manifest",
    )
    parser.add_argument(
        "--subset",
        nargs="*",
        default=[],
        help="Optional specific filenames or stems to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of eligible images to process (0 = all)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    print("========================================================================================")
    print("AI CHESS NAVIGATOR - EXTERNAL REAL-IMAGE BENCHMARK ADAPTER")
    print("========================================================================================")
    print(f"Dataset Raw Directory: {raw_dir}")
    print(f"Output Directory:      {out_dir}")
    print(f"Manifest Output:       {manifest_out}")

    # Load deployed piece classifier weights (read-only)
    W_model, b_model, temp_model = None, None, 1.0
    if os.path.exists(args.weights_file):
        W_model, b_model, temp_model = load_deployed_weights_from_ts(args.weights_file)
        print(f"Loaded deployed weights: W={W_model.shape}, b={b_model.shape}, temp={temp_model}")

    # Find image files
    all_images = sorted(
        p for p in raw_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.subset:
        subset_set = set(args.subset)
        images = [p for p in all_images if p.name in subset_set or p.stem in subset_set]
    else:
        images = all_images

    print(f"Total Candidate Images: {len(images)}")

    records: List[Dict[str, Any]] = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    reasons: Counter[str] = Counter()

    t_start = time.perf_counter()

    for idx, img_path in enumerate(images, 1):
        ann_path = raw_dir / f"{img_path.stem}.txt"
        res = process_external_image(
            img_path, ann_path, out_dir, W_model, b_model, temp_model
        )

        status = res["status"]
        if status == "PROCESSED":
            processed_count += 1
            records.append(res)
            if args.limit > 0 and processed_count >= args.limit:
                break
        elif status == "SKIPPED":
            skipped_count += 1
            reasons[res["reason"]] += 1
            records.append(res)
        else:
            failed_count += 1
            reasons[res["reason"]] += 1
            records.append(res)

        if idx % 50 == 0 or idx == len(images):
            print(
                f"  [{idx}/{len(images)}] Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}",
                flush=True,
            )

    t_elapsed = time.perf_counter() - t_start

    # Write JSONL manifest
    with manifest_out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print("\n========================================================================================")
    print(" BENCHMARK ADAPTER EXECUTION SUMMARY")
    print("========================================================================================")
    print(f"Total Evaluated:       {len(records)}")
    print(f"Successfully Processed:{processed_count}")
    print(f"Skipped / Ineligible:  {skipped_count}")
    print(f"Geometric Failures:    {failed_count}")
    print(f"Total Run Time:        {t_elapsed:.2f} s")
    print(f"Manifest Generated:    {manifest_out}")

    print("\n--- Skip / Ineligibility Reasons ---")
    for reason, count in reasons.most_common():
        print(f"  {reason:<45}: {count}")

    if processed_count > 0:
        processed_recs = [r for r in records if r["status"] == "PROCESSED"]
        avg_mode1 = np.mean([r["placement_match_score"] for r in processed_recs])
        avg_mode2 = np.mean([r["evaluation_modes"]["mode2_image_driven"]["placement_match_score"] for r in processed_recs])
        agrees_optimal_cnt = sum(1 for r in processed_recs if r["evaluation_modes"]["mode2_image_driven"]["orientation_agrees_with_oracle"])
        unique_optimal_cnt = sum(1 for r in processed_recs if r["evaluation_modes"]["mode2_image_driven"].get("is_unique_optimal", False))
        tied_optimal_cnt = sum(1 for r in processed_recs if r["evaluation_modes"]["mode2_image_driven"].get("is_tied", False) and r["evaluation_modes"]["mode2_image_driven"]["orientation_agrees_with_oracle"])
        exact_mode1 = sum(1 for r in processed_recs if r["placement_match_score"] == 1.0)
        exact_mode2 = sum(1 for r in processed_recs if r["evaluation_modes"]["mode2_image_driven"]["placement_match_score"] == 1.0)

        orientations = Counter(r["resolved_orientation"] for r in processed_recs)
        total_crops = sum(r["derived_crops_count"] for r in processed_recs)

        total_pieces_eval = sum(r["deployed_classifier_eval"]["occupied_squares"] for r in processed_recs)
        total_pieces_correct = sum(r["deployed_classifier_eval"]["correct_predictions"] for r in processed_recs)
        total_accepted = sum(r["deployed_classifier_eval"].get("accepted_count", 0) for r in processed_recs)
        total_abstentions = sum(r["deployed_classifier_eval"]["abstain_count"] for r in processed_recs)
        total_accepted_and_correct = sum(r["deployed_classifier_eval"].get("accepted_and_correct", 0) for r in processed_recs)
        total_accepted_and_incorrect = sum(r["deployed_classifier_eval"].get("accepted_and_incorrect", 0) for r in processed_recs)
        total_abstained_and_correct = sum(r["deployed_classifier_eval"].get("abstained_and_correct", 0) for r in processed_recs)
        total_abstained_and_incorrect = sum(r["deployed_classifier_eval"].get("abstained_and_incorrect", 0) for r in processed_recs)

        overall_piece_acc = (
            (total_pieces_correct / total_pieces_eval * 100.0) if total_pieces_eval > 0 else 0.0
        )
        selective_acc = (
            (total_accepted_and_correct / total_accepted * 100.0)
            if total_accepted > 0
            else 0.0
        )
        error_rate_accepted = (
            (total_accepted_and_incorrect / total_accepted * 100.0)
            if total_accepted > 0
            else 0.0
        )
        abstained_diag_acc = (
            (total_abstained_and_correct / total_abstentions * 100.0)
            if total_abstentions > 0
            else 0.0
        )

        print("\n--- Disaggregated Evaluation Modes (Milestones 5.2J / 5.2K / 5.2L) ---")
        print(f"Mode 1 (Ground-Truth-Assisted / Oracle Orientation):")
        print(f"  Mean FEN Placement Match Score:    {avg_mode1 * 100:.2f}% (YOLO box to FEN square alignment)")
        print(f"  Exact Placement Matches (100%):    {exact_mode1}/{len(processed_recs)} ({exact_mode1/len(processed_recs)*100:.2f}%)")
        print(f"  Resolved Orientations:             {dict(orientations)}")
        print(f"  [DISCLOSURE: Mode 1 uses filename FEN to select best 4-way corner orientation. Not autonomous.]")
        print(f"\nMode 2 (Image-Driven Screen-Space Orientation):")
        print(f"  Mean FEN Placement Match Score:    {avg_mode2 * 100:.2f}% (screen-space corner ordering)")
        print(f"  Exact Placement Matches (100%):    {exact_mode2}/{len(processed_recs)} ({exact_mode2/len(processed_recs)*100:.2f}%)")
        print(f"  Orientation Agreement with Oracle:")
        print(f"    - Unique Strict Agreement:       {unique_optimal_cnt}/{len(processed_recs)} ({unique_optimal_cnt/len(processed_recs)*100:.2f}%)")
        print(f"    - Optimal or Tied Agreement:     {agrees_optimal_cnt}/{len(processed_recs)} ({agrees_optimal_cnt/len(processed_recs)*100:.2f}%)")
        print(f"    - Tied Cases (Multi-Optimal):    {tied_optimal_cnt}/{len(processed_recs)} ({tied_optimal_cnt/len(processed_recs)*100:.2f}%)")
        print(f"  [DISCLOSURE: Reconciles 5.2J (71.17% optimal/tied) vs 5.2K (65.97% unique strict). Drops { (avg_mode1 - avg_mode2)*100:.2f} pp from oracle.]")
        print(f"\nMode 3 (End-to-End Autonomous Pipeline):")
        print(f"  Autonomous Raw Board Detection:    Requires deep-learning pose keypoints or manual calibration.")
        print(f"  Classical Contour Detection:       Not sufficient for autonomous use; see")
        print(f"                                     docs/BOARD_GEOMETRY_AND_CALIBRATION_BENCHMARK.md for explicit per-image accounting.")

        print("\n--- Deployed Piece Classifier Statistics (src/vision/pieceModelWeights.ts) ---")
        print(f"Total Occupied Squares Evaluated:   {total_pieces_eval}")
        print(f"Overall Top-1 Correct:              {total_pieces_correct} (overall accuracy: {overall_piece_acc:.2f}%)")
        print(f"Accepted Predictions (Coverage):    {total_accepted}/{total_pieces_eval} ({total_accepted/total_pieces_eval*100:.2f}%)")
        print(f"Abstained Predictions (Rate):       {total_abstentions}/{total_pieces_eval} ({total_abstentions/total_pieces_eval*100:.2f}%)")
        print(f"Accepted & Correct (Numerator):     {total_accepted_and_correct}")
        print(f"Accepted & Incorrect:               {total_accepted_and_incorrect}")
        print(f"Selective Accuracy (Correct/Accepted): {selective_acc:.2f}% ({total_accepted_and_correct}/{total_accepted})")
        print(f"Error Rate among Accepted:          {error_rate_accepted:.2f}% ({total_accepted_and_incorrect}/{total_accepted})")
        print(f"Abstained Correct (Diagnostic Only): {total_abstained_and_correct}/{total_abstentions} ({abstained_diag_acc:.2f}%)")
        print(f"[METRIC CORRECTION: Milestone 5.2K corrects selective accuracy numerator from {total_pieces_correct} to {total_accepted_and_correct}. True selective accuracy is {selective_acc:.2f}%, not 16.20%.]")
        print(f"Total Derived Square Crops:         {total_crops} (64 per full board outside Git)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
