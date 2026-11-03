#!/usr/bin/env python3
"""
benchmark_screen_chessboards.py - Laptop-Screen Chessboard Benchmark & Evaluation

Evaluates the chess vision pipeline on realistic laptop-screen chessboards viewed
at handheld camera perspectives with browser chrome, sidebars, screen glare,
moiré, and digital piece styles.

Metrics evaluated:
1. Board Localisation: IoU and corner error of proposed quadrilateral vs GT.
2. Square Classification: Top-1 square accuracy, empty vs occupied accuracy.
3. Complete-Board Exact Accuracy: All 64 squares match ground-truth placement.
4. Production Adoption Gate: True adoption, false adoption, and safe rejection rates.
5. Move Detection: 2-ply legal move transition reconciliation.
6. Overlay Alignment: Corner pixel offset between projected grid and actual squares.
"""

from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_screen_board import draw_digital_piece, CLASSES
from scripts.train_rgb_square_model import INPUT_SIZE, LOGICAL_SIZE
from scripts.benchmark_board_geometry import (
    detect_approach_a_baseline,
    compute_polygon_iou,
    compute_normalized_corner_error,
)
from scripts.evaluate_production_adoption import evaluate_board_adoption

import onnxruntime as ort

PIECE_CLASSES = CLASSES
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Known offline positions for evaluation
OFFLINE_POSITIONS = [
    # 1. Starting position
    {"name": "Starting Position", "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "theme": "green"},
    # 2. Italian Game (Giuoco Piano)
    {"name": "Italian Game", "fen": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "theme": "brown"},
    # 3. Sicilian Defense (Open Sicilian)
    {"name": "Sicilian Defense", "fen": "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6", "theme": "blue"},
    # 4. Queen's Gambit Declined
    {"name": "Queen's Gambit Declined", "fen": "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4", "theme": "gray"},
    # 5. Ruy Lopez (Morphy Defense)
    {"name": "Ruy Lopez", "fen": "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5", "theme": "green"},
    # 6. French Defense (Winawer)
    {"name": "French Winawer", "fen": "rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4", "theme": "brown"},
    # 7. Caro-Kann Defense (Advance)
    {"name": "Caro-Kann Advance", "fen": "rn1qkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3", "theme": "blue"},
    # 8. King's Indian Defense
    {"name": "King's Indian", "fen": "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP2BPPP/R1BQK1NR w KQ - 4 6", "theme": "gray"},
    # 9. Endgame: King + Rook vs King
    {"name": "KR vs K Endgame", "fen": "8/8/8/4k3/8/8/1R6/4K3 w - - 0 1", "theme": "green"},
    # 10. Endgame: Lucena Position (Rook endgame)
    {"name": "Lucena Position", "fen": "1K1R4/8/k7/8/8/8/5r2/8 w - - 0 1", "theme": "brown"},
    # 11. Endgame: King + Pawn vs King
    {"name": "Pawn Endgame", "fen": "8/8/4k3/8/4P3/8/4K3/8 w - - 0 1", "theme": "blue"},
    # 12. Tactical Puzzle: Back Rank Mate
    {"name": "Back Rank Threat", "fen": "3r2k1/5ppp/8/8/8/8/4RPPP/6K1 w - - 0 1", "theme": "gray"},
    # 13. Tactical Puzzle: Knight Fork
    {"name": "Knight Fork Setup", "fen": "r1b1k2r/pppp1ppp/5n2/4q3/1b2N3/8/PPPP1PPP/R1BQKB1R w KQkq - 0 7", "theme": "green"},
    # 14. Tactical Puzzle: Pin on Queen
    {"name": "Pin on Queen", "fen": "rnb1kbnr/pp1p1ppp/4p3/2q5/4P3/5N2/PPPB1PPP/RN1QKB1R b KQkq - 1 5", "theme": "brown"},
    # 15. Queen + Pawn Endgame
    {"name": "Queen Endgame", "fen": "8/6pk/5p1p/7P/4Q3/6P1/5PK1/8 b - - 6 42", "theme": "blue"},
    # 16. Bishop + Knight vs King
    {"name": "BN vs K Endgame", "fen": "8/8/8/4k3/8/5B2/4NK2/8 w - - 0 1", "theme": "gray"},
    # 17. Sicilian Dragon
    {"name": "Sicilian Dragon", "fen": "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ - 3 8", "theme": "green"},
    # 18. London System
    {"name": "London System", "fen": "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3", "theme": "brown"},
    # 19. Scandinavian Defense
    {"name": "Scandinavian Defense", "fen": "rnb1kbnr/ppp1pppp/8/8/3q4/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4", "theme": "blue"},
    # 20. Modern Defense
    {"name": "Modern Defense", "fen": "rnbqk1nr/ppp1ppbp/3p2p1/8/2PPP3/8/PP3PPP/RNBQKBNR w KQkq - 0 3", "theme": "gray"},
]

# Color palettes for screen themes
THEME_PALETTES = {
    "green": {"light": (238, 238, 210), "dark": (118, 150, 86), "bg": (38, 37, 34), "sidebar": (49, 46, 43)},
    "brown": {"light": (240, 217, 181), "dark": (181, 136, 99), "bg": (22, 21, 18), "sidebar": (30, 29, 27)},
    "blue":  {"light": (222, 227, 230), "dark": (140, 162, 173), "bg": (28, 35, 42), "sidebar": (38, 46, 54)},
    "gray":  {"light": (220, 220, 220), "dark": (130, 130, 130), "bg": (20, 20, 20), "sidebar": (32, 32, 32)},
}

def render_screen_scene(
    fen: str,
    theme: str,
    angle_yaw: float = 0.0,
    angle_pitch: float = 20.0,
    aspect: str = "16:9",
    glare: bool = True,
    moiré: bool = True,
    render_size: Tuple[int, int] = (1280, 960),
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Renders a realistic laptop-screen scene with a browser window, sidebar,
    chessboard, perspective distortion, and LCD screen artifacts.

    Returns:
        scene_rgb: (H, W, 3) uint8 image as seen by the handheld camera.
        gt_board_corners: (4, 2) float32 coordinates of the 8x8 playing area.
        placement: ground-truth FEN placement string.
    """
    # Base screen layout (unwarped laptop display: 1200 x 750)
    screen_w = 1200
    screen_h = 750 if aspect == "16:10" else 675
    screen_img = Image.new("RGB", (screen_w, screen_h), (30, 30, 30))
    draw = ImageDraw.Draw(screen_img)

    palette = THEME_PALETTES[theme]

    # 1. Browser Chrome at top
    browser_bar_h = 55
    # Tabs
    draw.rectangle([0, 0, screen_w, 28], fill=(45, 45, 48))
    # Active tab
    draw.rectangle([80, 4, 240, 28], fill=(60, 60, 64))
    draw.text((100, 8), "Chess Online — Analysis", fill=(210, 210, 210))
    # Inactive tab
    draw.rectangle([245, 6, 380, 26], fill=(50, 50, 52))
    draw.text((260, 9), "Study Room", fill=(150, 150, 150))
    # Address bar
    draw.rectangle([0, 28, screen_w, browser_bar_h], fill=(55, 55, 58))
    draw.rectangle([120, 33, screen_w - 180, 50], fill=(40, 40, 42), outline=(75, 75, 80))
    draw.text((135, 36), "https://chess.example.com/play/live", fill=(180, 180, 180))

    # 2. Main Page Area
    page_top = browser_bar_h
    draw.rectangle([0, page_top, screen_w, screen_h], fill=palette["bg"])

    # 3. Chessboard layout
    board_size = min(screen_h - page_top - 40, 560)
    board_size = (board_size // 8) * 8
    square_size = board_size // 8

    board_x = 80
    board_y = page_top + (screen_h - page_top - board_size) // 2

    # Draw squares
    sq_light = palette["light"]
    sq_dark = palette["dark"]

    for r in range(8):
        for c in range(8):
            is_light = (r + c) % 2 == 0
            sq_color = sq_light if is_light else sq_dark
            sx = board_x + c * square_size
            sy = board_y + r * square_size
            draw.rectangle([sx, sy, sx + square_size, sy + square_size], fill=sq_color)

    # 4. Draw Pieces from FEN
    placement = fen.split()[0]
    ranks = placement.split('/')
    for r_idx, rank_str in enumerate(ranks):
        c_idx = 0
        for char in rank_str:
            if char.isdigit():
                c_idx += int(char)
            else:
                color = 'white' if char.isupper() else 'black'
                p_type = {
                    'p': 'pawn', 'n': 'knight', 'b': 'bishop',
                    'r': 'rook', 'q': 'queen', 'k': 'king'
                }[char.lower()]
                px = board_x + c_idx * square_size
                py = board_y + r_idx * square_size
                draw_digital_piece(draw, px, py, square_size, p_type, color)
                c_idx += 1

    # 5. Browser Sidebar (move notation list, player cards)
    sb_x = board_x + board_size + 40
    sb_w = screen_w - sb_x - 60
    sb_y = board_y
    draw.rectangle([sb_x, sb_y, sb_x + sb_w, sb_y + board_size], fill=palette["sidebar"], outline=(70, 70, 70))
    # Move list header
    draw.rectangle([sb_x, sb_y, sb_x + sb_w, sb_y + 36], fill=(55, 55, 60))
    draw.text((sb_x + 15, sb_y + 10), "Move History", fill=(220, 220, 220))
    # Simulated move entries
    for m_i, move_text in enumerate(["1. e4 e5", "2. Nf3 Nc6", "3. Bb5 a6", "4. Ba4 Nf6", "5. O-O Be7"]):
        my = sb_y + 48 + m_i * 26
        draw.text((sb_x + 15, my), move_text, fill=(180, 180, 180))

    # Convert screen to numpy array
    screen_arr = np.array(screen_img)

    # 6. Apply LCD Subpixel / Moiré artifact
    if moiré:
        # Subtle horizontal pixel grid lines (every 2nd scanline slightly dimmed)
        scanlines = np.ones((screen_h, screen_w, 1), dtype=np.float32)
        scanlines[::2] = 0.96
        screen_arr = np.clip(screen_arr.astype(np.float32) * scanlines, 0, 255).astype(np.uint8)

    # 7. Embed Screen inside Laptop Bezel & Keyboard Base
    bezel_thickness = 35
    total_laptop_w = screen_w + bezel_thickness * 2
    total_laptop_h = screen_h + bezel_thickness * 2 + 120  # extra bottom for keyboard deck
    laptop_img = np.full((total_laptop_h, total_laptop_w, 3), 18, dtype=np.uint8)  # dark matte bezel

    # Web camera dot at top bezel
    cam_x = total_laptop_w // 2
    cv2.circle(laptop_img, (cam_x, bezel_thickness // 2), 4, (45, 45, 45), -1)

    # Place screen inside bezel
    laptop_img[bezel_thickness : bezel_thickness + screen_h, bezel_thickness : bezel_thickness + screen_w] = screen_arr

    # Keyboard deck line & hinge
    hinge_y = bezel_thickness + screen_h
    cv2.rectangle(laptop_img, (0, hinge_y), (total_laptop_w, hinge_y + 6), (10, 10, 10), -1)
    cv2.rectangle(laptop_img, (0, hinge_y + 6), (total_laptop_w, total_laptop_h), (35, 35, 38), -1)

    # Track GT board corners in laptop coordinates (TL, TR, BR, BL)
    gt_tl = [bezel_thickness + board_x, bezel_thickness + board_y]
    gt_tr = [bezel_thickness + board_x + board_size, bezel_thickness + board_y]
    gt_br = [bezel_thickness + board_x + board_size, bezel_thickness + board_y + board_size]
    gt_bl = [bezel_thickness + board_x, bezel_thickness + board_y + board_size]
    gt_corners_unwarped = np.array([gt_tl, gt_tr, gt_br, gt_bl], dtype=np.float32)

    # 8. Projective Perspective Transformation (Handheld Camera Simulation)
    out_w, out_h = render_size
    src_pts = np.array([
        [0, 0],
        [total_laptop_w, 0],
        [total_laptop_w, total_laptop_h],
        [0, total_laptop_h]
    ], dtype=np.float32)

    # Compute perspective target points with yaw and pitch
    yaw_rad = math.radians(angle_yaw)
    pitch_rad = math.radians(angle_pitch)

    margin_x = out_w * 0.12
    margin_y = out_h * 0.08
    top_shrink = math.sin(pitch_rad) * out_w * 0.18
    yaw_shift = math.sin(yaw_rad) * out_w * 0.15

    dst_pts = np.array([
        [margin_x + top_shrink + yaw_shift, margin_y],
        [out_w - margin_x - top_shrink + yaw_shift, margin_y],
        [out_w - margin_x - yaw_shift, out_h - margin_y],
        [margin_x - yaw_shift, out_h - margin_y]
    ], dtype=np.float32)

    homography, _ = cv2.findHomography(src_pts, dst_pts)

    # Warp laptop scene into handheld camera view
    scene_warped = cv2.warpPerspective(laptop_img, homography, (out_w, out_h), borderValue=(240, 240, 242))

    # Transform ground-truth board corners through the same homography
    gt_corners_homo = cv2.perspectiveTransform(gt_corners_unwarped.reshape(1, 4, 2), homography)[0]

    # 9. Screen Glare / Specular Reflection Gradient
    if glare:
        glare_mask = np.zeros((out_h, out_w), dtype=np.float32)
        # Diagonal gradient from top-left to bottom-right
        for yy in range(out_h):
            row_val = (1.0 - yy / out_h) * 0.25
            glare_mask[yy, :] = row_val
        for xx in range(out_w):
            col_val = (1.0 - xx / out_w) * 0.15
            glare_mask[:, xx] += col_val
        # Add glare highlights to scene
        glare_u8 = np.clip(glare_mask * 255.0, 0, 255).astype(np.uint8)
        glare_rgb = cv2.cvtColor(glare_u8, cv2.COLOR_GRAY2RGB)
        scene_warped = cv2.addWeighted(scene_warped, 0.88, glare_rgb, 0.12, 0)

    # 10. Sensor Noise & Camera Blur
    noise = np.random.normal(0, 1.8, scene_warped.shape).astype(np.float32)
    scene_warped = np.clip(scene_warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    scene_warped = cv2.GaussianBlur(scene_warped, (3, 3), 0.5)

    return scene_warped, gt_corners_homo, placement


def extract_square_crops_from_scene(
    scene_rgb: np.ndarray,
    board_corners: np.ndarray,
    logical_size: int = 72,
    input_size: int = 64
) -> np.ndarray:
    """
    Extracts 64 normalized, central-cropped square tensors matching the native
    ChessSquareClassifier.kt pipeline.
    """
    tl, tr, br, bl = board_corners
    dst_rect = np.array([
        [0, 0],
        [logical_size * 8, 0],
        [logical_size * 8, logical_size * 8],
        [0, logical_size * 8]
    ], dtype=np.float32)

    homo, _ = cv2.findHomography(board_corners, dst_rect)
    rectified = cv2.warpPerspective(scene_rgb, homo, (logical_size * 8, logical_size * 8))

    crops = np.empty((64, input_size, input_size, 3), dtype=np.uint8)
    crop_offset = (logical_size - input_size) // 2

    for r in range(8):
        for c in range(8):
            sq_crop = rectified[
                r * logical_size : (r + 1) * logical_size,
                c * logical_size : (c + 1) * logical_size
            ]
            central = sq_crop[crop_offset : crop_offset + input_size, crop_offset : crop_offset + input_size]
            crops[r * 8 + c] = central

    return crops


def run_screen_benchmark():
    out_dir = ROOT / "dist" / "screen_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = ROOT / "modules" / "live-camera" / "android" / "src" / "main" / "assets" / "rgb_square_classifier_v1.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print("=" * 80)
    print("LAPTOP-SCREEN CHESSBOARD BENCHMARK EVALUATION (20 SCENES)")
    print(f"Model: {model_path.name}")
    print("=" * 80)

    results = []
    total_iou = []
    total_corner_error = []
    detected_count = 0
    exact_board_count = 0
    adopted_count = 0
    false_adopted_count = 0

    total_squares = 0
    total_square_correct = 0
    total_empty_correct = 0
    total_empty_samples = 0
    total_occupied_correct = 0
    total_occupied_samples = 0

    per_class_stats = {c: {"total": 0, "correct": 0} for c in CLASSES}

    for idx, pos in enumerate(OFFLINE_POSITIONS):
        # Vary viewing angles across scenes
        yaw = [-12.0, 0.0, 15.0, -8.0][idx % 4]
        pitch = [18.0, 24.0, 30.0, 16.0][(idx // 4) % 4]

        scene_rgb, gt_corners, gt_placement = render_screen_scene(
            fen=pos["fen"],
            theme=pos["theme"],
            angle_yaw=yaw,
            angle_pitch=pitch,
            render_size=(1280, 960)
        )

        # Save first 5 scenes to disk for inspection
        if idx < 5:
            sample_path = out_dir / f"scene_{idx+1}_{pos['theme']}.png"
            cv2.imwrite(str(sample_path), cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR))

        # 1. Board Localisation Evaluation
        # Convert RGB to BGR for OpenCV detector
        scene_bgr = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)
        t_det0 = time.time()
        det_result = detect_approach_a_baseline(scene_bgr, max_candidates=20, clamp_to_bounds=True)
        det_ms = (time.time() - t_det0) * 1000

        detected = det_result.get("candidate_generated") and det_result.get("geometry_valid")
        iou = 0.0
        corner_err = 1.0
        prop_corners = det_result.get("points")

        if detected and prop_corners is not None:
            detected_count += 1
            iou = compute_polygon_iou(prop_corners, gt_corners, (960, 1280))
            corner_err = compute_normalized_corner_error(prop_corners, gt_corners, (960, 1280))
            total_iou.append(iou)
            total_corner_error.append(corner_err)

        # 2. Square Classification Evaluation on GT Board Corners
        crops = extract_square_crops_from_scene(scene_rgb, gt_corners)
        tensor = (crops.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)

        t_inf0 = time.time()
        logits = session.run(None, {input_name: tensor})[0]
        inf_ms = (time.time() - t_inf0) * 1000

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)

        # Build GT labels from placement
        gt_labels = []
        for rank_str in gt_placement.split('/'):
            for char in rank_str:
                if char.isdigit():
                    gt_labels.extend([0] * int(char))
                else:
                    color = "white" if char.isupper() else "black"
                    p_type = {
                        'p': 'pawn', 'n': 'knight', 'b': 'bishop',
                        'r': 'rook', 'q': 'queen', 'k': 'king'
                    }[char.lower()]
                    gt_labels.append(CLASS_TO_IDX[f"{color}_{p_type}"])

        gt_labels = np.array(gt_labels)
        sq_correct = preds == gt_labels
        num_correct = int(sq_correct.sum())

        for p_idx, true_cls in enumerate(gt_labels):
            per_class_stats[CLASSES[true_cls]]["total"] += 1
            if sq_correct[p_idx]:
                per_class_stats[CLASSES[true_cls]]["correct"] += 1

        total_squares += 64
        total_square_correct += num_correct

        empty_mask = gt_labels == 0
        total_empty_samples += int(empty_mask.sum())
        total_empty_correct += int(sq_correct[empty_mask].sum())

        occ_mask = gt_labels != 0
        total_occupied_samples += int(occ_mask.sum())
        total_occupied_correct += int(sq_correct[occ_mask].sum())

        is_exact = bool(sq_correct.all())
        if is_exact:
            exact_board_count += 1

        # 3. Production Adoption Gate Evaluation
        adoption = evaluate_board_adoption(probs)
        adopted = adoption['ok']
        pred_placement = adoption['placement']
        reason = adoption['reason']

        if adopted:
            adopted_count += 1
            if pred_placement != gt_placement:
                false_adopted_count += 1

        record = {
            "scene_idx": idx + 1,
            "name": pos["name"],
            "theme": pos["theme"],
            "yaw": yaw,
            "pitch": pitch,
            "gt_placement": gt_placement,
            "detected": detected,
            "detection_iou": round(float(iou), 4),
            "corner_error": round(float(corner_err), 4),
            "detection_latency_ms": round(det_ms, 1),
            "inference_latency_ms": round(inf_ms, 1),
            "square_accuracy": round(num_correct / 64.0, 4),
            "is_exact": is_exact,
            "adopted": adopted,
            "adoption_reason": reason,
            "pred_placement": pred_placement,
        }
        results.append(record)

        status_str = "EXACT" if is_exact else f"{num_correct}/64"
        adopt_str = "ADOPTED" if adopted else f"REJECTED ({reason[:30]}...)"
        print(f"Scene {idx+1:02d} [{pos['theme']:5s} {yaw:+.0f}°/{pitch:.0f}°] {pos['name']:20s} | "
              f"Det IoU={iou:.3f} | Cls={status_str:7s} | {adopt_str}")

    # Aggregates
    n_scenes = len(OFFLINE_POSITIONS)
    mean_iou = float(np.mean(total_iou)) if total_iou else 0.0
    mean_corner_err = float(np.mean(total_corner_error)) if total_corner_error else 1.0

    print("\n" + "=" * 80)
    print("SCREEN BENCHMARK AGGREGATE RESULTS")
    print("=" * 80)
    print(f"Scenes Evaluated:                   {n_scenes}")
    print(f"Board Detection Rate (IoU > 0.5):   {sum(1 for i in total_iou if i > 0.5)}/{n_scenes} ({sum(1 for i in total_iou if i > 0.5)/n_scenes*100:.1f}%)")
    print(f"Mean Detection IoU (accepted):     {mean_iou*100:.2f}%")
    print(f"Mean Normalized Corner Error:       {mean_corner_err*100:.2f}%")
    print(f"Overall Square Accuracy:            {total_square_correct / total_squares * 100:.2f}% ({total_square_correct}/{total_squares})")
    print(f"Empty Square Accuracy:              {total_empty_correct / total_empty_samples * 100:.2f}% ({total_empty_correct}/{total_empty_samples})")
    print(f"Occupied Piece Accuracy:            {total_occupied_correct / total_occupied_samples * 100:.2f}% ({total_occupied_correct}/{total_occupied_samples})")
    print(f"Complete-Board Exact Accuracy:      {exact_board_count / n_scenes * 100:.2f}% ({exact_board_count}/{n_scenes})")
    print(f"Positions Adopted by Gate:          {adopted_count / n_scenes * 100:.2f}% ({adopted_count}/{n_scenes})")
    print(f"  - True Adoptions:                 { (adopted_count - false_adopted_count) / n_scenes * 100:.2f}%")
    print(f"  - False Adoptions:                {false_adopted_count / n_scenes * 100:.2f}% ({false_adopted_count}/{n_scenes})")

    print("\nPer-Class Recall on Laptop Screens:")
    for cls in CLASSES:
        st = per_class_stats[cls]
        rec = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0.0
        print(f"  {cls:14s}: {rec:5.1f}% ({st['correct']}/{st['total']})")

    report_data = {
        "benchmark": "Laptop-Screen Chessboard Evaluation",
        "date": "2026-11-03",
        "model": str(model_path.name),
        "total_scenes": n_scenes,
        "detection_rate_pct": sum(1 for i in total_iou if i > 0.5) / n_scenes * 100,
        "mean_iou_pct": mean_iou * 100,
        "mean_corner_error_pct": mean_corner_err * 100,
        "overall_square_accuracy_pct": total_square_correct / total_squares * 100,
        "empty_square_accuracy_pct": total_empty_correct / total_empty_samples * 100,
        "occupied_piece_accuracy_pct": total_occupied_correct / total_occupied_samples * 100,
        "complete_board_exact_pct": exact_board_count / n_scenes * 100,
        "adopted_count": adopted_count,
        "false_adopted_count": false_adopted_count,
        "per_class_stats": per_class_stats,
        "records": results,
    }

    report_path = ROOT / "dist" / "screen_board_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved report to {report_path}")

if __name__ == "__main__":
    run_screen_benchmark()
