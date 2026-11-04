#!/usr/bin/env python3
"""
test_starting_position_verification.py - Strict standard-start verification tests.

Mirrors the TypeScript verifier exactly: every home-rank square must be occupied
by the correct colour, all 32 central squares must be empty, and the board must
contain exactly 32 pieces. Near-start positions after one or two plies must be
rejected. A verified layout is still only a candidate; the app never adopts
STARTING_FEN without explicit user confirmation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rgb_square_model import (  # noqa: E402
    load_fenit,
    build_cache,
    CLASSES,
    LOGICAL_SIZE,
    INPUT_SIZE,
)
from scripts.benchmark_screen_chessboards import (  # noqa: E402
    OFFLINE_POSITIONS,
    render_screen_scene,
    extract_square_crops_from_scene,
)

# All 20 legal first moves. Generated from chess.js in the TypeScript test.
FIRST_MOVES = [
    ("a3", "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1"),
    ("a4", "rnbqkbnr/pppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq - 0 1"),
    ("b3", "rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1"),
    ("b4", "rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1"),
    ("c3", "rnbqkbnr/pppppppp/8/8/8/2P5/PP1PPPPP/RNBQKBNR b KQkq - 0 1"),
    ("c4", "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1"),
    ("d3", "rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1"),
    ("d4", "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"),
    ("e3", "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
    ("e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
    ("f3", "rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1"),
    ("f4", "rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 0 1"),
    ("g3", "rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1"),
    ("g4", "rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq - 0 1"),
    ("h3", "rnbqkbnr/pppppppp/8/8/8/7P/PPPPPPP1/RNBQKBNR b KQkq - 0 1"),
    ("h4", "rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR b KQkq - 0 1"),
    ("Na3", "rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR b KQkq - 1 1"),
    ("Nc3", "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1"),
    ("Nf3", "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"),
    ("Nh3", "rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1"),
]

TWO_PLY_LINES = [
    ("1.e4 e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("1.e4 c5", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("1.d4 d5", "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"),
    ("1.Nf3 Nf6", "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"),
    ("1.e4 d5", "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("1.e4 d5 2.exd5", "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"),
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_starting_position_from_probs(board_probs: np.ndarray) -> dict:
    """Strict Python mirror of verifyStartingPosition in pieceRecognition.ts."""
    white_home = 0
    black_home = 0
    center_empty = 0
    center_occupied = 0
    wrong_color_errors = 0
    total_occupied = 0
    matching_squares = 0
    occupied_center_squares = []
    missing_white_home = []
    missing_black_home = []

    for row in range(8):
        for col in range(8):
            rank = 8 - row
            square = f"{chr(97 + col)}{rank}"
            pred_class = CLASSES[int(np.argmax(board_probs[row * 8 + col]))]
            is_white = pred_class.startswith("white_")
            is_black = pred_class.startswith("black_")
            is_empty = pred_class == "empty"

            if not is_empty:
                total_occupied += 1

            if rank in (1, 2):
                if is_white:
                    white_home += 1
                    matching_squares += 1
                else:
                    missing_white_home.append(square)
                    if is_black:
                        wrong_color_errors += 1
            elif rank in (7, 8):
                if is_black:
                    black_home += 1
                    matching_squares += 1
                else:
                    missing_black_home.append(square)
                    if is_white:
                        wrong_color_errors += 1
            else:
                if is_empty:
                    center_empty += 1
                    matching_squares += 1
                else:
                    center_occupied += 1
                    occupied_center_squares.append(square)

    ok = (
        center_occupied == 0
        and white_home == 16
        and black_home == 16
        and wrong_color_errors == 0
        and total_occupied == 32
        and matching_squares == 64
    )

    reason = None
    if not ok:
        if center_occupied > 0:
            reason = (
                f"Piece detected in center ({', '.join(occupied_center_squares)}); "
                "not an untouched starting position."
            )
        elif wrong_color_errors > 0:
            reason = (
                f"Inverted piece color detected on home rank ({wrong_color_errors} pieces). "
                "Check White/Black bottom toggle."
            )
        elif missing_white_home:
            reason = (
                f"White home square {', '.join(missing_white_home)} is not occupied by White."
            )
        elif missing_black_home:
            reason = (
                f"Black home square {', '.join(missing_black_home)} is not occupied by Black."
            )
        elif total_occupied != 32:
            reason = f"Detected piece count is {total_occupied}/32."
        else:
            reason = f"Board layout matches {matching_squares}/64 starting squares."

    return {
        "ok": ok,
        "reason": reason,
        "matching_squares": matching_squares,
        "white_home": white_home,
        "black_home": black_home,
        "center_empty": center_empty,
        "center_occupied": center_occupied,
        "occupied_center_squares": occupied_center_squares,
        "total_occupied": total_occupied,
    }


def classify_scene(session, input_name, fen: str, theme: str, yaw: float, pitch: float):
    scene_rgb, gt_corners, _ = render_screen_scene(fen, theme, yaw, pitch)
    crops = extract_square_crops_from_scene(scene_rgb, gt_corners)
    tensor = (crops.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    logits = session.run(None, {input_name: tensor})[0]
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)


def main() -> None:
    model_path = Path("modules/live-camera/android/src/main/assets/rgb_square_classifier_v1.onnx")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    manifest_path = Path(
        r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl"
    )
    cache_dir = Path(r"C:\dev\chess\dist\cache")

    print("=" * 88)
    print("STRICT STARTING-POSITION VERIFICATION (MILESTONE 5.4I.1)")
    print(f"Model: {model_path} SHA-256 {sha256_of(model_path)}")
    print("=" * 88)

    # A. Laptop-screen benchmark: one true start, 19 non-start boards.
    print("\nA. LAPTOP-SCREEN SCENES (1 true start + 19 non-start):")
    screen_start_ok = 0
    screen_false_start = 0
    for idx, pos in enumerate(OFFLINE_POSITIONS):
        yaw = [-12.0, 0.0, 15.0, -8.0][idx % 4]
        pitch = [18.0, 24.0, 30.0, 16.0][(idx // 4) % 4]
        probs = classify_scene(session, input_name, pos["fen"], pos["theme"], yaw, pitch)
        res = verify_starting_position_from_probs(probs)
        is_true_start = pos["name"] == "Starting Position"
        if res["ok"] and is_true_start:
            screen_start_ok += 1
            print(
                f"  Scene {idx + 1:02d} [TRUE START]: accepted "
                f"(matches {res['matching_squares']}/64, wHome={res['white_home']}/16, "
                f"bHome={res['black_home']}/16, cEmpty={res['center_empty']}/32)"
            )
        elif res["ok"]:
            screen_false_start += 1
            print(f"  Scene {idx + 1:02d} [FALSE START]: accepted {pos['name']}!")
        elif is_true_start:
            print(f"  Scene {idx + 1:02d} [TRUE START]: rejected: {res['reason']}")
    print(
        f"  Result: true starts accepted {screen_start_ok}/1; "
        f"false starts {screen_false_start}/19"
    )

    # B. Every legal first move and representative two-ply lines, rendered as
    # real laptop-screen scenes through the production classifier.
    print("\nB. LAPTOP-SCREEN NEAR-START POSITIONS (rendered, not hand-built):")
    first_move_false = 0
    for idx, (name, fen) in enumerate(FIRST_MOVES):
        theme = ["green", "brown", "blue", "gray"][idx % 4]
        yaw = [-12.0, 0.0, 15.0, -8.0][idx % 4]
        pitch = [18.0, 24.0, 30.0, 16.0][(idx // 4) % 4]
        probs = classify_scene(session, input_name, fen, theme, yaw, pitch)
        res = verify_starting_position_from_probs(probs)
        if res["ok"]:
            first_move_false += 1
            print(f"  {name:4s}: FALSE START ACCEPTED ({res['matching_squares']}/64)")
        else:
            print(f"  {name:4s}: rejected ({res['reason']})")
    print(f"  Result: false starts after one legal opening move {first_move_false}/20")

    line_false = 0
    for idx, (name, fen) in enumerate(TWO_PLY_LINES):
        theme = ["green", "brown", "blue", "gray"][idx % 4]
        probs = classify_scene(session, input_name, fen, theme, 0.0, 24.0)
        res = verify_starting_position_from_probs(probs)
        if res["ok"]:
            line_false += 1
            print(f"  {name:20s}: FALSE START ACCEPTED ({res['matching_squares']}/64)")
        else:
            print(f"  {name:20s}: rejected ({res['reason']})")
    print(f"  Result: false starts after two plies {line_false}/{len(TWO_PLY_LINES)}")

    # C. FENiT physical non-start boards.
    print("\nC. FENiT PHYSICAL VALIDATION BOARDS (40 non-start photos):")
    _, val_samples = load_fenit(manifest_path)
    val_images = build_cache(val_samples, "val_rgb_fenit", cache_dir)
    offset = (LOGICAL_SIZE - INPUT_SIZE) // 2
    cropped = val_images[:, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
    tensor = (cropped.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)

    physical_false = 0
    for b in range(40):
        logits = session.run(None, {input_name: tensor[b * 64 : (b + 1) * 64]})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        res = verify_starting_position_from_probs(probs)
        if res["ok"]:
            physical_false += 1
            print(f"  Board {b + 1:02d}: FALSE START ACCEPTED!")
    print(f"  Result: false starts {physical_false}/40")

    # D. The one FENiT training photo that is a real starting position.
    print("\nD. FENiT PHYSICAL REAL STARTING BOARD (training board 319):")
    train_samples, _ = load_fenit(manifest_path)
    train_images = build_cache(train_samples, "train_rgb_balanced_digital", cache_dir)
    b319 = train_images[319 * 64 : 320 * 64, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
    b319_tensor = (b319.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    logits = session.run(None, {input_name: b319_tensor})[0]
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    res = verify_starting_position_from_probs(probs)
    print(
        f"  Verified={res['ok']} matches={res['matching_squares']}/64 "
        f"wHome={res['white_home']}/16 bHome={res['black_home']}/16 "
        f"cEmpty={res['center_empty']}/32 total={res['total_occupied']}/32"
    )
    if res["reason"]:
        print(f"  Reason: {res['reason']}")
    print("\nNote: a verified layout is only a candidate. The app requires an")
    print("explicit Confirm Start tap before adopting STARTING_FEN.")


if __name__ == "__main__":
    main()
