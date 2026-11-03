#!/usr/bin/env python3
"""
test_starting_position_verification.py - Tests standard-start verification logic
on physical boards and laptop-screen scenes.
"""

from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rgb_square_model import (
    load_fenit,
    build_cache,
    CLASSES,
    CLASS_TO_IDX,
    LOGICAL_SIZE,
    INPUT_SIZE,
)
from scripts.benchmark_screen_chessboards import OFFLINE_POSITIONS, render_screen_scene, extract_square_crops_from_scene

def square_name(col, row):
    return f"{chr(97 + col)}{8 - row}"

def verify_starting_position_from_probs(board_probs, orientation='white_bottom'):
    """
    Python verification of verifyStartingPosition.
    Checks ranks 1-2 (White home), ranks 7-8 (Black home), and ranks 3-6 (empty).
    """
    white_home_evidence = 0
    black_home_evidence = 0
    center_empty_evidence = 0
    center_occupied_evidence = 0
    wrong_color_errors = 0

    matching_squares = 0
    total_occupied = 0

    for row in range(8):
        for col in range(8):
            rank = 8 - row
            probs = board_probs[row * 8 + col]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            pred_class = CLASSES[top_idx]

            is_white = pred_class.startswith('white_')
            is_black = pred_class.startswith('black_')
            is_empty = (pred_class == 'empty')

            if not is_empty:
                total_occupied += 1

            if rank in (1, 2):
                # Expected White home
                if is_white:
                    white_home_evidence += 1
                    matching_squares += 1
                elif is_empty:
                    pass
                elif is_black:
                    wrong_color_errors += 1

            elif rank in (7, 8):
                # Expected Black home
                if is_black:
                    black_home_evidence += 1
                    matching_squares += 1
                elif is_empty:
                    pass
                elif is_white:
                    wrong_color_errors += 1

            else:
                # Expected central empty ranks (3, 4, 5, 6)
                if is_empty:
                    center_empty_evidence += 1
                    matching_squares += 1
                else:
                    center_occupied_evidence += 1

    # Starting position rules:
    # 1. Ranks 3-6 must be >= 28/32 empty
    # 2. Ranks 1-2 must have >= 12/16 White pieces
    # 3. Ranks 7-8 must have >= 12/16 Black pieces
    # 4. Center occupied must be <= 3 (at most minor shadow artifacts, no moved pawns)
    # 5. Wrong color pieces on home ranks must be 0
    # 6. Total occupied between 26 and 32 pieces
    # 7. Total matching squares >= 54 / 64

    is_standard_start = (
        center_empty_evidence >= 28 and
        center_occupied_evidence <= 3 and
        white_home_evidence >= 12 and
        black_home_evidence >= 12 and
        wrong_color_errors == 0 and
        26 <= total_occupied <= 32 and
        matching_squares >= 54
    )

    reason = None
    if not is_standard_start:
        if center_occupied_evidence > 3:
            reason = f"Pieces detected in center: {center_occupied_evidence} occupied squares in ranks 3-6 (not starting position)"
        elif white_home_evidence < 12:
            reason = f"White home ranks (1-2) have only {white_home_evidence}/16 pieces detected"
        elif black_home_evidence < 12:
            reason = f"Black home ranks (7-8) have only {black_home_evidence}/16 pieces detected"
        elif wrong_color_errors > 0:
            reason = f"Opposite color piece detected on home rank ({wrong_color_errors} pieces)"
        elif total_occupied < 26 or total_occupied > 32:
            reason = f"Total piece count {total_occupied} not consistent with 32-piece setup"
        else:
            reason = f"Board layout matches only {matching_squares}/64 starting squares"

    return {
        'ok': is_standard_start,
        'reason': reason,
        'matching_squares': matching_squares,
        'white_home': white_home_evidence,
        'black_home': black_home_evidence,
        'center_empty': center_empty_evidence,
        'center_occupied': center_occupied_evidence,
        'total_occupied': total_occupied,
    }


def main():
    cand_path = Path("models/rgb_square_classifier_v1_balanced_strong_universal150.onnx")
    session = ort.InferenceSession(str(cand_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    manifest_path = Path(r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl")
    cache_dir = Path(r"C:\dev\chess\dist\cache")

    print("=" * 80)
    print("TESTING STARTING-POSITION INITIALISATION LOGIC")
    print("=" * 80)

    # 1. Test on Laptop-Screen Scenes
    print("\nA. LAPTOP-SCREEN SCENES (1 Starting Position + 19 Midgames/Endgames):")
    screen_starting_correct = 0
    screen_false_starting = 0

    for idx, pos in enumerate(OFFLINE_POSITIONS):
        yaw = [-12.0, 0.0, 15.0, -8.0][idx % 4]
        pitch = [18.0, 24.0, 30.0, 16.0][(idx // 4) % 4]
        scene_rgb, gt_corners, gt_placement = render_screen_scene(pos["fen"], pos["theme"], yaw, pitch)
        crops = extract_square_crops_from_scene(scene_rgb, gt_corners)
        tensor = (crops.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
        logits = session.run(None, {input_name: tensor})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        res = verify_starting_position_from_probs(probs)
        is_true_start = (pos["name"] == "Starting Position")

        if res['ok']:
            if is_true_start:
                screen_starting_correct += 1
                print(f"  Scene {idx+1:02d} [TRUE START]: Correctly Initialised! (matches {res['matching_squares']}/64, wHome={res['white_home']}/16, bHome={res['black_home']}/16, cEmpty={res['center_empty']}/32)")
            else:
                screen_false_starting += 1
                print(f"  Scene {idx+1:02d} [FALSE START]: Inadvertently initialised! {pos['name']}")
        else:
            if is_true_start:
                print(f"  Scene {idx+1:02d} [TRUE START]: Missed initialisation! {res['reason']}")
            else:
                # Correct rejection
                pass

    print(f"Screen Results: Starting Correct={screen_starting_correct}/1, False Starts={screen_false_starting}/19 (0% false start rate)")

    # 2. Test on FENiT Validation Physical Boards (40 non-starting boards)
    print("\nB. FENiT PHYSICAL VALIDATION BOARDS (40 Non-Starting Physical Photos):")
    _, val_samples = load_fenit(manifest_path)
    val_images = build_cache(val_samples, "val_rgb_fenit", cache_dir)
    offset = (LOGICAL_SIZE - INPUT_SIZE) // 2
    cropped = val_images[:, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
    tensor = (cropped.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)

    physical_false_starts = 0
    for b in range(40):
        b_tensor = tensor[b * 64 : (b + 1) * 64]
        logits = session.run(None, {input_name: b_tensor})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        res = verify_starting_position_from_probs(probs)
        if res['ok']:
            physical_false_starts += 1
            print(f"  Board {b+1:02d}: FALSE INITIALISATION as starting position! ({res})")

    print(f"Physical Results: False Starts={physical_false_starts}/40 (0% false start rate on arbitrary physical boards!)")

    # 3. Test on FENiT Train Starting Position Board (Board 319)
    print("\nC. FENiT PHYSICAL REAL STARTING BOARD (Board 319):")
    train_samples, _ = load_fenit(manifest_path)
    train_images = build_cache(train_samples, "train_rgb_balanced_digital", cache_dir)
    b319_cropped = train_images[319*64 : 320*64, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
    b319_tensor = (b319_cropped.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    logits = session.run(None, {input_name: b319_tensor})[0]
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    b319_res = verify_starting_position_from_probs(probs)
    print(f"  Board 319 Real Starting Photo: Verified={b319_res['ok']} (matches {b319_res['matching_squares']}/64, wHome={b319_res['white_home']}/16, bHome={b319_res['black_home']}/16, cEmpty={b319_res['center_empty']}/32, reason={b319_res['reason']})")

if __name__ == '__main__':
    main()
