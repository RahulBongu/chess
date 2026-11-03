import json
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort

from scripts.train_rgb_square_model import (
    load_fenit,
    build_cache,
    CLASSES,
    CLASS_TO_IDX,
    LOGICAL_SIZE,
    INPUT_SIZE,
)

PIECE_CLASSES = CLASSES
FEN_CHAR = {
    'white_pawn': 'P', 'white_knight': 'N', 'white_bishop': 'B',
    'white_rook': 'R', 'white_queen': 'Q', 'white_king': 'K',
    'black_pawn': 'p', 'black_knight': 'n', 'black_bishop': 'b',
    'black_rook': 'r', 'black_queen': 'q', 'black_king': 'k',
    'empty': None,
}

def square_name(col, row):
    return f"{chr(97 + col)}{8 - row}"

def evaluate_board_adoption(
    board_probs,  # 64 x 13 float array in row-major order (row 0=rank 8, row 7=rank 1)
    min_piece_prob=0.34,
    min_piece_margin=0.06,
    max_uncertain=3,
    turn='w'
):
    """
    Exact Python reproduction of recognizePositionFromObservations from src/vision/pieceRecognition.ts.
    """
    board = [[None for _ in range(8)] for _ in range(8)]
    uncertain_squares = []
    occupied = 0
    confidence_sum = 0.0
    white_king_conf = 0.0
    black_king_conf = 0.0

    for row in range(8):
        for col in range(8):
            sq = square_name(col, row)
            probs = board_probs[row * 8 + col]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            sorted_probs = np.sort(probs)
            second_prob = float(sorted_probs[-2])
            piece_class = PIECE_CLASSES[top_idx]

            occupied_candidate = (
                piece_class != 'empty' and
                top_prob >= min_piece_prob and
                (top_prob - second_prob) >= min_piece_margin
            )

            if piece_class != 'empty' and not occupied_candidate:
                uncertain_squares.append(sq)
                board[row][col] = 'empty'
            else:
                board[row][col] = piece_class

            if occupied_candidate:
                occupied += 1
                confidence_sum += top_prob
                if piece_class == 'white_king':
                    white_king_conf = top_prob
                elif piece_class == 'black_king':
                    black_king_conf = top_prob

    if len(uncertain_squares) > max_uncertain:
        return {
            'ok': False,
            'reason': f"{len(uncertain_squares)} squares are too uncertain ({', '.join(uncertain_squares[:5])})",
            'placement': '',
            'occupied': occupied,
            'uncertain': uncertain_squares,
        }

    if white_king_conf <= 0 or black_king_conf <= 0:
        return {
            'ok': False,
            'reason': f"Missing king confidence (wK={white_king_conf:.2f}, bK={black_king_conf:.2f})",
            'placement': '',
            'occupied': occupied,
            'uncertain': uncertain_squares,
        }

    # Counts check
    counts = {}
    for r in range(8):
        for c in range(8):
            cls = board[r][c]
            if cls and cls != 'empty':
                counts[cls] = counts.get(cls, 0) + 1

    if occupied < 2 or occupied > 32:
        return {
            'ok': False,
            'reason': f"Implausible occupied count: {occupied}",
            'placement': '',
            'occupied': occupied,
            'uncertain': uncertain_squares,
        }

    for color in ('white', 'black'):
        color_count = sum(v for k, v in counts.items() if k.startswith(color))
        pawn_count = counts.get(f"{color}_pawn", 0)
        king_count = counts.get(f"{color}_king", 0)
        if color_count > 16:
            return {'ok': False, 'reason': f"{color} has {color_count} pieces (max 16)", 'placement': '', 'occupied': occupied, 'uncertain': uncertain_squares}
        if pawn_count > 8:
            return {'ok': False, 'reason': f"{color} has {pawn_count} pawns (max 8)", 'placement': '', 'occupied': occupied, 'uncertain': uncertain_squares}
        if king_count != 1:
            return {'ok': False, 'reason': f"{color} has {king_count} kings (expected 1)", 'placement': '', 'occupied': occupied, 'uncertain': uncertain_squares}

    # Pawns on rank 1 or 8
    for col in range(8):
        for row, rank in ((0, 8), (7, 1)):
            cls = board[row][col]
            if cls and cls != 'empty' and cls.endswith('_pawn'):
                return {'ok': False, 'reason': f"Pawn on rank {rank}", 'placement': '', 'occupied': occupied, 'uncertain': uncertain_squares}

    # Placement FEN
    placement_rows = []
    for r in range(8):
        row_str = ''
        empty_run = 0
        for c in range(8):
            cls = board[r][c]
            if not cls or cls == 'empty':
                empty_run += 1
            else:
                if empty_run > 0:
                    row_str += str(empty_run)
                    empty_run = 0
                row_str += FEN_CHAR[cls]
        if empty_run > 0:
            row_str += str(empty_run)
        placement_rows.append(row_str)
    placement = '/'.join(placement_rows)

    return {
        'ok': True,
        'reason': None,
        'placement': placement,
        'occupied': occupied,
        'uncertain': uncertain_squares,
        'average_confidence': confidence_sum / occupied if occupied > 0 else 0.0,
    }

def main():
    manifest_path = Path(r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl")
    cache_dir = Path(r"C:\dev\chess\dist\cache")
    model_path = Path(r"C:\dev\chess\modules\live-camera\android\src\main\assets\rgb_square_classifier_v1.onnx")

    print(f"Loading FENiT validation from {manifest_path}...")
    _, val_samples = load_fenit(manifest_path)
    print(f"Loaded {len(val_samples)} validation samples (40 boards).")

    print(f"Loading images into cache...")
    val_images = build_cache(val_samples, "val_rgb_fenit", cache_dir)
    val_labels = np.array([s.label for s in val_samples])
    val_groups = [s.group for s in val_samples]

    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Center crop 64x64 from 72x72 cache
    offset = (LOGICAL_SIZE - INPUT_SIZE) // 2
    cropped = val_images[:, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
    tensor = (cropped.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)

    # Run inference in batches of 64
    all_probs = []
    t0 = time.time()
    for start in range(0, len(tensor), 64):
        batch = tensor[start : start + 64]
        logits = session.run(None, {input_name: batch})[0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        all_probs.append(probs)
    total_time = time.time() - t0
    all_probs = np.vstack(all_probs)
    predictions = np.argmax(all_probs, axis=1)

    # 1. Standalone square accuracy
    correct = predictions == val_labels
    occupied_mask = val_labels != 0
    empty_mask = val_labels == 0

    print("\n" + "=" * 65)
    print("STANDALONE CLASSIFIER ACCURACY (40 FENiT BOARDS, 2560 SQUARES)")
    print("=" * 65)
    print(f"Overall Top-1 Square Accuracy: {correct.mean()*100:.2f}% ({correct.sum()}/{len(correct)})")
    print(f"Empty Square Accuracy:        {correct[empty_mask].mean()*100:.2f}% ({correct[empty_mask].sum()}/{empty_mask.sum()})")
    print(f"Occupied Piece Accuracy:      {correct[occupied_mask].mean()*100:.2f}% ({correct[occupied_mask].sum()}/{occupied_mask.sum()})")
    print(f"Inference Latency per Batch-64: {total_time / (len(tensor)//64) * 1000:.1f} ms")

    # 2. Complete board evaluation
    board_count = len(val_samples) // 64
    exact_match_count = 0
    true_adoptions = 0
    false_adoptions = 0
    true_rejections = 0
    false_rejections = 0

    rejection_reasons = {}
    details = []

    for b in range(board_count):
        b_slice = slice(b * 64, (b + 1) * 64)
        b_correct = correct[b_slice]
        b_labels = val_labels[b_slice]
        b_preds = predictions[b_slice]
        b_probs = all_probs[b_slice]
        gt_placement = val_groups[b * 64]

        is_exact = bool(b_correct.all())
        if is_exact:
            exact_match_count += 1

        adoption = evaluate_board_adoption(b_probs)
        adopted = adoption['ok']
        pred_placement = adoption['placement']
        reason = adoption['reason']

        if adopted:
            placement_match = (pred_placement == gt_placement)
            if placement_match:
                true_adoptions += 1
            else:
                false_adoptions += 1
        else:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            if is_exact:
                false_rejections += 1
            else:
                true_rejections += 1

        details.append({
            'board_idx': b + 1,
            'gt_placement': gt_placement,
            'pred_placement': pred_placement,
            'is_exact': is_exact,
            'adopted': adopted,
            'reason': reason,
            'correct_squares': int(b_correct.sum()),
        })

    print("\n" + "=" * 65)
    print("PRODUCTION ADOPTION GATE EVALUATION (40 BOARDS)")
    print("=" * 65)
    print(f"Total Held-out Boards:               {board_count}")
    print(f"Complete-Board Exact Accuracy:       {exact_match_count / board_count * 100:.2f}% ({exact_match_count}/{board_count})")
    print(f"Positions Adopted by Gate (Coverage): { (true_adoptions + false_adoptions) / board_count * 100:.2f}% ({true_adoptions + false_adoptions}/{board_count})")
    print(f"  - True Adoptions (Exact Matches):   {true_adoptions / board_count * 100:.2f}% ({true_adoptions}/{board_count})")
    print(f"  - FALSE ADOPTIONS (Accepted Wrong): {false_adoptions / board_count * 100:.2f}% ({false_adoptions}/{board_count})")
    print(f"Positions Rejected by Gate (Uncertain): { (true_rejections + false_rejections) / board_count * 100:.2f}% ({true_rejections + false_rejections}/{board_count})")
    print(f"  - True Rejections (Saved from error): {true_rejections / board_count * 100:.2f}% ({true_rejections}/{board_count})")
    print(f"  - False Rejections (Exact rejected): {false_rejections / board_count * 100:.2f}% ({false_rejections}/{board_count})")

    print("\nRejection Reasons Breakdown:")
    for r, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        print(f"  [{count:2d}] {r}")

    # Output detailed report json
    out_report = {
        'model': str(model_path.name),
        'benchmark': 'FENiT validation (40 full-board photos)',
        'boards_evaluated': board_count,
        'exact_match_count': exact_match_count,
        'exact_match_pct': exact_match_count / board_count * 100,
        'coverage_pct': (true_adoptions + false_adoptions) / board_count * 100,
        'true_adoptions': true_adoptions,
        'false_adoptions': false_adoptions,
        'false_adoption_pct': false_adoptions / board_count * 100,
        'true_rejections': true_rejections,
        'false_rejections': false_rejections,
        'square_accuracy_top1': float(correct.mean()),
        'empty_accuracy_top1': float(correct[empty_mask].mean()),
        'occupied_accuracy_top1': float(correct[occupied_mask].mean()),
        'rejection_reasons': rejection_reasons,
        'board_details': details,
    }
    with open("dist/production_adoption_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2)
    print("\nSaved report to dist/production_adoption_benchmark.json")

if __name__ == '__main__':
    main()
