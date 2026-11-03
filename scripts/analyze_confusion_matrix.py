import json
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

def main():
    manifest_path = Path(r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl")
    cache_dir = Path(r"C:\dev\chess\dist\cache")
    model_path = Path(r"C:\dev\chess\modules\live-camera\android\src\main\assets\rgb_square_classifier_v1.onnx")

    _, val_samples = load_fenit(manifest_path)
    val_images = build_cache(val_samples, "val_rgb_fenit", cache_dir)
    val_labels = np.array([s.label for s in val_samples])

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    offset = (LOGICAL_SIZE - INPUT_SIZE) // 2
    cropped = val_images[:, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
    tensor = (cropped.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)

    all_logits = []
    for start in range(0, len(tensor), 64):
        batch = tensor[start : start + 64]
        logits = session.run(None, {input_name: batch})[0]
        all_logits.append(logits)
    all_logits = np.vstack(all_logits)
    predictions = np.argmax(all_logits, axis=1)

    # 13x13 confusion matrix: rows = true label, cols = predicted label
    cm = np.zeros((13, 13), dtype=np.int32)
    for t, p in zip(val_labels, predictions):
        cm[t, p] += 1

    print("=" * 90)
    print("CONFUSION MATRIX ON HELD-OUT FENiT VALIDATION (40 BOARDS, 2560 CROPS)")
    print("=" * 90)
    short_classes = ['emp', 'wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    header = "True \\ Pred  " + "  ".join(f"{c:>4s}" for c in short_classes) + "   Total   Recall"
    print(header)
    print("-" * len(header))

    for idx, name in enumerate(CLASSES):
        row = cm[idx]
        total = row.sum()
        recall = (row[idx] / total * 100) if total > 0 else 0.0
        row_str = f"{short_classes[idx]:>10s}   " + "  ".join(f"{v:4d}" if v > 0 else "   ." for v in row)
        print(f"{row_str}    {total:4d}   {recall:5.1f}%")

    print("\n" + "=" * 90)
    print("TOP CONFUSION PAIRS FOR WEAKEST CLASSES")
    print("=" * 90)
    weak_classes = ['white_knight', 'white_bishop', 'white_rook', 'white_king',
                    'black_knight', 'black_bishop', 'black_rook', 'black_king']
    for cls in weak_classes:
        idx = CLASS_TO_IDX[cls]
        row = cm[idx].copy()
        row[idx] = 0  # ignore correct
        top_err_indices = np.argsort(-row)[:3]
        total = cm[idx].sum()
        print(f"\n{cls} (N={total}, correct={cm[idx, idx]}):")
        for err_idx in top_err_indices:
            count = row[err_idx]
            if count > 0:
                print(f"  -> misclassified as {CLASSES[err_idx]:14s}: {count:2d} times ({count/total*100:.1f}%)")

if __name__ == '__main__':
    main()
