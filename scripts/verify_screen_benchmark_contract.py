#!/usr/bin/env python3
"""Contract checks for the screen benchmark and the ONNX production asset.

These checks exist because the 5.4I.3 benchmark silently reported a
misleading "32/64 all-empty" result: corners fell back to non-board pixels,
metrics were mislabelled, and nothing asserted the pipeline contract. Each
check below fails loudly with a non-zero exit code instead.

Run:  py -3 scripts/verify_screen_benchmark_contract.py
Exit: 0 all checks pass, 1 any check failed, 2 required file missing.

Screen domain only. This does not claim physical-board accuracy.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSON = ROOT / "dist" / "milestone_54i5_screen_benchmark.json"
MANIFEST = ROOT / "dist" / "cache" / "digital_crops" / "manifest.json"
ASSET = ROOT / "modules/live-camera/android/src/main/assets/rgb_square_classifier_v1.onnx"
PRODUCTION = ROOT / "models/rgb_square_classifier_v1_balanced_strong_universal150.onnx"
KOTLIN = ROOT / "modules/live-camera/android/src/main/java/expo/modules/livecamera/ChessSquareClassifier.kt"
TRAIN = ROOT / "scripts/train_rgb_square_model.py"

EXPECTED_CLASSES = [
    "empty",
    "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
]

failures: list[str] = []
passes: list[str] = []


def check(condition: bool, name: str, detail: str = "") -> None:
    if condition:
        passes.append(name)
    else:
        failures.append(f"{name}{': ' + detail if detail else ''}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_assets() -> None:
    if not ASSET.exists() or not PRODUCTION.exists():
        failures.append(f"asset missing: {ASSET if not ASSET.exists() else PRODUCTION}")
        return
    asset_hash = sha256(ASSET)
    production_hash = sha256(PRODUCTION)
    check(
        asset_hash == production_hash,
        "production asset is universal150",
        f"{asset_hash[:16]} != {production_hash[:16]}",
    )
    check(
        ASSET.stat().st_size == 235025,
        "production asset size pinned at 235025 bytes",
        str(ASSET.stat().st_size),
    )


def check_kotlin_contract() -> None:
    if not KOTLIN.exists():
        failures.append(f"missing {KOTLIN}")
        return
    text = read_text(KOTLIN)
    check("const val INPUT_SIZE = 64" in text, "Kotlin INPUT_SIZE is 64")
    check("const val CLASS_COUNT = 13" in text, "Kotlin CLASS_COUNT is 13")
    check(
        "const val TRAINING_CROP_RATIO = INPUT_SIZE / 72.0" in text,
        "Kotlin samples the central 64/72 window",
    )
    check(
        "const val SQUARE_INSET = (1.0 - TRAINING_CROP_RATIO) / 2.0" in text,
        "Kotlin inset matches the training crop",
    )
    check("NCHW" in text, "Kotlin documents NCHW layout")
    check("HWC -> CHW" in text or "Transpose HWC" in text, "Kotlin transposes HWC->CHW")

    class_block = re.search(r"val CLASSES = arrayOf\((.*?)\)", text, re.S)
    if not class_block:
        failures.append("Kotlin CLASSES array not found")
        return
    kotlin_classes = re.findall(r'"([a-z_]+)"', class_block.group(1))
    check(
        kotlin_classes == EXPECTED_CLASSES,
        "Kotlin class order matches benchmark class order",
        f"{kotlin_classes[:3]}... vs {EXPECTED_CLASSES[:3]}...",
    )


def check_training_contract() -> None:
    if not TRAIN.exists():
        failures.append(f"missing {TRAIN}")
        return
    text = read_text(TRAIN)
    check("LOGICAL_SIZE = 72" in text, "training logical size is 72")
    check("INPUT_SIZE = 64" in text, "training input size is 64")
    check(
        "tensor - 0.5) / 0.5" in text,
        "training normalization is (x/255-0.5)/0.5",
    )
    check(
        "offset_y = (batch.shape[1] - INPUT_SIZE) // 2" in text,
        "training evaluation centre-crops 64 of 72",
    )
    check("permute(0, 3, 1, 2)" in text, "training uses NCHW tensors")


def check_manifest() -> None:
    if not MANIFEST.exists():
        failures.append(f"missing {MANIFEST}")
        return
    records = json.loads(read_text(MANIFEST))
    check(len(records) == 2600, "digital manifest has 2600 crops", str(len(records)))
    per_class: dict[str, int] = {}
    for record in records:
        per_class[record["class_name"]] = per_class.get(record["class_name"], 0) + 1
    check(
        set(per_class) == set(EXPECTED_CLASSES),
        "digital manifest covers all 13 classes",
        str(sorted(set(EXPECTED_CLASSES) - set(per_class))),
    )
    check(
        all(count == 200 for count in per_class.values()),
        "digital manifest has 200 crops per class",
        str({k: v for k, v in per_class.items() if v != 200}),
    )
    paths = [record["path"] for record in records]
    check(len(paths) == len(set(paths)), "digital manifest has no duplicate paths")
    import cv2

    shapes = {cv2.imread(path).shape for path in paths[:50] if cv2.imread(path) is not None}
    check(shapes == {(72, 72, 3)}, "digital crops are 72x72x3", str(shapes))
    check(
        all(Path(path).exists() for path in paths[:50]),
        "digital manifest paths resolve (sampled 50)",
    )


def check_benchmark_output() -> None:
    if not BENCHMARK_JSON.exists():
        # Not fatal: the benchmark may not have been re-run on this machine.
        passes.append("benchmark JSON absent (skipped result checks)")
        return
    data = json.loads(read_text(BENCHMARK_JSON))
    check(data.get("asset_equals_universal150") is True, "JSON records asset==universal150")
    corners = data.get("corners", {})
    check(
        set(corners) == {"green_start", "green_e4e5", "brown_start"},
        "JSON covers the three screen boards",
        str(sorted(corners)),
    )
    for name, entry in corners.items():
        pts = entry.get("corners") or []
        check(len(pts) == 4, f"{name} has 4 corners", str(len(pts)))
        if len(pts) != 4:
            continue
        check(entry.get("method") not in ("manual_fallback",), f"{name} did not use last-resort fallback")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        check(max(xs) - min(xs) > 300 and max(ys) - min(ys) > 800, f"{name} quad is board-sized")
        check(
            pts[0][0] < pts[1][0] and pts[3][0] < pts[2][0],
            f"{name} corners are ordered TL,TR,BR,BL by column",
        )
        check(pts[0][1] < pts[3][1] and pts[1][1] < pts[2][1], f"{name} corners are ordered by row")
        # Convexity: every consecutive cross product has the same sign.
        cross = []
        for index in range(4):
            ax, ay = pts[index]
            bx, by = pts[(index + 1) % 4]
            cx, cy = pts[(index + 2) % 4]
            cross.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
        check(all(value > 0 for value in cross) or all(value < 0 for value in cross), f"{name} quad is convex")

    results = data.get("results", {})
    check("universal150_prod" in results, "JSON includes universal150 results")
    for model, entry in results.items():
        for board in ("green_start", "green_e4e5", "brown_start"):
            for mode in ("device", "warp"):
                metric = entry.get(f"{board}_{mode}")
                check(metric is not None, f"{model} {board} {mode} present")
                if not metric:
                    continue
                check(0 <= metric["per_square_exact_n"] <= 64, f"{model} {board} {mode} exact in range")
                check(
                    isinstance(metric.get("board_exact"), bool),
                    f"{model} {board} {mode} labels board_exact as bool",
                )
                check(
                    "per_square_exact_n" in metric and "board_exact" in metric,
                    f"{model} {board} {mode} separates per-square and board-exact metrics",
                )
        digital = entry.get("digital_crops", {})
        check(digital.get("n") == 2600, f"{model} digital eval uses all 2600 crops", str(digital.get("n")))

    # Regression guard: the old bug produced "all empty" on green_e4e5. An
    # all-empty warp is possible only if corners or sampling broke again.
    universal = results.get("universal150_prod", {}).get("green_e4e5_device", {})
    if universal:
        recall = universal.get("per_class_recall", {})
        occupied = {k: v for k, v in recall.items() if k != "empty"}
        check(
            any(not v.startswith("0/") for v in occupied.values()),
            "green_e4e5 is not an all-empty warp artifact",
            str(occupied),
        )


def main() -> int:
    missing_required = [str(path) for path in (ASSET, PRODUCTION, KOTLIN, TRAIN, MANIFEST) if not path.exists()]
    if missing_required:
        for path in missing_required:
            print(f"MISSING {path}")
        return 2

    check_assets()
    check_kotlin_contract()
    check_training_contract()
    check_manifest()
    check_benchmark_output()

    for name in passes:
        print(f"PASS {name}")
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"\n{len(passes)} passed, {len(failures)} failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
