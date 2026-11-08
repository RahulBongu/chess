#!/usr/bin/env python3
"""Evaluate the Milestone 5.4I.5 vivo screen-capture corpus.

This consumes a manifest produced by `scripts/build_screen_capture_manifest.py`
and evaluates one or more ONNX candidates on the exact same crops. It exists so
the Part B corpus cannot be scored with the wrong split, the wrong sampling
path, or an unlocked holdout.

Contract:
  * Session splits are taken from the manifest; frames are never re-split.
  * The crop path replicates `ChessSquareClassifier.classify` (bilinear sample
    at SQUARE_INSET over the 72->64 training ratio), via the same functions as
    `scripts/benchmark_screen_i5.py`. No corner re-detection.
  * A run mixes only one source (`native-dump` or `screencap`). The protocol
    forbids comparing them numerically.
  * `holdout_screen` is refused unless a freeze manifest lists the SHA-256 of
    every evaluated model and the one-shot use log does not exist. The log is
    written only after a successful evaluation.
  * Metrics are reported per split and per condition. Splits are never pooled.
  * The start-gate replica mirrors `verifyStartingPosition`: uncertainty from
    occupancy/colour probabilities, at most three prior-resolved squares, and
    confident contradictions blocked at p >= 0.8 and margin >= 0.4.

Usage:
  py -3 scripts/evaluate_screen_capture_benchmark.py --selftest
  py -3 scripts/evaluate_screen_capture_benchmark.py \
      --capture-root <dir> --manifest dist/capture_i5/manifest.jsonl \
      --models universal150_prod,balanced150fix,zenodo_long,wide150 \
      --splits val_screen --json-out dist/milestone_54i5_vivo_benchmark.json

Screen crops only. Never mix these numbers with physical 3D-board results.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_screen_i5 as benchmark
from build_screen_capture_manifest import (
    SPLITS,
    sha256_file,
    validate_corners,
    validate_fen,
)

UNCERT_CONF = 0.55
UNCERT_MARGIN = 0.08
START_MAX_UNCERTAIN = 3
CONTRADICTION_PROB = 0.80
CONTRADICTION_MARGIN = 0.40
CALIBRATION_BINS = 10
SOURCE_VALUES = ("native-dump", "screencap")
CONDITION_FIELDS = ("position_id", "theme", "brightness_pct", "distance", "angle_deg", "source")
CLASS_TO_INDEX = {name: index for index, name in enumerate(benchmark.CLASSES)}
WHITE_INDICES = [index for index, name in enumerate(benchmark.CLASSES) if name.startswith("white_")]
BLACK_INDICES = [index for index, name in enumerate(benchmark.CLASSES) if name.startswith("black_")]
EMPTY_INDEX = CLASS_TO_INDEX["empty"]


class FatalError(Exception):
    pass


# ---------------------------------------------------------------------------
# Labels and start gate
# ---------------------------------------------------------------------------

def parse_fen_to_classes(fen: str) -> list[str]:
    """Returns 64 classes row-major for a white-bottom board view."""
    fields = fen.strip().split()
    if len(fields) != 6:
        raise FatalError(f"FEN must have 6 fields: {fen!r}")
    grid: list[str] = []
    for rank in fields[0].split("/"):
        for symbol in rank:
            if symbol.isdigit():
                grid.extend(["empty"] * int(symbol))
                continue
            colour = "white" if symbol.isupper() else "black"
            kind = {
                "p": "pawn",
                "n": "knight",
                "b": "bishop",
                "r": "rook",
                "q": "queen",
                "k": "king",
            }[symbol.lower()]
            grid.append(f"{colour}_{kind}")
    if len(grid) != 64:
        raise FatalError(f"FEN placement expands to {len(grid)} squares: {fen!r}")
    unknown = sorted(set(grid) - set(benchmark.CLASSES))
    if unknown:
        raise FatalError(f"FEN contains unsupported classes: {unknown}")
    return grid


def orient_truth(white_bottom_grid: list[str], orientation: str) -> list[str]:
    """Rotates the white-bottom truth into the orientation the camera sees."""
    if orientation == "white_bottom":
        return list(white_bottom_grid)
    if orientation == "black_bottom":
        return [white_bottom_grid[(7 - row) * 8 + (7 - col)] for row in range(8) for col in range(8)]
    raise FatalError(f"unsupported orientation {orientation!r}")


def square_grid_index(row: int, col: int, orientation: str) -> int:
    """Row-major index of the square visible at (row, col)."""
    del orientation  # the index is defined in the camera grid, not algebraic
    return row * 8 + col


def algebraic_square(row: int, col: int, orientation: str) -> str:
    if orientation == "white_bottom":
        return f"{chr(97 + col)}{8 - row}"
    return f"{chr(97 + (7 - col))}{row + 1}"


START_BY_SQUARE = {
    algebraic_square(row, col, "white_bottom"): benchmark.START[row][col]
    for row in range(8)
    for col in range(8)
}


def top_two(probs: np.ndarray) -> tuple[int, float, float]:
    order = np.argsort(probs)[-2:]
    second, first = int(order[0]), int(order[1])
    return first, float(probs[first]), float(probs[first] - probs[second])


def square_uncertainty(probs: np.ndarray, square: str) -> bool:
    """Mirrors `isStartingSquareUncertain` using summed head probabilities."""
    rank = int(square[1])
    occ = 1.0 - float(probs[EMPTY_INDEX])
    white = float(np.sum(probs[WHITE_INDICES]))
    black = float(np.sum(probs[BLACK_INDICES]))
    if 3 <= rank <= 6:
        return occ > 0.30
    if rank in (1, 2):
        return occ < 0.70 or white < 0.70
    return occ < 0.70 or black < 0.70


def start_gate(probs: np.ndarray, orientation: str) -> dict:
    """Replicates the device start gate, including the bounded prior.

    Diagnostic counts are still reported when the gate fails, which is why a
    board with more than three uncertain squares can show a mix of observed
    values instead of the all-zero `failedStartingVerification` payload.
    """
    uncertain: list[str] = []
    tops: list[tuple[int, float, float, str]] = []
    for row in range(8):
        for col in range(8):
            index = square_grid_index(row, col, orientation)
            square = algebraic_square(row, col, orientation)
            vector = probs[index]
            top, p, margin = top_two(vector)
            tops.append((top, p, margin, square))
            if square_uncertainty(vector, square):
                uncertain.append(square)

    prior_allowed = len(uncertain) <= START_MAX_UNCERTAIN
    contradictions: list[str] = []
    resolved: list[str] = []
    if prior_allowed:
        for top, p, margin, square in tops:
            if square not in uncertain:
                continue
            expected = START_BY_SQUARE[square]
            observed = benchmark.CLASSES[top]
            if observed != expected and p >= CONTRADICTION_PROB and margin >= CONTRADICTION_MARGIN:
                contradictions.append(square)

    white_home = black_home = centre_occupied = wrong_colour = total_occupied = matching = 0
    occupied_centre: list[str] = []
    missing_white: list[str] = []
    missing_black: list[str] = []
    for top, _p, _margin, square in tops:
        rank = int(square[1])
        observed = benchmark.CLASSES[top]
        if prior_allowed and square in uncertain:
            effective = START_BY_SQUARE[square]
            resolved.append(square)
        else:
            effective = observed
        is_white = effective.startswith("white_")
        is_black = effective.startswith("black_")
        is_empty = effective == "empty"
        if not is_empty:
            total_occupied += 1
        if rank in (1, 2):
            if is_white:
                white_home += 1
                matching += 1
            else:
                missing_white.append(square)
                if is_black:
                    wrong_colour += 1
        elif rank in (7, 8):
            if is_black:
                black_home += 1
                matching += 1
            else:
                missing_black.append(square)
                if is_white:
                    wrong_colour += 1
        else:
            if is_empty:
                matching += 1
            else:
                centre_occupied += 1
                occupied_centre.append(square)

    ok = (
        not contradictions
        and len(uncertain) <= START_MAX_UNCERTAIN
        and centre_occupied == 0
        and white_home == 16
        and black_home == 16
        and wrong_colour == 0
        and total_occupied == 32
        and matching == 64
    )
    reason = None
    if contradictions:
        reason = f"Confident contradiction on {contradictions[0]}."
    elif len(uncertain) > START_MAX_UNCERTAIN:
        reason = f"{len(uncertain)} squares too uncertain to verify a starting position."
    elif centre_occupied:
        reason = f"Centre squares occupied: {', '.join(occupied_centre)}."
    elif wrong_colour:
        reason = f"{wrong_colour} inverted-colour pieces on home ranks."
    elif missing_white:
        reason = f"White home square not white: {', '.join(missing_white)}."
    elif missing_black:
        reason = f"Black home square not black: {', '.join(missing_black)}."
    elif total_occupied != 32:
        reason = f"Detected piece count is {total_occupied}/32."
    elif matching != 64:
        reason = f"Board layout matches {matching}/64 starting squares."

    return {
        "ok": ok,
        "reason": reason,
        "uncertain": len(uncertain),
        "uncertain_squares": sorted(uncertain),
        "prior_allowed": prior_allowed,
        "counts_use_prior": prior_allowed,
        "prior_resolved": sorted(resolved),
        "confident_contradictions": sorted(contradictions),
        "white_home": white_home,
        "black_home": black_home,
        "centre_occupied": centre_occupied,
        "wrong_colour_home": wrong_colour,
        "total_occupied": total_occupied,
        "matching_squares": matching,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def square_metrics(probs: np.ndarray, truth: list[str]) -> dict:
    """One frame of 64 square probability vectors against the labelled FEN."""
    true_idx = np.array([CLASS_TO_INDEX[name] for name in truth], dtype=np.int64)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    correct = preds == true_idx
    occupied_true = true_idx != EMPTY_INDEX
    occupied_pred = preds != EMPTY_INDEX
    occupied_correct = int(np.sum(correct & occupied_true))
    occupied_n = int(np.sum(occupied_true))
    empty_correct = int(np.sum(correct & ~occupied_true))
    empty_n = int(np.sum(~occupied_true))
    both_occupied = occupied_true & occupied_pred
    colour_n = int(np.sum(both_occupied))
    colour_correct = int(
        np.sum((true_idx[both_occupied] // 7) == (preds[both_occupied] // 7))
    ) if colour_n else 0
    uncertain = int(np.sum((confs < UNCERT_CONF) | (margins < UNCERT_MARGIN)))
    per_class_true: dict[int, int] = defaultdict(int)
    per_class_correct: dict[int, int] = defaultdict(int)
    for true, pred in zip(true_idx.tolist(), preds.tolist()):
        per_class_true[true] += 1
        if true == pred:
            per_class_correct[true] += 1
    return {
        "squares": 64,
        "correct": int(np.sum(correct)),
        "board_exact": bool(np.all(correct)),
        "occupancy_correct": int(np.sum(occupied_true == occupied_pred)),
        "occupied_correct": occupied_correct,
        "occupied_n": occupied_n,
        "empty_correct": empty_correct,
        "empty_n": empty_n,
        "colour_correct": colour_correct,
        "colour_n": colour_n,
        "confidence_sum": float(np.sum(confs)),
        "uncertain": uncertain,
        "per_class_true": dict(per_class_true),
        "per_class_correct": dict(per_class_correct),
        "calibration": [(float(conf), bool(ok)) for conf, ok in zip(confs, correct)],
    }


def expected_calibration_error(pairs: list[tuple[float, bool]], bins: int = CALIBRATION_BINS) -> dict:
    if not pairs:
        return {"ece": None, "bins": []}
    counts = [0] * bins
    conf_sum = [0.0] * bins
    correct_sum = [0] * bins
    for confidence, correct in pairs:
        index = min(int(confidence * bins), bins - 1)
        counts[index] += 1
        conf_sum[index] += confidence
        correct_sum[index] += int(correct)
    total = len(pairs)
    ece = 0.0
    detail = []
    for index in range(bins):
        if not counts[index]:
            continue
        mean_conf = conf_sum[index] / counts[index]
        accuracy = correct_sum[index] / counts[index]
        ece += counts[index] / total * abs(accuracy - mean_conf)
        detail.append(
            {
                "lo": round(index / bins, 2),
                "hi": round((index + 1) / bins, 2),
                "n": counts[index],
                "confidence": round(mean_conf, 4),
                "accuracy": round(accuracy, 4),
            }
        )
    return {"ece": round(ece, 4), "bins": detail}


def new_accumulator() -> dict:
    return {
        "frames": 0,
        "correct": 0,
        "boards_exact": 0,
        "occupancy_correct": 0,
        "occupied_correct": 0,
        "occupied_n": 0,
        "empty_correct": 0,
        "empty_n": 0,
        "colour_correct": 0,
        "colour_n": 0,
        "confidence_sum": 0.0,
        "uncertain_sum": 0,
        "per_class_true": defaultdict(int),
        "per_class_correct": defaultdict(int),
        "calibration": [],
        "gate_frames": 0,
        "gate_pass": 0,
        "gate_uncertain_sum": 0,
        "gate_centre_sum": 0,
        "gate_white_home_sum": 0,
        "gate_black_home_sum": 0,
        "gate_wrong_colour_sum": 0,
        "gate_occupied_sum": 0,
        "gate_prior_sum": 0,
    }


def add_frame(acc: dict, metrics: dict, gate: dict | None) -> None:
    acc["frames"] += 1
    acc["correct"] += metrics["correct"]
    acc["boards_exact"] += int(metrics["board_exact"])
    acc["occupancy_correct"] += metrics["occupancy_correct"]
    acc["occupied_correct"] += metrics["occupied_correct"]
    acc["occupied_n"] += metrics["occupied_n"]
    acc["empty_correct"] += metrics["empty_correct"]
    acc["empty_n"] += metrics["empty_n"]
    acc["colour_correct"] += metrics["colour_correct"]
    acc["colour_n"] += metrics["colour_n"]
    acc["confidence_sum"] += metrics["confidence_sum"]
    acc["uncertain_sum"] += metrics["uncertain"]
    for index, count in metrics["per_class_true"].items():
        acc["per_class_true"][index] += count
    for index, count in metrics["per_class_correct"].items():
        acc["per_class_correct"][index] += count
    acc["calibration"].extend(metrics["calibration"])
    if gate is not None:
        acc["gate_frames"] += 1
        acc["gate_pass"] += int(gate["ok"])
        acc["gate_uncertain_sum"] += gate["uncertain"]
        acc["gate_centre_sum"] += gate["centre_occupied"]
        acc["gate_white_home_sum"] += gate["white_home"]
        acc["gate_black_home_sum"] += gate["black_home"]
        acc["gate_wrong_colour_sum"] += gate["wrong_colour_home"]
        acc["gate_occupied_sum"] += gate["total_occupied"]
        acc["gate_prior_sum"] += len(gate["prior_resolved"])


def finalize(acc: dict) -> dict:
    frames = acc["frames"]
    squares = frames * 64
    occupied_n = acc["occupied_n"]
    empty_n = acc["empty_n"]
    colour_n = acc["colour_n"]
    per_class = {}
    for index, total in acc["per_class_true"].items():
        if total:
            per_class[benchmark.CLASSES[index]] = f"{acc['per_class_correct'].get(index, 0)}/{total}"
    gate_frames = acc["gate_frames"]
    return {
        "frames": frames,
        "per_square_exact_n": f"{acc['correct']}/{squares}" if squares else "0/0",
        "per_square_exact_rate": round(acc["correct"] / squares, 4) if squares else None,
        "board_exact": f"{acc['boards_exact']}/{frames}" if frames else "0/0",
        "board_exact_rate": round(acc["boards_exact"] / frames, 4) if frames else None,
        "occupancy_rate": round(acc["occupancy_correct"] / squares, 4) if squares else None,
        "occupied_exact": f"{acc['occupied_correct']}/{occupied_n}" if occupied_n else "0/0",
        "occupied_exact_rate": round(acc["occupied_correct"] / occupied_n, 4) if occupied_n else None,
        "empty_exact": f"{acc['empty_correct']}/{empty_n}" if empty_n else "0/0",
        "empty_exact_rate": round(acc["empty_correct"] / empty_n, 4) if empty_n else None,
        "colour": f"{acc['colour_correct']}/{colour_n}" if colour_n else "0/0",
        "colour_rate": round(acc["colour_correct"] / colour_n, 4) if colour_n else None,
        "mean_confidence": round(acc["confidence_sum"] / squares, 4) if squares else None,
        "mean_uncertain": round(acc["uncertain_sum"] / frames, 2) if frames else None,
        "calibration": expected_calibration_error(acc["calibration"]),
        "per_class_recall": per_class,
        "start_gate": {
            "frames": gate_frames,
            "pass": f"{acc['gate_pass']}/{gate_frames}" if gate_frames else "0/0",
            "mean_uncertain": round(acc["gate_uncertain_sum"] / gate_frames, 2) if gate_frames else None,
            "mean_centre_occupied": round(acc["gate_centre_sum"] / gate_frames, 2) if gate_frames else None,
            "mean_white_home": round(acc["gate_white_home_sum"] / gate_frames, 2) if gate_frames else None,
            "mean_black_home": round(acc["gate_black_home_sum"] / gate_frames, 2) if gate_frames else None,
            "mean_wrong_colour": round(acc["gate_wrong_colour_sum"] / gate_frames, 2) if gate_frames else None,
            "mean_occupied": round(acc["gate_occupied_sum"] / gate_frames, 2) if gate_frames else None,
            "mean_prior_resolved": round(acc["gate_prior_sum"] / gate_frames, 2) if gate_frames else None,
        },
    }


def mean_or_none(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(usable) / len(usable), 4) if usable else None


def cell_weighted(cells: list[dict]) -> dict:
    """Equal weight per (session, position, condition) burst, not per frame."""
    if not cells:
        return {"cells": 0}
    return {
        "cells": len(cells),
        "per_square_exact_rate": mean_or_none([cell["per_square_exact_rate"] for cell in cells]),
        "board_exact_rate": mean_or_none([cell["board_exact_rate"] for cell in cells]),
        "occupancy_rate": mean_or_none([cell["occupancy_rate"] for cell in cells]),
        "occupied_exact_rate": mean_or_none([cell["occupied_exact_rate"] for cell in cells]),
        "empty_exact_rate": mean_or_none([cell["empty_exact_rate"] for cell in cells]),
        "colour_rate": mean_or_none([cell["colour_rate"] for cell in cells]),
        "mean_confidence": mean_or_none([cell["mean_confidence"] for cell in cells]),
        "mean_uncertain": mean_or_none([cell["mean_uncertain"] for cell in cells]),
        "ece": mean_or_none([cell["calibration"]["ece"] for cell in cells]),
    }


# ---------------------------------------------------------------------------
# Manifest, source selection, holdout guard
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path, capture_root: Path) -> list[dict]:
    if not manifest_path.is_file():
        raise FatalError(f"manifest not found: {manifest_path}")
    root = capture_root.resolve()
    records = []
    errors: list[str] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: invalid JSON: {error}")
            continue
        required = (
            "session_id",
            "split",
            "fen",
            "corners_norm",
            "source",
            "orientation",
            "position_id",
        )
        missing = [field for field in required if record.get(field) is None]
        if missing:
            errors.append(f"line {line_number}: missing {missing}")
            continue
        if record["split"] not in SPLITS:
            errors.append(f"line {line_number}: invalid split {record['split']!r}")
            continue
        feature_errors = validate_fen(str(record["fen"]))
        if feature_errors:
            errors.append(f"line {line_number}: {feature_errors[0]}")
            continue
        _corners, corner_errors = validate_corners(record["corners_norm"])
        if corner_errors:
            errors.append(f"line {line_number}: {corner_errors[0]}")
            continue
        relpath = record.get("relpath") or f"{record['session_id']}/{record.get('frame', '')}"
        frame_path = (root / relpath).resolve()
        try:
            frame_path.relative_to(root)
        except ValueError:
            errors.append(f"line {line_number}: frame escapes capture root: {relpath}")
            continue
        if not frame_path.is_file():
            errors.append(f"line {line_number}: frame not found: {relpath}")
            continue
        expected_hash = record.get("sha256")
        if expected_hash and sha256_file(frame_path) != expected_hash:
            errors.append(f"line {line_number}: frame hash changed since manifest build: {relpath}")
            continue
        record["_path"] = frame_path
        records.append(record)
    if errors:
        preview = "\n".join(errors[:10])
        raise FatalError(f"manifest rejected ({len(errors)} problems):\n{preview}")
    if not records:
        raise FatalError("manifest contains no usable frames")
    return records


def select_records(records: list[dict], source: str | None) -> tuple[list[dict], str]:
    sources = sorted({record["source"] for record in records})
    unknown = sorted(set(sources) - set(SOURCE_VALUES))
    if unknown:
        raise FatalError(f"unknown capture source(s) {unknown}; allowed {SOURCE_VALUES}")
    if len(sources) > 1 and source is None:
        raise FatalError(
            f"manifest mixes sources {sources}; select one with --source. "
            "The protocol forbids comparing native-dump and screencap numerically."
        )
    if source is not None and source not in sources:
        raise FatalError(f"source {source!r} not present; manifest has {sources}")
    chosen = source or sources[0]
    return [record for record in records if record["source"] == chosen], chosen


def coverage_report(records: list[dict], splits: list[str]) -> tuple[dict, list[str]]:
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    warnings: list[str] = []
    for split in splits:
        split_records = [record for record in records if record["split"] == split]
        coverage[split] = {"frames": len(split_records), "positions": {}, "conditions": {}}
        for field in CONDITION_FIELDS:
            counts: dict[str, int] = defaultdict(int)
            for record in split_records:
                counts[str(record.get(field))] += 1
            coverage[split]["conditions"][field] = dict(sorted(counts.items()))
            if field == "position_id":
                coverage[split]["positions"] = dict(sorted(counts.items()))
    for field in CONDITION_FIELDS:
        values = sorted(
            {
                value
                for split in splits
                for value in coverage[split]["conditions"][field]
            }
        )
        for value in values:
            missing = [
                split
                for split in splits
                if coverage[split]["frames"] and value not in coverage[split]["conditions"][field]
            ]
            if missing:
                warnings.append(
                    f"{field}={value} missing from split(s) {missing}; "
                    "a split must not be a proxy for a condition."
                )
    return coverage, warnings


def model_hashes(model_paths: dict[str, str]) -> dict[str, str]:
    return {name: sha256_file(Path(path)) for name, path in model_paths.items()}


def check_holdout_guard(
    splits: list[str],
    hashes: dict[str, str],
    freeze_path: Path | None,
    use_log: Path,
) -> dict | None:
    if "holdout_screen" not in splits:
        return None
    if freeze_path is None:
        raise FatalError("holdout_screen requires --freeze-manifest with pinned model SHA-256 values")
    if not freeze_path.is_file():
        raise FatalError(f"freeze manifest not found: {freeze_path}")
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FatalError(f"invalid freeze manifest: {error}") from error
    frozen = freeze.get("models")
    if not isinstance(frozen, dict):
        raise FatalError("freeze manifest must contain a 'models' object of name -> sha256")
    for name, digest in hashes.items():
        pinned = frozen.get(name)
        if pinned is None:
            raise FatalError(f"model {name!r} is not pinned in the freeze manifest")
        if str(pinned).lower() != digest:
            raise FatalError(
                f"model {name!r} hash differs from the freeze manifest: "
                f"{digest} != {pinned}"
            )
    if use_log.exists():
        raise FatalError(
            f"holdout_screen was already consumed according to {use_log}; "
            "a second look is forbidden by the capture protocol"
        )
    return {"freeze_manifest": str(freeze_path), "use_log": str(use_log)}


def record_holdout_use(use_log: Path, payload: dict) -> None:
    use_log.parent.mkdir(parents=True, exist_ok=True)
    use_log.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def batch_frames(records: list[dict], chunk_size: int):
    for start in range(0, len(records), chunk_size):
        yield records[start : start + chunk_size]


def structure_groups() -> dict:
    return {
        "by_session": {},
        "by_position": {},
        "by_condition": {field: {} for field in CONDITION_FIELDS},
        "by_cell": {},
    }


def add_to_groups(groups: dict, record: dict, metrics: dict, gate: dict | None) -> None:
    session = record["session_id"]
    position = record["position_id"]
    cell = (
        session,
        position,
        str(record.get("theme")),
        str(record.get("brightness_pct")),
        str(record.get("distance")),
        str(record.get("angle_deg")),
        str(record.get("source")),
    )
    add_frame(groups["by_session"].setdefault(session, new_accumulator()), metrics, gate)
    add_frame(groups["by_position"].setdefault(position, new_accumulator()), metrics, gate)
    add_frame(groups["by_cell"].setdefault(cell, new_accumulator()), metrics, gate)
    for field in CONDITION_FIELDS:
        value = str(record.get(field))
        add_frame(groups["by_condition"][field].setdefault(value, new_accumulator()), metrics, gate)


def run_models(
    selected_records: list[dict],
    splits: list[str],
    model_paths: dict[str, str],
    chunk_size: int,
) -> dict:
    sessions = {
        name: benchmark.ort.InferenceSession(path)
        for name, path in model_paths.items()
    }
    results = {
        split: {
            "models": {
                name: {
                    "frame_weighted": None,
                    "cell_weighted": None,
                    "by_session": {},
                    "by_position": {},
                    "by_condition": {field: {} for field in CONDITION_FIELDS},
                }
                for name in model_paths
            }
        }
        for split in splits
    }
    split_acc = {
        split: {name: new_accumulator() for name in model_paths}
        for split in splits
    }
    groups = {
        split: {name: structure_groups() for name in model_paths}
        for split in splits
    }

    frames_done = 0
    for chunk in batch_frames(selected_records, chunk_size):
        images = []
        truths: list[list[str]] = []
        for record in chunk:
            image = cv2.imread(str(record["_path"]))
            if image is None:
                raise FatalError(f"cannot read frame: {record['_path']}")
            height, width = image.shape[:2]
            corners = np.array(record["corners_norm"], dtype=np.float32).reshape(4, 2)
            corners[:, 0] *= width
            corners[:, 1] *= height
            images.append(benchmark.sample_device_batch(image, corners))
            white_grid = parse_fen_to_classes(record["fen"])
            truths.append(orient_truth(white_grid, record.get("orientation", "white_bottom")))
        batch = np.concatenate(images, axis=0)
        for name, session in sessions.items():
            _logits, probs = benchmark.infer_batch(session, batch)
            for index, record in enumerate(chunk):
                split = record["split"]
                if split not in split_acc:
                    continue
                frame_probs = probs[index * 64 : (index + 1) * 64]
                metrics = square_metrics(frame_probs, truths[index])
                gate = (
                    start_gate(frame_probs, record.get("orientation", "white_bottom"))
                    if record["position_id"] == "start"
                    else None
                )
                add_frame(split_acc[split][name], metrics, gate)
                add_to_groups(groups[split][name], record, metrics, gate)
        frames_done += len(chunk)
        print(f"evaluated {frames_done}/{len(selected_records)} frames", flush=True)

    for split in splits:
        if not split_acc[split]:
            continue
        reference = next(iter(model_paths))
        split_frames = split_acc[split][reference]["frames"]
        if split_frames == 0:
            continue
        for name in model_paths:
            finished = finalize(split_acc[split][name])
            entry = results[split]["models"][name]
            entry["frame_weighted"] = finished
            entry["cell_weighted"] = cell_weighted(
                [finalize(acc) for acc in groups[split][name]["by_cell"].values()]
            )
            entry["by_session"] = {
                key: finalize(acc) for key, acc in groups[split][name]["by_session"].items()
            }
            entry["by_position"] = {
                key: finalize(acc) for key, acc in groups[split][name]["by_position"].items()
            }
            entry["by_condition"] = {
                field: {
                    key: finalize(acc)
                    for key, acc in groups[split][name]["by_condition"][field].items()
                }
                for field in CONDITION_FIELDS
            }
        results[split]["frames"] = split_frames
    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def perfect_start_probs() -> np.ndarray:
    probs = np.full((64, len(benchmark.CLASSES)), 0.01 / 12.0, dtype=np.float32)
    for row in range(8):
        for col in range(8):
            square = algebraic_square(row, col, "white_bottom")
            probs[row * 8 + col][CLASS_TO_INDEX[START_BY_SQUARE[square]]] = 0.99
    return probs


def selftest() -> int:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    white_grid = parse_fen_to_classes(start_fen)
    check(white_grid[0] == "black_rook", "FEN row 0 col 0 must be a8 black rook")
    check(white_grid[60] == "white_king", "FEN row 7 col 4 must be e1 white king")
    black_grid = orient_truth(white_grid, "black_bottom")
    check(black_grid[0] == "white_rook", "black_bottom row 0 col 0 must be h1 white rook")
    check(black_grid[63] == "black_rook", "black_bottom row 7 col 7 must be a8 black rook")
    check(algebraic_square(7, 4, "white_bottom") == "e1", "white_bottom e1 mapping")
    check(algebraic_square(7, 7, "black_bottom") == "a8", "black_bottom a8 mapping")

    probs = perfect_start_probs()
    truth = orient_truth(white_grid, "white_bottom")
    metrics = square_metrics(probs, truth)
    check(metrics["correct"] == 64, f"perfect frame scored {metrics['correct']}/64")
    check(metrics["board_exact"] is True, "perfect frame must be board-exact")
    check(metrics["occupied_correct"] == 32, "perfect frame occupied exact must be 32")
    check(metrics["colour_correct"] == 32, "perfect frame colour must be 32/32")
    check(metrics["uncertain"] == 0, "perfect frame must have no uncertain squares")

    gate = start_gate(probs, "white_bottom")
    check(gate["ok"] is True, f"perfect start must pass: {gate['reason']}")
    check(gate["uncertain"] == 0, "perfect start uncertain must be 0")
    check(gate["white_home"] == 16 and gate["black_home"] == 16, "perfect start homes")
    check(gate["total_occupied"] == 32, "perfect start occupancy")
    check(gate["matching_squares"] == 64, "perfect start matching squares")

    one_wrong = probs.copy()
    one_wrong[60] = 0.01 / 12.0
    one_wrong[60][CLASS_TO_INDEX["white_pawn"]] = 0.99
    metrics = square_metrics(one_wrong, truth)
    check(metrics["correct"] == 63, "one misclassified square must score 63/64")
    check(metrics["board_exact"] is False, "one misclassified square must break board exact")

    # A home square with weak occupancy is uncertain, then resolved from the prior.
    weak = probs.copy()
    weak[56] = 0.01 / 12.0
    weak[56][CLASS_TO_INDEX["empty"]] = 0.40
    weak[56][CLASS_TO_INDEX["black_pawn"]] = 0.60
    gate = start_gate(weak, "white_bottom")
    check(gate["uncertain"] == 1, f"weak home square uncertain={gate['uncertain']}")
    check(gate["prior_resolved"] == ["a1"], f"weak home prior={gate['prior_resolved']}")
    check(gate["ok"] is True, f"one weak home square must be prior-resolved: {gate['reason']}")

    # Four weak squares exceed the bounded prior.
    four = probs.copy()
    for index in (56, 57, 58, 59):
        four[index] = 0.01 / 12.0
        four[index][CLASS_TO_INDEX["empty"]] = 0.40
        four[index][CLASS_TO_INDEX["black_pawn"]] = 0.60
    gate = start_gate(four, "white_bottom")
    check(gate["uncertain"] == 4, f"four weak squares uncertain={gate['uncertain']}")
    check(gate["ok"] is False, "four uncertain squares must block the gate")
    check("too uncertain" in (gate["reason"] or ""), "four uncertain reason")

    # A confident colour inversion is a contradiction, not a prior-resolvable blip.
    inverted = probs.copy()
    inverted[56] = 0.01 / 12.0
    inverted[56][CLASS_TO_INDEX["black_pawn"]] = 0.95
    gate = start_gate(inverted, "white_bottom")
    check(gate["uncertain"] == 1, f"inverted home square uncertain={gate['uncertain']}")
    check(gate["ok"] is False, "confident contradiction must block the gate")
    check(gate["confident_contradictions"] == ["a1"], "contradiction square must be reported")

    occupied_centre = four.copy()
    occupied_centre[27] = 0.01 / 12.0
    occupied_centre[27][CLASS_TO_INDEX["black_pawn"]] = 0.99
    gate = start_gate(occupied_centre, "white_bottom")
    check(gate["centre_occupied"] == 1, "occupied centre must be counted")
    check(gate["ok"] is False, "occupied centre must block the gate")

    calibration = expected_calibration_error([(1.0, True)] * 100)
    check(calibration["ece"] == 0.0, f"perfect calibration ECE={calibration['ece']}")
    calibration = expected_calibration_error([(1.0, False)] * 100)
    check(abs(calibration["ece"] - 1.0) < 1e-9, f"fully wrong ECE={calibration['ece']}")

    # Aggregation: two frames, one exact and one off by a square.
    acc = new_accumulator()
    add_frame(acc, square_metrics(probs, truth), start_gate(probs, "white_bottom"))
    add_frame(acc, square_metrics(one_wrong, truth), start_gate(four, "white_bottom"))
    finished = finalize(acc)
    check(finished["per_square_exact_n"] == "127/128", finished["per_square_exact_n"])
    check(finished["board_exact"] == "1/2", finished["board_exact"])
    check(finished["start_gate"]["pass"] == "1/2", finished["start_gate"]["pass"])

    # Source mixing is refused; selecting a source filters.
    records = [
        {"source": "native-dump", "split": "train_screen"},
        {"source": "screencap", "split": "val_screen"},
    ]
    try:
        select_records(records, None)
        failures.append("mixed sources were not refused")
    except FatalError:
        pass
    filtered, source = select_records(records, "screencap")
    check(source == "screencap" and len(filtered) == 1, "source selection failed")

    # Holdout guard: missing freeze, wrong hash, existing log, then success.
    temp = Path(tempfile.mkdtemp(prefix="i5eval_"))
    try:
        model_a = temp / "a.onnx"
        model_a.write_bytes(b"model-a")
        hashes = {"universal150_prod": sha256_file(model_a)}
        freeze = temp / "freeze.json"
        use_log = temp / "used.json"
        try:
            check_holdout_guard(["holdout_screen"], hashes, None, use_log)
            failures.append("holdout without a freeze manifest was not refused")
        except FatalError:
            pass
        freeze.write_text(json.dumps({"models": {"universal150_prod": "0" * 64}}), encoding="utf-8")
        try:
            check_holdout_guard(["holdout_screen"], hashes, freeze, use_log)
            failures.append("holdout with a wrong pinned hash was not refused")
        except FatalError:
            pass
        freeze.write_text(json.dumps({"models": hashes}), encoding="utf-8")
        guard = check_holdout_guard(["holdout_screen"], hashes, freeze, use_log)
        check(guard is not None, "valid holdout guard returned nothing")
        check(not use_log.exists(), "guard must not write the use log before evaluation")
        record_holdout_use(use_log, {"models": hashes})
        try:
            check_holdout_guard(["holdout_screen"], hashes, freeze, use_log)
            failures.append("second holdout use was not refused")
        except FatalError:
            pass
        check(
            check_holdout_guard(["val_screen"], hashes, None, use_log) is None,
            "non-holdout evaluation must not require a freeze manifest",
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    if failures:
        for failure in failures:
            print(f"SELFTEST FAIL {failure}")
        return 1
    print("SELFTEST OK evaluator labels, metrics, gate, source and holdout guards")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_splits(raw: str) -> list[str]:
    splits = [part.strip() for part in raw.split(",") if part.strip()]
    if not splits:
        raise FatalError("--splits must name at least one split")
    unknown = sorted(set(splits) - SPLITS)
    if unknown:
        raise FatalError(f"unknown splits {unknown}; allowed {sorted(SPLITS)}")
    return splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--capture-root", help="root that manifest relpaths are relative to")
    parser.add_argument("--manifest", help="manifest.jsonl from build_screen_capture_manifest.py")
    parser.add_argument("--models", default="all", help="comma list or 'all'")
    parser.add_argument("--splits", default="train_screen,val_screen")
    parser.add_argument("--source", choices=SOURCE_VALUES)
    parser.add_argument("--json-out", default="dist/milestone_54i5_vivo_benchmark.json")
    parser.add_argument("--freeze-manifest", help="required for holdout_screen; pins model SHA-256")
    parser.add_argument(
        "--write-freeze",
        metavar="PATH",
        help="write a freeze manifest for --models and exit (no evaluation)",
    )
    parser.add_argument("--holdout-use-log", default="dist/holdout_screen_i5_used.json")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--limit-frames", type=int, help="debug only: evaluate the first N frames")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.write_freeze:
        try:
            model_paths = benchmark.resolve_models(args.models)
            hashes = model_hashes(model_paths)
        except (FatalError, SystemExit) as error:
            print(f"FATAL {error}")
            return 2
        payload = {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "models": hashes,
            "note": "Models frozen before the one-shot holdout_screen evaluation.",
        }
        freeze_path = Path(args.write_freeze)
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote freeze manifest {freeze_path} for {list(hashes)}")
        return 0
    if not args.capture_root or not args.manifest:
        parser.error("--capture-root and --manifest are required unless --selftest or --write-freeze is used")

    try:
        splits = parse_splits(args.splits)
        model_paths = benchmark.resolve_models(args.models)
        capture_root = Path(args.capture_root)
        records = load_manifest(Path(args.manifest), capture_root)
        selected, source = select_records(records, args.source)
        selected = [record for record in selected if record["split"] in splits]
        if args.limit_frames is not None:
            limited = []
            for split in splits:
                limited.extend(
                    [record for record in selected if record["split"] == split][
                        : max(0, args.limit_frames)
                    ]
                )
            selected = limited
        if not selected:
            raise FatalError(f"no frames for splits {splits} and source {source!r}")
        empty_splits = [split for split in splits if not any(record["split"] == split for record in selected)]
        if empty_splits:
            raise FatalError(f"requested split(s) have no frames: {empty_splits}")
        hashes = model_hashes(model_paths)
        guard = check_holdout_guard(
            splits,
            hashes,
            Path(args.freeze_manifest) if args.freeze_manifest else None,
            Path(args.holdout_use_log),
        )
        coverage, coverage_warnings = coverage_report(selected, splits)
        results = run_models(selected, splits, model_paths, max(1, args.chunk_size))
    except FatalError as error:
        print(f"FATAL {error}")
        return 2

    output = {
        "milestone": "5.4I.5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capture_root": str(capture_root),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(Path(args.manifest)),
        "source": source,
        "splits": splits,
        "thresholds": {
            "uncertain_conf": UNCERT_CONF,
            "uncertain_margin": UNCERT_MARGIN,
            "start_max_uncertain": START_MAX_UNCERTAIN,
            "contradiction_probability": CONTRADICTION_PROB,
            "contradiction_margin": CONTRADICTION_MARGIN,
            "crop": "device bilinear inset (64/72 centre crop)",
        },
        "models": {
            name: {"path": path, "sha256": hashes[name]}
            for name, path in model_paths.items()
        },
        "coverage": coverage,
        "coverage_warnings": coverage_warnings,
        "holdout": guard,
        "results": results,
        "notes": [
            "Screen-camera crops only; never pool with physical 3D-board metrics.",
            "Splits are session-level by construction; frames are never re-split.",
            "Per-square exact is correct/64 per frame; board_exact requires all 64.",
            "holdout_screen is one-shot: the use log is written only after this run.",
        ],
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if guard is not None:
        record_holdout_use(
            Path(args.holdout_use_log),
            {
                "used_at": output["generated_at"],
                "json_out": str(out_path),
                "json_sha256": sha256_file(out_path),
                "manifest_sha256": output["manifest_sha256"],
                "models": hashes,
                "splits": splits,
            },
        )

    for split in splits:
        split_result = results.get(split, {})
        if not split_result.get("models"):
            continue
        entry = split_result["models"].get("universal150_prod") or next(iter(split_result["models"].values()))
        frame_metrics = entry["frame_weighted"]
        if frame_metrics is None:
            continue
        print(
            f"{split}: frames={split_result.get('frames', 0)} "
            f"production per-square={frame_metrics['per_square_exact_rate']} "
            f"board={frame_metrics['board_exact']} "
            f"occupied={frame_metrics['occupied_exact']} "
            f"colour={frame_metrics['colour']} "
            f"gate={frame_metrics['start_gate']['pass']}"
        )
    print(f"models={list(model_paths)} source={source} json={out_path}")
    if coverage_warnings:
        for warning in coverage_warnings:
            print(f"WARN {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
