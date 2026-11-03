#!/usr/bin/env python3
"""
benchmark_board_geometry.py - Milestone 5.3D Real-Image Board Geometry Benchmark

Evaluates deterministic board detection and perspective calibration approaches on real
FENiT chessboard photographs (Mendeley Data DOI 10.17632/wpp9xnbcp6.1) without classifier training.

Approaches Compared:
  Approach A: Current Production Baseline (Canny 30/90 + 60/180 + dilate +
              approxPolyDP/minAreaRect + 8x8 checkerboard pattern evidence)
  Approach B: Multi-Scale Adaptive Edge & Contour Hierarchy (tuned Canny + convex hull +
              multi-epsilon). Reported twice: a relaxed any-convex-quad gate and a strict
              production gate that requires the same checkerboard pattern evidence as A.
  Approach C: Hough Line Intersections & Vanishing Point Quadrilateral Fitting
  Approach D: Ground-Truth-Seeded Assisted Calibration Simulation (coarse user taps +
              local Shi-Tomasi corner refinement)

Accounting contract (one record per image per approach):
  candidate_generated - the approach emitted a quadrilateral proposal.
  geometry_valid      - the proposal passes production structural checks (bounds,
                        non-self-intersection, strict convexity, area/span minimums).
  accepted            - the proposal passes the approach-specific acceptance gate.
  rejection_reason    - machine-readable code for the first failing stage, or null.
  rejection_stage     - "candidate", "geometry", or "acceptance".

Metric views (never conflated):
  accepted_quality      - conditional on acceptance (N = accepted).
  proposal_quality      - conditional on geometry-valid proposals (N = geometry_valid).
  all_input_penalized   - every input image counts; rejected or invalid proposals
                          contribute IoU 0.0 and corner error 1.0.

Outputs:
  board_geometry_benchmark.json          aggregate summary
  board_geometry_benchmark_records.jsonl reproducible per-image records

Default output directory:
  C:\\dev\\chess-capture-workspace\\external\\fenit\\reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# Production thresholds mirrored from src/vision.
# ---------------------------------------------------------------------------

MIN_DETECTION_CONFIDENCE = 0.32
MIN_QUADRILATERAL_AREA_RATIO = 0.05
MIN_CONVEX_TURN_SINE = 1e-6

PATTERN_GRID_SIZE = 8
PATTERN_CELL_INSET = 0.18
PATTERN_CORNER_SPREAD = 0.02
PATTERN_CORNER_SAMPLES = 3
PATTERN_SAMPLING_SCALES = [round(1.0 - index * 0.01, 2) for index in range(21)]

MIN_PATTERN_FIT = 0.55
MIN_PATTERN_ALTERNATION = 0.80
MIN_PATTERN_CONTRAST = 8.0
FULL_PATTERN_CONTRAST = 48.0
MIN_PATTERN_SCORE = 0.45

BENCHMARK_VERSION = "5.3D"


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------


def signed_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index in range(len(points)):
        current = points[index]
        following = points[(index + 1) % len(points)]
        total += float(current[0]) * float(following[1]) - float(following[0]) * float(current[1])
    return total / 2.0


def order_corners_screen_space(points: np.ndarray) -> np.ndarray:
    """Orders 4 corners clockwise starting from top-left in screen coordinates."""
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles, kind="stable")
    ordered = points[order]
    sums = ordered[:, 0] + ordered[:, 1]
    minimum_sum = float(np.min(sums))
    tied = [
        index
        for index, value in enumerate(sums)
        if math.isclose(float(value), minimum_sum, rel_tol=1e-9, abs_tol=1e-9)
    ]
    start = min(tied, key=lambda index: (float(ordered[index, 1]), float(ordered[index, 0])))
    return np.roll(ordered, -start, axis=0)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    if abs(o1) <= 1e-9 and on_segment(a, c, b):
        return True
    if abs(o2) <= 1e-9 and on_segment(a, d, b):
        return True
    if abs(o3) <= 1e-9 and on_segment(c, a, d):
        return True
    if abs(o4) <= 1e-9 and on_segment(c, b, d):
        return True
    return False


def is_self_intersecting(points: np.ndarray) -> bool:
    if len(points) != 4:
        return False
    return _segments_intersect(points[0], points[1], points[2], points[3]) or (
        _segments_intersect(points[1], points[2], points[3], points[0])
    )


def is_strictly_convex(points: np.ndarray) -> bool:
    """Scale-independent strict convexity, matching src/vision/boardDetection.ts."""
    if len(points) != 4 or not np.all(np.isfinite(points)):
        return False

    turn_sign = 0
    for index in range(4):
        a = points[index]
        b = points[(index + 1) % 4]
        c = points[(index + 2) % 4]
        first = b - a
        second = c - b
        first_length = float(np.hypot(first[0], first[1]))
        second_length = float(np.hypot(second[0], second[1]))
        if first_length < 1e-12 or second_length < 1e-12:
            return False
        cross = float(first[0] * second[1] - first[1] * second[0])
        turn_sine = cross / (first_length * second_length)
        if abs(turn_sine) < MIN_CONVEX_TURN_SINE:
            return False
        current_sign = 1 if turn_sine > 0 else -1
        if turn_sign == 0:
            turn_sign = current_sign
        elif current_sign != turn_sign:
            return False
    return True


def clamp_points_to_bounds(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Clamps quad vertices into the image bounds (Milestone 5.4C).

    Mirrors src/vision/boardDetection.ts clampCornersToBounds: detectors may
    propose corners 1-3 px outside the frame (rotated minAreaRect). Clamping
    them preserves otherwise-valid boards instead of rejecting them.
    """
    height, width = shape[:2]
    clamped = np.asarray(points, dtype=np.float64).copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0.0, float(width))
    clamped[:, 1] = np.clip(clamped[:, 1], 0.0, float(height))
    return clamped


# Milestone 5.4D: a clamp may absorb contour/rotation overshoot, but a
# candidate needing more than max(4 px, 2% of the shorter image side) of
# modification is rejected rather than silently moved into frame. Mirrors
# MAX_CLAMP_RATIO / MIN_CLAMP_TOLERANCE_PX in src/vision/boardDetection.ts.
MAX_CLAMP_RATIO = 0.02
MIN_CLAMP_TOLERANCE_PX = 4.0


def max_clamp_tolerance_px(shape: Tuple[int, int]) -> float:
    height, width = shape[:2]
    return max(MIN_CLAMP_TOLERANCE_PX, MAX_CLAMP_RATIO * min(width, height))


def measure_corner_clamp_distance(points: np.ndarray, shape: Tuple[int, int]) -> float:
    """Largest distance (px) any single corner would move under clamping."""
    height, width = shape[:2]
    array = np.asarray(points, dtype=np.float64)
    dx = np.clip(array[:, 0], 0.0, float(width)) - array[:, 0]
    dy = np.clip(array[:, 1], 0.0, float(height)) - array[:, 1]
    return float(np.max(np.hypot(dx, dy))) if array.size else 0.0


def validate_quadrilateral(
    points: Optional[np.ndarray], shape: Tuple[int, int]
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Returns (valid, reason_code, human_detail) using production structural checks."""
    height, width = shape[:2]
    if points is None:
        return False, "no_candidate", "The approach produced no quadrilateral proposal."
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (4, 2):
        return False, "geometry_wrong_point_count", "The proposal does not contain four corners."
    if not np.all(np.isfinite(array)):
        return False, "geometry_non_finite", "A proposed corner is not finite."

    if (
        np.any(array[:, 0] < 0)
        or np.any(array[:, 0] > width)
        or np.any(array[:, 1] < 0)
        or np.any(array[:, 1] > height)
    ):
        return False, "geometry_out_of_bounds", "A proposed corner lies outside the image."

    if is_self_intersecting(array):
        return (
            False,
            "geometry_self_intersecting",
            "The proposed corners form a self-intersecting quadrilateral.",
        )

    ordered = order_corners_screen_space(array)
    if not is_strictly_convex(ordered):
        return (
            False,
            "geometry_non_convex",
            "The proposed corners are concave or nearly collinear.",
        )
    if is_self_intersecting(ordered):
        return (
            False,
            "geometry_self_intersecting",
            "The ordered corners form a self-intersecting quadrilateral.",
        )

    area = abs(signed_area(ordered))
    image_area = float(width * height)
    if image_area <= 0 or area / image_area < MIN_QUADRILATERAL_AREA_RATIO:
        return (
            False,
            "geometry_too_small",
            "The proposed quadrilateral covers too little of the image.",
        )

    span_x = float(np.max(ordered[:, 0]) - np.min(ordered[:, 0]))
    span_y = float(np.max(ordered[:, 1]) - np.min(ordered[:, 1]))
    if span_x < width * 0.15 or span_y < height * 0.15:
        return (
            False,
            "geometry_insufficient_span",
            "The proposed corners do not span enough of the image.",
        )

    return True, None, None


def score_candidate(points: np.ndarray, shape: Tuple[int, int]) -> float:
    """Production boundary score: area ratio x rectangularity, clamped to [0, 1]."""
    height, width = shape[:2]
    ordered = order_corners_screen_space(points)
    area = abs(signed_area(ordered))
    image_area = float(width * height)
    area_ratio = area / image_area if image_area > 0 else 0.0

    side_lengths = []
    for index in range(4):
        following = ordered[(index + 1) % 4]
        side_lengths.append(float(np.hypot(following[0] - ordered[index, 0], following[1] - ordered[index, 1])))
    average_side = float(np.mean(side_lengths))
    side_area = average_side * average_side
    rectangularity = min(1.0, area / side_area) if side_area > 0 else 0.0
    return max(0.0, min(1.0, area_ratio * 1.6 * rectangularity))


def inset_corners(points: np.ndarray, scale: float) -> np.ndarray:
    ordered = order_corners_screen_space(points)
    center = ordered.mean(axis=0)
    return (center + (ordered - center) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# 8 x 8 checkerboard pattern evidence, mirroring src/vision/boardEvidence.ts.
# ---------------------------------------------------------------------------


def _square_to_quadrilateral(corners: np.ndarray) -> Optional[Dict[str, float]]:
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return None

    top_left, top_right, bottom_right, bottom_left = corners
    twice_area = abs(
        top_left[0] * top_right[1]
        - top_right[0] * top_left[1]
        + top_right[0] * bottom_right[1]
        - bottom_right[0] * top_right[1]
        + bottom_right[0] * bottom_left[1]
        - bottom_left[0] * bottom_right[1]
        + bottom_left[0] * top_left[1]
        - top_left[0] * bottom_left[1]
    )
    if twice_area < 1e-9:
        return None

    dx1 = top_right[0] - bottom_right[0]
    dx2 = bottom_left[0] - bottom_right[0]
    dy1 = top_right[1] - bottom_right[1]
    dy2 = bottom_left[1] - bottom_right[1]
    sx = top_left[0] - top_right[0] + bottom_right[0] - bottom_left[0]
    sy = top_left[1] - top_right[1] + bottom_right[1] - bottom_left[1]

    if abs(sx) < 1e-9 and abs(sy) < 1e-9:
        return {
            "a": float(top_right[0] - top_left[0]),
            "b": float(bottom_left[0] - top_left[0]),
            "c": float(top_left[0]),
            "d": float(top_right[1] - top_left[1]),
            "e": float(bottom_left[1] - top_left[1]),
            "f": float(top_left[1]),
            "g": 0.0,
            "h": 0.0,
        }

    denominator = dx1 * dy2 - dx2 * dy1
    if abs(denominator) < 1e-9:
        return None

    g = (sx * dy2 - dx2 * sy) / denominator
    h = (dx1 * sy - sx * dy1) / denominator

    return {
        "a": float(top_right[0] - top_left[0] + g * top_right[0]),
        "b": float(bottom_left[0] - top_left[0] + h * bottom_left[0]),
        "c": float(top_left[0]),
        "d": float(top_right[1] - top_left[1] + g * top_right[1]),
        "e": float(bottom_left[1] - top_left[1] + h * bottom_left[1]),
        "f": float(top_left[1]),
        "g": float(g),
        "h": float(h),
    }


def _project_square_points(
    homography: Dict[str, float], u: np.ndarray, v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    denominator = homography["g"] * u + homography["h"] * v + 1.0
    x = (homography["a"] * u + homography["b"] * v + homography["c"]) / denominator
    y = (homography["d"] * u + homography["e"] * v + homography["f"]) / denominator
    return x, y


def _sample_bilinear(gray: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite pattern sample coordinate.")

    clamped_x = np.clip(x, 0, width - 1)
    clamped_y = np.clip(y, 0, height - 1)
    x0 = np.floor(clamped_x).astype(np.int32)
    y0 = np.floor(clamped_y).astype(np.int32)
    x1 = np.minimum(width - 1, x0 + 1)
    y1 = np.minimum(height - 1, y0 + 1)
    fx = clamped_x - x0
    fy = clamped_y - y0

    top = gray[y0, x0] * (1 - fx) + gray[y0, x1] * fx
    bottom = gray[y1, x0] * (1 - fx) + gray[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def _pattern_grid_offsets() -> Tuple[np.ndarray, np.ndarray]:
    anchors = [PATTERN_CELL_INSET, 1.0 - PATTERN_CELL_INSET]
    half_spread = (PATTERN_CORNER_SAMPLES - 1) / 2
    offsets = [
        (sample - half_spread) * PATTERN_CORNER_SPREAD
        for sample in range(PATTERN_CORNER_SAMPLES)
    ]

    us: List[float] = []
    vs: List[float] = []
    for row in range(PATTERN_GRID_SIZE):
        for column in range(PATTERN_GRID_SIZE):
            for anchor_v in anchors:
                for anchor_u in anchors:
                    for sample_row in offsets:
                        for sample_column in offsets:
                            us.append((column + anchor_u + sample_column) / PATTERN_GRID_SIZE)
                            vs.append((row + anchor_v + sample_row) / PATTERN_GRID_SIZE)
    return np.asarray(us, dtype=np.float64), np.asarray(vs, dtype=np.float64)


_PATTERN_US, _PATTERN_VS = _pattern_grid_offsets()


def _invalid_pattern(reason: str, empty: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(empty)
    result["valid"] = False
    result["reason"] = reason
    return result


def measure_pattern_evidence(gray: np.ndarray, corners: np.ndarray) -> Dict[str, Any]:
    """Reimplementation of measureBoardPattern with identical evidence thresholds."""
    empty: Dict[str, Any] = {
        "valid": False,
        "score": 0.0,
        "fit": 0.0,
        "alternation": 0.0,
        "contrast": 0.0,
        "noise": 0.0,
        "orientation": None,
        "reason": None,
    }

    if gray.ndim != 2 or gray.shape[0] < 2 or gray.shape[1] < 2:
        return _invalid_pattern("pattern_invalid_grayscale", empty)

    homography = _square_to_quadrilateral(np.asarray(corners, dtype=np.float64))
    if homography is None:
        return _invalid_pattern("pattern_homography_invalid", empty)

    try:
        x, y = _project_square_points(homography, _PATTERN_US, _PATTERN_VS)
        values = _sample_bilinear(gray.astype(np.float64, copy=False), x, y)
    except ValueError:
        return _invalid_pattern("pattern_sampling_failed", empty)

    samples_per_cell = (PATTERN_CORNER_SAMPLES**2) * 4
    cell_values = values.reshape(PATTERN_GRID_SIZE * PATTERN_GRID_SIZE, samples_per_cell)
    means = cell_values.mean(axis=1)
    standard_deviations = cell_values.std(axis=1)

    base_mean = float(means.mean())
    total_variance = float(np.sum((means - base_mean) ** 2))

    parity = np.empty(PATTERN_GRID_SIZE * PATTERN_GRID_SIZE, dtype=np.float64)
    for index in range(parity.size):
        row = index // PATTERN_GRID_SIZE
        column = index % PATTERN_GRID_SIZE
        parity[index] = 1.0 if (row + column) % 2 == 0 else -1.0

    parity_sum = float(np.sum((means - base_mean) * parity))
    amplitude = parity_sum / means.size
    residual_variance = float(np.sum((means - base_mean - amplitude * parity) ** 2))
    fit = max(0.0, min(1.0, 1.0 - residual_variance / total_variance)) if total_variance > 1e-6 else 0.0

    adjacent_pairs: List[Tuple[int, int]] = []
    for row in range(PATTERN_GRID_SIZE):
        for column in range(PATTERN_GRID_SIZE):
            index = row * PATTERN_GRID_SIZE + column
            if column + 1 < PATTERN_GRID_SIZE:
                adjacent_pairs.append((index, index + 1))
            if row + 1 < PATTERN_GRID_SIZE:
                adjacent_pairs.append((index, index + PATTERN_GRID_SIZE))

    contrast_sum = 0.0
    agreeing = [0, 0]
    for first, second in adjacent_pairs:
        difference = float(means[first] - means[second])
        contrast_sum += abs(difference)

        first_row, first_column = divmod(first, PATTERN_GRID_SIZE)
        second_row, second_column = divmod(second, PATTERN_GRID_SIZE)
        first_parity = 1.0 if (first_row + first_column) % 2 == 0 else -1.0
        second_parity = 1.0 if (second_row + second_column) % 2 == 0 else -1.0
        parity_step = first_parity - second_parity

        for orientation in (0, 1):
            direction = 1.0 if orientation == 0 else -1.0
            if difference * parity_step * direction > 0:
                agreeing[orientation] += 1

    contrast = contrast_sum / len(adjacent_pairs)
    orientation = 0 if agreeing[0] >= agreeing[1] else 1
    alternation = agreeing[orientation] / len(adjacent_pairs)
    noise = float(np.median(standard_deviations))

    contrast_factor = max(
        0.0, min(1.0, (contrast - MIN_PATTERN_CONTRAST) / (FULL_PATTERN_CONTRAST - MIN_PATTERN_CONTRAST))
    )
    noise_factor = max(0.0, min(1.0, (contrast / (noise + 1.0) - 1.0) / 2.0))
    score = max(0.0, min(1.0, fit * alternation * contrast_factor * noise_factor))

    metrics: Dict[str, Any] = {
        "valid": False,
        "score": float(score),
        "fit": float(fit),
        "alternation": float(alternation),
        "contrast": float(contrast),
        "noise": float(noise),
        "orientation": orientation,
        "reason": None,
    }

    if contrast < MIN_PATTERN_CONTRAST:
        return _invalid_pattern("pattern_contrast_below_threshold", metrics)
    if fit < MIN_PATTERN_FIT:
        return _invalid_pattern("pattern_fit_below_threshold", metrics)
    if alternation < MIN_PATTERN_ALTERNATION:
        return _invalid_pattern("pattern_alternation_below_threshold", metrics)
    if contrast <= noise:
        return _invalid_pattern("pattern_contrast_not_above_noise", metrics)
    if score < MIN_PATTERN_SCORE:
        return _invalid_pattern("pattern_score_below_threshold", metrics)

    metrics["valid"] = True
    return metrics


def measure_pattern_best(
    gray: np.ndarray, processing_corners: np.ndarray
) -> Tuple[Dict[str, Any], float]:
    """Mirrors production multi-inset sampling; returns the strongest evidence.

    Production compares combined confidence, which is zero for invalid evidence,
    then breaks ties by pattern score. Boundary score is constant across sampling
    scales, so the equivalent ranking is (valid, pattern score).
    """
    best = measure_pattern_evidence(gray, processing_corners)
    best_scale = 1.0
    best_rank = (1 if best["valid"] else 0, best["score"])

    for scale in PATTERN_SAMPLING_SCALES:
        if scale == 1.0:
            continue
        inset = inset_corners(processing_corners, scale)
        evidence = measure_pattern_evidence(gray, inset)
        rank = (1 if evidence["valid"] else 0, evidence["score"])
        if rank > best_rank:
            best = evidence
            best_scale = scale
            best_rank = rank

    return best, best_scale


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def compute_polygon_iou(pts1: np.ndarray, pts2: np.ndarray, shape: Tuple[int, int]) -> float:
    """Computes exact intersection-over-union between two 4-point quadrilaterals."""
    h, w = shape[:2]
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask1, [np.int32(pts1)], 1)
    cv2.fillPoly(mask2, [np.int32(pts2)], 1)
    intersection = int(np.logical_and(mask1, mask2).sum())
    union = int(np.logical_or(mask1, mask2).sum())
    return float(intersection / union) if union > 0 else 0.0


def compute_normalized_corner_error(
    pts1: np.ndarray, pts2: np.ndarray, shape: Tuple[int, int]
) -> float:
    """
    Computes mean corner distance normalized by image diagonal D = sqrt(W^2 + H^2),
    minimized over all 4 cyclic corner permutations.
    """
    h, w = shape[:2]
    diag = math.sqrt(w * w + h * h)
    min_dist = float("inf")
    for shift in range(4):
        shifted = np.roll(pts1, -shift, axis=0)
        dists = np.sqrt(np.sum((shifted - pts2) ** 2, axis=1))
        mean_dist = float(np.mean(dists))
        if mean_dist < min_dist:
            min_dist = mean_dist
    return float(min_dist / diag) if diag > 0 else 1.0


def _empty_candidate(candidate_reason: str) -> Dict[str, Any]:
    return {
        "points": None,
        "candidate_generated": False,
        "candidate_rejection_reason": candidate_reason,
        "geometry_valid": False,
        "geometry_rejection_reason": "no_candidate",
        "geometry_detail": "The approach produced no quadrilateral proposal.",
        "approach_confidence": 0.0,
        "boundary_score": 0.0,
        "combined_confidence": 0.0,
        "pattern": None,
        "sampling_scale": None,
        "pattern_skipped_reason": None,
        "corner_clamp_px": 0.0,
    }


def _finish_candidate(
    points: np.ndarray,
    shape: Tuple[int, int],
    confidence: float,
    processing_gray: Optional[np.ndarray],
    processing_points: Optional[np.ndarray],
    measure_pattern: bool = True,
    clamp_to_bounds: bool = False,
) -> Dict[str, Any]:
    clamp_px = 0.0
    if clamp_to_bounds:
        # Milestone 5.4C production detector behaviour: clamp proposed corners
        # into the frame before structural validation instead of rejecting them.
        # Milestone 5.4D: reject candidates whose clamp would exceed the
        # documented tolerance instead of inventing in-frame geometry.
        clamp_px = measure_corner_clamp_distance(points, shape)
        if clamp_px > max_clamp_tolerance_px(shape):
            return {
                "points": None,
                "candidate_generated": True,
                "candidate_rejection_reason": None,
                "geometry_valid": False,
                "geometry_rejection_reason": "clamp_excessive",
                "geometry_detail": (
                    f"Clamping would move a corner {clamp_px:.2f}px, above the "
                    f"{max_clamp_tolerance_px(shape):.2f}px tolerance."
                ),
                "approach_confidence": float(confidence),
                "boundary_score": 0.0,
                "combined_confidence": 0.0,
                "pattern": None,
                "sampling_scale": None,
                "pattern_skipped_reason": None,
                "corner_clamp_px": clamp_px,
            }
        points = clamp_points_to_bounds(points, shape)
    valid, reason, detail = validate_quadrilateral(points, shape)
    result: Dict[str, Any] = {
        "points": points,
        "candidate_generated": True,
        "candidate_rejection_reason": None,
        "geometry_valid": valid,
        "geometry_rejection_reason": reason,
        "geometry_detail": detail,
        "approach_confidence": float(confidence),
        "boundary_score": 0.0,
        "combined_confidence": 0.0,
        "pattern": None,
        "sampling_scale": None,
        "pattern_skipped_reason": None,
        "corner_clamp_px": clamp_px,
    }
    if valid:
        result["boundary_score"] = float(score_candidate(points, shape))
        if measure_pattern and result["boundary_score"] < MIN_DETECTION_CONFIDENCE * 0.5:
            result["pattern_skipped_reason"] = "boundary_score_below_production_minimum"
        elif (
            measure_pattern
            and processing_gray is not None
            and processing_points is not None
        ):
            pattern, sampling_scale = measure_pattern_best(processing_gray, processing_points)
            result["pattern"] = pattern
            result["sampling_scale"] = sampling_scale
            result["combined_confidence"] = (
                result["boundary_score"] * pattern["score"] if pattern["valid"] else 0.0
            )
    return result


# ---------------------------------------------------------------------------
# Candidate detectors.
# ---------------------------------------------------------------------------


def detect_approach_a_baseline(
    img: np.ndarray, max_candidates: int = 20, clamp_to_bounds: bool = False
) -> Dict[str, Any]:
    """
    Replicates the production detectBoard.ts pipeline: collect quadrilateral
    candidates, apply structural checks, measure checkerboard pattern evidence,
    and rank the shortlist by combined boundary x pattern confidence.
    """
    h_orig, w_orig = img.shape[:2]
    target_w = min(720, w_orig)
    scale = target_w / w_orig
    target_h = int(round(h_orig * scale))

    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges_low = cv2.Canny(blurred, 30, 90)
    edges_high = cv2.Canny(blurred, 60, 180)
    edges = cv2.bitwise_or(edges_low, edges_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[float, np.ndarray]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        pts = None
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        elif len(contour) >= 4:
            rect = cv2.minAreaRect(contour)
            pts = cv2.boxPoints(rect).astype(np.float32)
        if pts is None:
            continue

        area = cv2.contourArea(np.int32(pts))
        if area < 0.05 * target_w * target_h:
            continue
        if not cv2.isContourConvex(np.int32(pts)):
            continue

        candidates.append((area, order_corners_screen_space(pts)))

    if not candidates:
        return _empty_candidate("no_valid_quadrilateral_contours")

    candidates.sort(key=lambda item: item[0], reverse=True)

    # Production evaluates every contour candidate and ranks by combined
    # boundary x pattern confidence. The benchmark caps the shortlist so
    # runtime stays bounded while still allowing a fallback when the largest
    # contour extends outside the frame.
    evaluated: List[Dict[str, Any]] = []
    first_invalid: Optional[Dict[str, Any]] = None
    for _, processing_points in candidates[:max_candidates]:
        points = processing_points / scale
        result = _finish_candidate(
            points, (h_orig, w_orig), 0.0, gray, processing_points,
            clamp_to_bounds=clamp_to_bounds,
        )
        result["approach_confidence"] = result["combined_confidence"]
        if not result["geometry_valid"]:
            if first_invalid is None:
                first_invalid = result
            continue
        evaluated.append(result)

    if evaluated:
        return max(
            evaluated,
            key=lambda item: (item["combined_confidence"], item["boundary_score"]),
        )
    if first_invalid is not None:
        return first_invalid
    return _empty_candidate("no_valid_quadrilateral_contours")


def detect_approach_b_multiscale(img: np.ndarray) -> Dict[str, Any]:
    """
    Approach B: Multi-scale edge detection with morphological closing and convex
    hull smoothing. Confidence 0.35 represents the relaxed any-convex-quad gate;
    the strict production gate is evaluated separately by the caller.
    """
    h_orig, w_orig = img.shape[:2]
    target_w = min(720, w_orig)
    scale = target_w / w_orig
    target_h = int(round(h_orig * scale))

    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edges1 = cv2.Canny(blurred, 40, 120)
    edges2 = cv2.Canny(blurred, 20, 80)
    edges = cv2.bitwise_or(edges1, edges2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.10 * target_w * target_h:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for epsilon in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                points = approx.reshape(4, 2).astype(np.float32)
                candidates.append((area, order_corners_screen_space(points)))
                break

    if not candidates:
        return _empty_candidate("no_convex_hull_quadrilateral_found")

    candidates.sort(key=lambda item: item[0], reverse=True)
    processing_points = candidates[0][1]
    points = processing_points / scale
    result = _finish_candidate(points, (h_orig, w_orig), 0.35, gray, processing_points)
    return result


def detect_approach_c_hough_lines(img: np.ndarray) -> Dict[str, Any]:
    """
    Approach C: Probabilistic Hough Line Transform and orthogonal bounding line
    intersections. This is an experimental proposal generator; it is not a
    production pipeline and does not use pattern evidence.
    """
    h_orig, w_orig = img.shape[:2]
    target_w = min(720, w_orig)
    scale = target_w / w_orig
    target_h = int(round(h_orig * scale))

    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=40, maxLineGap=10)
    if lines is None or len(lines) < 8:
        return _empty_candidate("insufficient_hough_lines")

    lines_arr = np.array(lines).reshape(-1, 4)
    horizontal: List[List[float]] = []
    vertical: List[List[float]] = []
    for x1, y1, x2, y2 in lines_arr:
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(math.atan2(dy, dx) * 180.0 / math.pi)
        if angle < 35 or angle > 145:
            horizontal.append([x1, y1, x2, y2])
        elif 55 < angle < 125:
            vertical.append([x1, y1, x2, y2])

    if len(horizontal) < 2 or len(vertical) < 2:
        return _empty_candidate("cannot_cluster_orthogonal_vanishing_directions")

    all_horizontal = np.array(
        [[line[0], line[1]] for line in horizontal] + [[line[2], line[3]] for line in horizontal]
    )
    all_vertical = np.array(
        [[line[0], line[1]] for line in vertical] + [[line[2], line[3]] for line in vertical]
    )
    all_points = np.vstack([all_horizontal, all_vertical])

    hull = cv2.convexHull(all_points)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.04 * perimeter, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        points = order_corners_screen_space(approx.reshape(4, 2).astype(np.float32) / scale)
        return _finish_candidate(points, (h_orig, w_orig), 0.30, None, None)

    rect = cv2.minAreaRect(hull)
    points = order_corners_screen_space(cv2.boxPoints(rect).astype(np.float32) / scale)
    return _finish_candidate(points, (h_orig, w_orig), 0.20, None, None)


def refine_corners_locally(
    img: np.ndarray, user_taps: np.ndarray
) -> Tuple[np.ndarray, float, str]:
    """Refines coarse corner taps using local Shi-Tomasi detection in a 5% diagonal window."""
    h_orig, w_orig = img.shape[:2]
    diag = math.sqrt(w_orig * w_orig + h_orig * h_orig)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    win_size = int(round(0.05 * diag))
    refined_pts = []

    for pt in user_taps:
        x, y = int(round(pt[0])), int(round(pt[1]))
        x0, x1 = max(0, x - win_size), min(w_orig, x + win_size)
        y0, y1 = max(0, y - win_size), min(h_orig, y + win_size)
        patch = gray[y0:y1, x0:x1]
        corners = cv2.goodFeaturesToTrack(patch, maxCorners=5, qualityLevel=0.04, minDistance=5)
        if corners is not None and len(corners) > 0:
            best_c = None
            best_d = float("inf")
            for corner in corners:
                cx_p, cy_p = corner[0]
                distance = (cx_p - (x - x0)) ** 2 + (cy_p - (y - y0)) ** 2
                if distance < best_d:
                    best_d = distance
                    best_c = (x0 + cx_p, y0 + cy_p)
            refined_pts.append(best_c)
        else:
            refined_pts.append((x, y))

    refined = np.array(refined_pts, dtype=np.float32)
    ordered = order_corners_screen_space(refined)
    return ordered, 0.85, "ok"


def detect_approach_d_assisted(
    img: np.ndarray, gt_corners: np.ndarray, noise_std_diag: float = 0.03
) -> Tuple[np.ndarray, float, str]:
    """
    Approach D: Ground-truth-seeded assisted-calibration simulation.
    Simulates user tapping 4 corners with human noise (std dev = 3% of diagonal),
    followed by local Shi-Tomasi / Harris corner refinement within a 5% window.
    """
    h_orig, w_orig = img.shape[:2]
    diag = math.sqrt(w_orig * w_orig + h_orig * h_orig)
    noise = np.random.normal(0, noise_std_diag * diag, gt_corners.shape).astype(np.float32)
    user_taps = gt_corners + noise
    return refine_corners_locally(img, user_taps)


# ---------------------------------------------------------------------------
# Acceptance gates.
# ---------------------------------------------------------------------------


def production_gate(result: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Production acceptance: structural validity plus combined pattern confidence."""
    if not result["candidate_generated"]:
        return False, result["candidate_rejection_reason"], "candidate"
    if not result["geometry_valid"]:
        return False, result["geometry_rejection_reason"], "geometry"

    pattern = result.get("pattern")
    if pattern is None:
        return (
            False,
            result.get("pattern_skipped_reason") or "pattern_not_measured",
            "acceptance",
        )
    if not pattern["valid"]:
        return False, pattern.get("reason") or "pattern_invalid", "acceptance"
    if result["combined_confidence"] < MIN_DETECTION_CONFIDENCE:
        return False, "combined_confidence_below_production_threshold", "acceptance"
    return True, None, None


def relaxed_geometry_gate(result: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Approach B relaxed gate: any candidate that passes structural geometry checks."""
    if not result["candidate_generated"]:
        return False, result["candidate_rejection_reason"], "candidate"
    if not result["geometry_valid"]:
        return False, result["geometry_rejection_reason"], "geometry"
    return True, None, None


def confidence_gate(result: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Approach C gate: its own confidence threshold, without pattern evidence."""
    if not result["candidate_generated"]:
        return False, result["candidate_rejection_reason"], "candidate"
    if not result["geometry_valid"]:
        return False, result["geometry_rejection_reason"], "geometry"
    if result["approach_confidence"] < MIN_DETECTION_CONFIDENCE:
        return False, "confidence_below_production_threshold", "acceptance"
    return True, None, None


# ---------------------------------------------------------------------------
# Per-image records and summaries.
# ---------------------------------------------------------------------------


def _metric_values(result: Dict[str, Any], gt: np.ndarray, shape: Tuple[int, int]) -> Tuple[float, float]:
    if not result["geometry_valid"]:
        return 0.0, 1.0
    points = result["points"]
    return (
        compute_polygon_iou(points, gt, shape),
        compute_normalized_corner_error(points, gt, shape),
    )


def _make_record(
    approach: str,
    file_name: str,
    view: str,
    result: Dict[str, Any],
    accepted: bool,
    rejection_reason: Optional[str],
    rejection_stage: Optional[str],
    iou: float,
    corner_error: float,
) -> Dict[str, Any]:
    return {
        "approach": approach,
        "file": file_name,
        "view": view,
        "candidate_generated": bool(result["candidate_generated"]),
        "geometry_valid": bool(result["geometry_valid"]),
        "accepted": bool(accepted),
        "rejection_reason": rejection_reason,
        "rejection_stage": rejection_stage,
        "candidate_rejection_reason": result.get("candidate_rejection_reason"),
        "geometry_rejection_reason": result.get("geometry_rejection_reason"),
        "geometry_rejection_detail": result.get("geometry_detail"),
        "iou": float(iou),
        "corner_error": float(corner_error),
        "approach_confidence": float(result.get("approach_confidence", 0.0)),
        "boundary_score": float(result.get("boundary_score", 0.0)),
        "combined_confidence": float(result.get("combined_confidence", 0.0)),
        "pattern": result.get("pattern"),
        "sampling_scale": result.get("sampling_scale"),
        "pattern_skipped_reason": result.get("pattern_skipped_reason"),
        "corner_clamp_px": float(result.get("corner_clamp_px") or 0.0),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if len(values) > 0 else None


def _pct(value: Optional[float]) -> Optional[float]:
    return round(float(value) * 100.0, 4) if value is not None else None


def _quality(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "denominator": len(rows),
        "mean_iou_pct": _pct(_mean([row["iou"] for row in rows])),
        "mean_corner_error_pct": _pct(_mean([row["corner_error"] for row in rows])),
    }


def _viewpoint_breakdown(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for view in ("frontal", "lateral"):
        rows = [record for record in records if record["view"] == view]
        penalized_iou = [
            record["iou"] if record["accepted"] else 0.0 for record in rows
        ]
        penalized_error = [
            record["corner_error"] if record["accepted"] else 1.0 for record in rows
        ]
        result[view] = {
            "denominator": len(rows),
            "accepted_count": sum(1 for record in rows if record["accepted"]),
            "all_input_penalized_mean_iou_pct": _pct(_mean(penalized_iou)),
            "all_input_penalized_mean_corner_error_pct": _pct(_mean(penalized_error)),
        }
    return result


def summarize_approach(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates one approach strictly from its per-image records."""
    total = len(records)
    generated = [record for record in records if record["candidate_generated"]]
    geometry_valid = [record for record in records if record["geometry_valid"]]
    accepted = [record for record in records if record["accepted"]]

    penalized_iou = [record["iou"] if record["accepted"] else 0.0 for record in records]
    penalized_error = [
        record["corner_error"] if record["accepted"] else 1.0 for record in records
    ]

    rejection_counts = Counter(
        record["rejection_reason"] for record in records if not record["accepted"]
    )
    stage_counts = Counter(
        record["rejection_stage"] for record in records if not record["accepted"]
    )

    return {
        "accounting": {
            "total_images": total,
            "candidate_generated_count": len(generated),
            "geometry_valid_count": len(geometry_valid),
            "accepted_count": len(accepted),
            "candidate_generated_pct": _pct(len(generated) / total) if total else None,
            "geometry_valid_pct": _pct(len(geometry_valid) / total) if total else None,
            "accepted_pct": _pct(len(accepted) / total) if total else None,
        },
        "accepted_quality": _quality(accepted),
        "proposal_quality": _quality(geometry_valid),
        "all_input_penalized": {
            "denominator": total,
            "mean_iou_pct": _pct(_mean(penalized_iou)),
            "mean_corner_error_pct": _pct(_mean(penalized_error)),
        },
        "rejection_reasons": dict(rejection_counts.most_common()),
        "rejection_stages": dict(stage_counts.most_common()),
        "viewpoint_breakdown": _viewpoint_breakdown(records),
    }


def summarize_approach_d(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Approach D views with explicit denominators: initial, paired, all-input."""
    total = len(records)
    if total == 0:
        return {"total_images": 0}

    initial_valid = [record for record in records if record["initial"]["geometry_valid"]]
    refined_valid = [record for record in records if record["geometry_valid"]]
    paired = [
        record
        for record in records
        if record["initial"]["geometry_valid"] and record["geometry_valid"]
    ]

    initial_penalized_iou = [
        record["initial"]["iou"] if record["initial"]["geometry_valid"] else 0.0
        for record in records
    ]
    initial_penalized_error = [
        record["initial"]["corner_error"] if record["initial"]["geometry_valid"] else 1.0
        for record in records
    ]
    refined_penalized_iou = [record["iou"] if record["accepted"] else 0.0 for record in records]
    refined_penalized_error = [
        record["corner_error"] if record["accepted"] else 1.0 for record in records
    ]

    paired_initial_iou = [record["initial"]["iou"] for record in paired]
    paired_refined_iou = [record["iou"] for record in paired]
    paired_initial_error = [record["initial"]["corner_error"] for record in paired]
    paired_refined_error = [record["corner_error"] for record in paired]
    deltas = [
        refined - initial for initial, refined in zip(paired_initial_iou, paired_refined_iou)
    ]

    return {
        "total_images": total,
        "initial_baseline_all_input": {
            "denominator": total,
            "geometry_valid_count": len(initial_valid),
            "mean_iou_pct": _pct(_mean(initial_penalized_iou)),
            "mean_corner_error_pct": _pct(_mean(initial_penalized_error)),
        },
        "refined_all_input_penalized": {
            "denominator": total,
            "accepted_count": sum(1 for record in records if record["accepted"]),
            "mean_iou_pct": _pct(_mean(refined_penalized_iou)),
            "mean_corner_error_pct": _pct(_mean(refined_penalized_error)),
        },
        "refined_accepted_quality": _quality(
            [record for record in records if record["accepted"]]
        ),
        "paired_before_after": {
            "denominator": len(paired),
            "excluded_count": total - len(paired),
            "excluded_reasons": dict(
                Counter(
                    (
                        "initial:"
                        + (
                            record["initial"].get("geometry_rejection_reason")
                            or "unknown"
                        )
                        if not record["initial"]["geometry_valid"]
                        else "refined:"
                        + (record.get("geometry_rejection_reason") or "unknown")
                    )
                    for record in records
                    if record not in paired
                ).most_common()
            ),
            "initial_mean_iou_pct": _pct(_mean(paired_initial_iou)),
            "refined_mean_iou_pct": _pct(_mean(paired_refined_iou)),
            "mean_delta_iou_pct_points": _pct(_mean(deltas)),
            "initial_mean_corner_error_pct": _pct(_mean(paired_initial_error)),
            "refined_mean_corner_error_pct": _pct(_mean(paired_refined_error)),
            "mean_delta_corner_error_pct_points": _pct(
                _mean(
                    [
                        refined - initial
                        for initial, refined in zip(paired_initial_error, paired_refined_error)
                    ]
                )
            ),
            "improved_iou_count": sum(1 for delta in deltas if delta > 1e-9),
            "worsened_iou_count": sum(1 for delta in deltas if delta < -1e-9),
            "unchanged_iou_count": sum(1 for delta in deltas if abs(delta) <= 1e-9),
        },
        "rejection_reasons": dict(
            Counter(
                record["rejection_reason"] for record in records if not record["accepted"]
            ).most_common()
        ),
        "simulation_disclosure": {
            "is_synthetic_simulation": True,
            "input_source": "FENiT ground-truth corner annotations with synthetic Gaussian touch perturbation",
            "touch_noise_std_diag": 0.03,
            "random_seed": 42,
            "refinement_method": "local_shi_tomasi_good_features_to_track_5pct_search_window",
            "disclosure_note": (
                "This metric evaluates algorithm snapping on perturbed ground-truth; "
                "it is NOT measured human-operator device performance."
            ),
        },
    }


def assert_summary_matches_records(
    section: Dict[str, Any], records: Sequence[Dict[str, Any]]
) -> None:
    """Internal guard: every aggregate field must be reproducible from records."""
    expected = summarize_approach(records)
    if section != expected:
        raise AssertionError(
            "Summary does not match its per-image records: "
            f"summary={json.dumps(section, sort_keys=True)} "
            f"records={json.dumps(expected, sort_keys=True)}"
        )


def assert_d_summary_matches_records(
    section: Dict[str, Any], records: Sequence[Dict[str, Any]]
) -> None:
    expected = summarize_approach_d(records)
    if section != expected:
        raise AssertionError("Approach D summary does not match its per-image records.")


# ---------------------------------------------------------------------------
# Benchmark driver.
# ---------------------------------------------------------------------------


def _unreadable_records(file_name: str, view: str) -> List[Dict[str, Any]]:
    def failure(approach: str) -> Dict[str, Any]:
        return {
            "approach": approach,
            "file": file_name,
            "view": view,
            "candidate_generated": False,
            "geometry_valid": False,
            "accepted": False,
            "rejection_reason": "image_unreadable",
            "rejection_stage": "candidate",
            "candidate_rejection_reason": "image_unreadable",
            "geometry_rejection_reason": "no_candidate",
            "geometry_rejection_detail": "The image could not be decoded.",
            "iou": 0.0,
            "corner_error": 1.0,
            "approach_confidence": 0.0,
            "boundary_score": 0.0,
            "combined_confidence": 0.0,
            "pattern": None,
            "sampling_scale": None,
            "pattern_skipped_reason": None,
        }

    records = [failure(approach) for approach in ("a_production", "b_relaxed", "c_hough")]
    records.append(
        {
            **failure("d_simulated"),
            "initial": {
                "candidate_generated": False,
                "geometry_valid": False,
                "geometry_rejection_reason": "no_candidate",
                "rejection_reason": "image_unreadable",
                "iou": 0.0,
                "corner_error": 1.0,
            },
        }
    )
    return records


def run_geometry_benchmark(
    manifest_path: Path, raw_dir: Path, sample_limit: int = 0
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    recs = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            recs.append(json.loads(line))

    processed = [record for record in recs if record.get("status") == "PROCESSED"]
    if sample_limit > 0:
        processed = processed[:sample_limit]

    total_images = len(processed)
    print(f"Benchmarking geometry across {total_images} eligible FENiT board scenes...")

    np.random.seed(42)

    records: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for idx, manifest_record in enumerate(processed, 1):
        file_name = manifest_record["file"]
        img_path = raw_dir / file_name
        img = cv2.imread(str(img_path))
        orientation = manifest_record.get("resolved_orientation", "unknown")
        view = "frontal" if orientation in {"white_perspective", "black_perspective"} else "lateral"

        if img is None:
            records.extend(_unreadable_records(file_name, view))
            continue

        height, width = img.shape[:2]
        shape = (height, width)

        gt = np.array(manifest_record["board_corners"], dtype=np.float32)
        gt[:, 0] *= width
        gt[:, 1] *= height
        gt_ordered = order_corners_screen_space(gt)

        # Approach A: production baseline.
        result_a = detect_approach_a_baseline(img)
        accepted_a, reason_a, stage_a = production_gate(result_a)
        iou_a, err_a = _metric_values(result_a, gt_ordered, shape)
        records.append(
            _make_record(
                "a_production", file_name, view, result_a, accepted_a, reason_a, stage_a, iou_a, err_a
            )
        )

        # Approach B: relaxed gate and strict production gate from the same proposal.
        result_b = detect_approach_b_multiscale(img)
        accepted_b_relaxed, reason_b_relaxed, stage_b_relaxed = relaxed_geometry_gate(result_b)
        accepted_b_strict, reason_b_strict, stage_b_strict = production_gate(result_b)
        iou_b, err_b = _metric_values(result_b, gt_ordered, shape)
        record_b = _make_record(
            "b_relaxed",
            file_name,
            view,
            result_b,
            accepted_b_relaxed,
            reason_b_relaxed,
            stage_b_relaxed,
            iou_b,
            err_b,
        )
        record_b["strict_production"] = {
            "accepted": bool(accepted_b_strict),
            "rejection_reason": reason_b_strict,
            "rejection_stage": stage_b_strict,
            "combined_confidence": float(result_b["combined_confidence"]),
        }
        records.append(record_b)

        # Approach C: experimental Hough proposal generator.
        result_c = detect_approach_c_hough_lines(img)
        accepted_c, reason_c, stage_c = confidence_gate(result_c)
        iou_c, err_c = _metric_values(result_c, gt_ordered, shape)
        records.append(
            _make_record(
                "c_hough", file_name, view, result_c, accepted_c, reason_c, stage_c, iou_c, err_c
            )
        )

        # Approach D: ground-truth-seeded assisted calibration simulation.
        diag = math.sqrt(width * width + height * height)
        noise = np.random.normal(0, 0.03 * diag, gt_ordered.shape).astype(np.float32)
        user_taps = gt_ordered + noise
        # A real preview touch cannot leave the image. Clamp simulated taps to
        # the decoded-image boundary so the initial baseline is not penalized by
        # an impossible out-of-frame tap that the UI could never produce.
        user_taps[:, 0] = np.clip(user_taps[:, 0], 0, width)
        user_taps[:, 1] = np.clip(user_taps[:, 1], 0, height)

        initial_valid, initial_reason, _ = validate_quadrilateral(user_taps, shape)
        initial_iou = (
            compute_polygon_iou(user_taps, gt_ordered, shape) if initial_valid else 0.0
        )
        initial_error = (
            compute_normalized_corner_error(user_taps, gt_ordered, shape) if initial_valid else 1.0
        )

        refined_points, _, _ = refine_corners_locally(img, user_taps)
        refined_valid, refined_reason, refined_detail = validate_quadrilateral(refined_points, shape)
        refined_iou = (
            compute_polygon_iou(refined_points, gt_ordered, shape) if refined_valid else 0.0
        )
        refined_error = (
            compute_normalized_corner_error(refined_points, gt_ordered, shape)
            if refined_valid
            else 1.0
        )

        records.append(
            {
                "approach": "d_simulated",
                "file": file_name,
                "view": view,
                "candidate_generated": True,
                "geometry_valid": bool(refined_valid),
                "accepted": bool(refined_valid),
                "rejection_reason": refined_reason,
                "rejection_stage": None if refined_valid else "geometry",
                "candidate_rejection_reason": None,
                "geometry_rejection_reason": refined_reason,
                "geometry_rejection_detail": refined_detail,
                "iou": float(refined_iou),
                "corner_error": float(refined_error),
                "approach_confidence": 0.85,
                "boundary_score": 0.0,
                "combined_confidence": 0.0,
                "pattern": None,
                "sampling_scale": None,
                "initial": {
                    "candidate_generated": True,
                    "geometry_valid": bool(initial_valid),
                    "geometry_rejection_reason": initial_reason,
                    "rejection_reason": initial_reason,
                    "iou": float(initial_iou),
                    "corner_error": float(initial_error),
                },
            }
        )

        if idx % 50 == 0 or idx == total_images:
            print(f"  [{idx}/{total_images}] completed...", flush=True)

    elapsed = time.perf_counter() - t0

    records_a = [record for record in records if record["approach"] == "a_production"]
    records_b = [record for record in records if record["approach"] == "b_relaxed"]
    records_c = [record for record in records if record["approach"] == "c_hough"]
    records_d = [record for record in records if record["approach"] == "d_simulated"]

    strict_b_records = [
        {
            **record,
            "accepted": record["strict_production"]["accepted"],
            "rejection_reason": record["strict_production"]["rejection_reason"],
            "rejection_stage": record["strict_production"]["rejection_stage"],
            "approach_confidence": record["strict_production"]["combined_confidence"],
        }
        for record in records_b
    ]

    summary = {
        "benchmark_metadata": {
            "benchmark_version": BENCHMARK_VERSION,
            "dataset": "FENiT (Mendeley Data DOI 10.17632/wpp9xnbcp6.1)",
            "images_evaluated": total_images,
            "records_written": len(records),
            "total_runtime_s": round(elapsed, 2),
            "date": "2026-11-03",
            "metric_definitions": {
                "accepted_quality": "Conditional mean over accepted records only.",
                "proposal_quality": "Conditional mean over geometry-valid proposals only.",
                "all_input_penalized": "Mean over every input; rejected or invalid proposals contribute IoU 0.0 and corner error 1.0.",
            },
            "acceptance_gates": {
                "a_production": "Production structural checks + 8x8 checkerboard pattern evidence + boundary*pattern >= 0.32.",
                "b_relaxed": "Any structurally valid convex quadrilateral (experimental; not production performance).",
                "b_strict_production": "Same structural checks and checkerboard pattern evidence as Approach A.",
                "c_hough": "Experimental Hough proposal confidence >= 0.32; no pattern evidence.",
                "d_simulated": "Ground-truth-seeded simulation; accepted when refined corners remain structurally valid.",
            },
        },
        "approach_a_production_baseline": summarize_approach(records_a),
        "approach_b_relaxed": summarize_approach(records_b),
        "approach_b_strict_production": summarize_approach(strict_b_records),
        "approach_c_hough_experimental": summarize_approach(records_c),
        "approach_d_simulated_assisted_calibration": summarize_approach_d(records_d),
    }

    assert_summary_matches_records(summary["approach_a_production_baseline"], records_a)
    assert_summary_matches_records(summary["approach_b_relaxed"], records_b)
    assert_summary_matches_records(summary["approach_b_strict_production"], strict_b_records)
    assert_summary_matches_records(summary["approach_c_hough_experimental"], records_c)
    assert_d_summary_matches_records(
        summary["approach_d_simulated_assisted_calibration"], records_d
    )

    return summary, records


def write_records(path: Path, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lines = [json.dumps(record, sort_keys=True) for record in records]
    payload = "\n".join(lines) + ("\n" if lines else "")
    payload_bytes = payload.encode("utf-8")
    path.write_bytes(payload_bytes)
    return {
        "records_file": str(path),
        "records_count": len(records),
        "records_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-image board geometry and calibration benchmark.")
    parser.add_argument(
        "--manifest",
        default=r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl",
        help="Path to external benchmark manifest",
    )
    parser.add_argument(
        "--raw-dir",
        default=r"C:\dev\chess-capture-workspace\external\fenit\raw",
        help="Path to FENiT raw images",
    )
    parser.add_argument(
        "--out",
        default=r"C:\dev\chess-capture-workspace\external\fenit\reports\board_geometry_benchmark.json",
        help="Path to output benchmark JSON report",
    )
    parser.add_argument(
        "--records-out",
        default="",
        help="Path to per-image JSONL records (default: <out stem>_records.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of boards to benchmark (0 = all 385)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records_path = (
        Path(args.records_out)
        if args.records_out
        else out_path.with_name(f"{out_path.stem}_records.jsonl")
    )

    print("========================================================================================")
    print("REAL-IMAGE BOARD GEOMETRY AND PERSPECTIVE CALIBRATION BENCHMARK")
    print("========================================================================================")
    print(f"Manifest: {manifest_path}")
    print(f"Raw Dir:  {raw_dir}")
    print(f"Output:   {out_path}")
    print(f"Records:  {records_path}")

    summary, records = run_geometry_benchmark(manifest_path, raw_dir, sample_limit=args.limit)
    summary["benchmark_metadata"].update(write_records(records_path, records))

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBenchmark completed successfully in {summary['benchmark_metadata']['total_runtime_s']} s.")
    print(f"Report written to: {out_path}")
    print(f"Records written to: {records_path}\n")

    print("========================================================================================")
    print(f" GEOMETRY BENCHMARK SUMMARY ({summary['benchmark_metadata']['images_evaluated']} Real FENiT Boards)")
    print("========================================================================================")
    print("\n--- Part 1: Per-Approach Accounting ---")
    header_acc = (
        f"{'Approach':<38} {'Candidates':>11} {'Geometry':>10} {'Accepted':>10} {'Accepted IoU':>14} {'All-input IoU':>15}"
    )
    print(header_acc)
    print("-" * len(header_acc))
    for name, key in [
        ("Approach A: Production", "approach_a_production_baseline"),
        ("Approach B: Relaxed (NOT production)", "approach_b_relaxed"),
        ("Approach B: Strict production gate", "approach_b_strict_production"),
        ("Approach C: Hough (experimental)", "approach_c_hough_experimental"),
    ]:
        section = summary[key]
        accounting = section["accounting"]
        accepted_iou = section["accepted_quality"]["mean_iou_pct"]
        all_input_iou = section["all_input_penalized"]["mean_iou_pct"]
        accepted_text = f"{accepted_iou:.2f}%" if accepted_iou is not None else "N/A"
        all_input_text = f"{all_input_iou:.2f}%" if all_input_iou is not None else "N/A"
        print(
            f"{name:<38} {accounting['candidate_generated_count']:>5}/{accounting['total_images']:<5} "
            f"{accounting['geometry_valid_count']:>4}/{accounting['total_images']:<5} "
            f"{accounting['accepted_count']:>4}/{accounting['total_images']:<5} "
            f"{accepted_text:>14} {all_input_text:>15}"
        )

    print("\n--- Part 2: Conditional vs All-Input Corner Error ---")
    header_err = f"{'Approach':<38} {'Accepted Err':>14} {'All-input Err':>15}"
    print(header_err)
    print("-" * len(header_err))
    for name, key in [
        ("Approach A: Production", "approach_a_production_baseline"),
        ("Approach B: Relaxed (NOT production)", "approach_b_relaxed"),
        ("Approach B: Strict production gate", "approach_b_strict_production"),
        ("Approach C: Hough (experimental)", "approach_c_hough_experimental"),
    ]:
        section = summary[key]
        accepted_err = section["accepted_quality"]["mean_corner_error_pct"]
        all_input_err = section["all_input_penalized"]["mean_corner_error_pct"]
        accepted_text = f"{accepted_err:.2f}%" if accepted_err is not None else "N/A"
        all_input_text = f"{all_input_err:.2f}%" if all_input_err is not None else "N/A"
        print(f"{name:<38} {accepted_text:>14} {all_input_text:>15}")

    print("\n--- Part 3: Rejection Reasons ---")
    for name, key in [
        ("Approach A: Production", "approach_a_production_baseline"),
        ("Approach B: Relaxed", "approach_b_relaxed"),
        ("Approach B: Strict production", "approach_b_strict_production"),
        ("Approach C: Hough", "approach_c_hough_experimental"),
    ]:
        print(f"{name}: {summary[key]['rejection_reasons']}")

    app_d = summary["approach_d_simulated_assisted_calibration"]
    print("\n--- Part 4: Ground-Truth-Seeded Assisted Calibration Simulation (Approach D) ---")
    initial = app_d["initial_baseline_all_input"]
    refined = app_d["refined_all_input_penalized"]
    paired = app_d["paired_before_after"]
    print(
        f"Initial baseline (N={initial['denominator']}): "
        f"IoU {initial['mean_iou_pct']:.2f}%, error {initial['mean_corner_error_pct']:.2f}%"
    )
    print(
        f"Refined all-input (N={refined['denominator']}, accepted={refined['accepted_count']}): "
        f"IoU {refined['mean_iou_pct']:.2f}%, error {refined['mean_corner_error_pct']:.2f}%"
    )
    print(
        f"Paired before/after (N={paired['denominator']}, excluded={paired['excluded_count']}): "
        f"IoU {paired['initial_mean_iou_pct']:.2f}% -> {paired['refined_mean_iou_pct']:.2f}% "
        f"(delta {paired['mean_delta_iou_pct_points']:+.2f} points), "
        f"error {paired['initial_mean_corner_error_pct']:.2f}% -> "
        f"{paired['refined_mean_corner_error_pct']:.2f}%"
    )
    print(
        f"Paired outcomes: improved {paired['improved_iou_count']}, "
        f"worsened {paired['worsened_iou_count']}, unchanged {paired['unchanged_iou_count']}"
    )
    print(f"Rejections: {app_d['rejection_reasons']}")
    print(
        "[DISCLOSURE: Ground-truth-seeded simulation only. "
        "This is NOT measured human-operator device performance.]"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
