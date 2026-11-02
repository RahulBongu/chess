#!/usr/bin/env python3
"""
extract_square_crops.py - Milestone 5.2H Offline Crop Extraction Engine

Extracts exactly the crops required by a capture category from an already-rectified
800x800 board image:

  full_board_context   -> 64 square crops from one full-board scene
  occupied_target_only -> 1 target crop from one isolated sweep/reference photo
  empty_board_grid     -> 64 empty-square crops from one empty-board photo

The tool refuses to invent physical piece identities. Full-board scenes require a
JSON piece-instance map (square -> canonical SETxx_W_Pxx identifier). Non-800x800
inputs require explicit corner coordinates; the tool never silently resizes an
oblique raw photograph into an 800x800 "rectified" board.

Usage:
    py scripts/extract_square_crops.py \\
        --image path/to/rectified_board.png \\
        --mode full_board_context \\
        --placement "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" \\
        --piece-map path/to/starting_piece_map.json \\
        --out-dir C:/dev/chess-capture-workspace/crops/train/SET01 \\
        --set-id SET01 \\
        --scene-id SET01_SCENE_01 \\
        --camera-angle zenith \\
        --lighting diffuse_daylight \\
        --manifest-out C:/dev/chess-capture-workspace/manifests/SET01_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from validate_capture_manifest import (
    canonical_to_physical_tag,
    count_pieces_in_placement,
    validate_complete_fen,
    validate_placement_only,
)

CANONICAL_BOARD_SIZE = 800
CANONICAL_SQUARE_SIZE = 100
GRID_DIM = 8

EMPTY_PLACEMENT = "8/8/8/8/8/8/8/8"
SQUARE_REGEX = re.compile(r"^[a-h][1-8]$")
CANONICAL_PIECE_REGEX = re.compile(r"^SET([0-9]{2})_([WB])_([PNBRQK])([0-9]{2})$")
SET_ID_REGEX = re.compile(r"^SET[0-9]{2}$")
SCENE_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")

CROP_TYPE_BY_MODE = {
    "full_board_context": "full_board_context",
    "occupied_target_only": "isolated_piece",
    "empty_board_grid": "empty_square",
}

CHAR_TO_PIECE_TYPE = {
    "p": "pawn",
    "P": "pawn",
    "n": "knight",
    "N": "knight",
    "b": "bishop",
    "B": "bishop",
    "r": "rook",
    "R": "rook",
    "q": "queen",
    "Q": "queen",
    "k": "king",
    "K": "king",
}

TYPE_CHAR_TO_PIECE_TYPE = {
    "P": "pawn",
    "N": "knight",
    "B": "bishop",
    "R": "rook",
    "Q": "queen",
    "K": "king",
}


class ExtractionError(ValueError):
    """Raised when capture metadata or extraction mode is invalid."""


def parse_placement_to_square_map(placement: str) -> Dict[str, Dict[str, str]]:
    """Strictly parse an 8-rank placement; malformed input raises ExtractionError."""
    valid, error = validate_placement_only(placement)
    if not valid:
        raise ExtractionError(f"Invalid placement: {error}")

    sq_map: Dict[str, Dict[str, str]] = {}
    ranks = placement.strip().split("/")

    for r_idx in range(8):
        rank_num = 8 - r_idx
        rank_str = ranks[r_idx]
        file_idx = 0

        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            elif ch in "pnbrqkPNBRQK":
                file_char = chr(97 + file_idx)
                sq = f"{file_char}{rank_num}"
                sq_map[sq] = {
                    "piece_char": ch,
                    "piece_type": CHAR_TO_PIECE_TYPE[ch],
                    "piece_color": "white" if ch.isupper() else "black",
                }
                file_idx += 1

    return sq_map


def create_empty_square_map() -> Dict[str, str]:
    return {
        f"{chr(97 + file_idx)}{rank}": "EMPTY"
        for rank in range(1, 9)
        for file_idx in range(8)
    }


def parse_canonical_piece_id(piece_instance_id: str) -> Dict[str, str]:
    match = CANONICAL_PIECE_REGEX.match(piece_instance_id)
    if not match:
        raise ExtractionError(
            f"Invalid piece_instance_id '{piece_instance_id}'. Expected canonical format SETxx_W_P01."
        )
    return {
        "set_id": f"SET{match.group(1)}",
        "color_char": match.group(2),
        "type_char": match.group(3),
        "piece_type": TYPE_CHAR_TO_PIECE_TYPE[match.group(3)],
    }


def assert_piece_identity(
    piece_instance_id: str,
    expected: Dict[str, str],
    set_id: Optional[str],
    square: str,
) -> None:
    parsed = parse_canonical_piece_id(piece_instance_id)
    expected_color_char = "W" if expected["piece_color"] == "white" else "B"
    if parsed["color_char"] != expected_color_char:
        raise ExtractionError(
            f"Piece identity '{piece_instance_id}' at {square} has color "
            f"'{parsed['color_char']}', expected '{expected_color_char}'."
        )
    if parsed["piece_type"] != expected["piece_type"]:
        raise ExtractionError(
            f"Piece identity '{piece_instance_id}' at {square} is a "
            f"{parsed['piece_type']}, expected {expected['piece_type']}."
        )
    if set_id and parsed["set_id"] != set_id:
        raise ExtractionError(
            f"Piece identity '{piece_instance_id}' belongs to {parsed['set_id']}, expected set {set_id}."
        )


def resolve_set_id(piece_map: Dict[str, str], set_id: Optional[str]) -> str:
    if set_id is not None:
        if not SET_ID_REGEX.match(set_id):
            raise ExtractionError(f"Invalid set_id '{set_id}'. Expected SET01-SET20.")
        return set_id

    first_id = next((value for value in piece_map.values() if value and value != "EMPTY"), None)
    if not first_id:
        raise ExtractionError("Cannot infer set_id from an empty piece instance map.")
    return parse_canonical_piece_id(first_id)["set_id"]


def validate_full_board_piece_map(
    piece_map: Dict[str, str],
    placement_map: Dict[str, Dict[str, str]],
    set_id: Optional[str],
) -> Dict[str, str]:
    if not isinstance(piece_map, dict):
        raise ExtractionError("full_board_context requires a pieceInstanceMap object.")

    resolved_set_id = resolve_set_id(piece_map, set_id)
    seen_instances = set()
    validated: Dict[str, str] = {}

    for square, raw_id in piece_map.items():
        if not SQUARE_REGEX.match(str(square)):
            raise ExtractionError(f"Invalid square '{square}' in pieceInstanceMap.")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ExtractionError(f"Empty piece instance ID for square {square}.")
        piece_id = raw_id.strip()
        expected = placement_map.get(square)

        if expected is None:
            if piece_id != "EMPTY":
                raise ExtractionError(
                    f"pieceInstanceMap assigns '{piece_id}' to empty square {square}; FEN says the square is empty."
                )
            validated[square] = "EMPTY"
            continue

        if piece_id == "EMPTY":
            raise ExtractionError(f"pieceInstanceMap leaves occupied square {square} as EMPTY.")

        assert_piece_identity(piece_id, expected, resolved_set_id, square)
        if piece_id in seen_instances:
            raise ExtractionError(
                f"piece_instance_id '{piece_id}' appears on multiple squares in one scene; "
                "a physical piece can occupy only one square."
            )
        seen_instances.add(piece_id)
        validated[square] = piece_id

    for square in placement_map:
        if square not in validated:
            raise ExtractionError(f"pieceInstanceMap is missing occupied square {square}.")

    # Empty squares default to EMPTY.
    for square, value in create_empty_square_map().items():
        validated.setdefault(square, value)

    return validated


def build_crop_filename(scene_id: str, square: str, piece_instance_id: str) -> str:
    if not SCENE_ID_REGEX.match(scene_id):
        raise ExtractionError(
            f"Invalid scene_id '{scene_id}'. Use letters, digits, underscore, or hyphen only."
        )
    if not SQUARE_REGEX.match(square):
        raise ExtractionError(f"Invalid square '{square}'.")
    if piece_instance_id != "EMPTY" and not CANONICAL_PIECE_REGEX.match(piece_instance_id):
        raise ExtractionError(f"Invalid piece_instance_id '{piece_instance_id}'.")
    return f"{scene_id}_{square}_{piece_instance_id}.png"


def extract_crops_from_rectified(
    image: np.ndarray,
    placement: Optional[str] = None,
    orientation: str = "white-bottom",
) -> List[Dict[str, Any]]:
    """Slices all 64 square patches. Mode selection happens in plan_crop_extraction."""
    if image.shape[0] != CANONICAL_BOARD_SIZE or image.shape[1] != CANONICAL_BOARD_SIZE:
        raise ExtractionError(
            f"Image dimensions {image.shape[1]}x{image.shape[0]} do not match canonical 800x800."
        )

    piece_map = parse_placement_to_square_map(placement) if placement else {}
    crops: List[Dict[str, Any]] = []

    for row in range(GRID_DIM):
        for col in range(GRID_DIM):
            if orientation == "white-bottom":
                file_idx = col
                rank_idx = 7 - row
            elif orientation == "black-bottom":
                file_idx = 7 - col
                rank_idx = row
            else:
                raise ExtractionError(
                    f"Unknown orientation '{orientation}'. Expected white-bottom or black-bottom."
                )

            file_char = chr(97 + file_idx)
            rank_num = rank_idx + 1
            square = f"{file_char}{rank_num}"

            start_x = col * CANONICAL_SQUARE_SIZE
            start_y = row * CANONICAL_SQUARE_SIZE
            patch = image[
                start_y : start_y + CANONICAL_SQUARE_SIZE,
                start_x : start_x + CANONICAL_SQUARE_SIZE,
            ]

            piece_info = piece_map.get(
                square,
                {"piece_char": None, "piece_type": "empty", "piece_color": "none"},
            )
            crops.append(
                {
                    "square": square,
                    "file_index": file_idx,
                    "rank_index": rank_idx,
                    "col": col,
                    "row": row,
                    "bounds": {
                        "x": start_x,
                        "y": start_y,
                        "width": CANONICAL_SQUARE_SIZE,
                        "height": CANONICAL_SQUARE_SIZE,
                    },
                    "patch": patch,
                    "piece_type": piece_info["piece_type"],
                    "piece_color": piece_info["piece_color"],
                    "piece_char": piece_info["piece_char"],
                }
            )

    return crops


def _placement_from_ground_truth(
    placement: Optional[str], board_fen: Optional[str]
) -> Optional[str]:
    if placement and board_fen:
        raise ExtractionError("Provide --placement or --board-fen, not both.")
    if board_fen:
        valid, error = validate_complete_fen(board_fen)
        if not valid:
            raise ExtractionError(f"Invalid complete board_fen: {error}")
        return board_fen.strip().split()[0]
    if placement:
        return placement.strip()
    return None


def plan_crop_extraction(
    image: np.ndarray,
    mode: str,
    placement: Optional[str] = None,
    board_fen: Optional[str] = None,
    orientation: str = "white-bottom",
    set_id: Optional[str] = None,
    target_square: Optional[str] = None,
    target_piece_instance_id: Optional[str] = None,
    piece_instance_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Validate ground truth and return exactly the crops required by the mode."""
    if mode not in CROP_TYPE_BY_MODE:
        raise ExtractionError(f"Unknown extraction mode '{mode}'.")

    placement_source = _placement_from_ground_truth(placement, board_fen)

    if mode == "empty_board_grid":
        placement_source = placement_source or EMPTY_PLACEMENT
        placement_map = parse_placement_to_square_map(placement_source)
        if placement_map:
            raise ExtractionError("empty_board_grid requires an empty placement with zero occupied squares.")
        id_map = create_empty_square_map()
    else:
        if not placement_source:
            raise ExtractionError(f"{mode} requires --placement or --board-fen ground truth.")
        placement_map = parse_placement_to_square_map(placement_source)

        if mode == "occupied_target_only":
            if not target_square or not SQUARE_REGEX.match(target_square):
                raise ExtractionError("occupied_target_only requires a valid --target-square (a1-h8).")
            if not target_piece_instance_id:
                raise ExtractionError("occupied_target_only requires --piece-instance-id.")
            if len(placement_map) != 1:
                raise ExtractionError(
                    "occupied_target_only requires exactly one occupied square, "
                    f"got {len(placement_map)}."
                )
            occupied_square = next(iter(placement_map))
            if occupied_square != target_square:
                raise ExtractionError(
                    f"Target square '{target_square}' does not match the only occupied square "
                    f"'{occupied_square}'."
                )
            assert_piece_identity(
                target_piece_instance_id,
                placement_map[occupied_square],
                set_id,
                occupied_square,
            )
            id_map = create_empty_square_map()
            id_map[occupied_square] = target_piece_instance_id
        else:  # full_board_context
            if not placement_map:
                raise ExtractionError("full_board_context requires at least one occupied square.")
            if piece_instance_map is None:
                raise ExtractionError(
                    "full_board_context requires --piece-map; refusing to invent physical "
                    "piece identities from FEN types."
                )
            id_map = validate_full_board_piece_map(piece_instance_map, placement_map, set_id)

    crops = extract_crops_from_rectified(image, placement=placement_source, orientation=orientation)

    selected: List[Dict[str, Any]] = []
    for crop in crops:
        square = crop["square"]
        if mode == "occupied_target_only" and square != target_square:
            continue
        piece_instance_id = id_map[square]
        physical_tag_id = canonical_to_physical_tag(piece_instance_id)
        if physical_tag_id == "INVALID":
            raise ExtractionError(f"Invalid canonical piece identity '{piece_instance_id}' for {square}.")
        selected.append(
            {
                **crop,
                "piece_instance_id": piece_instance_id,
                "physical_tag_id": physical_tag_id,
                "crop_type": CROP_TYPE_BY_MODE[mode],
                "extraction_mode": mode,
            }
        )

    expected_count = 1 if mode == "occupied_target_only" else 64
    if len(selected) != expected_count:
        raise ExtractionError(f"{mode} must emit {expected_count} crops, got {len(selected)}.")

    return {
        "mode": mode,
        "crop_type": CROP_TYPE_BY_MODE[mode],
        "placement": placement_source,
        "source_photo_count": 1,
        "derived_crop_count": len(selected),
        "crops": selected,
    }


def load_piece_map(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as error:
        raise ExtractionError(f"Could not read piece map '{path}': {error}") from error
    except json.JSONDecodeError as error:
        raise ExtractionError(f"Invalid JSON in piece map '{path}': {error}") from error

    if not isinstance(data, dict):
        raise ExtractionError("Piece map must be a JSON object mapping squares to piece instance IDs.")
    return data


def rectify_with_corners(image: np.ndarray, corners_arg: str) -> np.ndarray:
    try:
        points = [float(value.strip()) for value in corners_arg.split(",")]
    except ValueError as error:
        raise ExtractionError("--corners must contain numeric x,y values.") from error
    if len(points) != 8:
        raise ExtractionError("--corners must contain exactly 8 numbers (x1,y1,...,x4,y4).")

    src = np.array(points, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [
            [0, 0],
            [CANONICAL_BOARD_SIZE, 0],
            [CANONICAL_BOARD_SIZE, CANONICAL_BOARD_SIZE],
            [0, CANONICAL_BOARD_SIZE],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (CANONICAL_BOARD_SIZE, CANONICAL_BOARD_SIZE))


def write_crops_and_manifest(
    plan: Dict[str, Any],
    image_path: str,
    out_dir: str,
    scene_id: str,
    set_id: str,
    split: str,
    board_fen: Optional[str],
    camera_angle: str,
    lighting_setup: str,
    photographer: str,
    rights_holder: str,
    release_record: str,
    occlusion_level: str,
    annotation_status: str,
    quality_notes: str,
    manifest_out: Optional[str],
    manifest_mode: str,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []
    filenames = set()
    output_paths = set()

    for crop in plan["crops"]:
        filename = build_crop_filename(scene_id, crop["square"], crop["piece_instance_id"])
        if filename in filenames:
            raise ExtractionError(f"Duplicate derived crop filename '{filename}'.")
        filenames.add(filename)

        crop_path = os.path.abspath(os.path.join(out_dir, filename))
        if crop_path in output_paths:
            raise ExtractionError(f"Duplicate derived crop path '{crop_path}'.")
        output_paths.add(crop_path)

        if os.path.exists(crop_path) and not overwrite:
            raise ExtractionError(
                f"Refusing to overwrite existing crop '{crop_path}'. Pass --overwrite to replace it."
            )

        if not cv2.imwrite(crop_path, crop["patch"]):
            raise ExtractionError(f"Failed to write crop '{crop_path}'.")

        records.append(
            {
                "dataset_version": "1.0.0",
                "source_id": "author_capture_2026",
                "set_id": set_id,
                "split": split,
                "piece_instance_id": crop["piece_instance_id"],
                "physical_tag_id": crop["physical_tag_id"],
                "piece_type": crop["piece_type"],
                "piece_color": crop["piece_color"],
                "scene_id": scene_id,
                "board_fen": board_fen,
                "piece_placement": plan["placement"],
                "square_coord": crop["square"],
                "camera_angle": camera_angle,
                "lighting_setup": lighting_setup,
                "original_image_path": os.path.abspath(image_path),
                "derived_crop_path": crop_path,
                "crop_type": plan["crop_type"],
                "extraction_mode": plan["mode"],
                "occlusion_level": occlusion_level,
                "photographer": photographer,
                "rights_holder": rights_holder,
                "release_record": release_record,
                "data_license": "CC0-1.0",
                "code_license": "MIT",
                "permission_status": "verified_author_consent",
                "annotation_status": annotation_status,
                "quality_notes": quality_notes,
            }
        )

    if manifest_out:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_out)), exist_ok=True)
        file_mode = "w" if manifest_mode == "replace" else "a"
        with open(manifest_out, file_mode, encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract mode-specific square crops from one rectified board image."
    )
    parser.add_argument("--image", required=True, help="Input 800x800 rectified board image.")
    parser.add_argument(
        "--corners",
        help="Optional x1,y1,...,x4,y4 corner coordinates. Required for non-800x800 inputs.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(CROP_TYPE_BY_MODE),
        help="Extraction mode determining exact output count and labels.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for derived 100x100 crops.")
    parser.add_argument("--set-id", default="SET01", help="Set identifier (e.g. SET01).")
    parser.add_argument("--scene-id", required=True, help="Unique source-scene identifier.")
    parser.add_argument("--split", default="train", choices=["train", "val", "holdout"])
    parser.add_argument("--placement", help="Placement-only ground truth (8 ranks).")
    parser.add_argument("--board-fen", help="Complete six-field FEN ground truth.")
    parser.add_argument(
        "--piece-map",
        help="JSON object mapping occupied squares to canonical piece instance IDs (full-board mode).",
    )
    parser.add_argument(
        "--target-square",
        help="Target square for occupied_target_only (e.g. e4).",
    )
    parser.add_argument(
        "--piece-instance-id",
        help="Canonical target piece identity for occupied_target_only (e.g. SET01_W_P01).",
    )
    parser.add_argument(
        "--orientation",
        default="white-bottom",
        choices=["white-bottom", "black-bottom"],
    )
    parser.add_argument("--camera-angle", default="zenith")
    parser.add_argument("--lighting-setup", default="diffuse_daylight")
    parser.add_argument("--photographer", default="Rahul Bongu")
    parser.add_argument("--rights-holder", default="Rahul Bongu")
    parser.add_argument(
        "--release-record",
        default="author_property_release_20261102.pdf",
    )
    parser.add_argument("--occlusion-level", default="none", choices=["none", "partial", "severe"])
    parser.add_argument(
        "--annotation-status",
        default="verified",
        choices=["verified", "provisional", "unlabeled"],
    )
    parser.add_argument("--quality-notes", default="")
    parser.add_argument("--manifest-out", help="Optional manifest JSONL output path.")
    parser.add_argument(
        "--manifest-mode",
        default="append",
        choices=["append", "replace"],
        help="Append to or replace the manifest file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing derived crop files.",
    )
    args = parser.parse_args()

    try:
        if not SCENE_ID_REGEX.match(args.scene_id):
            raise ExtractionError(
                f"Invalid --scene-id '{args.scene_id}'. Use letters, digits, underscore, or hyphen only."
            )

        image = cv2.imread(args.image)
        if image is None:
            raise ExtractionError(f"Could not read image '{args.image}'.")

        if args.corners:
            image = rectify_with_corners(image, args.corners)
        elif image.shape[0] != CANONICAL_BOARD_SIZE or image.shape[1] != CANONICAL_BOARD_SIZE:
            raise ExtractionError(
                f"Input image is {image.shape[1]}x{image.shape[0]}, not 800x800. "
                "Pass --corners for a documented homography or provide an already-rectified image. "
                "The tool will not silently resize an oblique photograph."
            )

        piece_map = load_piece_map(args.piece_map) if args.piece_map else None
        plan = plan_crop_extraction(
            image,
            mode=args.mode,
            placement=args.placement,
            board_fen=args.board_fen,
            orientation=args.orientation,
            set_id=args.set_id,
            target_square=args.target_square,
            target_piece_instance_id=args.piece_instance_id,
            piece_instance_map=piece_map,
        )

        records = write_crops_and_manifest(
            plan=plan,
            image_path=args.image,
            out_dir=args.out_dir,
            scene_id=args.scene_id,
            set_id=args.set_id,
            split=args.split,
            board_fen=args.board_fen,
            camera_angle=args.camera_angle,
            lighting_setup=args.lighting_setup,
            photographer=args.photographer,
            rights_holder=args.rights_holder,
            release_record=args.release_record,
            occlusion_level=args.occlusion_level,
            annotation_status=args.annotation_status,
            quality_notes=args.quality_notes,
            manifest_out=args.manifest_out,
            manifest_mode=args.manifest_mode,
            overwrite=args.overwrite,
        )
    except ExtractionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

    print("================================================================================")
    print("CROP EXTRACTION REPORT")
    print("================================================================================")
    print(f"Extraction mode:           {plan['mode']}")
    print(f"Crop type:                 {plan['crop_type']}")
    print(f"Source photographs:        1")
    print(f"Derived crops:             {len(records)}")
    print(f"Manifest records:          {len(records) if args.manifest_out else 0}")
    print(f"Output directory:          {os.path.abspath(args.out_dir)}")
    print("================================================================================")


if __name__ == "__main__":
    main()
