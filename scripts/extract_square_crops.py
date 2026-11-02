#!/usr/bin/env python3
"""
extract_square_crops.py - Milestone 5.2G Offline Crop Extraction Engine

Slices 64 canonical 100x100 square crops from an 800x800 rectified chessboard image,
assigns piece ground-truth from FEN/placement annotations, and optionally writes
manifest JSONL records.

Usage:
    py scripts/extract_square_crops.py \\
        --image path/to/rectified_board.png \\
        --out-dir C:/dev/chess-capture-workspace/crops/train/SET01 \\
        --set-id SET01 \\
        --scene-id SET01_SCENE_01 \\
        --placement "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" \\
        --camera-angle zenith \\
        --lighting diffuse_daylight
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

CANONICAL_BOARD_SIZE = 800
CANONICAL_SQUARE_SIZE = 100
GRID_DIM = 8

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


def parse_placement_to_square_map(placement: str) -> Dict[str, Dict[str, str]]:
    sq_map: Dict[str, Dict[str, str]] = {}
    if not placement:
        return sq_map

    first_token = placement.strip().split()[0]
    ranks = first_token.split("/")
    if len(ranks) != 8:
        return sq_map

    for r_idx in range(8):
        rank_num = 8 - r_idx
        rank_str = ranks[r_idx]
        file_idx = 0

        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            elif ch in "pnbrqkPNBRQK":
                if file_idx < 8:
                    file_char = chr(97 + file_idx)
                    sq = f"{file_char}{rank_num}"
                    is_white = ch.isupper()
                    sq_map[sq] = {
                        "piece_char": ch,
                        "piece_type": CHAR_TO_PIECE_TYPE.get(ch, "empty"),
                        "piece_color": "white" if is_white else "black",
                    }
                    file_idx += 1
    return sq_map


def extract_crops_from_rectified(
    image: np.ndarray,
    placement: Optional[str] = None,
    orientation: str = "white-bottom",
) -> List[Dict[str, Any]]:
    if image.shape[0] != CANONICAL_BOARD_SIZE or image.shape[1] != CANONICAL_BOARD_SIZE:
        raise ValueError(
            f"Image dimensions {image.shape[1]}x{image.shape[0]} do not match canonical 800x800."
        )

    piece_map = parse_placement_to_square_map(placement) if placement else {}
    crops = []

    for row in range(GRID_DIM):
        for col in range(GRID_DIM):
            if orientation == "white-bottom":
                file_idx = col
                rank_idx = 7 - row
            else:
                file_idx = 7 - col
                rank_idx = row

            file_char = chr(97 + file_idx)
            rank_num = rank_idx + 1
            square = f"{file_char}{rank_num}"

            start_x = col * CANONICAL_SQUARE_SIZE
            start_y = row * CANONICAL_SQUARE_SIZE

            patch = image[start_y : start_y + CANONICAL_SQUARE_SIZE, start_x : start_x + CANONICAL_SQUARE_SIZE]

            piece_info = piece_map.get(
                square, {"piece_char": None, "piece_type": "empty", "piece_color": "none"}
            )

            crops.append({
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
            })

    return crops


def main():
    parser = argparse.ArgumentParser(description="Extract 64 canonical square crops from a board image.")
    parser.add_argument("--image", required=True, help="Path to input 800x800 rectified image or full board.")
    parser.add_argument("--corners", help="Comma-separated x1,y1,x2,y2,x3,y3,x4,y4 if rectification is needed.")
    parser.add_argument("--out-dir", required=True, help="Directory to save extracted 100x100 crops.")
    parser.add_argument("--set-id", default="SET01", help="Set identifier (e.g. SET01).")
    parser.add_argument("--scene-id", default="SET01_SCENE_01", help="Scene identifier.")
    parser.add_argument("--split", default="train", choices=["train", "val", "holdout"], help="Dataset split.")
    parser.add_argument("--placement", help="FEN placement string (8 ranks).")
    parser.add_argument("--board-fen", help="Complete 6-field FEN.")
    parser.add_argument("--camera-angle", default="zenith", help="Camera viewpoint angle.")
    parser.add_argument("--lighting-setup", default="diffuse_daylight", help="Lighting condition.")
    parser.add_argument("--photographer", default="Rahul Bongu", help="Photographer attribution.")
    parser.add_argument("--rights-holder", default="Rahul Bongu", help="Rights holder.")
    parser.add_argument("--manifest-out", help="Path to append or write manifest JSONL records.")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image at '{args.image}'", file=sys.stderr)
        sys.exit(1)

    if args.corners:
        pts = [float(v.strip()) for v in args.corners.split(",")]
        if len(pts) != 8:
            print("Error: --corners must contain exactly 8 numbers (x1,y1..x4,y4).", file=sys.stderr)
            sys.exit(1)
        src_pts = np.array(pts, dtype=np.float32).reshape(4, 2)
        dst_pts = np.array([
            [0, 0],
            [CANONICAL_BOARD_SIZE, 0],
            [CANONICAL_BOARD_SIZE, CANONICAL_BOARD_SIZE],
            [0, CANONICAL_BOARD_SIZE],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(img, M, (CANONICAL_BOARD_SIZE, CANONICAL_BOARD_SIZE))

    if img.shape[0] != CANONICAL_BOARD_SIZE or img.shape[1] != CANONICAL_BOARD_SIZE:
        print(
            f"Resizing input from {img.shape[1]}x{img.shape[0]} to {CANONICAL_BOARD_SIZE}x{CANONICAL_BOARD_SIZE}..."
        )
        img = cv2.resize(img, (CANONICAL_BOARD_SIZE, CANONICAL_BOARD_SIZE), interpolation=cv2.INTER_AREA)

    placement = args.placement
    if not placement and args.board_fen:
        placement = args.board_fen.strip().split()[0]

    crops = extract_crops_from_rectified(img, placement=placement)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest_records = []
    print(f"Extracting 64 crops to '{args.out_dir}'...")

    for c in crops:
        sq = c["square"]
        pt = c["piece_type"]
        pc = c["piece_color"]

        if pt == "empty":
            tag_suffix = "EMPTY"
            inst_id = "EMPTY"
            phys_tag = "NONE"
        else:
            color_char = "W" if pc == "white" else "B"
            type_char = c["piece_char"].upper()
            inst_id = f"{args.set_id}_{color_char}_{type_char}01"
            phys_tag = f"S{args.set_id[3:5]}-{color_char}-{type_char}01"
            tag_suffix = f"{color_char}_{type_char}"

        filename = f"{args.scene_id}_{sq}_{tag_suffix}.png"
        crop_path = os.path.join(args.out_dir, filename)
        cv2.imwrite(crop_path, c["patch"])

        record = {
            "dataset_version": "1.0.0",
            "source_id": "author_capture_2026",
            "set_id": args.set_id,
            "split": args.split,
            "piece_instance_id": inst_id,
            "physical_tag_id": phys_tag,
            "piece_type": pt,
            "piece_color": pc,
            "scene_id": args.scene_id,
            "board_fen": args.board_fen,
            "piece_placement": placement,
            "square_coord": sq,
            "camera_angle": args.camera_angle,
            "lighting_setup": args.lighting_setup,
            "original_image_path": args.image,
            "derived_crop_path": crop_path,
            "crop_type": "full_board_context",
            "occlusion_level": "none",
            "photographer": args.photographer,
            "rights_holder": args.rights_holder,
            "release_record": "author_property_release_20261102.pdf",
            "data_license": "CC0-1.0",
            "code_license": "MIT",
            "permission_status": "verified_author_consent",
            "annotation_status": "verified",
        }
        manifest_records.append(record)

    if args.manifest_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)), exist_ok=True)
        with open(args.manifest_out, "a", encoding="utf-8") as f:
            for rec in manifest_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(manifest_records)} records to '{args.manifest_out}'.")

    print(f"Successfully extracted {len(crops)} 100x100 square crops.")


if __name__ == "__main__":
    main()
