#!/usr/bin/env python3
"""
validate_capture_manifest.py - Milestone 5.2D Offline Manifest Validator

Validates physical capture manifest JSONL files against schema constraints,
piece instance naming, FEN placement vs 6-field complete FEN syntax,
licensing provenance, and strict cross-split leakage prevention.

Usage:
    python scripts/validate_capture_manifest.py <manifest_file.jsonl>
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

SET_ID_REGEX = re.compile(r"^SET[0-9]{2}$")
PIECE_INSTANCE_REGEX = re.compile(r"^SET[0-9]{2}_[WB]_[PNBRQK][0-9]{2}$")
PHYSICAL_TAG_REGEX = re.compile(r"^S[0-9]{2}-[WB]-[PNBRQK][0-9]{2}$")
SQUARE_REGEX = re.compile(r"^[a-h][1-8]$")

def canonical_to_physical_tag(canonical_id: str) -> str:
    if not canonical_id or canonical_id == "EMPTY":
        return "NONE"
    m = re.match(r"^SET([0-9]{2})_([WB])_([PNBRQK][0-9]{2})$", canonical_id)
    if not m:
        return "INVALID"
    return f"S{m.group(1)}-{m.group(2)}-{m.group(3)}"


def physical_tag_to_canonical(physical_tag: str) -> str:
    if not physical_tag or physical_tag in ("NONE", "EMPTY"):
        return "EMPTY"
    m = re.match(r"^S([0-9]{2})-([WB])-([PNBRQK][0-9]{2})$", physical_tag)
    if not m:
        return "INVALID"
    return f"SET{m.group(1)}_{m.group(2)}_{m.group(3)}"

VALID_PIECE_TYPES = {"pawn", "knight", "bishop", "rook", "queen", "king", "empty"}
VALID_PIECE_COLORS = {"white", "black", "none"}
VALID_ANGLES = {"zenith", "white_perspective", "black_perspective", "lateral_flank"}
VALID_LIGHTING = {"diffuse_daylight", "overhead_specular", "directional_oblique"}
VALID_CROP_TYPES = {"isolated_piece", "empty_square", "full_board_context", "full_board_raw"}
VALID_EXTRACTION_MODES = {"full_board_context", "occupied_target_only", "empty_board_grid"}
EXTRACTION_MODE_TO_CROP_TYPE = {
    "full_board_context": "full_board_context",
    "occupied_target_only": "isolated_piece",
    "empty_board_grid": "empty_square",
}
VALID_OCCLUSIONS = {"none", "partial", "severe"}
VALID_SPLITS = {"train", "val", "holdout"}
VALID_PERMISSIONS = {"verified_author_consent", "verified_owner_release"}
VALID_ANNOTATION_STATUSES = {"verified", "provisional", "unlabeled"}

TYPE_CHAR_MAP = {
    "pawn": "P",
    "knight": "N",
    "bishop": "B",
    "rook": "R",
    "queen": "Q",
    "king": "K",
    "empty": "E",
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


def get_expected_split(set_id: str) -> str:
    m = re.match(r"^SET([0-9]{2})$", set_id)
    if not m:
        return ""
    num = int(m.group(1))
    if 1 <= num <= 14:
        return "train"
    if 15 <= num <= 17:
        return "val"
    if 18 <= num <= 20:
        return "holdout"
    return ""


def validate_placement_only(placement: str) -> Tuple[bool, str]:
    if not placement or not isinstance(placement, str):
        return False, "Placement string is empty or invalid."
    trimmed = placement.strip()
    if " " in trimmed:
        return False, "Placement string must not contain spaces (use complete FEN for 6-field strings)."
    ranks = trimmed.split("/")
    if len(ranks) != 8:
        return False, f"Placement must contain exactly 8 ranks separated by '/', got {len(ranks)}."
    for r_idx, rank in enumerate(ranks):
        count = 0
        for ch in rank:
            if ch.isdigit():
                count += int(ch)
            elif ch in "pnbrqkPNBRQK":
                count += 1
            else:
                return False, f"Invalid character '{ch}' in rank {8 - r_idx}."
        if count != 8:
            return False, f"Rank {8 - r_idx} has {count} squares instead of 8."
    return True, ""


def validate_complete_fen(fen: str) -> Tuple[bool, str]:
    if not fen or not isinstance(fen, str):
        return False, "FEN string is empty or invalid."
    tokens = fen.strip().split()
    if len(tokens) != 6:
        return False, f"Complete FEN must contain exactly 6 space-separated fields, got {len(tokens)}."

    placement, active_color, castling, en_passant, halfmove, fullmove = tokens

    valid_place, err = validate_placement_only(placement)
    if not valid_place:
        return False, err

    if active_color not in ("w", "b"):
        return False, f"Active color must be 'w' or 'b', got '{active_color}'."

    if castling != "-":
        valid_chars = {"K", "Q", "k", "q"}
        seen = set()
        for ch in castling:
            if ch not in valid_chars or ch in seen:
                return False, f"Invalid castling rights string: '{castling}'."
            seen.add(ch)

    if en_passant != "-":
        if not re.match(r"^[a-h][36]$", en_passant):
            return False, f"Invalid en-passant square: '{en_passant}'."

    try:
        half = int(halfmove)
        if half < 0 or str(half) != halfmove:
            return False, f"Invalid halfmove clock: '{halfmove}'."
    except ValueError:
        return False, f"Invalid halfmove clock: '{halfmove}'."

    try:
        full = int(fullmove)
        if full < 1 or str(full) != fullmove:
            return False, f"Invalid fullmove number: '{fullmove}'."
    except ValueError:
        return False, f"Invalid fullmove number: '{fullmove}'."

    return True, ""


def validate_fen(fen: str) -> bool:
    if not fen:
        return False
    tokens = fen.strip().split()
    if len(tokens) == 6:
        valid, _ = validate_complete_fen(fen)
        return valid
    placement = tokens[0]
    valid, _ = validate_placement_only(placement)
    return valid


def count_pieces_in_placement(placement: str) -> Dict[str, Any]:
    white: Dict[str, int] = defaultdict(int)
    black: Dict[str, int] = defaultdict(int)
    empty_squares = 0
    if not placement:
        return {"white": {}, "black": {}, "total_white": 0, "total_black": 0, "total_pieces": 0, "empty_squares": 64}

    first_token = placement.strip().split()[0]
    ranks = first_token.split("/")
    for rank in ranks:
        for ch in rank:
            if ch.isdigit():
                empty_squares += int(ch)
            elif ch in "PNBRQK":
                white[ch] += 1
            elif ch in "pnbrqk":
                black[ch] += 1

    total_w = sum(white.values())
    total_b = sum(black.values())
    return {
        "white": dict(white),
        "black": dict(black),
        "total_white": total_w,
        "total_black": total_b,
        "total_pieces": total_w + total_b,
        "empty_squares": empty_squares,
    }


def get_placement_occupant(placement: str, square: str) -> Dict[str, str] | None:
    ranks = placement.strip().split("/")
    if len(ranks) != 8 or not SQUARE_REGEX.match(square):
        return None

    file_index = ord(square[0]) - 97
    rank = int(square[1])
    rank_str = ranks[8 - rank]
    index = 0

    for ch in rank_str:
        if ch.isdigit():
            index += int(ch)
            continue
        if index == file_index:
            if ch not in "pnbrqkPNBRQK":
                return None
            return {
                "piece_char": ch,
                "piece_type": CHAR_TO_PIECE_TYPE[ch],
                "piece_color": "white" if ch.isupper() else "black",
            }
        index += 1
    return None


def validate_record(r: Dict[str, Any], idx: int) -> List[str]:
    errors = []
    prefix = f"[Record #{idx} Set:{r.get('set_id', 'UNKNOWN')} Piece:{r.get('piece_instance_id', 'UNKNOWN')}]"

    # Required fields
    for field in ["dataset_version", "source_id", "scene_id", "original_image_path", "derived_crop_path"]:
        if not str(r.get(field, "")).strip():
            errors.append(f"{prefix} Missing required '{field}'")

    set_id = str(r.get("set_id", ""))
    if not SET_ID_REGEX.match(set_id):
        errors.append(f"{prefix} Invalid 'set_id': '{set_id}'. Must match SET01-SET20 format.")

    split = str(r.get("split", ""))
    if split not in VALID_SPLITS:
        errors.append(f"{prefix} Invalid 'split': '{split}'. Expected train, val, or holdout.")
    else:
        expected = get_expected_split(set_id)
        if expected and split != expected:
            errors.append(f"{prefix} Split mismatch for set '{set_id}': assigned '{split}', expected '{expected}'.")

    piece_type = str(r.get("piece_type", ""))
    piece_color = str(r.get("piece_color", ""))
    instance_id = str(r.get("piece_instance_id", ""))

    if piece_type not in VALID_PIECE_TYPES:
        errors.append(f"{prefix} Unknown 'piece_type': '{piece_type}'")
    if piece_color not in VALID_PIECE_COLORS:
        errors.append(f"{prefix} Unknown 'piece_color': '{piece_color}'")

    if piece_type == "empty":
        if piece_color != "none":
            errors.append(f"{prefix} Empty square must have piece_color 'none', got '{piece_color}'")
        if instance_id != "EMPTY":
            errors.append(f"{prefix} Empty square must have piece_instance_id 'EMPTY', got '{instance_id}'")
    else:
        if piece_color not in ("white", "black"):
            errors.append(f"{prefix} Non-empty piece must have piece_color 'white' or 'black', got '{piece_color}'")
        if not PIECE_INSTANCE_REGEX.match(instance_id):
            errors.append(f"{prefix} Invalid 'piece_instance_id': '{instance_id}'. Expected format SETxx_W_P01.")
        else:
            if instance_id[:5] != set_id:
                errors.append(f"{prefix} Instance prefix '{instance_id[:5]}' does not match set_id '{set_id}'")
            expected_color_char = "W" if piece_color == "white" else "B"
            if instance_id[6] != expected_color_char:
                errors.append(f"{prefix} Instance color char '{instance_id[6]}' does not match '{piece_color}'")
            if piece_type in TYPE_CHAR_MAP and instance_id[8] != TYPE_CHAR_MAP[piece_type]:
                errors.append(f"{prefix} Instance type char '{instance_id[8]}' does not match '{piece_type}'")

    # Physical tag ID validation & deterministic mapping (Gate 1)
    physical_tag = r.get("physical_tag_id")
    if physical_tag is not None and str(physical_tag).strip():
        tag = str(physical_tag).strip()
        if piece_type == "empty":
            if tag not in ("NONE", "EMPTY"):
                errors.append(f"{prefix} Empty square 'physical_tag_id' must be 'NONE' or 'EMPTY', got '{tag}'")
        else:
            if not PHYSICAL_TAG_REGEX.match(tag):
                errors.append(f"{prefix} Invalid 'physical_tag_id': '{tag}'. Expected format Sxx-W-P01 (e.g. S01-W-P01).")
            else:
                expected_tag = canonical_to_physical_tag(instance_id)
                if tag != expected_tag:
                    errors.append(
                        f"{prefix} Physical tag ID '{tag}' does not deterministically map to piece_instance_id '{instance_id}' (expected '{expected_tag}')."
                    )

    crop_type = str(r.get("crop_type", ""))
    if crop_type not in VALID_CROP_TYPES:
        errors.append(f"{prefix} Unknown 'crop_type': '{crop_type}'")

    square = str(r.get("square_coord", ""))
    if crop_type != "full_board_raw":
        if not SQUARE_REGEX.match(square):
            errors.append(f"{prefix} Invalid 'square_coord': '{square}'. Must be a1-h8.")

    # FEN and placement validation
    if crop_type in ("full_board_context", "full_board_raw"):
        board_fen = r.get("board_fen")
        piece_placement = r.get("piece_placement")
        if board_fen:
            tokens = str(board_fen).strip().split()
            if len(tokens) == 6:
                valid_fen, fen_err = validate_complete_fen(str(board_fen))
                if not valid_fen:
                    errors.append(f"{prefix} Invalid complete 6-field 'board_fen': {fen_err}")
            else:
                valid_place, place_err = validate_placement_only(tokens[0])
                if not valid_place:
                    errors.append(f"{prefix} Invalid placement syntax in 'board_fen': {place_err}")
        elif piece_placement:
            valid_place, place_err = validate_placement_only(str(piece_placement))
            if not valid_place:
                errors.append(f"{prefix} Invalid 'piece_placement': {place_err}")
        else:
            errors.append(f"{prefix} Missing required 'board_fen' or 'piece_placement' for full-board crop.")

    if r.get("piece_placement"):
        valid_place, place_err = validate_placement_only(str(r["piece_placement"]))
        if not valid_place:
            errors.append(f"{prefix} Invalid standalone 'piece_placement': {place_err}")

    # Extraction mode and square-label consistency (Milestone 5.2H)
    extraction_mode = str(r.get("extraction_mode", "")).strip()
    if extraction_mode:
        if extraction_mode not in VALID_EXTRACTION_MODES:
            errors.append(f"{prefix} Unknown 'extraction_mode': '{extraction_mode}'")
        elif crop_type != EXTRACTION_MODE_TO_CROP_TYPE[extraction_mode]:
            errors.append(
                f"{prefix} 'extraction_mode' '{extraction_mode}' requires crop_type "
                f"'{EXTRACTION_MODE_TO_CROP_TYPE[extraction_mode]}', got '{crop_type}'."
            )

    occupancy_placement = None
    if r.get("piece_placement"):
        valid_place, _ = validate_placement_only(str(r["piece_placement"]))
        if valid_place:
            occupancy_placement = str(r["piece_placement"]).strip()
    elif r.get("board_fen"):
        tokens = str(r["board_fen"]).strip().split()
        if len(tokens) == 6:
            valid_fen, _ = validate_complete_fen(str(r["board_fen"]))
            if valid_fen:
                occupancy_placement = tokens[0]
        elif len(tokens) == 1:
            valid_place, _ = validate_placement_only(tokens[0])
            if valid_place:
                occupancy_placement = tokens[0]

    if occupancy_placement:
        counts = count_pieces_in_placement(occupancy_placement)
        if extraction_mode == "occupied_target_only" and counts["total_pieces"] != 1:
            errors.append(
                f"{prefix} 'occupied_target_only' requires exactly one occupied square, "
                f"got {counts['total_pieces']}."
            )
        if extraction_mode == "empty_board_grid" and counts["total_pieces"] != 0:
            errors.append(
                f"{prefix} 'empty_board_grid' requires zero occupied squares, "
                f"got {counts['total_pieces']}."
            )

        if crop_type != "full_board_raw" and SQUARE_REGEX.match(square):
            occupant = get_placement_occupant(occupancy_placement, square)
            if piece_type == "empty":
                if occupant:
                    errors.append(
                        f"{prefix} '{square}' is labelled empty but placement contains "
                        f"'{occupant['piece_char']}'."
                    )
            elif occupant is None:
                errors.append(
                    f"{prefix} '{square}' is labelled {piece_type} but placement says it is empty."
                )
            else:
                if occupant["piece_type"] != piece_type:
                    errors.append(
                        f"{prefix} '{square}' is labelled {piece_type} but placement contains "
                        f"{occupant['piece_type']}."
                    )
                if occupant["piece_color"] != piece_color:
                    errors.append(
                        f"{prefix} '{square}' is labelled {piece_color} but placement contains "
                        f"{occupant['piece_color']}."
                    )

    if r.get("camera_angle") not in VALID_ANGLES:
        errors.append(f"{prefix} Unknown 'camera_angle': '{r.get('camera_angle')}'")
    if r.get("lighting_setup") not in VALID_LIGHTING:
        errors.append(f"{prefix} Unknown 'lighting_setup': '{r.get('lighting_setup')}'")
    if r.get("occlusion_level") not in VALID_OCCLUSIONS:
        errors.append(f"{prefix} Unknown 'occlusion_level': '{r.get('occlusion_level')}'")

    # Licensing & Consent
    if not str(r.get("photographer", "")).strip():
        errors.append(f"{prefix} Missing required 'photographer'")
    if not str(r.get("rights_holder", "")).strip():
        errors.append(f"{prefix} Missing required 'rights_holder'")
    if not str(r.get("release_record", "")).strip():
        errors.append(f"{prefix} Missing required 'release_record'")
    if r.get("data_license") != "CC0-1.0":
        errors.append(f"{prefix} Non-compliant 'data_license': '{r.get('data_license')}'. Must be 'CC0-1.0'.")
    if r.get("code_license") != "MIT":
        errors.append(f"{prefix} Non-compliant 'code_license': '{r.get('code_license')}'. Must be 'MIT'.")
    if r.get("permission_status") not in VALID_PERMISSIONS:
        errors.append(f"{prefix} Invalid 'permission_status': '{r.get('permission_status')}'")
    if r.get("annotation_status") not in VALID_ANNOTATION_STATUSES:
        errors.append(f"{prefix} Unknown 'annotation_status': '{r.get('annotation_status')}'")

    return errors


def validate_manifest(
    records: List[Dict[str, Any]], check_files: bool = False
) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    sets_seen: Set[str] = set()
    instances_seen: Set[str] = set()
    tags_seen: Set[str] = set()
    scenes_seen: Set[str] = set()
    paths_seen: Dict[str, int] = {}
    source_paths_seen: Set[str] = set()
    scene_source_map: Dict[str, str] = {}
    source_scene_map: Dict[str, str] = {}
    missing_paths_reported: Set[str] = set()
    set_split_map: Dict[str, str] = {}
    piece_split_map: Dict[str, str] = {}
    tag_piece_map: Dict[str, str] = {}
    tag_split_map: Dict[str, str] = {}

    crops_by_type: Dict[str, int] = defaultdict(int)
    crops_by_split: Dict[str, int] = defaultdict(int)
    crops_by_mode: Dict[str, int] = defaultdict(int)

    if not records:
        return False, ["Manifest is empty."], {}

    for i, r in enumerate(records):
        errs = validate_record(r, i)
        errors.extend(errs)

        set_id = str(r.get("set_id", ""))
        split = str(r.get("split", ""))
        instance_id = str(r.get("piece_instance_id", ""))
        crop_path = str(r.get("derived_crop_path", ""))
        source_path = str(r.get("original_image_path", ""))
        physical_tag = r.get("physical_tag_id")

        if set_id:
            sets_seen.add(set_id)
            if split:
                if set_id in set_split_map and set_split_map[set_id] != split:
                    errors.append(
                        f"[Set Leakage] Set '{set_id}' appears in multiple splits: "
                        f"'{set_split_map[set_id]}' and '{split}'"
                    )
                else:
                    set_split_map[set_id] = split

        if instance_id and instance_id != "EMPTY":
            instances_seen.add(instance_id)
            if split:
                if instance_id in piece_split_map and piece_split_map[instance_id] != split:
                    errors.append(
                        f"[Piece Leakage] Piece '{instance_id}' appears in multiple splits: "
                        f"'{piece_split_map[instance_id]}' and '{split}'"
                    )
                else:
                    piece_split_map[instance_id] = split

        if physical_tag and str(physical_tag).strip() not in ("NONE", "EMPTY"):
            tag_str = str(physical_tag).strip()
            tags_seen.add(tag_str)
            if instance_id:
                if tag_str in tag_piece_map and tag_piece_map[tag_str] != instance_id:
                    errors.append(
                        f"[Tag Conflict] Physical tag '{tag_str}' mapped to multiple pieces: '{tag_piece_map[tag_str]}' and '{instance_id}'"
                    )
                else:
                    tag_piece_map[tag_str] = instance_id
            if split:
                if tag_str in tag_split_map and tag_split_map[tag_str] != split:
                    errors.append(
                        f"[Tag Leakage] Physical tag '{tag_str}' appears in multiple splits: '{tag_split_map[tag_str]}' and '{split}'"
                    )
                else:
                    tag_split_map[tag_str] = split

        if crop_path:
            if crop_path in paths_seen:
                errors.append(
                    f"[Duplicate Path] Derived crop path '{crop_path}' appears in records #{paths_seen[crop_path]} and #{i}"
                )
            else:
                paths_seen[crop_path] = i

        if source_path:
            source_paths_seen.add(source_path)
            if r.get("scene_id"):
                scene_id = str(r["scene_id"])
                mapped_source = scene_source_map.get(scene_id)
                if mapped_source and mapped_source != source_path:
                    errors.append(
                        f"[Scene Source Conflict] Scene '{scene_id}' maps to multiple source images: "
                        f"'{mapped_source}' and '{source_path}'"
                    )
                else:
                    scene_source_map[scene_id] = source_path

                mapped_scene = source_scene_map.get(source_path)
                if mapped_scene and mapped_scene != scene_id:
                    errors.append(
                        f"[Source Scene Conflict] Source image '{source_path}' maps to multiple scenes: "
                        f"'{mapped_scene}' and '{scene_id}'"
                    )
                else:
                    source_scene_map[source_path] = scene_id

        if check_files:
            for role, path in (("source image", source_path), ("derived crop", crop_path)):
                if path and not os.path.isfile(path) and path not in missing_paths_reported:
                    errors.append(f"[Missing File] {role} '{path}' does not exist.")
                    missing_paths_reported.add(path)
        if source_path and crop_path and source_path == crop_path:
            errors.append(f"[Path Conflict] Source image and derived crop share path '{crop_path}'.")

        if r.get("scene_id"):
            scenes_seen.add(str(r["scene_id"]))
        if r.get("crop_type"):
            crops_by_type[str(r["crop_type"])] += 1
        if split:
            crops_by_split[split] += 1
        if r.get("extraction_mode"):
            crops_by_mode[str(r["extraction_mode"])] += 1

    summary = {
        "total_records": len(records),
        "source_photographs": len(source_paths_seen),
        "derived_crops": len(records),
        "distinct_sets": sorted(list(sets_seen)),
        "distinct_piece_instances": sorted(list(instances_seen)),
        "distinct_physical_tags": sorted(list(tags_seen)),
        "distinct_scenes": sorted(list(scenes_seen)),
        "crops_by_type": dict(crops_by_type),
        "crops_by_split": dict(crops_by_split),
        "crops_by_extraction_mode": dict(crops_by_mode),
        "set_split_map": set_split_map,
    }
    return len(errors) == 0, errors, summary


def main():
    parser = argparse.ArgumentParser(description="Validate physical capture manifest JSONL.")
    parser.add_argument("manifest_path", help="Path to manifest JSONL file.")
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Require every original_image_path and derived_crop_path to exist on disk.",
    )
    args = parser.parse_args()

    records = []
    with open(args.manifest_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_no}: {e}", file=sys.stderr)
                sys.exit(1)

    valid, errors, summary = validate_manifest(records, check_files=args.check_files)

    print("================================================================================")
    print("PHYSICAL CHESS CAPTURE MANIFEST VALIDATION REPORT")
    print("================================================================================")
    print(f"Total Records:             {summary.get('total_records', 0)}")
    print(f"Source Photographs:        {summary.get('source_photographs', 0)}")
    print(f"Derived Crops:             {summary.get('derived_crops', 0)}")
    print(f"Distinct Physical Sets:    {len(summary.get('distinct_sets', []))} {summary.get('distinct_sets', [])}")
    print(f"Distinct Piece Instances:  {len(summary.get('distinct_piece_instances', []))}")
    print(f"Distinct Physical Tags:    {len(summary.get('distinct_physical_tags', []))}")
    print(f"Distinct Scenes:           {len(summary.get('distinct_scenes', []))}")
    print("Crops by Type:")
    for ct, count in summary.get("crops_by_type", {}).items():
        print(f"  - {ct:<24}: {count}")
    if summary.get("crops_by_extraction_mode"):
        print("Crops by Extraction Mode:")
        for mode, count in summary.get("crops_by_extraction_mode", {}).items():
            print(f"  - {mode:<24}: {count}")
    print("Crops by Split:")
    for sp, count in summary.get("crops_by_split", {}).items():
        print(f"  - {sp:<24}: {count}")
    print("--------------------------------------------------------------------------------")

    if valid:
        print("GATE STATUS: PASS (Zero errors detected. 100% leak-free and compliant.)")
        print("================================================================================")
        sys.exit(0)
    else:
        print(f"GATE STATUS: FAIL ({len(errors)} errors detected)")
        print("--------------------------------------------------------------------------------")
        for err in errors[:25]:
            print(f"  ERROR: {err}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} additional errors.")
        print("================================================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
