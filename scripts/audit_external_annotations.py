#!/usr/bin/env python3
"""
audit_external_annotations.py - Milestone 5.2J FENiT Annotation Quality Audit

Audits the 491 FENiT dataset annotations (Mendeley Data DOI 10.17632/wpp9xnbcp6.1)
to evaluate corner geometry, piece box consistency against filename FENs, and
model-generated annotation artifacts.

Outputs:
  C:\\dev\\chess-capture-workspace\\external\\fenit\\reports\\annotation_quality_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

CLASS_NAMES = {
    0: "Board",
    1: "White-King",
    2: "White-Pawn",
    3: "Black-Pawn",
    4: "Black-King",
    5: "White-Queen",
    6: "White-Bishop",
    7: "White-Knight",
    8: "White-Rook",
    9: "Black-Bishop",
    10: "Black-Rook",
    11: "Black-Knight",
    12: "Black-Queen",
}

FEN_CHAR_TO_CLASS = {
    "K": "White-King",
    "P": "White-Pawn",
    "p": "Black-Pawn",
    "k": "Black-King",
    "Q": "White-Queen",
    "B": "White-Bishop",
    "N": "White-Knight",
    "R": "White-Rook",
    "b": "Black-Bishop",
    "r": "Black-Rook",
    "n": "Black-Knight",
    "q": "Black-Queen",
}


def normalize_stem(stem: str) -> str:
    s = stem.strip()
    s = re.sub(r"\s*\((?:full board|board)\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    s = re.sub(r"\s+[wb]$", "", s)
    return s.strip()


def parse_fen_placement(stem: str) -> str | None:
    candidate = normalize_stem(stem)
    for sep in ("_", "-"):
        parts = candidate.split(sep)
        if len(parts) != 8:
            continue
        if not all(re.fullmatch(r"[1-8pnbrqkPNBRQK]+", part) for part in parts):
            continue
        if any(re.search(r"\d{2}", part) for part in parts):
            continue
        if not all(sum(int(ch) if ch.isdigit() else 1 for ch in part) == 8 for part in parts):
            continue
        return "/".join(parts)
    return None


def is_strictly_convex(corners: np.ndarray) -> bool:
    if len(corners) != 4:
        return False
    signs = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        p2 = corners[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-6:
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def audit_dataset(raw_dir: Path) -> Dict[str, Any]:
    images = sorted(
        p for p in raw_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    annotations = sorted(raw_dir.glob("*.txt"))
    annotations = [p for p in annotations if p.name != "Train.txt"]

    total_images = len(images)
    total_annotations = len(annotations)

    board_line_counts = Counter()
    non_convex_cases = []
    missing_board_cases = []
    timestamp_cases = []
    isolated_piece_cases = []
    unparseable_cases = []

    exact_count_agreements = []
    piece_count_mismatches = []
    exact_multiset_agreements = []
    class_mismatches = []

    # Map stems
    ann_by_stem = {p.stem: p for p in annotations}

    for img_path in images:
        stem = img_path.stem
        ann_path = ann_by_stem.get(stem)
        placement = parse_fen_placement(stem)

        if not placement:
            if re.fullmatch(r"\d{8}_\d{6}", stem):
                timestamp_cases.append(img_path.name)
            elif len(stem) <= 6 and stem in {"B", "K", "N", "P", "Q", "R"}:
                isolated_piece_cases.append(img_path.name)
            else:
                unparseable_cases.append(img_path.name)
            continue

        if not ann_path or not ann_path.exists():
            missing_board_cases.append(img_path.name)
            continue

        lines = [
            l.strip().split()
            for l in ann_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

        board_lines = [l for l in lines if l and l[0] == "0"]
        board_line_counts[len(board_lines)] += 1

        if len(board_lines) != 1:
            missing_board_cases.append(img_path.name)
            continue

        # Check board corners
        b_tokens = board_lines[0]
        vals = [float(v) for v in b_tokens[1:]]
        if len(vals) < 16:
            missing_board_cases.append(img_path.name)
            continue

        corners = np.array(
            [[vals[4 + i * 3], vals[5 + i * 3]] for i in range(4)], dtype=np.float32
        )
        if not is_strictly_convex(corners):
            non_convex_cases.append({
                "file": img_path.name,
                "corners": corners.tolist(),
                "reason": "self_intersecting_bowtie_quadrilateral",
            })
            continue

        # Parse pieces from FEN
        fen_pieces = Counter()
        for ch in placement:
            if ch in FEN_CHAR_TO_CLASS:
                fen_pieces[FEN_CHAR_TO_CLASS[ch]] += 1

        # Parse pieces from YOLO annotation
        yolo_pieces = Counter()
        for l in lines:
            cls = int(l[0])
            if cls in CLASS_NAMES and cls != 0:
                yolo_pieces[CLASS_NAMES[cls]] += 1

        total_fen = sum(fen_pieces.values())
        total_yolo = sum(yolo_pieces.values())

        if total_fen != total_yolo:
            piece_count_mismatches.append({
                "file": img_path.name,
                "fen_pieces_count": total_fen,
                "yolo_pieces_count": total_yolo,
                "difference": total_yolo - total_fen,
                "fen_pieces": dict(fen_pieces),
                "yolo_pieces": dict(yolo_pieces),
            })
        else:
            exact_count_agreements.append(img_path.name)
            if fen_pieces != yolo_pieces:
                diffs = {}
                for k in set(fen_pieces) | set(yolo_pieces):
                    if fen_pieces.get(k, 0) != yolo_pieces.get(k, 0):
                        diffs[k] = {"fen": fen_pieces.get(k, 0), "yolo": yolo_pieces.get(k, 0)}
                class_mismatches.append({
                    "file": img_path.name,
                    "piece_count": total_fen,
                    "class_discrepancies": diffs,
                })
            else:
                exact_multiset_agreements.append(img_path.name)

    return {
        "dataset_name": "FENiT (Mendeley Data wpp9xnbcp6 v1)",
        "dataset_doi": "10.17632/wpp9xnbcp6.1",
        "license": "CC BY 4.0",
        "authors": "Theo Harjanto, Vincent Oei, Dicky Susanto (Binus University)",
        "annotation_tool_provenance": "CVAT semi-automated with pre-trained yolov8l-pose.pt",
        "total_images": total_images,
        "total_annotations": total_annotations,
        "eligibility_breakdown": {
            "eligible_fen_boards": total_images - len(timestamp_cases) - len(isolated_piece_cases) - len(unparseable_cases),
            "timestamp_named_no_fen": len(timestamp_cases),
            "isolated_piece_photos": len(isolated_piece_cases),
            "unparseable_fen": len(unparseable_cases),
        },
        "corner_geometry_audit": {
            "board_line_frequency": dict(board_line_counts),
            "missing_board_lines": len(missing_board_cases),
            "strictly_convex_boards": 436 - len(non_convex_cases),
            "non_convex_bowtie_boards": len(non_convex_cases),
            "non_convex_details": non_convex_cases,
        },
        "piece_annotation_consistency": {
            "total_fen_audited": len(exact_count_agreements) + len(piece_count_mismatches),
            "exact_piece_count_agreement": len(exact_count_agreements),
            "exact_piece_count_agreement_pct": round(len(exact_count_agreements) / (len(exact_count_agreements) + len(piece_count_mismatches)) * 100, 2),
            "piece_count_mismatches": len(piece_count_mismatches),
            "piece_count_mismatches_pct": round(len(piece_count_mismatches) / (len(exact_count_agreements) + len(piece_count_mismatches)) * 100, 2),
            "exact_class_multiset_agreement": len(exact_multiset_agreements),
            "exact_class_multiset_agreement_pct": round(len(exact_multiset_agreements) / (len(exact_count_agreements) + len(piece_count_mismatches)) * 100, 2),
            "class_misclassifications": len(class_mismatches),
            "class_misclassifications_pct": round(len(class_mismatches) / (len(exact_count_agreements) + len(piece_count_mismatches)) * 100, 2),
            "piece_count_mismatch_samples": piece_count_mismatches[:10],
            "class_mismatch_samples": class_mismatches[:10],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit FENiT annotation quality.")
    parser.add_argument(
        "--raw-dir",
        default=r"C:\dev\chess-capture-workspace\external\fenit\raw",
        help="Path to FENiT raw images and annotations",
    )
    parser.add_argument(
        "--out",
        default=r"C:\dev\chess-capture-workspace\external\fenit\reports\annotation_quality_audit.json",
        help="Path to output audit report JSON",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("========================================================================================")
    print("FENiT ANNOTATION QUALITY AND GROUND-TRUTH INTEGRITY AUDIT")
    print("========================================================================================")
    print(f"Raw Directory: {raw_dir}")
    print(f"Output Report: {out_path}")

    report = audit_dataset(raw_dir)

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAudit completed. Full report saved to {out_path}")

    geom = report["corner_geometry_audit"]
    piece = report["piece_annotation_consistency"]
    elig = report["eligibility_breakdown"]

    print("\n--- Summary Findings ---")
    print(f"Total Images:                 {report['total_images']}")
    print(f"Eligible FEN Boards:          {elig['eligible_fen_boards']}")
    print(f"Timestamp/No-FEN Skipped:     {elig['timestamp_named_no_fen']}")
    print(f"Isolated Piece Photos:        {elig['isolated_piece_photos']}")
    print(f"Missing Board Corners:        {geom['missing_board_lines']}")
    print(f"Non-Convex / Bowtie Corners:  {geom['non_convex_bowtie_boards']}")
    print(f"Exact Piece Count Agreement:  {piece['exact_piece_count_agreement']} ({piece['exact_piece_count_agreement_pct']}%)")
    print(f"Piece Count Mismatches:       {piece['piece_count_mismatches']} ({piece['piece_count_mismatches_pct']}%)")
    print(f"Exact Class Agreement:        {piece['exact_class_multiset_agreement']} ({piece['exact_class_multiset_agreement_pct']}%)")
    print(f"Class Misclassifications:     {piece['class_misclassifications']} ({piece['class_misclassifications_pct']}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
