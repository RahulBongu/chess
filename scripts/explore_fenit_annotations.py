"""Explore FENiT annotation structure: class ids, board corners, orientation.

Reads raw/*.txt (one line per annotated object). Class 0 is the board with four
corner keypoints; classes 1..12 are pieces with a centre point. The filename
stem encodes the true FEN placement. We recover which dihedral transform maps
the geometrically ordered board quad to the placement, and which class id
corresponds to which piece character. This is ground truth derived from the
annotations, never from our classifier.
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\dev\chess-capture-workspace\external\fenit")
RAW = ROOT / "raw"

PLACEMENT_RE = re.compile(r"^([1-8rnbqkpRNBQKP]+_)+[1-8rnbqkpRNBQKP]+$")


def parse_placement(stem: str) -> str | None:
    head = stem.strip()
    if head.endswith(" w") or head.endswith(" b"):
        head = head[:-2]
    head = head.split()[0]
    if not PLACEMENT_RE.match(head.replace("/", "_")):
        return None
    groups = head.split("_")
    if len(groups) != 8:
        return None
    for group in groups:
        total = 0
        for ch in group:
            if ch.isdigit():
                total += int(ch)
            elif ch in "rnbqkpRNBQKP":
                total += 1
            else:
                return None
        if total != 8:
            return None
    return "/".join(groups)


def parse_annotation(path: Path):
    board = None
    pieces = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 16:
            continue
        cls = int(parts[0])
        cx, cy = float(parts[1]), float(parts[2])
        if cls == 0:
            # Layout: cls cx cy w h then four (x, y, visibility) keypoints.
            board = [(float(parts[5 + 3 * i]), float(parts[6 + 3 * i])) for i in range(4)]
        else:
            pieces.append((cls, cx, cy))
    return board, pieces


def order_quad(points):
    """Clockwise from the top-left-most corner, in unit-square coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    order = np.argsort(angles)
    pts = pts[order]
    # Ensure clockwise in image coordinates (y down).
    area = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    if area < 0:
        pts = pts[::-1]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_h(h, x, y):
    vec = h @ np.array([x, y, 1.0])
    return vec[0] / vec[2], vec[1] / vec[2]


# Eight dihedral transforms on an 8x8 grid.
def transform_cell(kind, col, row):
    if kind == "identity":
        return col, row
    if kind == "rot90":
        return 7 - row, col
    if kind == "rot180":
        return 7 - col, 7 - row
    if kind == "rot270":
        return row, 7 - col
    if kind == "flip_h":
        return 7 - col, row
    if kind == "flip_v":
        return col, 7 - row
    if kind == "transpose":
        return row, col
    if kind == "anti_transpose":
        return 7 - row, 7 - col
    raise ValueError(kind)


TRANSFORMS = [
    "identity", "rot90", "rot180", "rot270",
    "flip_h", "flip_v", "transpose", "anti_transpose",
]


def fen_char_at(placement: str, col: int, row: int) -> str:
    field = placement.split("/")[row]
    index = 0
    for ch in field:
        if ch.isdigit():
            index += int(ch)
        else:
            if index == col:
                return ch
            index += 1
    return ""


def main():
    stems = []
    for txt in sorted(RAW.glob("*.txt")):
        placement = parse_placement(txt.stem)
        if placement is None:
            continue
        board, pieces = parse_annotation(txt)
        if board is None or len(pieces) < 2:
            continue
        stems.append((txt.stem, placement, board, pieces))

    print("usable boards:", len(stems))

    class_votes = collections.defaultdict(collections.Counter)
    transform_scores = collections.Counter()
    transform_margins = []
    skipped = 0

    for stem, placement, board, pieces in stems:
        quad = order_quad(board)
        h = homography(quad, [(0, 0), (8, 0), (8, 8), (0, 8)])
        cells = []
        for cls, cx, cy in pieces:
            u, v = apply_h(h, cx, cy)
            col, row = int(math.floor(u)), int(math.floor(v))
            if not (0 <= col < 8 and 0 <= row < 8):
                continue
            cells.append((cls, col, row))
        if len(cells) < 2:
            skipped += 1
            continue

        best = None
        scores = {}
        for kind in TRANSFORMS:
            votes = collections.defaultdict(collections.Counter)
            for cls, col, row in cells:
                tc, tr = transform_cell(kind, col, row)
                ch = fen_char_at(placement, tc, tr)
                if ch:
                    votes[cls][ch] += 1
            score = sum(counter.most_common(1)[0][1] for counter in votes.values())
            scores[kind] = score
            if best is None or score > best[1]:
                best = (kind, score, votes)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        transform_scores[best[0]] += 1
        transform_margins.append(ranked[0][1] - ranked[1][1])
        for cls, counter in best[2].items():
            class_votes[cls].update(counter)

    print("orientation distribution:", dict(transform_scores))
    print("median margin:", float(np.median(transform_margins)))
    print("skipped:", skipped)
    print("class id -> piece char votes:")
    for cls in sorted(class_votes):
        print(cls, class_votes[cls].most_common(3), "total", sum(class_votes[cls].values()))


if __name__ == "__main__":
    main()
