#!/usr/bin/env python3
"""Build and validate the Milestone 5.4I.5 vivo screen-capture manifest.

Input layout (one directory per capture session):

    <capture-root>/
      sessions.json              {"s01": "train_screen", "s05": "val_screen", ...}
      s01/
        frame_0001.png           full classifier-input frame (native dump) or screencap
        frame_0001.json          sidecar: corners, FEN, condition, device, apk hash
        ...
      s05/
        ...

Sidecar fields (see docs/VIVO_SCREEN_CAPTURE_PROTOCOL_54I5.md):

    frame, session_id, split, position_id, fen, corners_norm,
    orientation, theme, brightness_pct, distance, angle_deg, source,
    device, apk_sha256, captured_at

The builder refuses to guess. Every frame must have a sidecar, a legal FEN,
a convex ordered corner quad, and a session that maps to exactly one split.
It writes JSONL plus a summary and exits non-zero on any error, or on a
warning unless `--allow-warnings` is passed. No metrics are produced here:
this tool only decides whether a corpus is safe to evaluate.

Usage:
  py -3 scripts/build_screen_capture_manifest.py --capture-root <dir> --out dist/capture_i5
  py -3 scripts/build_screen_capture_manifest.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SPLITS = {"train_screen", "val_screen", "holdout_screen"}
MIN_BURST_FRAMES = 10
# Frame corners must stay inside the sensor. A little float tolerance only.
CORNER_EPSILON = 1e-6


class FatalError(Exception):
    pass


@dataclass
class FrameIssue:
    path: str
    errors: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int, int] | None:
    """Reads PNG IHDR dimensions without an image library dependency."""
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def validate_fen(fen: str) -> list[str]:
    """Minimal six-field FEN checks needed before a label can be trusted."""
    errors: list[str] = []
    fields = fen.strip().split()
    if len(fields) != 6:
        return [f"FEN must have 6 fields, got {len(fields)}"]
    placement, turn, castling, ep, halfmove, fullmove = fields
    ranks = placement.split("/")
    if len(ranks) != 8:
        errors.append(f"FEN placement must have 8 ranks, got {len(ranks)}")
        return errors
    counts = {"white": 0, "black": 0}
    kings = {"w": 0, "b": 0}
    pawns_on_back = []
    for rank_index, rank in enumerate(ranks):
        rank_number = 8 - rank_index
        files = 0
        for symbol in rank:
            if symbol.isdigit():
                files += int(symbol)
                continue
            if symbol not in "pnbrqkPNBRQK":
                errors.append(f"invalid piece symbol {symbol!r}")
                continue
            files += 1
            colour = "white" if symbol.isupper() else "black"
            counts[colour] += 1
            if symbol.lower() == "k":
                kings["w" if symbol.isupper() else "b"] += 1
            if symbol.lower() == "p" and rank_number in (1, 8):
                pawns_on_back.append(f"{symbol}@{rank_number}")
        if files != 8:
            errors.append(f"rank {rank_number} covers {files} files, expected 8")
    if kings["w"] != 1 or kings["b"] != 1:
        errors.append(f"expected exactly one king per side, got w={kings['w']} b={kings['b']}")
    if counts["white"] > 16 or counts["black"] > 16:
        errors.append(f"too many pieces: {counts}")
    if pawns_on_back:
        errors.append(f"pawns on back rank: {', '.join(pawns_on_back)}")
    if turn not in ("w", "b"):
        errors.append(f"invalid side to move {turn!r}")
    if ep != "-":
        if len(ep) != 2 or ep[0] not in "abcdefgh" or ep[1] not in "36":
            errors.append(f"invalid en-passant target {ep!r}")
    if not halfmove.isdigit() or not fullmove.isdigit() or int(fullmove) < 1:
        errors.append("invalid halfmove/fullmove counters")
    return errors


def order_corners(points: list[list[float]]) -> list[list[float]]:
    """Orders four points TL, TR, BR, BL using the sum/difference rule."""
    sums = [p[0] + p[1] for p in points]
    diffs = [p[0] - p[1] for p in points]
    tl = points[sums.index(min(sums))]
    br = points[sums.index(max(sums))]
    tr = points[diffs.index(max(diffs))]
    bl = points[diffs.index(min(diffs))]
    return [tl, tr, br, bl]


def is_convex(points: list[list[float]]) -> bool:
    cross = []
    for index in range(4):
        ax, ay = points[index]
        bx, by = points[(index + 1) % 4]
        cx, cy = points[(index + 2) % 4]
        cross.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    return all(value > 0 for value in cross) or all(value < 0 for value in cross)


def validate_corners(raw: object) -> tuple[list[float] | None, list[str]]:
    """Validates the documented flat `[TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy]` form."""
    errors: list[str] = []
    if not isinstance(raw, list) or len(raw) != 8:
        return None, ["corners_norm must be a flat list of 8 numbers (TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy)"]
    values: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"corner value {index} must be a number")
            continue
        coordinate = float(value)
        if not (-CORNER_EPSILON <= coordinate <= 1 + CORNER_EPSILON):
            errors.append(f"corner value {index} outside the normalized frame: {coordinate:.4f}")
        values.append(coordinate)
    if len(values) != 8:
        return None, errors
    points = [[values[i], values[i + 1]] for i in range(0, 8, 2)]
    ordered = order_corners(points)
    if ordered != points:
        errors.append("corners_norm must already be ordered TL, TR, BR, BL")
    if not is_convex(points):
        errors.append("corner quad is not convex")
    width = max(p[0] for p in points) - min(p[0] for p in points)
    height = max(p[1] for p in points) - min(p[1] for p in points)
    if width < 0.2 or height < 0.2:
        errors.append(f"corner quad too small to be a board: {width:.3f} x {height:.3f}")
    return (values if not errors else None), errors


def load_sessions(root: Path) -> dict[str, str]:
    sessions_path = root / "sessions.json"
    if not sessions_path.exists():
        raise FatalError(f"missing {sessions_path}")
    try:
        data = json.loads(sessions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FatalError(f"invalid sessions.json: {error}") from error
    if not isinstance(data, dict):
        raise FatalError("sessions.json must map session_id -> split")
    for session, split in data.items():
        if split not in SPLITS:
            raise FatalError(f"session {session}: split must be one of {sorted(SPLITS)}, got {split!r}")
    return data


def build(capture_root: Path, out_dir: Path) -> tuple[int, dict]:
    sessions = load_sessions(capture_root)
    records: list[dict] = []
    issues: list[FrameIssue] = []
    warnings: list[str] = []
    seen_hashes: dict[str, str] = {}
    per_session_counts: dict[str, int] = {}
    per_position_counts: dict[tuple[str, str], int] = {}

    session_dirs = sorted(p for p in capture_root.iterdir() if p.is_dir())
    for session_dir in session_dirs:
        session_id = session_dir.name
        if session_id not in sessions:
            warnings.append(f"{session_id}: directory has no entry in sessions.json; skipped")
            continue
        split = sessions[session_id]
        frames = sorted(session_dir.glob("*.png"))
        if not frames:
            issues.append(FrameIssue(str(session_dir), ["session has no PNG frames"]))
            continue
        for frame_path in frames:
            issue = FrameIssue(str(frame_path))
            sidecar_path = frame_path.with_suffix(".json")
            if not sidecar_path.exists():
                issue.errors.append("missing sidecar JSON")
                issues.append(issue)
                continue
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                issue.errors.append(f"invalid sidecar JSON: {error}")
                issues.append(issue)
                continue

            if sidecar.get("session_id") != session_id:
                issue.errors.append(f"sidecar session_id {sidecar.get('session_id')!r} != directory {session_id!r}")
            if sidecar.get("split") != split:
                issue.errors.append(f"sidecar split {sidecar.get('split')!r} != sessions.json {split!r}")
            if sidecar.get("frame") not in (None, frame_path.name):
                issue.errors.append(f"sidecar frame {sidecar.get('frame')!r} != {frame_path.name!r}")

            fen = str(sidecar.get("fen", "")).strip()
            fen_errors = validate_fen(fen)
            issue.errors.extend(f"FEN: {error}" for error in fen_errors)

            corners, corner_errors = validate_corners(sidecar.get("corners_norm"))
            issue.errors.extend(corner_errors)

            size = png_size(frame_path)
            if size is None:
                issue.errors.append("frame is not a readable PNG")
            else:
                declared = sidecar.get("frame_size")
                if declared is not None and [int(declared[0]), int(declared[1])] != list(size):
                    issue.errors.append(f"frame_size {declared} != PNG {list(size)}")

            for required in (
                "position_id",
                "orientation",
                "theme",
                "brightness_pct",
                "distance",
                "angle_deg",
                "source",
                "device",
                "apk_sha256",
            ):
                if sidecar.get(required) is None:
                    issue.errors.append(f"missing required field {required!r}")

            source = sidecar.get("source")
            if source is not None and source not in ("native-dump", "screencap"):
                issue.errors.append(f"source must be 'native-dump' or 'screencap', got {source!r}")
            apk_sha = sidecar.get("apk_sha256")
            if apk_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(apk_sha)):
                issue.errors.append("apk_sha256 must be 64 lowercase hex characters")
            brightness = sidecar.get("brightness_pct")
            if brightness is not None and (
                isinstance(brightness, bool)
                or not isinstance(brightness, (int, float))
                or not 0 <= float(brightness) <= 100
            ):
                issue.errors.append(f"brightness_pct must be 0-100, got {brightness!r}")
            distance = sidecar.get("distance")
            if distance is not None and distance not in ("near", "nominal", "far"):
                issue.errors.append(f"distance must be near/nominal/far, got {distance!r}")
            angle = sidecar.get("angle_deg")
            if angle is not None and (
                isinstance(angle, bool) or not isinstance(angle, (int, float))
            ):
                issue.errors.append(f"angle_deg must be numeric, got {angle!r}")

            digest = sha256_file(frame_path)
            if digest in seen_hashes:
                issue.errors.append(f"duplicate frame content (same bytes as {seen_hashes[digest]})")
            else:
                seen_hashes[digest] = str(frame_path)

            if issue.errors:
                issues.append(issue)
                continue

            position_id = str(sidecar["position_id"])
            per_session_counts[session_id] = per_session_counts.get(session_id, 0) + 1
            key = (session_id, position_id)
            per_position_counts[key] = per_position_counts.get(key, 0) + 1
            records.append(
                {
                    "frame": frame_path.name,
                    "relpath": frame_path.relative_to(capture_root).as_posix(),
                    "session_id": session_id,
                    "split": split,
                    "position_id": position_id,
                    "fen": fen,
                    "corners_norm": corners,
                    "orientation": sidecar.get("orientation", "white_bottom"),
                    "theme": sidecar.get("theme"),
                    "brightness_pct": sidecar.get("brightness_pct"),
                    "distance": sidecar.get("distance"),
                    "angle_deg": sidecar.get("angle_deg"),
                    "source": sidecar["source"],
                    "device": sidecar["device"],
                    "apk_sha256": sidecar["apk_sha256"],
                    "captured_at": sidecar.get("captured_at"),
                    "sha256": digest,
                }
            )

    # A session directory maps to exactly one split in sessions.json, so a
    # frame can never inherit a split from a different session.
    for split in sorted(SPLITS):
        if not any(record["split"] == split for record in records):
            warnings.append(f"split {split} has no valid frames")

    for (session_id, position_id), count in sorted(per_position_counts.items()):
        if count < MIN_BURST_FRAMES:
            warnings.append(f"{session_id}/{position_id}: only {count} frames (burst target {MIN_BURST_FRAMES})")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = {
        "capture_root": str(capture_root),
        "manifest": str(manifest_path),
        "sessions": sessions,
        "frames": len(records),
        "per_session": per_session_counts,
        "per_split": {
            split: sum(1 for record in records if record["split"] == split) for split in sorted(SPLITS)
        },
        "positions": {
            f"{session}/{position}": count for (session, position), count in sorted(per_position_counts.items())
        },
        "errors": [{"path": issue.path, "errors": issue.errors} for issue in issues],
        "warnings": warnings,
        "ok": not issues,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return len(issues), summary


def write_png(path: Path, width: int, height: int, shade: int = 0x7F) -> None:
    """Writes a tiny flat PNG using only the standard library.

    The shade lets a fixture produce distinct frame bytes, so the duplicate
    guard can be exercised against genuinely different captures.
    """
    import zlib

    pixel = bytes((shade, shade, shade))
    raw = b"".join(b"\x00" + pixel * width for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def selftest() -> int:
    root = Path(tempfile.mkdtemp(prefix="i5capture_"))
    failures: list[str] = []
    try:
        (root / "sessions.json").write_text(
            json.dumps({"s01": "train_screen", "s05": "val_screen"}), encoding="utf-8"
        )
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        base_sidecar = {
            "session_id": "s01",
            "split": "train_screen",
            "position_id": "start",
            "fen": start_fen,
            "corners_norm": [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
            "orientation": "white_bottom",
            "theme": "brown",
            "brightness_pct": 70,
            "distance": "nominal",
            "angle_deg": 0,
            "source": "native-dump",
            "device": "test-device",
            "apk_sha256": "0" * 64,
            "captured_at": "2026-11-07T18:00:00+05:30",
        }
        session_dir = root / "s01"
        session_dir.mkdir()
        for index in range(MIN_BURST_FRAMES):
            frame = session_dir / f"f{index:03d}.png"
            write_png(frame, 8, 8, shade=16 + index * 8)
            sidecar = dict(base_sidecar)
            sidecar["frame"] = frame.name
            sidecar["frame_size"] = [8, 8]
            frame.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")

        issues, summary = build(root, root / "_out")
        if issues or not summary["ok"]:
            failures.append(f"clean corpus rejected: {summary['errors']} {summary['warnings']}")
        if summary["frames"] != MIN_BURST_FRAMES:
            failures.append(f"expected {MIN_BURST_FRAMES} frames, got {summary['frames']}")

        # 1. Unknown session directory is a warning, not silent inclusion.
        (root / "s99").mkdir()
        write_png(root / "s99" / "x.png", 8, 8)
        _, summary = build(root, root / "_out")
        if not any("s99" in warning for warning in summary["warnings"]):
            failures.append("unknown session was not warned about")
        shutil.rmtree(root / "s99")

        # 2. Illegal FEN must fail.
        bad = session_dir / "badfen.png"
        write_png(bad, 8, 8)
        sidecar = dict(base_sidecar)
        sidecar["frame"] = bad.name
        sidecar["fen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0"
        bad.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        issues, summary = build(root, root / "_out")
        if not any("6 fields" in error for issue in summary["errors"] for error in issue["errors"]):
            failures.append("short FEN was not rejected")
        bad.unlink()
        bad.with_suffix(".json").unlink()

        # 3. Unordered / tiny corners must fail.
        bad = session_dir / "badcorners.png"
        write_png(bad, 8, 8)
        sidecar = dict(base_sidecar)
        sidecar["frame"] = bad.name
        sidecar["corners_norm"] = [0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9]
        bad.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        _, summary = build(root, root / "_out")
        if not any("ordered" in error for issue in summary["errors"] for error in issue["errors"]):
            failures.append("unordered corners were not rejected")
        bad.unlink()
        bad.with_suffix(".json").unlink()

        # 4. Missing sidecar must fail.
        orphan = session_dir / "orphan.png"
        write_png(orphan, 8, 8)
        _, summary = build(root, root / "_out")
        if not any("sidecar" in error for issue in summary["errors"] for error in issue["errors"]):
            failures.append("missing sidecar was not rejected")
        orphan.unlink()

        # 5. A session split mismatch must fail.
        mismatch = root / "s05"
        mismatch.mkdir()
        frame = mismatch / "f000.png"
        write_png(frame, 8, 8)
        sidecar = dict(base_sidecar)
        sidecar.update({"session_id": "s05", "split": "train_screen", "frame": frame.name})
        frame.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        _, summary = build(root, root / "_out")
        if not any("sessions.json" in error for issue in summary["errors"] for error in issue["errors"]):
            failures.append("session split mismatch was not rejected")
        shutil.rmtree(mismatch)

        # 6. Duplicate bytes must fail.
        duplicate = session_dir / "dup.png"
        shutil.copyfile(session_dir / "f000.png", duplicate)
        sidecar = dict(base_sidecar)
        sidecar["frame"] = duplicate.name
        duplicate.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        _, summary = build(root, root / "_out")
        if not any("duplicate" in error for issue in summary["errors"] for error in issue["errors"]):
            failures.append("duplicate frame bytes were not rejected")
        duplicate.unlink()
        duplicate.with_suffix(".json").unlink()

        # 7. PNG dimension mismatch must fail.
        frame = session_dir / "f001.png"
        sidecar_path = frame.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["frame_size"] = [64, 64]
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        _, summary = build(root, root / "_out")
        if not any("frame_size" in error for issue in summary["errors"] for error in issue["errors"]):
            failures.append("frame_size mismatch was not rejected")
        sidecar["frame_size"] = [8, 8]
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

        # 8. The documented flat corner array is required; nested pairs drift.
        bad = session_dir / "badshape.png"
        write_png(bad, 8, 8)
        sidecar = dict(base_sidecar)
        sidecar["frame"] = bad.name
        sidecar["corners_norm"] = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        bad.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        _, summary = build(root, root / "_out")
        if not any("flat list" in error for issue in summary["errors"] for error in issue["errors"]):
            failures.append("nested corner pairs were not rejected")
        bad.unlink()
        bad.with_suffix(".json").unlink()

        # 9. Missing protocol fields and malformed hashes must fail.
        bad = session_dir / "badmeta.png"
        write_png(bad, 8, 8)
        sidecar = dict(base_sidecar)
        sidecar["frame"] = bad.name
        sidecar.pop("theme")
        sidecar["apk_sha256"] = "not-a-hash"
        sidecar["source"] = "screenshot"
        bad.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        _, summary = build(root, root / "_out")
        errors = [error for issue in summary["errors"] for error in issue["errors"]]
        for expected in ("missing required field 'theme'", "apk_sha256", "source must be"):
            if not any(expected in error for error in errors):
                failures.append(f"metadata drift not rejected: {expected}")
        bad.unlink()
        bad.with_suffix(".json").unlink()

        # 10. A valid build emits the documented flat corners and relpath.
        _, summary = build(root, root / "_out")
        manifest_lines = (root / "_out" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        first = json.loads(manifest_lines[0])
        if len(first["corners_norm"]) != 8 or not all(isinstance(v, float) for v in first["corners_norm"]):
            failures.append("manifest corners_norm is not the flat 8-number form")
        if first["relpath"] != f"s01/{first['frame']}":
            failures.append(f"manifest relpath is wrong: {first['relpath']!r}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if failures:
        for failure in failures:
            print(f"SELFTEST FAIL {failure}")
        return 1
    print("SELFTEST OK capture manifest builder rejects every malformed fixture")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", help="directory containing sessions.json and session subdirectories")
    parser.add_argument("--out", default="dist/capture_i5", help="output directory for manifest.jsonl + summary.json")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--allow-warnings", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.capture_root:
        parser.error("--capture-root is required unless --selftest is used")

    capture_root = Path(args.capture_root)
    if not capture_root.is_dir():
        print(f"capture root not found: {capture_root}")
        return 2
    try:
        issue_count, summary = build(capture_root, Path(args.out))
    except FatalError as error:
        print(f"FATAL {error}")
        return 2

    for warning in summary["warnings"]:
        print(f"WARN {warning}")
    for issue in summary["errors"]:
        print(f"ERROR {issue['path']}: {'; '.join(issue['errors'])}")
    print(
        f"frames={summary['frames']} splits={summary['per_split']} "
        f"errors={issue_count} warnings={len(summary['warnings'])} manifest={summary['manifest']}"
    )
    if issue_count:
        return 1
    if summary["warnings"] and not args.allow_warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
