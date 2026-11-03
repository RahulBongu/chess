#!/usr/bin/env python3
"""Milestone 5.4C — A2 bounds-clamped production detector benchmark.

Re-runs ONLY the production-baseline detector with proposed corners clamped
into the image bounds before structural validation (mirrors the Milestone
5.4C fix in src/vision/detectBoard.ts + src/vision/boardDetection.ts).

Provenance preservation: the original Milestone 5.3D report
(board_geometry_benchmark.json / _records.jsonl) and its SHA-256 are never
read, modified or overwritten by this script. Results are written to a new,
distinct report file.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

import benchmark_board_geometry as BG

DEFAULT_MANIFEST = r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl"
DEFAULT_RAW_DIR = r"C:\dev\chess-capture-workspace\external\fenit\raw"
DEFAULT_OUT = r"C:\dev\chess-capture-workspace\external\fenit\reports\board_geometry_benchmark_a2_bounds.json"
DEFAULT_RECORDS = r"C:\dev\chess-capture-workspace\external\fenit\reports\board_geometry_benchmark_a2_bounds_records.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="A2 (bounds-clamped) detector benchmark")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--records-out", default=DEFAULT_RECORDS)
    parser.add_argument("--limit", type=int, default=0, help="0 = all eligible boards")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_records = [json.loads(line) for line in handle]
    processed = [r for r in manifest_records if r.get("status") == "PROCESSED"]
    if args.limit > 0:
        processed = processed[: args.limit]
    total = len(processed)
    print(f"A2 bounds-clamped benchmark across {total} eligible FENiT board scenes...")

    t0 = time.perf_counter()
    records = []
    for idx, manifest_record in enumerate(processed, 1):
        file_name = manifest_record["file"]
        img = cv2.imread(str(raw_dir / file_name))
        orientation = manifest_record.get("resolved_orientation", "unknown")
        view = "frontal" if orientation in {"white_perspective", "black_perspective"} else "lateral"
        if img is None:
            records.extend(BG._unreadable_records(file_name, view))
            continue
        height, width = img.shape[:2]
        shape = (height, width)
        gt = np.array(manifest_record["board_corners"], dtype=np.float32)
        gt[:, 0] *= width
        gt[:, 1] *= height
        gt_ordered = BG.order_corners_screen_space(gt)

        result_a2 = BG.detect_approach_a_baseline(img, clamp_to_bounds=True)
        accepted, reason, stage = BG.production_gate(result_a2)
        iou, err = BG._metric_values(result_a2, gt_ordered, shape)
        records.append(
            BG._make_record(
                "a2_production_clamped_bounds",
                file_name,
                view,
                result_a2,
                accepted,
                reason,
                stage,
                iou,
                err,
            )
        )
        if idx % 50 == 0 or idx == total:
            print(f"  [{idx}/{total}] completed...", flush=True)

    elapsed = time.perf_counter() - t0
    summary = BG.summarize_approach(records)
    BG.assert_summary_matches_records(summary, records)

    report = {
        "benchmark_metadata": {
            "benchmark_version": "milestone-5.4c-a2-bounds-v1",
            "dataset": "FENiT (Mendeley Data DOI 10.17632/wpp9xnbcp6.1)",
            "images_evaluated": total,
            "records_written": len(records),
            "total_runtime_s": round(elapsed, 2),
            "date": "2026-11-03",
            "derived_from": "Same detector as milestone-5.3d approach a_production with corners clamped to image bounds before validation.",
            "provenance": "Original milestone-5.3d report (board_geometry_benchmark.json) is preserved and unmodified.",
            "acceptance_gates": {
                "a2_production_clamped_bounds": "Production structural checks + 8x8 pattern evidence + boundary*pattern >= 0.32, on corners clamped to the image frame."
            },
        },
        "approach_a2_production_clamped_bounds": summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    records_path = Path(args.records_out)
    meta = BG.write_records(records_path, records)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["records"] = meta
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nA2 summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nReport: {out_path}")
    print(f"Records: {records_path} (sha256 {meta['records_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
