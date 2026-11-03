#!/usr/bin/env python3
"""Milestone 5.4D — A3 detector benchmark: bounds clamp WITH maximum tolerance.

Re-runs the production-baseline detector with the Milestone 5.4D clamp
tolerance applied (clamped corners may move at most
max(4 px, 2% of the shorter image side); candidates needing more are rejected
with reason ``clamp_excessive`` instead of being moved into frame).

This script:
  1. Evaluates every eligible FENiT image once with the A3 detector.
  2. Measures the corner-clamp distance distribution (the empirical basis for
     the chosen tolerance).
  3. Reads - read-only, never writes - the Milestone 5.3D report and the
     Milestone 5.4C A2 report plus their records to build a before/after
     comparison table, embedding each source file's SHA-256 for provenance.
  4. Writes a NEW report and NEW records; the original 5.3D and A2 artefacts
     are never modified.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import benchmark_board_geometry as BG

DEFAULT_MANIFEST = r"C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl"
DEFAULT_RAW_DIR = r"C:\dev\chess-capture-workspace\external\fenit\raw"
REPORTS_DIR = Path(r"C:\dev\chess-capture-workspace\external\fenit\reports")
DEFAULT_OUT = REPORTS_DIR / "board_geometry_benchmark_a3_clamp_tolerance.json"
DEFAULT_RECORDS = REPORTS_DIR / "board_geometry_benchmark_a3_clamp_tolerance_records.jsonl"
DEFAULT_BASELINE = REPORTS_DIR / "board_geometry_benchmark.json"
DEFAULT_BASELINE_RECORDS = REPORTS_DIR / "board_geometry_benchmark_records.jsonl"
DEFAULT_A2 = REPORTS_DIR / "board_geometry_benchmark_a2_bounds.json"
DEFAULT_A2_RECORDS = REPORTS_DIR / "board_geometry_benchmark_a2_bounds_records.jsonl"

FALSE_POSITIVE_IOU = 0.5


def sha256_of(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 4)


def variant_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Metrics required by the 5.4D report, derived only from per-image records."""
    total = len(records)
    generated = [r for r in records if r.get("candidate_generated")]
    geometry_valid = [r for r in records if r.get("geometry_valid")]
    accepted = [r for r in records if r.get("accepted")]
    false_positives = [r for r in accepted if r["iou"] < FALSE_POSITIVE_IOU]

    reasons: Dict[str, int] = {}
    for record in records:
        if record.get("accepted"):
            continue
        key = record.get("rejection_reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    reasons_sorted = dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))

    mean_accepted_iou = (
        round(float(np.mean([r["iou"] for r in accepted])) * 100.0, 4) if accepted else None
    )
    mean_accepted_err = (
        round(float(np.mean([r["corner_error"] for r in accepted])) * 100.0, 4)
        if accepted
        else None
    )
    penalized_iou = [r["iou"] if r["accepted"] else 0.0 for r in records]
    penalized_err = [r["corner_error"] if r["accepted"] else 1.0 for r in records]

    return {
        "accounting": {
            "total_images": total,
            "candidate_generated_count": len(generated),
            "geometry_valid_count": len(geometry_valid),
            "accepted_count": len(accepted),
            "candidate_generated_pct": round(len(generated) / total * 100.0, 4) if total else None,
            "geometry_valid_pct": round(len(geometry_valid) / total * 100.0, 4) if total else None,
            "accepted_pct": round(len(accepted) / total * 100.0, 4) if total else None,
        },
        "false_positives": {
            "definition": f"accepted with IoU < {FALSE_POSITIVE_IOU} against FENiT ground truth",
            "count": len(false_positives),
            "rate_of_accepted_pct": (
                round(len(false_positives) / len(accepted) * 100.0, 4) if accepted else None
            ),
            "rate_of_all_inputs_pct": (
                round(len(false_positives) / total * 100.0, 4) if total else None
            ),
        },
        "mean_accepted_iou_pct": mean_accepted_iou,
        "mean_accepted_corner_error_pct": mean_accepted_err,
        "all_input_penalized_mean_iou_pct": (
            round(float(np.mean(penalized_iou)) * 100.0, 4) if records else None
        ),
        "all_input_penalized_mean_corner_error_pct": (
            round(float(np.mean(penalized_err)) * 100.0, 4) if records else None
        ),
        "rejection_reasons": reasons_sorted,
    }


def load_variant(label: str, report_path: Path, records_path: Path, approach_filter: Optional[str] = None) -> Dict[str, Any]:
    """Read-only load of a previous benchmark's records for comparison."""
    if not records_path.exists():
        return {"label": label, "available": False, "reason": "records file missing"}
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    if approach_filter:
        records = [r for r in records if r.get("approach") == approach_filter]
    entry: Dict[str, Any] = {
        "label": label,
        "available": True,
        "report_path": str(report_path),
        "report_sha256": sha256_of(report_path),
        "records_path": str(records_path),
        "records_sha256": sha256_of(records_path),
        "metrics": variant_metrics(records),
    }
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entry["benchmark_version"] = report.get("benchmark_metadata", {}).get("benchmark_version")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="A3 clamp-tolerance detector benchmark")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--records-out", default=str(DEFAULT_RECORDS))
    parser.add_argument("--baseline-records", default=str(DEFAULT_BASELINE_RECORDS))
    parser.add_argument("--a2-records", default=str(DEFAULT_A2_RECORDS))
    parser.add_argument("--limit", type=int, default=0, help="0 = all eligible boards")
    parser.add_argument("--from-records", action="store_true", help="Re-generate report from existing records")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    records_path = Path(args.records_out)

    if args.from_records and records_path.exists():
        print(f"Loading existing records from {records_path}...")
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
        total = len(records)
        elapsed = 0.0
    else:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest_records = [json.loads(line) for line in handle]
        processed = [r for r in manifest_records if r.get("status") == "PROCESSED"]
        if args.limit > 0:
            processed = processed[: args.limit]
        total = len(processed)
        print(f"A3 clamp-tolerance benchmark across {total} eligible FENiT board scenes...")
        print(
            f"  tolerance = max({BG.MIN_CLAMP_TOLERANCE_PX}px, "
            f"{BG.MAX_CLAMP_RATIO * 100:.0f}% of shorter image side)"
        )

        t0 = time.perf_counter()
        records: List[Dict[str, Any]] = []
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

            result = BG.detect_approach_a_baseline(img, clamp_to_bounds=True)
            accepted, reason, stage = BG.production_gate(result)
            iou, err = BG._metric_values(result, gt_ordered, shape)
            records.append(
                BG._make_record(
                    "a3_production_clamp_tolerance",
                    file_name,
                    view,
                    result,
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

    # Empirical clamp-distance distribution over the chosen candidate per image.
    clamp_values = [float(r.get("corner_clamp_px") or 0.0) for r in records]
    clamped_nonzero = [v for v in clamp_values if v > 0]
    excessive = [
        r for r in records if r.get("geometry_rejection_reason") == "clamp_excessive"
    ]
    clamp_distribution = {
        "images_with_nonzero_clamp": len(clamped_nonzero),
        "max_px": round(max(clamp_values), 4) if clamp_values else None,
        "p99_px": percentile(clamp_values, 99),
        "p95_px": percentile(clamp_values, 95),
        "p90_px": percentile(clamp_values, 90),
        "p50_px": percentile(clamp_values, 50),
        "tolerance_px_at_640_shorter_side": round(BG.max_clamp_tolerance_px((480, 640)), 4),
        "clamp_excessive_rejections": len(excessive),
        "clamp_excessive_files": [r["file"] for r in excessive][:20],
    }

    comparison = {
        "note": (
            "Previous variants loaded read-only for comparison; their files were "
            "not modified by this run."
        ),
        "milestone_5_3d_baseline_no_clamp": load_variant(
            "5.3D a_production (no clamp)", Path(DEFAULT_BASELINE), Path(args.baseline_records), approach_filter="a_production"
        ),
        "milestone_5_4c_a2_clamp_only": load_variant(
            "5.4C a2 clamp-to-bounds", DEFAULT_A2, Path(args.a2_records)
        ),
        "milestone_5_4d_a3_clamp_with_tolerance": {
            "label": "5.4D a3 clamp + max tolerance",
            "available": True,
            "metrics": variant_metrics(records),
        },
    }

    report = {
        "benchmark_metadata": {
            "benchmark_version": "milestone-5.4d-a3-clamp-tolerance-v1",
            "dataset": "FENiT (Mendeley Data DOI 10.17632/wpp9xnbcp6.1)",
            "images_evaluated": total,
            "records_written": len(records),
            "total_runtime_s": round(elapsed, 2),
            "date": "2026-11-03",
            "derived_from": (
                "Same detector as milestone-5.3d approach a_production, with corners "
                "clamped to image bounds (5.4C) AND a maximum clamp tolerance "
                "(5.4D): candidates needing more than max(4px, 2% of shorter side) "
                "of modification are rejected as clamp_excessive."
            ),
            "provenance": (
                "Original milestone-5.3d report (board_geometry_benchmark.json) and "
                "the milestone-5.4c A2 report are preserved and unmodified; both are "
                "read here only for the comparison table."
            ),
            "acceptance_gates": {
                "a3_production_clamp_tolerance": (
                    "Production structural checks + 8x8 pattern evidence + "
                    "boundary*pattern >= 0.32, on corners clamped to the image frame "
                    "within the maximum clamp tolerance."
                )
            },
        },
        "clamp_tolerance": {
            "max_clamp_ratio": BG.MAX_CLAMP_RATIO,
            "min_clamp_tolerance_px": BG.MIN_CLAMP_TOLERANCE_PX,
            "rationale": (
                "A clamp may absorb contour/rotation overshoot; a candidate needing "
                "more than max(4px, 2% of shorter side) is inventing geometry. "
                "2% of the shorter side is well under half a board square, so a "
                "tolerated clamp cannot move a corner across a square boundary."
            ),
            "measured_distribution": clamp_distribution,
        },
        "before_after_comparison": comparison,
        "approach_a3_production_clamp_tolerance": summary,
        "approach_a3_false_positives": variant_metrics(records)["false_positives"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = BG.write_records(records_path, records)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["records"] = meta
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nA3 summary:")
    print(json.dumps(summary, indent=2))
    print("\nClamp distribution:")
    print(json.dumps(clamp_distribution, indent=2))
    print(f"\nReport: {out_path}")
    print(f"Records: {records_path} (sha256 {meta['records_sha256']})")
    print(f"5.3D report sha256 (read-only): {comparison['milestone_5_3d_baseline_no_clamp'].get('report_sha256')}")
    print(f"A2 report sha256 (read-only):   {comparison['milestone_5_4c_a2_clamp_only'].get('report_sha256')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
