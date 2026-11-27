#!/usr/bin/env python3
"""Regression guard for the measured laptop-screen accuracy baseline.

The values here are the audited 2026-11-06 production (`universal150`) results
on the three laptop frames. They are pinned so a future benchmark change cannot
silently "fix" a number by changing corners, labels or sampling. Missing JSON
is reported as skipped, never as a pass.

This tool is screen-domain only. It does not claim physical-board accuracy and
does not imply the vivo corpus has been captured.

Usage:
  py -3 scripts/check_screen_accuracy_regression.py
  py -3 scripts/check_screen_accuracy_regression.py --selftest
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

DEFAULT_JSON = Path("dist/milestone_54i5_screen_benchmark.json")
MODEL = "universal150_prod"

# Audited device-path values from docs/MILESTONE_54I5_SCREEN_BENCHMARK_AUDIT.md.
EXPECTED = {
    "brown_start_device": {
        "per_square_exact_n": 51,
        "occ": 64,
        "occupied_exact": "19/32",
        "colour": "32/32",
        "uncertain": 16,
    },
    "green_start_device": {
        "per_square_exact_n": 23,
        "uncertain": 43,
    },
    "green_e4e5_device": {
        "per_square_exact_n": 31,
        "uncertain": 39,
    },
}


def check_report(data: dict) -> list[str]:
    """Returns a list of failures; empty means the measured baseline is intact."""
    failures: list[str] = []
    results = data.get("results", {})
    model = results.get(MODEL)
    if not model:
        return [f"missing results for {MODEL}"]
    if data.get("asset_equals_universal150") is not True:
        failures.append("asset_equals_universal150 is not true")
    for board, expected in EXPECTED.items():
        metric = model.get(board)
        if not metric:
            failures.append(f"missing {board}")
            continue
        for key, value in expected.items():
            if metric.get(key) != value:
                failures.append(
                    f"{board}.{key} changed: got {metric.get(key)!r}, expected {value!r}"
                )
        if not isinstance(metric.get("board_exact"), bool):
            failures.append(f"{board}.board_exact must be a bool")
        if metric.get("board_exact") is True:
            failures.append(f"{board} unexpectedly claims a fully exact board")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    args = parser.parse_args()

    if args.selftest:
        path = Path(tempfile.mkdtemp(prefix="i5accuracy_")) / "benchmark.json"
        good = {
            "asset_equals_universal150": True,
            "results": {MODEL: {board: dict(values) for board, values in EXPECTED.items()}},
        }
        for board in EXPECTED:
            good["results"][MODEL][board]["board_exact"] = False
        path.write_text(json.dumps(good), encoding="utf-8")
        if check_report(good):
            print("SELFTEST FAIL valid baseline rejected")
            return 1
        tampered = json.loads(json.dumps(good))
        tampered["results"][MODEL]["brown_start_device"]["per_square_exact_n"] = 64
        failures = check_report(tampered)
        if not any("per_square_exact_n changed" in failure for failure in failures):
            print("SELFTEST FAIL tampered baseline accepted")
            return 1
        path.unlink()
        path.parent.rmdir()
        print("SELFTEST OK accuracy baseline guard detects tampered metrics")
        return 0

    json_path = Path(args.json)
    if not json_path.is_file():
        print(f"SKIP {json_path} not present; no laptop-screen result to check")
        return 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    failures = check_report(data)
    for failure in failures:
        print(f"FAIL {failure}")
    print(
        f"accuracy baseline {'INTACT' if not failures else 'CHANGED'} "
        f"({len(EXPECTED)} boards, {len(failures)} failures)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
