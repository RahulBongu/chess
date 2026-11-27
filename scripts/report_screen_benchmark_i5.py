#!/usr/bin/env python3
"""Render a concise accuracy report from a Milestone 5.4I.5 evaluator JSON.

This is a reporting tool only. It never changes labels, thresholds, models or
the manifest. It prints screen-crop results with explicit n, separates splits,
and applies the documented promotion rule (a candidate must win on holdout
per-square exact, occupied-exact and colour, without more wrong-colour home
squares). With no holdout, the verdict is always "keep production".

Usage:
  py -3 scripts/report_screen_benchmark_i5.py --json dist/milestone_54i5_vivo_benchmark.json
  py -3 scripts/report_screen_benchmark_i5.py --json <file> --out dist/accuracy_i5.md
  py -3 scripts/report_screen_benchmark_i5.py --selftest
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

PRODUCTION = "universal150_prod"


def rate(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def count(value) -> str:
    return "n/a" if value is None else str(value)


def metric_row(model: str, data: dict) -> str:
    frame = data.get("frame_weighted") or {}
    gate = frame.get("start_gate") or {}
    burst = (data.get("burst") or {}).get("summary") or {}
    burst_gate = burst.get("start_gate") or {}
    return "| " + " | ".join(
        [
            f"`{model}`",
            count(frame.get("frames")),
            rate(frame.get("per_square_exact_rate")),
            count(frame.get("board_exact")),
            rate(frame.get("occupancy_rate")),
            count(frame.get("occupied_exact")),
            count(frame.get("colour")),
            rate(frame.get("mean_confidence")),
            count(frame.get("mean_uncertain")),
            rate((frame.get("calibration") or {}).get("ece")),
            count(gate.get("pass")),
            rate(burst.get("per_square_exact_rate")),
            count(burst_gate.get("pass")),
        ]
    ) + " |"


def promotion_verdict(report: dict) -> dict:
    """Applies the documented holdout promotion rule without relaxing it."""
    results = report.get("results", {})
    if "holdout_screen" not in results:
        return {
            "keep_production": PRODUCTION,
            "reason": "No holdout_screen evaluation; models stay unchanged.",
        }
    models = results["holdout_screen"].get("models", {})
    prod = models.get(PRODUCTION, {}).get("frame_weighted")
    if not prod:
        return {
            "keep_production": PRODUCTION,
            "reason": "Holdout has no production result; models stay unchanged.",
        }

    def number(metric: dict, key: str) -> float | None:
        value = metric.get(key)
        return None if value is None else float(value)

    prod_gate = prod.get("start_gate") or {}
    prod_wrong = prod_gate.get("mean_wrong_colour")
    prod_wrong = float(prod_wrong) if prod_wrong is not None else None
    winners = []
    for name, entry in models.items():
        if name == PRODUCTION or not entry.get("frame_weighted"):
            continue
        metrics = entry["frame_weighted"]
        wins = (
            number(metrics, "per_square_exact_rate") is not None
            and number(metrics, "per_square_exact_rate") > number(prod, "per_square_exact_rate")
            and number(metrics, "occupied_exact_rate") is not None
            and number(metrics, "occupied_exact_rate") > number(prod, "occupied_exact_rate")
            and number(metrics, "colour_rate") is not None
            and number(metrics, "colour_rate") > number(prod, "colour_rate")
        )
        wrong = (metrics.get("start_gate") or {}).get("mean_wrong_colour")
        wrong = float(wrong) if wrong is not None else None
        if wins and wrong is not None and prod_wrong is not None and wrong <= prod_wrong:
            winners.append(name)
    if winners:
        return {
            "keep_production": None,
            "promote": sorted(winners)[0],
            "reason": "Holdout wins on all three metrics without more wrong-colour home squares.",
        }
    return {
        "keep_production": PRODUCTION,
        "reason": "No candidate wins all three holdout metrics without more wrong-colour home squares.",
    }


def decision_payload(report: dict) -> dict:
    """Machine-readable version of the promotion decision and its inputs."""
    verdict = promotion_verdict(report)
    return {
        "milestone": "5.4I.5",
        "generated_at": report.get("generated_at"),
        "manifest": report.get("manifest"),
        "manifest_sha256": report.get("manifest_sha256"),
        "source": report.get("source"),
        "splits": report.get("splits", []),
        "models": report.get("models", {}),
        "verdict": verdict,
    }


def render(report: dict) -> str:
    verdict = promotion_verdict(report)
    lines = [
        "# Milestone 5.4I.5 screen accuracy report",
        "",
        f"- Source: `{report.get('manifest', 'n/a')}` (SHA-256 `{report.get('manifest_sha256', 'n/a')}`)",
        f"- Capture source: `{report.get('source', 'n/a')}`",
        f"- Splits: {', '.join(f'`{split}`' for split in report.get('splits', [])) or 'n/a'}",
        f"- Models: {', '.join(f'`{name}`' for name in report.get('models', {})) or 'n/a'}",
        f"- Generated: {report.get('generated_at', 'n/a')}",
        "",
        "## Per split (frame-weighted; `burst` averages probabilities per burst first)",
        "",
        "| Split | Model | Frames | per-square | board-exact | occupancy | occupied-exact | colour | mean conf | mean uncertain | ECE | gate pass | burst per-square | burst gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, split_data in report.get("results", {}).items():
        models = split_data.get("models", {})
        if not models:
            lines.append(f"| `{split}` | — | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        for name in [PRODUCTION] + sorted(set(models) - {PRODUCTION}):
            if name in models:
                lines.append(f"| `{split}` | " + metric_row(name, models[name])[2:])
    lines.extend(["", "## Promotion decision", "", f"- {verdict['reason']}"])
    if verdict.get("promote"):
        lines.append(f"- Candidate: `{verdict['promote']}` (holdout evidence supports a swap).")
    else:
        lines.append(f"- Production stays `{verdict['keep_production']}`.")
    warnings = report.get("coverage_warnings", [])
    lines.extend(["", "## Coverage warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Screen-camera crops only. Never pool these numbers with physical 3D-board results.",
            "- Start-gate pass rate is conservative (per frame); on-device acceptance remains authoritative.",
            "",
        ]
    )
    return "\n".join(lines)


def selftest() -> int:
    temp = Path(tempfile.mkdtemp(prefix="i5report_"))
    try:
        base = {
            "generated_at": "2026-11-08T00:00:00+00:00",
            "manifest": "dist/capture_i5/manifest.jsonl",
            "manifest_sha256": "abc",
            "source": "native-dump",
            "splits": ["val_screen"],
            "models": {PRODUCTION: {"sha256": "0" * 64}},
            "results": {
                "val_screen": {
                    "models": {
                        PRODUCTION: {
                            "frame_weighted": {
                                "frames": 10,
                                "per_square_exact_rate": 0.8,
                                "board_exact": "2/10",
                                "occupancy_rate": 0.9,
                                "occupied_exact": "20/32",
                                "occupied_exact_rate": 0.625,
                                "colour": "30/32",
                                "colour_rate": 0.9375,
                                "mean_confidence": 0.88,
                                "mean_uncertain": 5.2,
                                "calibration": {"ece": 0.07},
                                "start_gate": {"pass": "8/10", "mean_wrong_colour": 0.0},
                            },
                            "burst": {
                                "summary": {
                                    "per_square_exact_rate": 0.82,
                                    "start_gate": {"pass": "1/1"},
                                }
                            },
                        }
                    }
                }
            },
        }
        text = render(base)
        for expected in ("val_screen", "per-square", "burst per-square", "Production stays", "Coverage warnings"):
            if expected not in text:
                print(f"SELFTEST FAIL missing {expected!r}")
                return 1
        if promotion_verdict(base)["keep_production"] != PRODUCTION:
            print("SELFTEST FAIL verdict changed without holdout")
            return 1
        holdout = json.loads(json.dumps(base))
        holdout["results"]["holdout_screen"] = holdout["results"]["val_screen"]
        if promotion_verdict(holdout)["keep_production"] != PRODUCTION:
            print("SELFTEST FAIL tie incorrectly promotes")
            return 1
        winner = json.loads(json.dumps(holdout))
        candidate = winner["results"]["holdout_screen"]["models"]["candidate"] = json.loads(
            json.dumps(winner["results"]["holdout_screen"]["models"][PRODUCTION])
        )
        candidate["frame_weighted"].update(
            per_square_exact_rate=0.9, occupied_exact_rate=0.75, colour_rate=0.98
        )
        if promotion_verdict(winner).get("promote") != "candidate":
            print("SELFTEST FAIL valid holdout winner not promoted")
            return 1
    finally:
        temp.rmdir()
    print("SELFTEST OK report renderer and promotion rule")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--json", help="evaluator JSON to summarize")
    parser.add_argument("--out", help="write Markdown here instead of stdout")
    parser.add_argument("--decision-json", help="write the machine-readable promotion decision here")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.json:
        parser.error("--json is required unless --selftest is used")
    report = json.loads(Path(args.json).read_text(encoding="utf-8"))
    text = render(report)
    if args.decision_json:
        path = Path(args.decision_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision_payload(report), indent=2), encoding="utf-8")
        print(f"wrote decision {path}")
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
