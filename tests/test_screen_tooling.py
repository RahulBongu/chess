#!/usr/bin/env python3
"""Unit tests for the Milestone 5.4I.5 screen tooling.

These are synthetic software tests. They validate manifest, evaluator and
preflight logic only; they are not vivo capture evidence and never produce
accuracy claims for the physical board.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_screen_i5 as benchmark  # noqa: E402
import build_screen_capture_manifest as manifest  # noqa: E402
import evaluate_screen_capture_benchmark as evaluator  # noqa: E402
import preflight_vivo_capture as preflight  # noqa: E402
import report_screen_benchmark_i5 as reporter  # noqa: E402
import verify_screen_benchmark_contract as contract  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class ManifestBuilderTests(unittest.TestCase):
    def test_fen_validation_checks_fields_and_illegal_layout(self):
        self.assertEqual(manifest.validate_fen(START_FEN), [])
        self.assertTrue(any("6 fields" in error for error in manifest.validate_fen(" ".join(START_FEN.split()[:5]))))
        no_kings = "8/8/8/8/8/8/8/8 w - - 0 1"
        self.assertTrue(any("king" in error for error in manifest.validate_fen(no_kings)))
        pawns_back = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".replace(
            "RNBQKBNR", "PNBRQKNP"
        )
        self.assertTrue(any("back rank" in error for error in manifest.validate_fen(pawns_back)))
        bad_castling = START_FEN.replace("KQkq", "KKQQ")
        self.assertTrue(any("castling" in error for error in manifest.validate_fen(bad_castling)))

    def test_source_warnings_flag_unknown_and_mixed_sources(self):
        self.assertEqual(
            manifest.source_warnings([{"source": "native-dump"}, {"source": "native-dump"}]), []
        )
        warnings = manifest.source_warnings([{"source": "native-dump"}, {"source": "screencap"}])
        self.assertTrue(any("mixes sources" in warning for warning in warnings))
        warnings = manifest.source_warnings([{"source": "screenshot"}])
        self.assertTrue(any("unknown capture source" in warning for warning in warnings))

    def test_corner_validation_requires_flat_ordered_quad(self):
        flat = [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]
        corners, errors = manifest.validate_corners(flat)
        self.assertEqual(errors, [])
        self.assertEqual(corners, flat)
        _, errors = manifest.validate_corners([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        self.assertTrue(any("flat list" in error for error in errors))
        _, errors = manifest.validate_corners([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9])
        self.assertTrue(any("ordered" in error for error in errors))

    def test_build_keeps_sessions_in_their_declared_split(self):
        temp = Path(tempfile.mkdtemp(prefix="i5tool_manifest_"))
        try:
            sessions = {"s01": "train_screen", "s05": "val_screen"}
            (temp / "sessions.json").write_text(json.dumps(sessions), encoding="utf-8")
            for session, split in sessions.items():
                folder = temp / session
                folder.mkdir()
                frame = folder / "f000.png"
                manifest.write_png(frame, 8, 8, shade=32 if session == "s01" else 64)
                sidecar = {
                    "frame": frame.name,
                    "frame_size": [8, 8],
                    "session_id": session,
                    "split": split,
                    "position_id": "start",
                    "fen": START_FEN,
                    "corners_norm": [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
                    "orientation": "white_bottom",
                    "theme": "brown",
                    "brightness_pct": 70,
                    "distance": "nominal",
                    "angle_deg": 0,
                    "source": "native-dump",
                    "device": "synthetic",
                    "apk_sha256": "0" * 64,
                    "captured_at": "2026-11-08T10:00:00+05:30",
                }
                frame.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
            issues, summary = manifest.build(temp, temp / "out")
            self.assertEqual(issues, 0)
            self.assertEqual(summary["frames"], 2)
            lines = [json.loads(line) for line in (temp / "out" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({record["session_id"]: record["split"] for record in lines}, sessions)
            self.assertTrue(all(record["relpath"].startswith(record["session_id"]) for record in lines))
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class EvaluatorTests(unittest.TestCase):
    def test_truth_orientation_and_metrics(self):
        white_grid = evaluator.parse_fen_to_classes(START_FEN)
        self.assertEqual(white_grid[0], "black_rook")
        self.assertEqual(evaluator.algebraic_square(7, 4, "white_bottom"), "e1")
        self.assertEqual(evaluator.algebraic_square(7, 7, "black_bottom"), "a8")
        black_grid = evaluator.orient_truth(white_grid, "black_bottom")
        self.assertEqual(black_grid[0], "white_rook")
        probs = evaluator.perfect_start_probs()
        metrics = evaluator.square_metrics(probs, white_grid)
        self.assertEqual(metrics["correct"], 64)
        self.assertTrue(metrics["board_exact"])
        self.assertEqual(metrics["colour_correct"], 32)
        one_wrong = probs.copy()
        one_wrong[60] = 0.01 / 12.0
        one_wrong[60][evaluator.CLASS_TO_INDEX["white_pawn"]] = 0.99
        metrics = evaluator.square_metrics(one_wrong, white_grid)
        self.assertEqual(metrics["correct"], 63)
        self.assertFalse(metrics["board_exact"])

    def test_start_gate_prior_and_confident_contradiction(self):
        probs = evaluator.perfect_start_probs()
        gate = evaluator.start_gate(probs, "white_bottom")
        self.assertTrue(gate["ok"])
        self.assertEqual((gate["white_home"], gate["black_home"], gate["total_occupied"]), (16, 16, 32))
        weak = probs.copy()
        for index in (56, 57, 58, 59):
            weak[index] = 0.01 / 12.0
            weak[index][evaluator.CLASS_TO_INDEX["black_pawn"]] = 0.60
            weak[index][evaluator.CLASS_TO_INDEX["empty"]] = 0.40
        gate = evaluator.start_gate(weak, "white_bottom")
        self.assertEqual(gate["uncertain"], 4)
        self.assertFalse(gate["ok"])
        inverted = probs.copy()
        inverted[56] = 0.01 / 12.0
        inverted[56][evaluator.CLASS_TO_INDEX["black_pawn"]] = 0.95
        gate = evaluator.start_gate(inverted, "white_bottom")
        self.assertFalse(gate["ok"])
        self.assertEqual(gate["confident_contradictions"], ["a1"])

    def test_burst_averaged_metrics_and_gate(self):
        probs = evaluator.perfect_start_probs()
        truth = evaluator.parse_fen_to_classes(START_FEN)
        metrics = evaluator.burst_metrics([probs, probs.copy()], truth)
        self.assertEqual(metrics["correct"], 64)
        self.assertTrue(evaluator.burst_gate([probs, probs.copy()], "white_bottom")["ok"])
        weak = probs.copy()
        for index in (56, 57, 58, 59):
            weak[index] = 0.01 / 12.0
            weak[index][evaluator.CLASS_TO_INDEX["black_pawn"]] = 0.60
            weak[index][evaluator.CLASS_TO_INDEX["empty"]] = 0.40
        self.assertFalse(evaluator.burst_gate([weak, weak.copy()], "white_bottom")["ok"])
        with self.assertRaises(evaluator.FatalError):
            evaluator.burst_probabilities([])

    def test_strict_coverage_detects_missing_condition_values(self):
        records = [
            {"split": "train_screen", "theme": "brown"},
            {"split": "val_screen", "theme": "brown"},
            {"split": "holdout_screen", "theme": "green"},
        ]
        errors = evaluator.strict_coverage_errors(records, ["train_screen", "val_screen", "holdout_screen"])
        self.assertTrue(any("theme=brown missing" in error for error in errors))
        self.assertTrue(any("theme=green missing" in error for error in errors))
        self.assertEqual(
            evaluator.strict_coverage_errors(records[:2], ["train_screen", "val_screen"]), []
        )

    def test_manifest_sessions_are_cross_checked(self):
        temp = Path(tempfile.mkdtemp(prefix="i5tool_sessions_"))
        try:
            (temp / "sessions.json").write_text(
                json.dumps({"s01": "train_screen"}), encoding="utf-8"
            )
            records = [{"session_id": "s01", "split": "train_screen"}]
            self.assertEqual(evaluator.validate_manifest_sessions(records, temp), [])
            errors = evaluator.validate_manifest_sessions(
                [{"session_id": "s01", "split": "val_screen"}], temp
            )
            self.assertTrue(any("sessions.json" in error for error in errors))
            errors = evaluator.validate_manifest_sessions(
                [{"session_id": "s99", "split": "train_screen"}], temp
            )
            self.assertTrue(any("not declared" in error for error in errors))
            self.assertTrue(evaluator.validate_manifest_sessions(records, temp / "missing"))
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_calibration_source_and_holdout_guards(self):
        self.assertEqual(evaluator.expected_calibration_error([(1.0, True)] * 50)["ece"], 0.0)
        self.assertAlmostEqual(evaluator.expected_calibration_error([(1.0, False)] * 50)["ece"], 1.0)
        with self.assertRaises(evaluator.FatalError):
            evaluator.select_records(
                [{"source": "native-dump"}, {"source": "screencap"}], None
            )
        temp = Path(tempfile.mkdtemp(prefix="i5tool_eval_"))
        try:
            model = temp / "m.onnx"
            model.write_bytes(b"model")
            hashes = {"universal150_prod": evaluator.sha256_file(model)}
            freeze = temp / "freeze.json"
            use_log = temp / "used.json"
            with self.assertRaises(evaluator.FatalError):
                evaluator.check_holdout_guard(["holdout_screen"], hashes, None, use_log)
            freeze.write_text(json.dumps({"models": hashes}), encoding="utf-8")
            self.assertIsNotNone(evaluator.check_holdout_guard(["holdout_screen"], hashes, freeze, use_log))
            evaluator.record_holdout_use(use_log, {"models": hashes})
            with self.assertRaises(evaluator.FatalError):
                evaluator.check_holdout_guard(["holdout_screen"], hashes, freeze, use_log)
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class PreflightTests(unittest.TestCase):
    def test_device_and_package_parsing(self):
        devices = preflight.parse_devices(
            "List of devices attached\ndevice-1\tdevice product:vivo\nother\toffline\n\n"
        )
        self.assertEqual([device["serial"] for device in devices], ["device-1", "other"])
        self.assertEqual(
            preflight.parse_package_path("package:/data/app/base.apk\npackage:/data/app/split.apk\n"),
            "/data/app/base.apk",
        )

    def test_preflight_blocks_wrong_device_and_hash(self):
        temp = Path(tempfile.mkdtemp(prefix="i5tool_preflight_"))
        try:
            model_bytes = b"model"
            model = temp / "model.onnx"
            model.write_bytes(model_bytes)
            apk = temp / "app.apk"
            with zipfile.ZipFile(apk, "w") as archive:
                archive.writestr(preflight.APK_ASSET_MODEL, model_bytes)
            apk_bytes = apk.read_bytes()
            apk_hash = preflight.sha256_bytes(apk_bytes)
            model_hash = preflight.sha256_bytes(model_bytes)
            root = temp / "capture"
            preflight.ensure_capture_root(root)

            def runner(args, timeout=60):
                if args == ["devices", "-l"]:
                    return 0, b"List of devices attached\ndevice-1\tdevice product:vivo\n\n", b""
                if args[-1] == "ro.product.model":
                    return 0, b"VivoTest\n", b""
                if "pm" in args and "path" in args:
                    return 0, b"package:/data/app/base.apk\n", b""
                if "sha256sum" in args:
                    return 0, (apk_hash + "  /data/app/base.apk\n").encode(), b""
                if "test" in args and "-w" in args:
                    return 0, b"", b""
                return 1, b"", b"unexpected"

            def run(**overrides):
                options = dict(
                    serial="device-1",
                    package=preflight.PACKAGE,
                    apk=apk,
                    asset_model=model,
                    capture_root=root,
                    expect_apk_sha256=apk_hash,
                    expect_model_sha256=model_hash,
                    run_adb=runner,
                    adb_available=True,
                )
                options.update(overrides)
                return preflight.evaluate_preflight(**options)

            self.assertTrue(run()["ok"])
            empty = lambda args, timeout=60: (0, b"List of devices attached\n\n", b"")
            self.assertFalse(run(run_adb=empty)["ok"])
            bad_hash = lambda args, timeout=60: (
                (0, ("0" * 64 + "  /data/app/base.apk\n").encode(), b"")
                if "sha256sum" in args else runner(args)
            )
            report = run(run_adb=bad_hash)
            self.assertFalse(report["ok"])
            self.assertTrue(any(check["name"] == "installed-matches-local" for check in report["blockers"]))

            partial_root = temp / "partial_capture"
            partial_root.mkdir()
            (partial_root / "sessions.json").write_text(
                json.dumps({"s01": "train_screen", "s05": "val_screen"}), encoding="utf-8"
            )
            (partial_root / "s01").mkdir()
            (partial_root / "s05").mkdir()
            report = run(capture_root=partial_root)
            self.assertFalse(report["ok"])
            self.assertTrue(any(check["name"] == "split-coverage" for check in report["blockers"]))

            unwritable = lambda args, timeout=60: (
                (1, b"", b"") if "test" in args and "-w" in args else runner(args)
            )
            report = run(run_adb=unwritable)
            self.assertFalse(report["ok"])
            self.assertTrue(any(check["name"] == "dump-target" for check in report["blockers"]))
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class ReportTests(unittest.TestCase):
    def test_report_keeps_production_without_holdout_and_enforces_promotion_rule(self):
        def metrics(exact, occupied, colour, wrong=0.0):
            return {
                "frame_weighted": {
                    "frames": 10,
                    "per_square_exact_rate": exact,
                    "board_exact": "1/10",
                    "occupancy_rate": 0.9,
                    "occupied_exact": "20/32",
                    "occupied_exact_rate": occupied,
                    "colour": "30/32",
                    "colour_rate": colour,
                    "mean_confidence": 0.8,
                    "mean_uncertain": 4.0,
                    "calibration": {"ece": 0.08},
                    "start_gate": {"pass": "8/10", "mean_wrong_colour": wrong},
                }
            }

        report = {
            "manifest": "m.jsonl",
            "manifest_sha256": "a" * 64,
            "source": "native-dump",
            "splits": ["val_screen"],
            "models": {"universal150_prod": {}},
            "results": {"val_screen": {"models": {"universal150_prod": metrics(0.8, 0.6, 0.9)}}},
        }
        text = reporter.render(report)
        self.assertIn("val_screen", text)
        self.assertIn("Production stays", text)
        self.assertEqual(reporter.promotion_verdict(report)["keep_production"], "universal150_prod")

        holdout = json.loads(json.dumps(report))
        holdout["results"]["holdout_screen"] = holdout["results"]["val_screen"]
        holdout["results"]["holdout_screen"]["models"]["candidate"] = metrics(0.7, 0.7, 0.95)
        self.assertEqual(reporter.promotion_verdict(holdout)["keep_production"], "universal150_prod")
        holdout["results"]["holdout_screen"]["models"]["candidate"] = metrics(0.9, 0.7, 0.95, wrong=1.0)
        self.assertEqual(reporter.promotion_verdict(holdout)["keep_production"], "universal150_prod")
        holdout["results"]["holdout_screen"]["models"]["candidate"] = metrics(0.9, 0.7, 0.95)
        self.assertEqual(reporter.promotion_verdict(holdout).get("promote"), "candidate")


    def test_decision_json_cli_writes_machine_readable_verdict(self):
        temp = Path(tempfile.mkdtemp(prefix="i5tool_decision_"))
        try:
            source = temp / "report.json"
            source.write_text(
                json.dumps(
                    {
                        "manifest": "m.jsonl",
                        "manifest_sha256": "a" * 64,
                        "source": "native-dump",
                        "splits": ["val_screen"],
                        "models": {"universal150_prod": {}},
                        "results": {},
                    }
                ),
                encoding="utf-8",
            )
            decision = temp / "decision.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/report_screen_benchmark_i5.py"),
                 "--json", str(source), "--decision-json", str(decision), "--out", str(temp / "report.md")],
                capture_output=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
            payload = json.loads(decision.read_text(encoding="utf-8"))
            self.assertEqual(payload["milestone"], "5.4I.5")
            self.assertEqual(payload["verdict"]["keep_production"], "universal150_prod")
            self.assertTrue((temp / "report.md").exists())
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class ContractTests(unittest.TestCase):
    def test_contract_detects_tampered_evaluator_threshold(self):
        temp = Path(tempfile.mkdtemp(prefix="i5tool_contract_"))
        original = contract.EVALUATOR
        try:
            tampered = temp / "evaluate_screen_capture_benchmark.py"
            text = original.read_text(encoding="utf-8").replace("UNCERT_CONF = 0.55", "UNCERT_CONF = 0.99")
            tampered.write_text(text, encoding="utf-8")
            contract.EVALUATOR = tampered
            contract.failures.clear()
            contract.passes.clear()
            contract.check_tooling_contract()
            self.assertTrue(
                any("uncertainty confidence" in failure for failure in contract.failures),
                contract.failures,
            )
        finally:
            contract.EVALUATOR = original
            contract.failures.clear()
            contract.passes.clear()
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
