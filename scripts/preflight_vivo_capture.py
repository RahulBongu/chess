#!/usr/bin/env python3
"""Preflight the Milestone 5.4I.5 vivo screen-capture session.

A capture session is expensive: the phone must be connected, the installed
APK must match the locally built one, the APK must carry the production
universal150 model, and the host capture root must be ready. This tool checks
those facts before the first frame and writes a JSON evidence file.

It does not capture frames, install anything, or change the device. It only
runs read-only adb commands and reads local files.

Usage:
  py -3 scripts/preflight_vivo_capture.py
  py -3 scripts/preflight_vivo_capture.py --init-capture-root
  py -3 scripts/preflight_vivo_capture.py --selftest
  py -3 scripts/preflight_vivo_capture.py --json-out dist/preflight_i5.json

Exit codes: 0 ready, 1 self-test failure, 2 blocked (or adb missing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

EXPECTED_SERIAL = "10BE2A1B5Q00054"
PACKAGE = "com.anonymous.aichessnavigator"
PRODUCTION_MODEL_SHA256 = "4a78b7ad6145f2849f6d7ef18be6c1c4b124f909f984131f860820f90ddfffab"
APK_ASSET_MODEL = "assets/rgb_square_classifier_v1.onnx"
APK_PATH = Path("android/app/build/outputs/apk/release/app-release.apk")
MODEL_ASSET_PATH = Path("modules/live-camera/android/src/main/assets/rgb_square_classifier_v1.onnx")
CAPTURE_ROOT = Path("dist/capture_i5_raw")
CAPTURE_SESSIONS = {"s01": "train_screen", "s05": "val_screen", "s07": "holdout_screen"}
SPLITS = {"train_screen", "val_screen", "holdout_screen"}


class PreflightError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adb_run(args: list[str], timeout: int = 60) -> tuple[int, bytes, bytes]:
    try:
        result = subprocess.run(
            ["adb", *args],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError as error:
        raise PreflightError("adb was not found on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise PreflightError(f"adb timed out after {timeout}s: adb {' '.join(args)}") from error


def parse_devices(text: str) -> list[dict]:
    devices = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("List of devices"):
            continue
        if stripped.startswith("*"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            devices.append({"serial": parts[0], "state": parts[1], "detail": " ".join(parts[2:])})
    return devices


def parse_package_path(text: str) -> str | None:
    paths = []
    for line in text.splitlines():
        match = re.match(r"^package:(.+)$", line.strip())
        if match:
            paths.append(match.group(1))
    if not paths:
        return None
    for path in paths:
        if path.endswith("base.apk"):
            return path
    return paths[0]


def installed_apk_hash(serial: str, apk_path: str, run_adb) -> str | None:
    code, stdout, _stderr = run_adb(["-s", serial, "shell", "sha256sum", apk_path])
    if code == 0:
        match = re.match(r"^([0-9a-fA-F]{64})\s", stdout.decode("utf-8", "replace").strip())
        if match:
            return match.group(1).lower()
    code, stdout, _stderr = run_adb(["-s", serial, "exec-out", "cat", apk_path], timeout=300)
    if code == 0 and stdout:
        return sha256_bytes(stdout)
    return None


def ensure_capture_root(root: Path) -> list[str]:
    """Creates the session template only when it is absent. Never overwrites."""
    created = []
    root.mkdir(parents=True, exist_ok=True)
    sessions_path = root / "sessions.json"
    if sessions_path.exists():
        try:
            sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            sessions = {}
    else:
        sessions = dict(CAPTURE_SESSIONS)
        sessions_path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
        created.append(str(sessions_path))
    for session in sessions:
        session_dir = root / session
        if not session_dir.exists():
            session_dir.mkdir()
            created.append(str(session_dir))
    return created


def evaluate_preflight(
    *,
    serial: str,
    package: str,
    apk: Path,
    asset_model: Path,
    capture_root: Path,
    expect_apk_sha256: str | None,
    expect_model_sha256: str,
    run_adb,
    adb_available: bool = True,
) -> dict:
    checks: list[dict] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    if adb_available:
        add("adb", "ok", "adb is available on PATH")
    else:
        add("adb", "block", "adb was not found on PATH")

    devices: list[dict] = []
    if adb_available:
        try:
            code, stdout, stderr = run_adb(["devices", "-l"])
        except PreflightError as error:
            add("device", "block", str(error))
            code, stdout = 1, b""
        if code == 0:
            devices = parse_devices(stdout.decode("utf-8", "replace"))
            expected = [device for device in devices if device["serial"] == serial]
            other = [device for device in devices if device["serial"] != serial]
            if not expected:
                add("device", "block", f"expected device {serial} is not connected; seen {devices}")
            elif expected[0]["state"] != "device":
                add("device", "block", f"device {serial} state is {expected[0]['state']!r}, not 'device'")
            else:
                add("device", "ok", f"{serial} connected{(' (' + expected[0]['detail'] + ')') if expected[0]['detail'] else ''}")
            if other:
                add("other-devices", "warn", f"other devices attached: {[d['serial'] for d in other]}")
        elif adb_available:
            message = stderr.decode("utf-8", "replace").strip()
            add("device", "block", f"adb devices failed: {message or 'unknown error'}")

    device_ready = any(
        check["name"] == "device" and check["status"] == "ok" for check in checks
    )
    installed_path = None
    installed_hash = None
    if device_ready:
        code, stdout, stderr = run_adb(["-s", serial, "shell", "getprop", "ro.product.model"])
        if code == 0:
            add("device-model", "ok", stdout.decode("utf-8", "replace").strip() or "unknown model")
        else:
            add("device-model", "warn", "could not read ro.product.model")
        code, stdout, stderr = run_adb(["-s", serial, "shell", "pm", "path", package])
        if code == 0:
            installed_path = parse_package_path(stdout.decode("utf-8", "replace"))
        if not installed_path:
            add("installed-apk", "block", f"package {package} is not installed or has no base.apk")
        else:
            add("installed-apk", "ok", installed_path)
            installed_hash = installed_apk_hash(serial, installed_path, run_adb)
            if not installed_hash:
                add("installed-hash", "block", "could not hash the installed APK")
            else:
                add("installed-hash", "ok", installed_hash)

    local_apk_hash = None
    if not apk.is_file():
        add("local-apk", "block", f"missing {apk}; build the arm64 release APK first")
    else:
        local_apk_hash = sha256_file(apk)
        add("local-apk", "ok", f"{apk} sha256={local_apk_hash}")
        if expect_apk_sha256 and local_apk_hash != expect_apk_sha256.lower():
            add(
                "expected-apk-hash",
                "block",
                f"local APK differs from --expect-apk-sha256: {local_apk_hash} != {expect_apk_sha256}",
            )
        elif expect_apk_sha256:
            add("expected-apk-hash", "ok", "local APK matches --expect-apk-sha256")
        if installed_hash and installed_hash != local_apk_hash:
            add(
                "installed-matches-local",
                "block",
                f"installed APK {installed_hash} != local {local_apk_hash}; "
                "install the freshly built APK before capturing",
            )
        elif installed_hash:
            add("installed-matches-local", "ok", "installed APK matches the local build")

        apk_model_hash = None
        try:
            with zipfile.ZipFile(apk) as archive:
                apk_model_hash = sha256_bytes(archive.read(APK_ASSET_MODEL))
        except (KeyError, zipfile.BadZipFile, OSError) as error:
            add("apk-model", "block", f"cannot read {APK_ASSET_MODEL} from {apk}: {error}")
        if apk_model_hash:
            if asset_model.is_file() and sha256_file(asset_model) == apk_model_hash:
                add("apk-model", "ok", f"{APK_ASSET_MODEL} matches the production asset")
            else:
                add(
                    "apk-model",
                    "block",
                    f"{APK_ASSET_MODEL} in the APK does not match {asset_model}",
                )
            if apk_model_hash == expect_model_sha256.lower():
                add("production-model", "ok", f"model sha256={apk_model_hash}")
            else:
                add(
                    "production-model",
                    "block",
                    f"APK model {apk_model_hash} != expected {expect_model_sha256}",
                )

    if not asset_model.is_file():
        add("model-asset", "block", f"missing {asset_model}")
    else:
        asset_hash = sha256_file(asset_model)
        if asset_hash == expect_model_sha256.lower():
            add("model-asset", "ok", f"{asset_model} sha256={asset_hash}")
        else:
            add("model-asset", "block", f"asset model {asset_hash} != expected {expect_model_sha256}")

    sessions_path = capture_root / "sessions.json"
    if not capture_root.is_dir():
        add("capture-root", "block", f"missing {capture_root}; run with --init-capture-root")
    else:
        add("capture-root", "ok", str(capture_root))
        try:
            usage = shutil.disk_usage(capture_root)
            free_gb = usage.free / (1024**3)
            if free_gb < 0.2:
                add("disk-space", "block", f"only {free_gb:.2f} GiB free")
            elif free_gb < 1.0:
                add("disk-space", "warn", f"only {free_gb:.2f} GiB free")
            else:
                add("disk-space", "ok", f"{free_gb:.1f} GiB free")
        except OSError as error:
            add("disk-space", "warn", str(error))
    if not sessions_path.is_file():
        add("sessions-json", "block", f"missing {sessions_path}; run with --init-capture-root")
    else:
        try:
            sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
            if not isinstance(sessions, dict) or not sessions:
                raise ValueError("must be a non-empty object")
            invalid = {name: split for name, split in sessions.items() if split not in SPLITS}
            if invalid:
                raise ValueError(f"invalid splits: {invalid}")
            missing_dirs = [name for name in sessions if not (capture_root / name).is_dir()]
            if missing_dirs:
                add("session-dirs", "block", f"missing session directories: {missing_dirs}")
            else:
                add("sessions-json", "ok", f"{len(sessions)} sessions: {sessions}")
        except (json.JSONDecodeError, ValueError) as error:
            add("sessions-json", "block", f"invalid {sessions_path}: {error}")
    if capture_root.is_dir():
        frame_count = sum(1 for _ in capture_root.glob("*/[!._]*.png"))
        if frame_count:
            add("capture-root-clean", "warn", f"{frame_count} PNG frames already present; use a new session root")

    blockers = [check for check in checks if check["status"] == "block"]
    warnings = [check for check in checks if check["status"] == "warn"]
    return {
        "serial": serial,
        "package": package,
        "device": devices,
        "apk": str(apk),
        "apk_sha256": local_apk_hash,
        "capture_root": str(capture_root),
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "ok": not blockers,
    }


def selftest() -> int:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    temp = Path(tempfile.mkdtemp(prefix="i5preflight_"))
    try:
        model_bytes = b"production-model-bytes"
        model_path = temp / "rgb_square_classifier_v1.onnx"
        model_path.write_bytes(model_bytes)
        model_hash = sha256_bytes(model_bytes)
        apk_path = temp / "app-release.apk"
        with zipfile.ZipFile(apk_path, "w") as archive:
            archive.writestr(APK_ASSET_MODEL, model_bytes)
        apk_bytes = apk_path.read_bytes()
        apk_hash = sha256_bytes(apk_bytes)
        root = temp / "capture"
        ensure_capture_root(root)
        serial = "device-1"

        def base_runner(args, timeout=60):
            if args == ["devices", "-l"]:
                return 0, b"List of devices attached\ndevice-1\tdevice product:test model:VivoTest\n\n", b""
            if args[-1] == "ro.product.model":
                return 0, b"VivoTest\n", b""
            if "pm" in args and "path" in args:
                return 0, b"package:/data/app/base.apk\n", b""
            if "sha256sum" in args:
                return 0, (apk_hash + "  /data/app/base.apk\n").encode(), b""
            if "exec-out" in args:
                return 0, apk_bytes, b""
            return 1, b"", b"unexpected"

        def run(**overrides):
            options = {
                "serial": serial,
                "package": PACKAGE,
                "apk": apk_path,
                "asset_model": model_path,
                "capture_root": root,
                "expect_apk_sha256": apk_hash,
                "expect_model_sha256": model_hash,
                "run_adb": base_runner,
                "adb_available": True,
            }
            options.update(overrides)
            return evaluate_preflight(**options)

        report = run()
        check(report["ok"] is True, f"clean preflight blocked: {report['blockers']}")
        check(any(check["name"] == "device" for check in report["checks"]), "device check missing")
        check(any(check["name"] == "apk-model" for check in report["checks"]), "apk model check missing")

        report = run(run_adb=lambda args: (0, b"List of devices attached\n\n", b""))
        check(report["ok"] is False, "missing device was not blocked")

        report = run(
            run_adb=lambda args: (
                0,
                b"List of devices attached\ndevice-1\tunauthorized\n\n",
                b"",
            )
        )
        check(report["ok"] is False, "unauthorized device was not blocked")

        report = run(
            run_adb=lambda args: (
                0,
                b"List of devices attached\nother-device\tdevice\n\n",
                b"",
            )
        )
        check(report["ok"] is False, "wrong serial was not blocked")

        def no_package(args, timeout=60):
            if args == ["devices", "-l"]:
                return 0, b"List of devices attached\ndevice-1\tdevice\n\n", b""
            if args[-1] == "ro.product.model":
                return 0, b"VivoTest\n", b""
            if "pm" in args and "path" in args:
                return 0, b"", b""
            return 1, b"", b""

        report = run(run_adb=no_package)
        check(report["ok"] is False, "missing package was not blocked")

        def wrong_installed_hash(args, timeout=60):
            if "sha256sum" in args:
                return 0, (("0" * 64) + "  /data/app/base.apk\n").encode(), b""
            return base_runner(args)

        report = run(run_adb=wrong_installed_hash)
        check(report["ok"] is False, "installed APK hash mismatch was not blocked")
        check(
            any(check["name"] == "installed-matches-local" for check in report["blockers"]),
            "installed hash blocker missing",
        )

        def fallback_hash(args, timeout=60):
            if "sha256sum" in args:
                return 1, b"", b"sha256sum unavailable"
            return base_runner(args)

        report = run(run_adb=fallback_hash)
        check(report["ok"] is True, f"exec-out hash fallback blocked: {report['blockers']}")

        bad_model = temp / "other.onnx"
        bad_model.write_bytes(b"different-model")
        report = run(asset_model=bad_model)
        check(report["ok"] is False, "model asset mismatch was not blocked")

        report = run(expect_apk_sha256="1" * 64)
        check(report["ok"] is False, "expected APK hash mismatch was not blocked")

        empty_root = temp / "empty_root"
        report = run(capture_root=empty_root)
        check(report["ok"] is False, "missing capture root was not blocked")

        created = ensure_capture_root(empty_root)
        check(any(path.endswith("sessions.json") for path in created), "sessions.json was not created")
        report = run(capture_root=empty_root)
        check(report["ok"] is True, f"initialized capture root blocked: {report['blockers']}")
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    if failures:
        for failure in failures:
            print(f"SELFTEST FAIL {failure}")
        return 1
    print("SELFTEST OK preflight blocks missing/wrong device, APK hash, model and capture root")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=EXPECTED_SERIAL)
    parser.add_argument("--package", default=PACKAGE)
    parser.add_argument("--apk", default=str(APK_PATH))
    parser.add_argument("--asset-model", default=str(MODEL_ASSET_PATH))
    parser.add_argument("--capture-root", default=str(CAPTURE_ROOT))
    parser.add_argument("--expect-apk-sha256")
    parser.add_argument("--expect-model-sha256", default=PRODUCTION_MODEL_SHA256)
    parser.add_argument("--init-capture-root", action="store_true")
    parser.add_argument("--json-out", default="dist/preflight_i5.json")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    capture_root = Path(args.capture_root)
    if args.init_capture_root:
        created = ensure_capture_root(capture_root)
        for path in created:
            print(f"CREATED {path}")

    try:
        report = evaluate_preflight(
            serial=args.serial,
            package=args.package,
            apk=Path(args.apk),
            asset_model=Path(args.asset_model),
            capture_root=capture_root,
            expect_apk_sha256=args.expect_apk_sha256,
            expect_model_sha256=args.expect_model_sha256,
            run_adb=adb_run,
            adb_available=shutil.which("adb") is not None,
        )
    except PreflightError as error:
        print(f"BLOCKED {error}")
        return 2

    for check in report["checks"]:
        label = {"ok": "OK", "warn": "WARN", "block": "BLOCK"}[check["status"]]
        print(f"[{label}] {check['name']}: {check['detail']}")
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"preflight={'READY' if report['ok'] else 'BLOCKED'} "
        f"blockers={len(report['blockers'])} warnings={len(report['warnings'])} json={out_path}"
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
