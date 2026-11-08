# Milestone 5.4I.5 — vivo capture protocol (Part B)

Status: protocol only. No vivo crops have been captured yet; `adb devices`
was empty on 2026-11-07. Nothing here is a result, and no model swap is
authorised by this document.

This protocol is deliberately separate from
[`PHYSICAL_CAPTURE_PROTOCOL.md`](PHYSICAL_CAPTURE_PROTOCOL.md): that one
covers 20 physical sets and lands in the `train / val / holdout` manifest
schema validated by `scripts/validate_capture_manifest.py`. This one covers
**phone-camera crops of a screen**, used to measure and then close the
domain shift that blocks automatic start and the first ply. The two result
families must never be pooled.

## 1. Why a new corpus

Existing evidence on 2026-11-06:

| Evidence | Result |
|---|---|
| laptop screen, brown board, one frame, contour corners | universal150 51/64 per-square, occ 64, colour 32/32, 16 uncertain |
| laptop screen, green board, start | universal150 23/64, 43 uncertain (start gate fails: 31 uncertain) |
| laptop screen, green board, e4+e5 | universal150 31/64, 39 uncertain |
| digital crops (2600) | universal150 1.000 — train-set leakage, not screen generalisation |
| device session 2026-11-06 | auto-start FAIL, one-ply `e2–e4` FAIL, 0 epoch resets / 1107 s |

Three frames are not a benchmark. The remaining gap is sensor/exposure/
moiré/sub-pixel perspective, and it can only be attributed with a corpus
that has (a) real vivo pixels, (b) exact corners, (c) session-level splits.

## 2. Capture source of truth

Preferred (gold): a debug-only dump of the exact classifier input.

- Location: `modules/live-camera/android/src/main/java/expo/modules/livecamera/LiveCameraView.kt`,
  next to the existing `ChessSquareClassifier.classify(...)` call.
- Gated by a build flag or dev-only intent extra (never on in release).
- Writes, per classified frame: timestamp, frame `width`/`height`, the raw
  RGBA byte buffer (or a lossless PNG), `normalizedCorners`, classifier
  `scores` (64x13), and the current occupancy/colour fusion counters.
- Pull with `adb pull /sdcard/Android/data/<pkg>/files/i5capture/<session>/`.
- This is the only source that reproduces the classifier bit-exactly; it
  removes screen-recording re-encoding, preview cropping, and display
  rotation from the measurement.

Fallback (silver, allowed only if the dump is unavailable): 1080x2400
`adb exec-out screencap -p` frames plus HUD telemetry of `normalizedCorners`
captured in the same second. The offline replay must use those HUD corners,
not re-detected ones, or corner error is mixed into the result. Label the
corpus `source: screencap` and do not compare it numerically with
`source: native-dump` frames.

The debug dump is **not implemented yet**. It must be written, built and
verified on the device in the same session that produces the corpus; do not
pre-commit untested native capture code.

## 3. Session / position split

Split by **capture session**, never by crop. A session is one continuous
recording with unchanged rig, board, device position, and lighting plan.

| Split | Sessions (suggested) | Use |
|---|---|---|
| `train_screen` | 1–4 | fine-tune / calibration experiments only |
| `val_screen` | 5–6 | model selection, threshold tuning |
| `holdout_screen` | 7–8 | exactly once, after a candidate is frozen |

Rules:

- Every frame of a session stays in one split; every crop inherits it.
- Positions, themes, and lighting conditions must appear in all three
  splits so a split is not a proxy for a condition.
- `holdout_screen` is never used for early stopping, preprocessing choice,
  uncertainty-threshold choice, or model ranking.
- Root-level `session_id` and `split` are mandatory fields in the manifest.

## 4. Matrix per session

Positions (white bottom, standard start, laptop board reset each time):

1. `start` — untouched 32-piece layout.
2. `e4` — white `e2–e4` only.
3. `e5` — white `e2–e4`, black `e7–e5`.
4. `nf3` — `1. e4 e5 2. Nf3`.
5. `capture` — a legal capture with a clearly removed piece.
6. `empty_center` — start minus one centre pawn (hard negative for the
   "centre must be empty" check).

Conditions (vary one at a time, log the value):

- Distance: near (board fills the view), nominal, far (board ~60% of view).
- Angle: front, ~15° left tilt, ~15° right tilt, ~20° pitch.
- Laptop brightness: 40%, 70%, 100% (record actual `adb shell dumpsys power`
  where available).
- Theme: green, brown (the two screenshot themes already in `dist/`).
- Room light: normal and dim.

Per `(session, position, condition)` record a burst of **10 frames, 1 s
apart**, after the app reports a stable lock. Discard a burst if tracking
is `hold`/`none`, the lens is covered, or sharpness telemetry drops below
the value seen on a known-good frame; re-shoot it.

## 5. Labels

Ground truth comes from the physical position applied to the laptop board,
not from the model:

```json
{"session_id":"s05","split":"val_screen","frame":"s05_start_70_0003.png",
 "position_id":"start","fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
 "corners_norm":[0.096,0.304,0.907,0.287,0.898,0.672,0.108,0.666],
 "orientation":"white_bottom","theme":"brown","brightness_pct":70,"distance":"nominal",
 "angle_deg":0,"source":"native-dump","device":"10BE2A1B5Q00054",
 "apk_sha256":"<sha256>","captured_at":"2026-11-07T18:12:00+05:30"}
```

- `corners_norm` is `[TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy]`, matching
  `ChessSquareClassifier.classify` input.
- `fen` must be a complete six-field FEN; validate with
  `scripts/validate_capture_manifest.py` semantics (placement, both kings,
  no pawns on rank 1/8) before accepting the manifest.
- Never edit a label to match a prediction. Mislabelled frames are deleted
  with a note in the session log.

Build the corpus manifest with `scripts/build_screen_capture_manifest.py`.
It refuses a frame without a sidecar, a non-six-field or malformed FEN, an
unordered/non-convex/small corner quad, a session directory missing from
`sessions.json`, a split mismatch, a duplicate frame (including the same
bytes reused in another split), or malformed protocol metadata. It writes
`manifest.jsonl` plus `summary.json` and exits non-zero on errors, or on
warnings unless `--allow-warnings` is passed. Run its `--selftest` after any
change to the schema.

```powershell
py -3 scripts/build_screen_capture_manifest.py --selftest
py -3 scripts/build_screen_capture_manifest.py --capture-root <dir> --out dist/capture_i5
```

## 6. QC before any evaluation number is quoted

Before the first frame of a session, run:

```powershell
py -3 scripts/preflight_vivo_capture.py --init-capture-root
```

It blocks unless the expected vivo serial is connected and in `device` state,
the installed APK hash equals the local release APK, the APK and asset model
both equal the pinned `universal150` SHA-256, and the capture root has valid
`sessions.json` and session directories. It only reads the device; it never
installs or captures. Fix every `[BLOCK]` before recording.

After capture:

1. Corner residual: reproject the labelled corners onto the frame; the
   four edges must lie on the board border within 4 px at 1080-wide.
2. Convex, correctly ordered quad (same checks as
   `scripts/verify_screen_benchmark_contract.py`).
3. No duplicate frames (hash), no burst leaking across splits.
4. Confirm each claimed position by FEN replay, not by looking at the image.
5. Record frame count per class; classes with < 20 crops in a split cannot
   support a per-class claim (report `n` and "insufficient").

## 7. Evaluation

Run all four candidates on identical crops with the corpus evaluator. It
samples the classifier path from the manifest corners (no re-detection),
keeps session splits, refuses mixed `source` values, and reports the
start-gate replica:

```powershell
py -3 scripts/evaluate_screen_capture_benchmark.py `
  --capture-root <capture-root> `
  --manifest dist/capture_i5/manifest.jsonl `
  --models universal150_prod,balanced150fix,zenodo_long,wide150 `
  --splits val_screen `
  --json-out dist/milestone_54i5_vivo_benchmark.json
```

`holdout_screen` is refused unless every selected model is pinned by SHA-256
in a freeze manifest, and the one-shot use log does not exist. Pin the files
first; this writes hashes of the current files without evaluating anything:

```powershell
py -3 scripts/evaluate_screen_capture_benchmark.py `
  --models universal150_prod,balanced150fix `
  --write-freeze dist/freeze_i5.json
```

```json
{"frozen_at":"2026-11-08T12:00:00+05:30",
 "models":{"universal150_prod":"<sha256>","balanced150fix":"<sha256>"}}
```

```powershell
py -3 scripts/evaluate_screen_capture_benchmark.py `
  --capture-root <capture-root> --manifest dist/capture_i5/manifest.jsonl `
  --models universal150_prod,balanced150fix `
  --splits holdout_screen --freeze-manifest dist/freeze_i5.json `
  --holdout-use-log dist/holdout_screen_i5_used.json `
  --json-out dist/milestone_54i5_holdout.json
```

The use log is written only after a successful evaluation; a second run
against the same log is refused. Start-gate pass rate is per frame, which is
stricter than the device's burst-averaged fusion; it does not substitute for
the on-device automatic-start acceptance. The three laptop frames remain
scored by `scripts/benchmark_screen_i5.py`; their numbers are a baseline, not
this corpus.

Report, per split and per condition:

- per-square exact, true board-exact, occupancy, occupied-exact,
  empty-exact, colour, mean confidence, uncertain count;
- calibration (confidence vs correctness, ECE bins);
- start-gate replica: uncertain count, centre occupied, white/black home,
  wrong-colour, total occupied;
- comparison against the screen-frame baseline above, never against the
  physical 20-set numbers.

Promotion rule: a candidate may replace `universal150` only if it wins on
`holdout_screen` on per-square exact **and** occupied-exact **and** colour,
without more wrong-colour home squares, and still satisfies
`docs/MILESTONE_54I5_SCREEN_BENCHMARK_AUDIT.md`'s constraint that the
confident-contradiction rule stays at `p >= 0.8, margin >= 0.4`. A tie
keeps production unchanged.

## 8. Explicitly forbidden

- Raising `maxUncertainSquares` above 3, or loosening
  `isConfidentContradiction`, to make a gate pass.
- Random crop-level train/val splits, or training on frames from the same
  session used to report.
- Quoting the 2600 digital crops as screen accuracy (they are training data
  for `universal150`).
- Mixing screen and physical-board metrics in one number.
- Committing a candidate model before `holdout_screen` exists.
