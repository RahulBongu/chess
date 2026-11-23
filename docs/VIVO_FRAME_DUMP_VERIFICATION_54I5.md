# Milestone 5.4I.5 — vivo frame-dump verification (Part B prep)

Status: **uncommitted draft**. The native capture code below is written but has
NOT been built or run on a device. Per the capture protocol it must be built and
verified on the vivo in the capture session before it is committed. This file is
the verification checklist for that session.

## What was added (local, uncommitted)

- `modules/live-camera/android/.../LiveCameraView.kt`
  - `captureDumpAllowed`: hard gate on `ApplicationInfo.FLAG_DEBUGGABLE`. A
    release APK can never dump, even if the session is set.
  - `setCaptureDumpSession(session)` / `clearCaptureDump()`.
  - `maybeDumpCaptureFrame(...)`: snapshots the RGBA classifier-input frame,
    normalized corners and 64x13 scores, then queues a lossless write on a
    dedicated `dumpExecutor` so PNG compression never blocks the camera loop.
    All failures are swallowed; a dump problem cannot drop camera frames.
  - `writeCaptureFrame(...)`: writes `frame_<seq>.png` (ARGB_8888 -> lossless
    PNG) and `frame_<seq>.json` (corners_norm, scores, frame_size, timings,
    requestId, device, captured_at) to
    `getExternalFilesDir(null)/i5capture/<session>/`.
  - `stopCamera()` clears the session; `releaseResources()` shuts the executor.
- `modules/live-camera/android/.../LiveCameraModule.kt`: `setCaptureDumpSession`
  and `clearCaptureDump` AsyncFunctions.
- `modules/live-camera/src/index.ts`: the two methods on `LiveCameraViewHandle`.

The dump records only what native can measure. Ground-truth fields the manifest
builder requires (`fen`, `position_id`, `theme`, `brightness_pct`, `distance`,
`angle_deg`, `orientation`, `split`, `apk_sha256`) are added in the labelling
step, never guessed by the dump.

## Build a debuggable capture APK

The release acceptance APK is not debuggable, so the dump is intentionally inert
there. Capture uses a debug build:

```powershell
cd C:\dev\chess\android
$env:JAVA_HOME='C:\Users\rahul\.gradle\jdks\eclipse_adoptium-17-amd64-windows.2'
.\gradlew.bat assembleDebug -PreactNativeArchitectures=arm64-v8a
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Record the debug APK SHA-256 and use it for `--expect-apk-sha256` / the manifest
`apk_sha256` during this capture session (the preflight release-APK check is for
the acceptance build, not the capture build).

## Enable the dump from JS

With the live camera view ref (same ref used for `setClassificationCorners`):

```ts
await cameraRef.current?.setCaptureDumpSession('s01'); // start a session
// ... record the 10-frame burst for each (position, condition) ...
await cameraRef.current?.clearCaptureDump();           // stop
```

Switch the session string (`s01`..`s08`) per the protocol split table. The dump
only writes frames where classification actually ran (scores + corners present).

## Pull the corpus

```powershell
adb pull /sdcard/Android/data/com.anonymous.aichessnavigator/files/i5capture C:\dev\chess\dist\capture_i5_raw
```

## On-device verification (must all pass before committing)

1. **Lossless pixels.** Each PNG decodes to exactly `frame_size` from its
   sidecar; re-encoding the RGBA buffer reproduces the PNG byte-for-byte (no
   JPEG artefacts). Confirm a known square's colour matches the screen.
2. **Corners agree with the HUD.** For a frame, `corners_norm` in the sidecar
   equals the `classificationCorners` reported in `onFrame` for the same
   `classificationRequestId`. No re-detected corners.
3. **Scores agree with the bridge.** The sidecar `scores` (64x13 logits) equal
   the `squareScores` in `onFrame` for that frame (same requestId), within float
   tolerance. This proves the dump captures the exact model input/output.
4. **Bit-exact offline replay.** Run the evaluator's device sampling
   (`benchmark_screen_i5.sample_device_batch`) on the dumped PNG + corners and
   compare argmax + confidence to the dumped scores. They must match; this is
   the whole point of the native dump (removes re-encoding/rotation from the
   measurement).
5. **Frame continuity.** `frame_<seq>` is monotonic per session; no gaps from
   dropped dumps (`dumpDroppedCount` stays 0 for a slow 1 s burst).
6. **Release safety.** Install the release APK, call `setCaptureDumpSession`,
   and confirm NO files are written and logcat shows
   `capture dump ignored: not a debuggable build`.

## Labelling, then manifest

After verification, add the ground-truth fields to each sidecar (FEN replay of
the physical laptop position, condition values, split from `sessions.json`,
`orientation`, `apk_sha256`), then:

```powershell
py -3 scripts/build_screen_capture_manifest.py --capture-root dist/capture_i5_raw --out dist/capture_i5
py -3 scripts/evaluate_screen_capture_benchmark.py --capture-root dist/capture_i5_raw --manifest dist/capture_i5/manifest.jsonl --models universal150_prod,balanced150fix,zenodo_long,wide150 --splits val_screen --json-out dist/milestone_54i5_vivo_benchmark.json
```

## Commit rule

Commit the native dump + this checklist only after steps 1–6 pass on the vivo.
Until then this remains an uncommitted local draft.
