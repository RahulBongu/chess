# AI Chess Navigator

Android-first mobile app for analyzing and navigating a physical chess game.

## Current milestone

Milestone 3.7: offline Chal v2.0.0 engine integration. The app includes a playable local chessboard, user-initiated camera calibration, local OpenCV board detection (boundary and 8 × 8 pattern verification), and offline chess analysis in Board Study powered by an embedded MIT-licensed Chal v2.0.0 native engine (tag 2.0.0, commit c5410b5, embedded 49,346-byte NNUE weights) on Android ARM64. A 2-ply heuristic fallback is retained with full UI disclosure when the native engine is unavailable. Piece recognition, voice instructions, and backend services are not implemented.

## Technology stack

- Expo SDK 57
- React Native 0.86
- React 19.2
- TypeScript 6 (strict mode)
- Expo Router
- `expo-camera` with local OpenCV board-boundary detection
- `expo-file-system` for temporary capture cleanup
- In-process native JNI module `chal-engine` embedding Chal v2.0.0 (C11/C++20, Android NDK CMake) with embedded NNUE evaluation
- pnpm
- chess.js 1.4

Expo SDK 57 was selected because it is the current stable SDK available from `create-expo-app`, supports React Native 0.86, requires Node.js 22.13.x or newer (this machine uses Node.js 24.18.0), and targets Android SDK 36. It also leaves a clear upgrade path to SDK 58 later.

## Setup

```bash
pnpm install
```

On Windows, if pnpm reports a virtual-store path-length problem, scope the workaround to the current shell only:

```powershell
$env:PNPM_CONFIG_VIRTUAL_STORE_DIR_MAX_LENGTH = '32'
pnpm install
```

## Run

```bash
pnpm start
pnpm android
pnpm web
```

`pnpm android` starts the Expo dev server and attempts to open an Android emulator or connected device. A JDK is required for native Android builds; set `JAVA_HOME` to a JDK 17 installation for the current shell if Android native builds are needed. No global tooling was changed by this project.

The normal debug workflow keeps Metro, INTERNET, and cleartext traffic enabled. The offline test build described below is a separate, opt-in configuration.

## Privacy and camera image lifecycle

- Camera permission is requested only after the user taps **Enable camera**.
- A still image is captured only when the user taps **Auto-detect board**.
- The captured JPEG is processed locally by OpenCV on the device. The app has no upload, analytics, telemetry, crash-reporting, or gallery-save flow.
- `expo-camera` writes the JPEG to the app-private cache. The app deletes it after retake, after confirmed calibration, or when leaving the camera workflow. If deletion fails, the operating system may reclaim the cache file later; the app never uploads it.
- Calibration corners are kept in memory for the current session only.
- Camera availability is platform-aware. `CameraView.isAvailableAsync()` is a web-only API, so it is only called on web. On Android and iOS the permission-authorized `CameraView` mounts directly and reports real preview success or failure through `onCameraReady` and `onMountError`; a missing camera is never inferred from the web-only probe.
- The preview is mounted only while the Camera / Board Setup screen is focused. Leaving the tab unmounts it and releases the camera, following Expo Camera's guidance for screens with multiple tabs; this also prevents Android's CameraX from starting without a preview surface.

## Board detection and detection score

The detector runs in two stages on the locally captured still image:

1. **Boundary evidence.** The image is resized to at most 720 pixels wide, converted to grayscale, blurred, and edge-detected. External four-point contours are simplified and must form a convex quadrilateral that spans enough of the image. Boundary quality combines image coverage with a rotation-tolerant rectangularity measure (quadrilateral area divided by the square of its average side length).
2. **Pattern evidence.** The candidate is rectified through a homography and sampled as an 8 × 8 grid. Patches near each cell's corners are used so pieces standing in square centres do not hide the square colour. Because a detected outline often includes a board border or a surrounding table edge, the scorer also evaluates several inward sampling regions (scales from 1.00 to 0.82) and keeps the strongest board evidence. The outer corners are still returned for calibration. The detector measures how well the patch means fit two checkerboard parity orientations, how many adjacent cell pairs alternate in the expected direction, the average adjacent-cell contrast, and the variation inside each cell. Uniform, low-contrast, striped, incomplete, and non-board rectangles fail this stage.

A candidate is accepted only when the geometry is valid, the interior passes the pattern-evidence gates, and the combined score reaches `MIN_DETECTION_CONFIDENCE` (`0.32`). Pattern gates require checkerboard parity to explain at least 55% of cell-mean variance, at least 80% of adjacent cell pairs to alternate in the expected direction, at least 8 gray levels of mean adjacent-square contrast, and a combined pattern quality of at least `0.45`. These thresholds are exercised by the deterministic fixtures rather than chosen only by inspection. The candidate list is collected with `RETR_LIST`, so a board contained in a larger rectangle can still be evaluated. The app displays per-candidate diagnostics (corners, sampling scale, boundary score, pattern score, fit, alternation, contrast, noise, and rejection reason) for local testing. The reported **detection score** is:

```text
detection score = boundary quality × pattern quality
```

The score is a heuristic quality value from 0 to 1, **not a probability or a certainty**. A score of `1.00` requires both a near-ideal boundary and near-ideal alternating-square evidence; a large rectangle with no checkerboard pattern scores `0`. The UI therefore labels the value "Detection score" instead of "Confidence". Manual corner correction remains available after auto-detection, and confirming still validates the four corners.

The synthetic test fixtures are deterministic and cover dark/uniform/blank frames, non-board rectangles, low-contrast patterns, stripes, coarse and half-complete grids, rotations, perspective, uneven lighting, and different image scales. They do not replace testing with real camera images from a physical device.

## Offline test APK

The offline test build is opt-in and blocks `android.permission.INTERNET` in the final manifest. It is intended for release-mode testing with Wi-Fi and mobile data disabled, not for distribution.

```powershell
$env:CHESS_OFFLINE_BUILD = '1'
$env:JAVA_HOME = 'C:\path\to\jdk-17'
npx expo prebuild --platform android --no-install
Set-Location android
.\gradlew.bat :app:assembleRelease --console=plain
```

The resulting APK is at `android/app/build/outputs/apk/release/app-release.apk`. It embeds the JavaScript bundle, uses the local debug keystore, and must not be distributed.

On Windows, the full four-ABI release can exceed CMake's path limit for the 32-bit ABIs. The verified local test APK was built arm64-only with ABI splits (`-PreactNativeArchitectures=arm64-v8a` plus an `arm64-v8a` split) and with resource optimization disabled (`-Pandroid.enableResourceOptimizations=false`) to avoid AAPT2 short-name collisions on a case-insensitive filesystem. That build produces `android/app/build/outputs/apk/release/app-arm64-v8a-release.apk`.

The flag is evaluated when Expo generates the native project, so set it for both prebuild and Gradle. Do not leave the offline-generated manifest in place for normal Metro development: rerun `npx expo prebuild --platform android --no-install` without `CHESS_OFFLINE_BUILD` to restore the standard debug configuration. The normal `pnpm android` workflow keeps INTERNET and cleartext traffic for Metro.

## Quality checks

```bash
pnpm typecheck
pnpm test
npx expo-doctor
```

## Project structure

- `src/app/` — Expo Router routes and screens
- `src/calibration/` — four-corner geometry, ordering, and validation
- `src/chess/` — authoritative chess game state and domain helpers
- `src/components/` — reusable UI components, including `ChessBoard`
- `src/theme/` — design tokens
- `src/types/` — ambient type declarations
- `src/vision/` — OpenCV board detection, boundary and 8 × 8 pattern evidence, scoring, and coordinate normalization
- `tests/` — focused chess, calibration, camera-availability, and vision tests

## Milestone 3.7 additions

- Upgraded offline analysis in Board Study from the 2-ply TypeScript heuristic to the MIT-licensed Chal v2.0.0 native chess engine on Android ARM64.
- Embedded Chal v2.0.0 (commit `c5410b5`) with embedded 49,346-byte NNUE weights (`src/net.h`) into an in-process JNI shared library (`libchal-engine.so`).
- Implemented C11 atomic search cancellation (`chal_request_stop`, `chal_is_stopped`, `chal_reset_stop`) and monotonic search sequence IDs in `chal_bridge.cpp` to prevent stop-flag race conditions.
- Serialized search execution with `std::try_to_lock` to guarantee at most one active search and safe teardown without freeing native state during an in-flight search.
- Added strict FEN validation in the bridge before passing positions to upstream `parse_fen`.
- Created `ChalChessEngine` adapter implementing the `ChessEngine` interface with UCI PV-to-SAN conversion, mate-distance reporting, and illegal-move filtering.
- Implemented `createDefaultEngine` in `src/chess/engineSelection.ts` with explicit heuristic fallback disclosure.
- Verified native runtime on physical device (vivo V2336, Android 16) with 69 assertions, sub-millisecond cancellation (median 152 µs), and safe teardown under load.
- Packaged into an offline release APK with zero network permissions (`android.permission.INTERNET` blocked).

## Milestone 3.6 additions

- Added finer inward sampling scales from 1.00 to 0.80 so small symmetric borders can align with the 8 × 8 grid.
- Added minimum-area-rectangle candidates for contours that are not simplified to exactly four points.
- Added a second Canny threshold pair and unioned the edge images before dilation.
- Retained diagnostics for candidate corners, sampling scale, scores, and rejection reasons.

## Milestone 3.5 additions

- Diagnosed the real-device failure with a Python/OpenCV emulation of the native pipeline: the detected outer quadrilateral included a board border, so the 8 × 8 sampler was shifted by roughly 10% and produced fit 0.051, noise 51.4, and combined confidence 0.
- Shrinking the sampling region to 90% of the detected quadrilateral recovered fit 0.887, alternation 1.0, contrast 106.8, noise 3.6, and combined confidence 0.887.
- Detection now evaluates sampling scales from 1.00 to 0.82 for every candidate and keeps the strongest board evidence while returning the outer corners for calibration.
- Contour collection now uses `RETR_LIST`, allowing a board contained in a larger rectangle to be evaluated.
- Added per-candidate diagnostics to `BoardDetectionResult` and the camera setup screen.
- Added regression tests for a bordered board, including the outer-corner failure and inset-sampling recovery.

## Milestone 3.4 additions

- Added `src/vision/boardEvidence.ts`: homography rectification, bilinear grayscale sampling, 8 × 8 cell statistics, and checkerboard parity evidence.
- Detection now requires plausible alternating-square evidence; the combined score is boundary quality multiplied by pattern quality, and invalid patterns force a zero score.
- Boundary rectangularity is rotation-tolerant, so boards rotated away from the image axes are not penalized by an axis-aligned bounding box.
- Corner ordering uses angular order around the quadrilateral centroid, which avoids duplicate-corner selection on diamond-shaped boards.
- Added deterministic negative and positive fixtures in `tests/boardEvidence.test.ts`, including a regression for the dark-frame high-confidence failure.
- Renamed the displayed value to "Detection score" and documented that it is a heuristic quality score, not a probability.

## Milestone 3.3 additions

- `expo-camera` configured with `recordAudioAndroid: false` and `barcodeScannerEnabled: false`; Expo Autolinking forces `expo-camera` to build from source so the barcode setting is honored.
- Blocked unused permissions: `RECORD_AUDIO`, `READ_EXTERNAL_STORAGE`, `WRITE_EXTERNAL_STORAGE`, `SYSTEM_ALERT_WINDOW`, `VIBRATE`, and `ACCESS_NETWORK_STATE`.
- `android.allowBackup` disabled.
- Optional `CHESS_OFFLINE_BUILD=1` configuration blocks `INTERNET`, disables cleartext traffic, and produces a release APK with an embedded JavaScript bundle.
- Temporary camera captures are deleted after retake, confirmation, or leaving the camera workflow.
- Removed the unused `expo-device` dependency.

## Milestone 3 additions

- `react-native-fast-opencv@1.0.1` integration for local board-boundary detection
- `src/vision/` detection algorithm, validation, scoring, and coordinate normalization
- Auto-detect board workflow in camera setup with captured-image overlay and manual corner refinement
- Tests for candidate ranking, invalid geometry, ordering, bounds, confidence, and normalization (extended in Milestone 3.4 with boundary and 8 × 8 pattern evidence)

## Milestone 2 additions

- Runtime camera permission flow with denied/unavailable handling
- Live back-camera preview
- Manual four-corner board calibration with visible markers, connecting lines, undo, reset, and validation
- In-memory calibration state only; captured images are not saved to the gallery or uploaded
- Normalized preview coordinates: origin at top-left, x right, y down, range 0 to 1

## Offline chess analysis (Chal v2.0.0 & Heuristic Fallback)

Board Study (`src/app/study.tsx`) provides offline position analysis for imported FEN and PGN games:

- **Embedded Native Engine (Chal v2.0.0):** Powered by an in-process JNI shared library (`libchal-engine.so`, 144,152 bytes) built from the MIT-licensed Chal v2.0.0 engine (commit `c5410b5`, released 2026-09-08). It includes embedded 49,346-byte NNUE evaluation weights in `src/net.h` with zero external file or network dependencies.
- **Native Synchronization and Cancellation:** Uses C11 atomics (`atomic_int time_over_flag`) with acquire/release semantics in the engine loop. The JNI bridge serializes all search invocations with a mutex (`std::try_to_lock`), protects flag resets with monotonic search sequence IDs, and guarantees safe teardown without freeing native state while a search thread is active.
- **Explicit Fallback Disclosure:** When the native library is unavailable (such as during web development or in environments without native binary support), the app automatically falls back to `HeuristicChessEngine` (2-ply depth) and surfaces a clear UI banner disclosing the fallback.
- **Verification and Artifacts:** The native integration was verified on physical hardware (vivo V2336, Android 16 ARM64) across 69 runtime assertions covering engine accuracy, cancellation latency (median 152 µs), active teardown safety, and memory stability. Details, hashes, and reproduction steps are documented in [`modules/chal-engine/verification/README.md`](modules/chal-engine/verification/README.md).

## Features implemented

- Full 64-square board with White at the bottom
- Standard starting position
- Tap to select a piece and view legal destinations
- Legal moves executed through `chess.js`
- Illegal moves rejected
- Turn tracking and complete move history
- Captures, check, checkmate, stalemate, castling, en passant, and promotion
- Undo and Restart controls
- Terminal-position handling
- Board Study offline FEN and PGN import with move-by-move navigation
- On-device offline chess analysis powered by Chal v2.0.0 with embedded NNUE evaluation
- Explicit 2-ply heuristic analysis fallback with UI disclosure
- Camera / Board Setup with live preview, manual four-corner calibration, and local OpenCV board detection
- Temporary capture cleanup after retake, confirmation, or leaving the workflow
- Opt-in offline test build with INTERNET and unused permissions blocked

## Known limitations

- Board detection identifies the outer board boundary and verifies a plausible 8 × 8 alternating-square pattern; it does not recognize pieces or verify that the pattern is a regulation chessboard.
- A strong non-board checker pattern (for example a checkerboard tablecloth or printed calibration target) can still pass. The detector cannot distinguish those from a chessboard using boundary and square evidence alone.
- Heavy glare, deep shadows, severe perspective, occlusion that covers square corners, or a thick decorative border inside the detected outline can weaken pattern evidence and cause a retry request. The detection score is a quality heuristic, not a calibrated probability.
- Chal engine search is serialized to a single concurrent active search in-process.
- No piece recognition from camera frames; positions are imported via FEN or PGN in Board Study or played on the local board.
- No voice instructions.
- No backend, accounts, payments, analytics, telemetry, or remote storage.
- Release builds use the local debug keystore and are for personal device testing only; do not distribute them.
- Offline test builds block INTERNET and cannot use Metro or any network service.
- Android native builds depend on a locally available JDK, Android SDK, CMake, and NDK.
