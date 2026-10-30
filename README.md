# AI Chess Navigator

Android-first mobile app for analyzing and navigating a physical chess game.

## Current milestone

Milestone 3.4: board-detector reliability. The app includes a playable local chessboard, user-initiated camera calibration, and local OpenCV board detection that checks both the outer boundary and an alternating 8 × 8 square pattern. Stockfish analysis, piece recognition, voice instructions, and backend services are not implemented.

## Technology stack

- Expo SDK 57
- React Native 0.86
- React 19.2
- TypeScript 6 (strict mode)
- Expo Router
- `expo-camera` with local OpenCV board-boundary detection
- `expo-file-system` for temporary capture cleanup
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
2. **Pattern evidence.** The candidate is rectified through a homography and sampled as an 8 × 8 grid. Patches near each cell's corners are used so pieces standing in square centres do not hide the square colour. The detector measures how well the patch means fit two checkerboard parity orientations, how many adjacent cell pairs alternate in the expected direction, the average adjacent-cell contrast, and the variation inside each cell. Uniform, low-contrast, striped, incomplete, and non-board rectangles fail this stage.

A candidate is accepted only when the geometry is valid, the interior passes the pattern-evidence gates, and the combined score reaches `MIN_DETECTION_CONFIDENCE` (`0.32`). Pattern gates require checkerboard parity to explain at least 55% of cell-mean variance, at least 80% of adjacent cell pairs to alternate in the expected direction, at least 8 gray levels of mean adjacent-square contrast, and a combined pattern quality of at least `0.45`. These thresholds are exercised by the deterministic fixtures rather than chosen only by inspection. The reported **detection score** is:

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
- Analysis panel marked unavailable until Stockfish is integrated
- Camera / Board Setup with live preview, manual four-corner calibration, and local OpenCV board detection
- Temporary capture cleanup after retake, confirmation, or leaving the workflow
- Opt-in offline test build with INTERNET and unused permissions blocked

## Known limitations

- Board detection identifies the outer board boundary and verifies a plausible 8 × 8 alternating-square pattern; it does not recognize pieces or verify that the pattern is a regulation chessboard.
- A strong non-board checker pattern (for example a checkerboard tablecloth or printed calibration target) can still pass. The detector cannot distinguish those from a chessboard using boundary and square evidence alone.
- Heavy glare, deep shadows, severe perspective, occlusion that covers square corners, or a thick decorative border inside the detected outline can weaken pattern evidence and cause a retry request. The detection score is a quality heuristic, not a calibrated probability.
- No Stockfish analysis or move suggestions.
- No voice instructions.
- No backend, accounts, payments, analytics, telemetry, or remote storage.
- Release builds use the local debug keystore and are for personal device testing only; do not distribute them.
- Offline test builds block INTERNET and cannot use Metro or any network service.
- Android native builds depend on a locally available JDK, Android SDK, CMake, and NDK.
