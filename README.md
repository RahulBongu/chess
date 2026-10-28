# AI Chess Navigator

Android-first mobile app for analyzing and navigating a physical chess game.

## Current milestone

Milestone 3.3: privacy hardening and a standalone offline test build. The app includes a playable local chessboard, user-initiated camera calibration, and local OpenCV board-boundary detection. Stockfish analysis, piece recognition, voice instructions, and backend services are not implemented.

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
- `src/vision/` — OpenCV board-boundary detection, scoring, and coordinate normalization
- `tests/` — focused chess, calibration, and vision tests

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
- Tests for candidate ranking, invalid geometry, ordering, bounds, confidence, and normalization

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

- Board detection identifies the outer board boundary only; piece recognition is not implemented.
- No Stockfish analysis or move suggestions.
- No voice instructions.
- No backend, accounts, payments, analytics, telemetry, or remote storage.
- Release builds use the local debug keystore and are for personal device testing only; do not distribute them.
- Offline test builds block INTERNET and cannot use Metro or any network service.
- Android native builds depend on a locally available JDK, Android SDK, CMake, and NDK.
