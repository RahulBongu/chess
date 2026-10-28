# AI Chess Navigator

Android-first mobile app for analyzing and navigating a physical chess game.

## Current milestone

Milestone 1: scaffold and functional chessboard. This version includes a playable local chessboard with authoritative rules from `chess.js`. Camera recognition, board calibration, Stockfish analysis, voice instructions, and backend services are intentionally not implemented yet.

## Technology stack

- Expo SDK 57
- React Native 0.86
- React 19.2
- TypeScript 6 (strict mode)
- Expo Router
- pnpm
- chess.js 1.4

Expo SDK 57 was selected because it is the current stable SDK available from `create-expo-app`, supports React Native 0.86, requires Node.js 22.13.x or newer (this machine uses Node.js 24.18.0), and targets Android SDK 36. It also leaves a clear upgrade path to SDK 58 later.

## Setup

```bash
pnpm install
```

## Run

```bash
pnpm start
pnpm android
pnpm web
```

`pnpm android` starts the Expo dev server and attempts to open an Android emulator or connected device. A JDK is required for native Android builds; this machine has an Adoptium JDK 17 under the user profile. Set `JAVA_HOME` to that JDK for the current shell if Android native builds are needed. No global tooling was changed by this project.

## Quality checks

```bash
pnpm typecheck
pnpm test
npx expo-doctor
```

## Project structure

- `src/app/` — Expo Router routes and screens
- `src/chess/` — authoritative chess game state and domain helpers
- `src/components/` — reusable UI components, including `ChessBoard`
- `src/theme/` — design tokens
- `src/types/` — ambient type declarations
- `tests/` — focused chess-domain tests

## Milestone 3 additions

- `react-native-fast-opencv@1.0.1` integration for local board-boundary detection
- `src/vision/` detection algorithm, validation, scoring, and coordinate normalization
- Auto-detect board workflow in camera setup with captured-image overlay and manual corner refinement
- Tests for candidate ranking, invalid geometry, ordering, bounds, confidence, and normalization

## Milestone 2 additions

- Runtime camera permission flow with denied/unavailable handling
- Live back-camera preview
- Manual four-corner board calibration with visible markers, connecting lines, undo, reset, and validation
- In-memory calibration state only; no images are stored or uploaded
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
- Camera / Board Setup entry point marked as planned

## Known limitations

- No camera, computer vision, or board recognition
- No Stockfish or move suggestions
- No voice instructions
- No backend, accounts, payments, or analytics
- Android native builds depend on a locally available JDK and Android tooling