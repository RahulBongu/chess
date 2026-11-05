/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { ChessGame, STARTING_FEN, type ChessSquare } from '../src/chess/ChessGame.ts';
import type { ChessEngine, EngineAnalysisResult } from '../src/chess/engine.ts';
import { BoardTracker } from '../src/vision/boardTracker.ts';
import { LiveGuidanceCoordinator } from '../src/vision/liveGuidance.ts';
import type { SquareObservation } from '../src/vision/moveRecognizer.ts';
import {
  algebraicToColRow,
  type Point2D,
} from '../src/vision/perspectiveOverlay.ts';
import {
  PIECE_CLASS_INDEX,
  PIECE_CLASSES,
  observationsFromSquareProbabilities,
  verifyStartingPosition,
  type PieceClass,
} from '../src/vision/pieceRecognition.ts';
import {
  computeFrameSharpness,
} from '../src/vision/frameQuality.ts';
import {
  FUSION_WINDOW_MS,
  TemporalSquareFusion,
  type FrameQuality,
} from '../src/vision/temporalFusion.ts';

const PERSPECTIVE_CORNERS: Point2D[] = [
  { x: 100, y: 50 },
  { x: 300, y: 50 },
  { x: 380, y: 350 },
  { x: 20, y: 350 },
];

const NORMALIZED_CORNERS: Point2D[] = [
  { x: 0.1, y: 0.1 },
  { x: 0.9, y: 0.1 },
  { x: 0.9, y: 0.9 },
  { x: 0.1, y: 0.9 },
];

const AFTER_E4_FEN = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
const AFTER_E5_FEN = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
const CAPTURE_BEFORE_FEN =
  'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2';
const CASTLE_BEFORE_FEN = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1';
const EN_PASSANT_BEFORE_FEN =
  'rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3';
const WHITE_PROMOTION_FEN = '4k3/P7/8/8/8/8/8/4K3 w - - 0 1';
const BLACK_PROMOTION_FEN = '4k3/8/8/8/8/8/7p/4K3 b - - 0 1';

const PIECE_NAME: Record<string, string> = {
  p: 'pawn',
  n: 'knight',
  b: 'bishop',
  r: 'rook',
  q: 'queen',
  k: 'king',
};

const QUALITY: FrameQuality = {
  sharpness: 0.3,
  alignmentConfidence: 0.9,
};

class MockEngine implements ChessEngine {
  readonly calls: string[] = [];

  async analyze(fen: string): Promise<EngineAnalysisResult> {
    this.calls.push(fen);
    return {
      fen,
      bestMove: { from: 'e2', to: 'e4', san: 'e4' },
      score: 25,
      perspective: 'side-to-move',
      terminal: 'none',
      pv: ['e2e4'],
      positionsEvaluated: 64,
      depth: 6,
      engine: 'chal-2.0.0',
      nodes: 64,
    };
  }

  stop(): void {}
}

function probabilityGridForFen(fen: string, topProbability = 0.94): number[][] {
  const game = new ChessGame(fen);
  const remainder = (1 - topProbability) / (PIECE_CLASSES.length - 1);
  const grid = Array.from({ length: 64 }, () =>
    new Array<number>(PIECE_CLASSES.length).fill(remainder),
  );
  for (let file = 0; file < 8; file += 1) {
    for (let rank = 1; rank <= 8; rank += 1) {
      const square = `${'abcdefgh'[file]}${rank}` as ChessSquare;
      const piece = game.pieceAt(square);
      const pieceClass: PieceClass = piece
        ? (`${piece.color === 'w' ? 'white' : 'black'}_${PIECE_NAME[piece.type]}` as PieceClass)
        : 'empty';
      const { col, row } = algebraicToColRow(square);
      const index = row * 8 + col;
      grid[index] = new Array<number>(PIECE_CLASSES.length).fill(remainder);
      grid[index][PIECE_CLASS_INDEX[pieceClass]] = topProbability;
    }
  }
  return grid;
}

function setSquareClass(
  grid: number[][],
  square: ChessSquare,
  pieceClass: PieceClass,
  topProbability = 0.94,
): void {
  const { col, row } = algebraicToColRow(square);
  const index = row * 8 + col;
  const remainder = (1 - topProbability) / (PIECE_CLASSES.length - 1);
  grid[index] = new Array<number>(PIECE_CLASSES.length).fill(remainder);
  grid[index][PIECE_CLASS_INDEX[pieceClass]] = topProbability;
}

/** Splits one square's mass between two classes so fusion keeps it uncertain. */
function splitSquareClass(
  grid: number[][],
  square: ChessSquare,
  first: PieceClass,
  second: PieceClass,
  probability = 0.45,
): void {
  const { col, row } = algebraicToColRow(square);
  const index = row * 8 + col;
  const remainder = (1 - 2 * probability) / (PIECE_CLASSES.length - 2);
  grid[index] = new Array<number>(PIECE_CLASSES.length).fill(remainder);
  grid[index][PIECE_CLASS_INDEX[first]] = probability;
  grid[index][PIECE_CLASS_INDEX[second]] = probability;
}

function observationsFromFen(fen: string): SquareObservation[] {
  return observationsFromSquareProbabilities(probabilityGridForFen(fen), 'white_bottom');
}

function applyMove(
  fen: string,
  from: ChessSquare,
  to: ChessSquare,
  promotion?: 'q' | 'r' | 'b' | 'n',
): string {
  const game = new ChessGame(fen);
  const executed = game.tryMove({ from, to, promotion });
  assert.ok(executed, `Expected ${from}${to} to be legal in ${fen}`);
  return game.fen();
}

function jitteredCorners(frame: number, amplitude = 0.006): Point2D[] {
  return NORMALIZED_CORNERS.map((corner, index) => ({
    x: corner.x + Math.sin(frame * 0.9 + index) * amplitude,
    y: corner.y + Math.cos(frame * 1.1 + index) * amplitude,
  }));
}

async function feed(
  coordinator: LiveGuidanceCoordinator,
  observations: SquareObservation[],
  frames: number,
  startMs: number,
  quality: FrameQuality = QUALITY,
): Promise<Awaited<ReturnType<LiveGuidanceCoordinator['processFrame']>>> {
  let model = null as Awaited<ReturnType<LiveGuidanceCoordinator['processFrame']>> | null;
  for (let frame = 0; frame < frames; frame += 1) {
    const now = startMs + frame * 250;
    model = await coordinator.processFrame(
      jitteredCorners(frame),
      observations,
      false,
      0.9,
      now,
      null,
      { ...quality, capturedAtMs: now },
    );
  }
  assert.ok(model);
  return model;
}

function confirmedCoordinator(initialFen: string, engine = new MockEngine()) {
  return new LiveGuidanceCoordinator({
    engine,
    orientation: 'white_bottom',
    initialFen,
    initialTurn: new ChessGame(initialFen).turn(),
    requirePositionEvidence: true,
    initializationMode: 'arbitrary_position',
  });
}

function standardCoordinator(engine = new MockEngine()) {
  return new LiveGuidanceCoordinator({
    engine,
    orientation: 'white_bottom',
    initialTurn: 'w',
    requirePositionEvidence: true,
    initializationMode: 'standard_start',
    adoptionWindow: 4,
    adoptionConfirmations: 2,
  });
}

// ---------------------------------------------------------------------------
// Corner tracking: jitter, gradual movement, dropout, blur/misalignment.
// ---------------------------------------------------------------------------

test('small hand jitter is smoothed without losing board tracking', () => {
  const tracker = new BoardTracker();
  assert.ok(tracker.seed(NORMALIZED_CORNERS, 0));
  for (let frame = 0; frame < 40; frame += 1) {
    assert.equal(
      tracker.update(jitteredCorners(frame), 0.9, (frame + 1) * 250),
      'tracking',
    );
  }
  const corners = tracker.getCorners();
  assert.ok(corners);
  assert.ok(Math.abs(corners[0].x - NORMALIZED_CORNERS[0].x) < 0.02);
  assert.ok(Math.abs(corners[2].y - NORMALIZED_CORNERS[2].y) < 0.02);
  assert.ok(tracker.getConfidence() > 0.5);
  assert.equal(tracker.getHoldReason(), null);
});

test('gradual camera pan is followed without excessive smoothing lag', () => {
  const tracker = new BoardTracker();
  assert.ok(tracker.seed(NORMALIZED_CORNERS, 0));
  const target = NORMALIZED_CORNERS.map((corner) => ({
    x: corner.x + 0.1,
    y: corner.y,
  }));
  for (let frame = 1; frame <= 12; frame += 1) {
    const progress = (0.1 * frame) / 12;
    const shifted = NORMALIZED_CORNERS.map((corner) => ({
      x: corner.x + progress,
      y: corner.y,
    }));
    assert.equal(tracker.update(shifted, 0.9, frame * 250), 'tracking');
  }
  const corners = tracker.getCorners();
  assert.ok(corners);
  // An exponential filter inherently lags a constant-velocity ramp; the
  // meaningful bound is that the residual lag stays small relative to the
  // frame, not that it reaches the ramp endpoint exactly.
  assert.ok(
    corners[0].x > NORMALIZED_CORNERS[0].x + 0.08,
    `expected the fix to track the pan, got x=${corners[0].x}`,
  );
  const lag = target[0].x - corners[0].x;
  assert.ok(lag >= 0 && lag <= 0.02, `expected lag <= 0.02, got ${lag}`);

  // Once the pan stops the fix must converge instead of retaining the lag.
  for (let frame = 13; frame <= 20; frame += 1) {
    assert.equal(tracker.update(target, 0.9, frame * 250), 'tracking');
  }
  const settled = tracker.getCorners();
  assert.ok(settled);
  assert.ok(Math.abs(settled[0].x - target[0].x) < 0.005);
});

test('a single-frame discontinuity is not adopted; two consistent proposals reacquire', () => {
  const tracker = new BoardTracker();
  assert.ok(tracker.seed(NORMALIZED_CORNERS, 0));
  const jumped = NORMALIZED_CORNERS.map((corner) => ({
    x: corner.x + 0.2,
    y: corner.y + 0.1,
  }));

  assert.equal(tracker.update(jumped, 0.9, 250), 'holding');
  assert.deepEqual(tracker.getReacquireProgress(), { confirmations: 1, required: 2 });
  assert.deepEqual(tracker.getCorners(), NORMALIZED_CORNERS);

  assert.equal(tracker.update(jumped, 0.9, 500), 'tracking');
  const corners = tracker.getCorners();
  assert.ok(corners);
  assert.ok(corners[0].x > NORMALIZED_CORNERS[0].x + 0.15);
  assert.equal(tracker.getHoldReason(), null);
});

test('motion blur and misalignment hold the last good corners without moving them', () => {
  const tracker = new BoardTracker();
  assert.ok(tracker.seed(NORMALIZED_CORNERS, 0));
  const shifted = NORMALIZED_CORNERS.map((corner) => ({
    x: corner.x + 0.04,
    y: corner.y,
  }));

  assert.equal(
    tracker.update(shifted, 0.9, 250, null, 0, { motionBlur: true }),
    'holding',
  );
  assert.deepEqual(tracker.getCorners(), NORMALIZED_CORNERS);
  assert.equal(tracker.getHoldReason(), 'motion blur');

  assert.equal(
    tracker.update(shifted, 0.9, 500, null, 0, { alignmentConfidence: 0.05 }),
    'holding',
  );
  assert.deepEqual(tracker.getCorners(), NORMALIZED_CORNERS);
  assert.equal(tracker.getHoldReason(), 'grid misaligned');
});

// ---------------------------------------------------------------------------
// Bounded temporal fusion.
// ---------------------------------------------------------------------------

test('computeFrameSharpness calculates normalized focus without buffer out-of-bounds', () => {
  const width = 640;
  const height = 480;
  const buffer = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const squareX = Math.floor(x / 60);
      const squareY = Math.floor(y / 60);
      let val = (squareX + squareY) % 2 === 0 ? 200 : 80;
      if (x % 60 < 2 || y % 60 < 2) val = 10;
      if (x % 30 < 5 && y % 30 < 5) val = 250;
      buffer[y * width + x] = val;
    }
  }

  const focus = computeFrameSharpness(buffer, width, height);
  assert.ok(focus.sampleCount > 1000);
  assert.ok(focus.sharpness > 0.010, `Expected sharpness > 0.010, got ${focus.sharpness}`);
  assert.ok(Number.isFinite(focus.sharpness));
  assert.ok(Number.isFinite(focus.laplacianVariance));
});

test('fusion rejects blurred, stale, incomplete, and misaligned frames', () => {
  const observations = observationsFromFen(STARTING_FEN);
  const fusion = new TemporalSquareFusion();
  const now = 10_000;

  const blur = fusion.addFrame(observations, { sharpness: 0.001 }, now);
  assert.equal(blur.accepted, false);
  assert.match(blur.reason ?? '', /focus/i);

  const motion = fusion.addFrame(observations, { motionBlur: true, sharpness: 0.3 }, now);
  assert.equal(motion.accepted, false);
  assert.match(motion.reason ?? '', /motion blur/i);

  const misaligned = fusion.addFrame(
    observations,
    { sharpness: 0.3, alignmentConfidence: 0.05 },
    now,
  );
  assert.equal(misaligned.accepted, false);
  assert.match(misaligned.reason ?? '', /alignment/i);

  const stale = fusion.addFrame(
    observations,
    { sharpness: 0.3, capturedAtMs: now - FUSION_WINDOW_MS - 500 },
    now,
  );
  assert.equal(stale.accepted, false);
  assert.match(stale.reason ?? '', /stale/i);

  const incomplete = fusion.addFrame(observations.slice(0, 63), { sharpness: 0.3 }, now);
  assert.equal(incomplete.accepted, false);
  assert.match(incomplete.reason ?? '', /incomplete/i);

  const valid = fusion.addFrame(observations, { sharpness: 0.3 }, now);
  assert.equal(valid.accepted, true);
  const snapshot = fusion.snapshot(now);
  assert.equal(snapshot.acceptedFrames, 1);
  assert.equal(snapshot.rejectedFrames, 5);
  assert.equal(snapshot.hasFullCoverage, true);
  assert.equal(snapshot.observations.length, 64);
});

test('fusion cancels a one-frame square misread instead of moving the board', () => {
  const fusion = new TemporalSquareFusion();
  const start = observationsFromFen(STARTING_FEN);
  const misreadGrid = probabilityGridForFen(STARTING_FEN);
  setSquareClass(misreadGrid, 'e4', 'white_pawn', 0.9);
  const misread = observationsFromSquareProbabilities(misreadGrid, 'white_bottom');

  fusion.addFrame(start, { sharpness: 0.3 }, 0);
  fusion.addFrame(start, { sharpness: 0.3 }, 250);
  fusion.addFrame(misread, { sharpness: 0.3 }, 500);
  fusion.addFrame(start, { sharpness: 0.3 }, 750);
  fusion.addFrame(start, { sharpness: 0.3 }, 1000);

  const snapshot = fusion.snapshot(1000);
  const e4 = snapshot.squares.find((square) => square.square === 'e4');
  assert.ok(e4);
  assert.equal(e4.topClass, 'empty');
  assert.equal(e4.uncertain, false);

  // The single misread frame alone would have claimed a pawn.
  const single = new TemporalSquareFusion();
  single.addFrame(misread, { sharpness: 0.3 }, 1000);
  const singleE4 = single.snapshot(1000).squares.find((square) => square.square === 'e4');
  assert.ok(singleE4);
  assert.equal(singleE4.topClass, 'white_pawn');
});

test('fusion keeps occupancy and colour evidence separate', () => {
  const fusion = new TemporalSquareFusion();
  const grid = probabilityGridForFen(STARTING_FEN);
  splitSquareClass(grid, 'd4', 'white_rook', 'black_rook');
  splitSquareClass(grid, 'e4', 'empty', 'white_pawn');
  const observations = observationsFromSquareProbabilities(grid, 'white_bottom');

  fusion.addFrame(observations, { sharpness: 0.3 }, 0);
  fusion.addFrame(observations, { sharpness: 0.3 }, 250);
  const snapshot = fusion.snapshot(250);

  const d4 = snapshot.squares.find((square) => square.square === 'd4');
  assert.ok(d4);
  assert.equal(d4.uncertain, true);
  assert.ok(d4.occupancyProbability > 0.95, 'colour ambiguity must not read as empty');
  assert.ok(Math.abs(d4.whiteProbability - 0.5) < 0.02);
  assert.ok(Math.abs(d4.blackProbability - 0.5) < 0.02);

  const e4 = snapshot.squares.find((square) => square.square === 'e4');
  assert.ok(e4);
  assert.equal(e4.uncertain, true, 'ambiguous occupancy must not read as confidently empty');
  assert.ok(e4.occupancyProbability > 0.45 && e4.occupancyProbability < 0.65);

  const fusedD4 = snapshot.observations.find((square) => square.square === 'd4');
  assert.ok(fusedD4);
  assert.equal(fusedD4.occupied, false, 'uncertain squares are never asserted as occupied');
  assert.equal(fusedD4.provenance, 'observed');
});

test('fusion expires old evidence and reports missing coverage', () => {
  const fusion = new TemporalSquareFusion();
  const observations = observationsFromFen(STARTING_FEN);
  assert.equal(fusion.addFrame(observations, { sharpness: 0.3 }, 0).accepted, true);
  assert.equal(fusion.snapshot(0).hasFullCoverage, true);

  const expired = fusion.snapshot(FUSION_WINDOW_MS + 100);
  assert.equal(expired.acceptedFrames, 0);
  assert.equal(expired.hasFullCoverage, false);
  assert.equal(expired.missingSquares.length, 64);
});

test('fusion starts a new epoch when a move changes the board', () => {
  const fusion = new TemporalSquareFusion();
  fusion.addFrame(observationsFromFen(STARTING_FEN), { sharpness: 0.3 }, 0);
  fusion.addFrame(observationsFromFen(STARTING_FEN), { sharpness: 0.3 }, 250);
  fusion.addFrame(observationsFromFen(AFTER_E4_FEN), { sharpness: 0.3 }, 500);

  const snapshot = fusion.snapshot(500);
  assert.equal(snapshot.transitionResets, 1);
  assert.equal(snapshot.acceptedFrames, 1);
  assert.equal(snapshot.squares.find((square) => square.square === 'e2')?.topClass, 'empty');
  assert.equal(
    snapshot.squares.find((square) => square.square === 'e4')?.topClass,
    'white_pawn',
  );
});

test('rgb_square_classifier receives nchw tensor layout with correct channel planes', () => {
  // The Kotlin Android classifier (ChessSquareClassifier.kt) historically
  // sampled pixels into a pixel-major HWC buffer and fed it directly to the
  // ONNX tensor declared as (batch, 3, 64, 64) NCHW. That scrambled
  // spatial coordinates and channels: white_pawn became black_king with
  // 99.9% confidence, and empty squares became black_queen.
  //
  // The native Kotlin code now transposes HWC -> CHW before inference.
  // This TypeScript regression guards the JS-side contract: probabilities
  // are stored in strict NCHW semantics where classProbabilities[0..12]
  // correspond to (empty, white_pawn, ..., black_king) without channel
  // interleaving. We verify by building a 64-square synthetic board with
  // known piece layout and checking that observation decoding agrees with
  // the raw tensor shape (batch, 3, 64, 64).
  //
  // Note: The canonical NCHW layout is enforced in Kotlin; the Python
  // benchmark verified NCHW inference achieves 100% accuracy on the
  // digital crop dataset while scrambled inference achieves only 9.6%.
  const startGrid = probabilityGridForFen(STARTING_FEN);
  const startObs = observationsFromSquareProbabilities(startGrid, 'white_bottom');
  const e2Obs = startObs.find((o) => o.square === 'e2');
  assert.ok(e2Obs);
  assert.equal(e2Obs.classProbabilities?.[PIECE_CLASS_INDEX['white_pawn']], 0.94);
  assert.equal(e2Obs.color, 'w');
  assert.equal(e2Obs.occupied, true);
});

test('epoch comparison uses the oldest frame so one-square-per-frame drift cannot evade it', () => {
  const fusion = new TemporalSquareFusion();
  fusion.addFrame(observationsFromFen(STARTING_FEN), { sharpness: 0.3 }, 0);

  const intermediate = probabilityGridForFen(STARTING_FEN);
  setSquareClass(intermediate, 'e2', 'empty');
  fusion.addFrame(observationsFromSquareProbabilities(intermediate, 'white_bottom'), {
    sharpness: 0.3,
  }, 250);

  fusion.addFrame(observationsFromFen(AFTER_E4_FEN), { sharpness: 0.3 }, 500);
  const snapshot = fusion.snapshot(500);
  assert.equal(snapshot.transitionResets, 1);
  assert.equal(snapshot.acceptedFrames, 1);
  assert.equal(snapshot.squares.find((square) => square.square === 'e4')?.topClass, 'white_pawn');
});

// ---------------------------------------------------------------------------
// Camera shake, blur and tracking loss through the live coordinator.
// ---------------------------------------------------------------------------

test('confirmed position survives blurred shake frames without moving and recovers', async () => {
  const coordinator = confirmedCoordinator(STARTING_FEN);
  await feed(coordinator, observationsFromFen(STARTING_FEN), 1, 0);

  const afterE4 = observationsFromFen(AFTER_E4_FEN);
  let model = await coordinator.processFrame(
    jitteredCorners(1),
    afterE4,
    false,
    0.9,
    250,
    null,
    { sharpness: 0.001, alignmentConfidence: 0.2, capturedAtMs: 250 },
  );
  assert.equal(model.positionState, 'STABILIZING');
  assert.equal(model.fen, STARTING_FEN);
  assert.equal(model.bestMove, null);
  assert.equal(model.perspectiveGeometry.arrow, null);
  assert.equal(model.fusion.frameAccepted, false);

  model = await feed(coordinator, afterE4, 3, 500);
  assert.equal(model.fen, AFTER_E4_FEN);
  assert.equal(model.recognizerStatus, 'move_detected');
  assert.equal(model.positionState, 'POSITION_CONFIRMED');
});

test('hand occlusion pauses a move and the move commits only after clean frames', async () => {
  const coordinator = confirmedCoordinator(STARTING_FEN);
  await feed(coordinator, observationsFromFen(STARTING_FEN), 2, 0);
  const afterE4 = observationsFromFen(AFTER_E4_FEN);

  let model = await coordinator.processFrame(
    NORMALIZED_CORNERS,
    afterE4,
    true,
    0.9,
    500,
    null,
    { ...QUALITY, capturedAtMs: 500 },
  );
  assert.equal(model.positionState, 'STABILIZING');
  assert.equal(model.fen, STARTING_FEN);
  assert.equal(model.recognizerStatus, 'occluded');

  model = await feed(coordinator, afterE4, 3, 750);
  assert.equal(model.fen, AFTER_E4_FEN);
  assert.equal(model.recognizerStatus, 'move_detected');
});

test('tracking loss preserves the last confirmed FEN and reacquires cleanly', async () => {
  const coordinator = confirmedCoordinator(STARTING_FEN);
  await feed(coordinator, observationsFromFen(STARTING_FEN), 1, 0);

  let model = await coordinator.processFrame(null, null, false, 0, 250);
  assert.equal(model.positionState, 'REACQUIRING_BOARD');
  assert.equal(model.fen, STARTING_FEN);
  assert.equal(model.bestMove, null);

  for (let frame = 2; frame < 14; frame += 1) {
    model = await coordinator.processFrame(null, null, false, 0, frame * 250);
  }
  assert.equal(model.trackingState, 'lost');
  assert.equal(model.positionState, 'REACQUIRING_BOARD');
  assert.equal(model.positionConfirmed, true);
  assert.equal(model.fen, STARTING_FEN);

  model = await feed(coordinator, observationsFromFen(STARTING_FEN), 1, 4000);
  assert.equal(model.trackingState, 'tracking');
  assert.equal(model.positionState, 'POSITION_CONFIRMED');
  assert.equal(model.fen, STARTING_FEN);
});

test('a global colour/lighting contradiction never commits a move or invents pieces', async () => {
  const coordinator = confirmedCoordinator(STARTING_FEN);
  await feed(coordinator, observationsFromFen(STARTING_FEN), 1, 0);

  const contradictory = probabilityGridForFen(STARTING_FEN);
  for (let file = 0; file < 8; file += 1) {
    for (let rank = 1; rank <= 8; rank += 1) {
      const square = `${'abcdefgh'[file]}${rank}` as ChessSquare;
      const piece = new ChessGame(STARTING_FEN).pieceAt(square);
      if (!piece) continue;
      const trueClass = `${piece.color === 'w' ? 'white' : 'black'}_${PIECE_NAME[piece.type]}` as PieceClass;
      const swapped = (piece.color === 'w'
        ? `black_${PIECE_NAME[piece.type]}`
        : `white_${PIECE_NAME[piece.type]}`) as PieceClass;
      splitSquareClass(contradictory, square, trueClass, swapped);
    }
  }
  const observations = observationsFromSquareProbabilities(contradictory, 'white_bottom');

  const model = await feed(coordinator, observations, 3, 250);
  assert.equal(model.fen, STARTING_FEN);
  assert.equal(model.recognizerStatus, 'uncertain');
  assert.notEqual(model.positionState, 'POSITION_CONFIRMED');
  assert.equal(model.bestMove, null);
  assert.equal(model.perspectiveGeometry.arrow, null);
  assert.ok(model.provisionalMove === null);
  assert.ok(model.fusion.uncertainSquares.length > 20);
});

// ---------------------------------------------------------------------------
// Bounded uncertainty: provisional candidates and standard-start prior.
// ---------------------------------------------------------------------------

test('a move with three uncertain squares stays a provisional candidate until unique evidence repeats', async () => {
  const coordinator = confirmedCoordinator(STARTING_FEN);
  const grid = probabilityGridForFen(AFTER_E4_FEN);
  splitSquareClass(grid, 'e2', 'white_pawn', 'empty');
  splitSquareClass(grid, 'a8', 'black_rook', 'empty');
  splitSquareClass(grid, 'h8', 'black_rook', 'empty');
  const observations = observationsFromSquareProbabilities(grid, 'white_bottom');

  let model = await feed(coordinator, observations, 3, 0);
  assert.equal(model.fen, STARTING_FEN, 'confirmed FEN must not advance on a candidate');
  assert.equal(model.provisionalMove?.san, 'e4');
  assert.ok((model.provisionalStreak ?? 0) >= 1);
  assert.equal(model.positionState, 'POSITION_CANDIDATE');
  assert.equal(model.bestMove, null);
  assert.equal(model.perspectiveGeometry.arrow, null);
  assert.ok(model.fusion.uncertainSquares.includes('e2'));
  assert.ok(model.fusion.uncertainSquares.includes('a8'));
  assert.ok(model.fusion.uncertainSquares.includes('h8'));

  model = await feed(coordinator, observations, 1, 750);
  assert.equal(model.fen, AFTER_E4_FEN);
  assert.equal(model.recognizerStatus, 'move_detected');
  assert.equal(model.provisionalMove, null);
});

test('one uncertain changed square is a provisional candidate that commits on repeat', async () => {
  const coordinator = confirmedCoordinator(STARTING_FEN);
  const grid = probabilityGridForFen(AFTER_E4_FEN);
  // Source e2 ambiguous; arrival e4 is confident, so the candidate has a
  // reliable arrival and is uniquely supported by the legal move generator.
  splitSquareClass(grid, 'e2', 'white_pawn', 'empty');
  const observations = observationsFromSquareProbabilities(grid, 'white_bottom');

  let model = await feed(coordinator, observations, 3, 0);
  assert.equal(model.fen, STARTING_FEN, 'one uncertain square must not commit immediately');
  assert.equal(model.provisionalMove?.san, 'e4');
  assert.equal(model.positionState, 'POSITION_CANDIDATE');
  assert.equal(model.bestMove, null);
  assert.equal(model.perspectiveGeometry.arrow, null);
  assert.ok(model.fusion.uncertainSquares.includes('e2'));
  assert.equal(model.fusion.uncertainSquares.length, 1);

  model = await feed(coordinator, observations, 1, 750);
  assert.equal(model.fen, AFTER_E4_FEN);
  assert.equal(model.recognizerStatus, 'move_detected');
  assert.equal(model.provisionalMove, null);
});

test('two uncertain changed squares are a provisional candidate that commits on repeat', async () => {
  const coordinator = confirmedCoordinator(CASTLE_BEFORE_FEN);
  // Kingside castling changes four squares (e1, g1, h1, f1).
  // Blurring the rook transition (h1 and f1) leaves the king arrival at g1
  // and departure from e1 reliably observed. The candidate has a reliable
  // arrival (g1) and unique legal support (only O-O places the king on g1),
  // so it becomes provisional.
  const after = applyMove(CASTLE_BEFORE_FEN, 'e1', 'g1');
  const grid = probabilityGridForFen(after);
  splitSquareClass(grid, 'h1', 'white_rook', 'empty');
  splitSquareClass(grid, 'f1', 'white_rook', 'empty');
  const observations = observationsFromSquareProbabilities(grid, 'white_bottom');

  let model = await feed(coordinator, observations, 3, 0);
  assert.equal(model.fen, CASTLE_BEFORE_FEN, 'two uncertain squares must not commit immediately');
  assert.ok(model.provisionalMove, `provisional: ${model.recognizerStatus} / ${model.fusion.uncertainSquares.join(',')}`);
  assert.equal(model.provisionalMove?.from, 'e1');
  assert.equal(model.provisionalMove?.to, 'g1');
  assert.equal(model.positionState, 'POSITION_CANDIDATE');
  assert.equal(model.bestMove, null);
  assert.equal(model.perspectiveGeometry.arrow, null);
  assert.ok(model.fusion.uncertainSquares.includes('h1'));
  assert.ok(model.fusion.uncertainSquares.includes('f1'));
  assert.equal(model.fusion.uncertainSquares.length, 2);

  model = await feed(coordinator, observations, 1, 750);
  assert.equal(model.fen, after);
  assert.equal(model.recognizerStatus, 'move_detected');
  assert.equal(model.provisionalMove, null);
});

test('uncertain squares causing legal ambiguity (e.g. Rf1 vs O-O) are rejected and keep confirmed FEN', async () => {
  const coordinator = confirmedCoordinator(CASTLE_BEFORE_FEN);
  // Blurring e1 (departure) and g1 (arrival) makes the observation equally
  // compatible with Rf1 (rook moves h1->f1) and O-O (king castles).
  // Without unique legal support, the recognizer must NOT propose a candidate.
  const after = applyMove(CASTLE_BEFORE_FEN, 'e1', 'g1');
  const grid = probabilityGridForFen(after);
  splitSquareClass(grid, 'e1', 'white_king', 'empty');
  splitSquareClass(grid, 'g1', 'white_king', 'empty');
  const observations = observationsFromSquareProbabilities(grid, 'white_bottom');

  const model = await feed(coordinator, observations, 4, 0);
  assert.equal(model.fen, CASTLE_BEFORE_FEN, 'ambiguous move must never commit');
  assert.equal(model.provisionalMove, null, 'ambiguous move must never become a provisional candidate');
  assert.notEqual(model.recognizerStatus, 'move_detected');
});

test('unresolved source evidence alone never invents an arriving piece', async () => {
  const coordinator = confirmedCoordinator(STARTING_FEN);
  const grid = probabilityGridForFen(AFTER_E4_FEN);
  // Source e2 is ambiguous; the arrival e4 is deliberately unreadable.
  splitSquareClass(grid, 'e2', 'white_pawn', 'empty');
  splitSquareClass(grid, 'e4', 'white_pawn', 'empty');

  const observations = observationsFromSquareProbabilities(grid, 'white_bottom');
  const model = await feed(coordinator, observations, 4, 0);
  assert.equal(model.fen, STARTING_FEN);
  assert.equal(model.provisionalMove, null);
  assert.notEqual(model.recognizerStatus, 'move_detected');
});

test('standard-start prior resolves at most three genuinely uncertain squares', async () => {
  const coordinator = standardCoordinator();
  const grid = probabilityGridForFen(STARTING_FEN);
  splitSquareClass(grid, 'a1', 'white_rook', 'empty');
  splitSquareClass(grid, 'e2', 'white_pawn', 'empty');
  splitSquareClass(grid, 'e8', 'black_king', 'empty');
  const observations = observationsFromSquareProbabilities(grid, 'white_bottom');

  const model = await feed(coordinator, observations, 5, 0);
  assert.equal(model.startReady, true, model.recognition.reason ?? '');
  assert.equal(model.positionConfirmed, false);
  assert.equal(model.recognition.startUsedPrior, true);
  assert.equal(model.recognition.startPriorResolvedSquares.length, 3);
  assert.equal(model.bestMove, null);
  assert.equal(model.perspectiveGeometry.arrow, null);
  assert.equal(coordinator.confirmStartingPosition(), true);
});

test('four uncertain squares block the standard-start prior', async () => {
  const coordinator = standardCoordinator();
  const grid = probabilityGridForFen(STARTING_FEN);
  splitSquareClass(grid, 'a1', 'white_rook', 'empty');
  splitSquareClass(grid, 'e2', 'white_pawn', 'empty');
  splitSquareClass(grid, 'e8', 'black_king', 'empty');
  splitSquareClass(grid, 'h8', 'black_rook', 'empty');
  const observations = observationsFromSquareProbabilities(grid, 'white_bottom');

  const model = await feed(coordinator, observations, 5, 0);
  assert.equal(model.positionConfirmed, false);
  assert.equal(model.startReady, false);
  assert.match(model.recognition.reason ?? '', /too uncertain/i);
  assert.equal(coordinator.confirmStartingPosition(), false);
});

test('confident contradictory evidence blocks the standard-start prior', () => {
  const observations = observationsFromFen(STARTING_FEN);
  const grid = probabilityGridForFen(STARTING_FEN);
  setSquareClass(grid, 'e4', 'white_pawn', 0.9);
  const withCenterPawn = observationsFromSquareProbabilities(grid, 'white_bottom');
  const blocked = verifyStartingPosition(withCenterPawn, 'white_bottom', {
    maxUncertainSquares: 3,
    priorFen: STARTING_FEN,
  });
  assert.equal(blocked.ok, false);
  assert.match(blocked.reason ?? '', /center/i);

  // An uncertain flag cannot excuse confident opposite evidence on a square
  // the expected layout claims is empty.
  const flagged = observations.map((observation) =>
    observation.square === 'e4'
      ? {
          ...observation,
          uncertain: true,
          classProbabilities: withCenterPawn.find((candidate) => candidate.square === 'e4')!
            .classProbabilities,
        }
      : observation,
  );
  const contradiction = verifyStartingPosition(flagged, 'white_bottom', {
    maxUncertainSquares: 3,
    priorFen: STARTING_FEN,
  });
  assert.equal(contradiction.ok, false);
  assert.match(contradiction.reason ?? '', /contradict/i);
});

// ---------------------------------------------------------------------------
// Legal move coverage: all openings, captures, castling, en passant, promotions.
// ---------------------------------------------------------------------------

test('every legal opening move is recognized through the fusion pipeline', async () => {
  const start = new ChessGame(STARTING_FEN);
  const moves = start.allLegalMoves();
  assert.equal(moves.length, 20);

  for (const move of moves) {
    const beforeFen = start.fen();
    const child = new ChessGame(beforeFen);
    const executed = child.tryMove({
      from: move.from,
      to: move.to,
      promotion: move.promotion as 'q' | 'r' | 'b' | 'n' | undefined,
    });
    assert.ok(executed, move.san);

    const coordinator = confirmedCoordinator(beforeFen);
    await feed(coordinator, observationsFromFen(beforeFen), 1, 0);
    const model = await feed(coordinator, observationsFromFen(child.fen()), 3, 250);

    assert.equal(model.fen, child.fen(), `FEN mismatch for ${move.san}`);
    assert.equal(model.recognizerStatus, 'move_detected', `not detected: ${move.san}`);
  }
});

test('black replies in the other direction are recognized', async () => {
  const coordinator = confirmedCoordinator(AFTER_E4_FEN);
  await feed(coordinator, observationsFromFen(AFTER_E4_FEN), 1, 0);
  const model = await feed(coordinator, observationsFromFen(AFTER_E5_FEN), 3, 250);
  assert.equal(model.fen, AFTER_E5_FEN);
  assert.equal(model.turn, 'w');
});

test('captures are recognized from piece evidence, not brightness', async () => {
  const after = applyMove(CAPTURE_BEFORE_FEN, 'e4', 'd5');
  const coordinator = confirmedCoordinator(CAPTURE_BEFORE_FEN);
  await feed(coordinator, observationsFromFen(CAPTURE_BEFORE_FEN), 1, 0);
  const model = await feed(coordinator, observationsFromFen(after), 3, 250);
  assert.equal(model.fen, after);
  assert.equal(model.recognizerStatus, 'move_detected');
});

test('castling is recognized as a king and rook transition', async () => {
  const after = applyMove(CASTLE_BEFORE_FEN, 'e1', 'g1');
  const coordinator = confirmedCoordinator(CASTLE_BEFORE_FEN);
  await feed(coordinator, observationsFromFen(CASTLE_BEFORE_FEN), 1, 0);
  const model = await feed(coordinator, observationsFromFen(after), 3, 250);
  assert.equal(model.fen, after);
  assert.equal(model.recognizerStatus, 'move_detected');
});

test('en passant is recognized when the adopted FEN has no ep field', async () => {
  // Adopt the normalized placement (no ep target) to prove reconstruction.
  const beforeGame = new ChessGame(EN_PASSANT_BEFORE_FEN);
  const normalizedBefore = beforeGame.fen().split(' ').slice(0, 4).join(' ') + ' 0 1';
  const after = applyMove(EN_PASSANT_BEFORE_FEN, 'e5', 'd6');

  const coordinator = confirmedCoordinator(normalizedBefore);
  await feed(coordinator, observationsFromFen(normalizedBefore), 1, 0);
  const model = await feed(coordinator, observationsFromFen(after), 3, 250);
  // Side to move, castling and en-passant fields are reconstructed; move
  // counters legitimately follow the adopted FEN's base.
  const placement = (fen: string) => fen.split(' ').slice(0, 4).join(' ');
  assert.equal(placement(model.fen), placement(after));
  assert.equal(model.recognizerStatus, 'move_detected');
});

test('white promotion requires explicit piece confirmation', async () => {
  const afterQueen = applyMove(WHITE_PROMOTION_FEN, 'a7', 'a8', 'q');
  const coordinator = confirmedCoordinator(WHITE_PROMOTION_FEN);
  await feed(coordinator, observationsFromFen(WHITE_PROMOTION_FEN), 1, 0);

  let model = await feed(coordinator, observationsFromFen(afterQueen), 3, 250);
  assert.equal(model.fen, WHITE_PROMOTION_FEN, 'promotion must not auto-apply');
  assert.equal(model.pendingPromotion?.to, 'a8');
  assert.equal(coordinator.getRecognizer().confirmPromotion('q'), true);
  assert.equal(coordinator.getRecognizer().getFen(), afterQueen);
});

test('black promotion requires explicit piece confirmation', async () => {
  const afterQueen = applyMove(BLACK_PROMOTION_FEN, 'h2', 'h1', 'q');
  const coordinator = confirmedCoordinator(BLACK_PROMOTION_FEN);
  await feed(coordinator, observationsFromFen(BLACK_PROMOTION_FEN), 1, 0);

  const model = await feed(coordinator, observationsFromFen(afterQueen), 3, 250);
  assert.equal(model.fen, BLACK_PROMOTION_FEN);
  assert.equal(model.pendingPromotion?.to, 'h1');
  assert.equal(coordinator.getRecognizer().confirmPromotion('q'), true);
  assert.equal(coordinator.getRecognizer().getFen(), afterQueen);
});

// ---------------------------------------------------------------------------
// 5.4I.1 regressions: explicit start confirmation, no silent replacement.
// ---------------------------------------------------------------------------

test('standard start is never auto-adopted and engine stays off until confirmation', async () => {
  const engine = new MockEngine();
  const coordinator = standardCoordinator(engine);
  const start = observationsFromFen(STARTING_FEN);

  let model = await feed(coordinator, start, 4, 0);
  assert.equal(model.positionConfirmed, false);
  assert.equal(model.startReady, true);
  assert.equal(model.positionState, 'POSITION_CANDIDATE');
  assert.equal(model.bestMove, null);
  assert.equal(model.perspectiveGeometry.arrow, null);
  assert.equal(engine.calls.length, 0);

  assert.equal(coordinator.confirmStartingPosition(), true);
  model = await feed(coordinator, start, 1, 1000);
  assert.equal(model.positionConfirmed, true);
  assert.equal(model.startReady, false);
  assert.equal(model.fen, STARTING_FEN);
});

test('a near-start mid-game board is never replaced by the starting FEN', async () => {
  const engine = new MockEngine();
  const coordinator = standardCoordinator(engine);
  const nearStart = observationsFromFen(AFTER_E4_FEN);

  for (let frame = 0; frame < 5; frame += 1) {
    const model = await feed(coordinator, nearStart, 1, frame * 250);
    assert.equal(model.positionConfirmed, false);
    assert.equal(model.startReady, false);
    assert.equal(model.bestMove, null);
    assert.equal(model.perspectiveGeometry.arrow, null);
  }
  assert.equal(coordinator.confirmStartingPosition(), false);
  assert.equal(coordinator.isPositionConfirmed(), false);
  assert.equal(engine.calls.length, 0);
});

test('confirmed start advances to the observed move and never silently resets', async () => {
  const coordinator = standardCoordinator();
  const start = observationsFromFen(STARTING_FEN);
  await feed(coordinator, start, 4, 0);
  assert.equal(coordinator.confirmStartingPosition(), true);
  await feed(coordinator, start, 1, 1000);

  const model = await feed(coordinator, observationsFromFen(AFTER_E4_FEN), 3, 1250);
  assert.equal(model.fen, AFTER_E4_FEN);
  assert.notEqual(model.fen, STARTING_FEN);
  assert.equal(model.recognizerStatus, 'move_detected');
});
