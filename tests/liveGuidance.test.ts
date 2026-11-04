/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { LiveGuidanceCoordinator } from '../src/vision/liveGuidance.ts';
import type { Point2D } from '../src/vision/perspectiveOverlay.ts';
import { algebraicToColRow } from '../src/vision/perspectiveOverlay.ts';
import type { SquareObservation } from '../src/vision/moveRecognizer.ts';
import { ChessGame, STARTING_FEN, type ChessSquare } from '../src/chess/ChessGame.ts';
import type { ChessEngine, EngineAnalysisResult } from '../src/chess/engine.ts';
import {
  PIECE_CLASS_INDEX,
  PIECE_CLASSES,
  observationsFromSquareProbabilities,
  type PieceClass,
} from '../src/vision/pieceRecognition.ts';

const PERSPECTIVE_CORNERS: Point2D[] = [
  { x: 100, y: 50 },  // top-left
  { x: 300, y: 50 },  // top-right
  { x: 380, y: 350 }, // bottom-right
  { x: 20, y: 350 },  // bottom-left
];

class MockEngine implements ChessEngine {
  private stopped = false;

  async analyze(fen: string): Promise<EngineAnalysisResult> {
    return {
      fen,
      bestMove: {
        from: 'e2',
        to: 'e4',
        san: 'e4',
      },
      score: 35,
      perspective: 'side-to-move',
      terminal: 'none',
      pv: ['e2e4', 'e7e5'],
      positionsEvaluated: 100,
      depth: 6,
      engine: 'chal-2.0.0',
      nodes: 100,
    };
  }

  stop(): void {
    this.stopped = true;
  }
}

function createDummyObservations(): SquareObservation[] {
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const obs: SquareObservation[] = [];
  for (const file of files) {
    for (let rank = 1; rank <= 8; rank += 1) {
      obs.push({
        square: `${file}${rank}` as ChessSquare,
        occupied: rank <= 2 || rank >= 7,
        confidence: 0.9,
      });
    }
  }
  return obs;
}

const PIECE_NAME: Record<string, string> = {
  p: 'pawn',
  n: 'knight',
  b: 'bishop',
  r: 'rook',
  q: 'queen',
  k: 'king',
};

/** Builds the same 64 x 13 evidence the native RGB classifier returns. */
function classifiedObservations(fen: string): SquareObservation[] {
  const game = new ChessGame(fen);
  const remainder = (1 - 0.94) / (PIECE_CLASSES.length - 1);
  const probabilities = Array.from({ length: 64 }, () =>
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
      probabilities[row * 8 + col][PIECE_CLASS_INDEX[pieceClass]] = 0.94;
    }
  }
  return observationsFromSquareProbabilities(probabilities, 'white_bottom');
}

/** Same evidence with one square deliberately misclassified in this frame. */
function classifiedObservationsWithMistake(
  fen: string,
  square: ChessSquare,
  pieceClass: PieceClass,
): SquareObservation[] {
  const observations = classifiedObservations(fen);
  const remainder = 0.1 / (PIECE_CLASSES.length - 1);
  const probabilities = new Array<number>(PIECE_CLASSES.length).fill(remainder);
  probabilities[PIECE_CLASS_INDEX[pieceClass]] = 0.9;
  return observations.map((observation) =>
    observation.square === square
      ? { ...observation, occupied: true, color: 'w', classProbabilities: probabilities }
      : observation,
  );
}

const AFTER_E4_FEN = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

test('LiveGuidanceCoordinator produces perspective geometry and instructions', async () => {
  const mockEngine = new MockEngine();
  const coordinator = new LiveGuidanceCoordinator({
    engine: mockEngine,
    orientation: 'white_bottom',
  });

  const obs = createDummyObservations();
  // Frame 1 launches asynchronous engine analysis
  await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  // Yield to allow mock engine promise to resolve
  await new Promise((resolve) => setTimeout(resolve, 5));
  // Frame 2 receives the resolved engine guidance
  const model = await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);

  assert.equal(model.trackingState, 'tracking');
  assert.equal(model.turn, 'w');
  assert.ok(model.perspectiveGeometry.corners !== null);
  assert.ok(model.perspectiveGeometry.gridLines !== null);
  assert.equal(model.perspectiveGeometry.gridLines.length, 14);

  // Engine best move is e2 -> e4
  // Source quad (e2) and destination quad (e4) must be strictly convex
  assert.ok(model.perspectiveGeometry.sourceQuad !== null);
  assert.ok(model.perspectiveGeometry.destinationQuad !== null);
  assert.ok(model.perspectiveGeometry.arrow !== null);

  // Milestone 5.4F stage telemetry is populated for every processed frame.
  assert.ok(Number.isFinite(model.stageTimings.trackerMs));
  assert.ok(Number.isFinite(model.stageTimings.recognizerMs));
  assert.ok(Number.isFinite(model.stageTimings.geometryMs));
  assert.ok(Number.isFinite(model.stageTimings.totalMs));
  assert.ok(model.stageTimings.totalMs >= 0);
  assert.equal(model.engineResultCount, 1);
  assert.ok(model.engineLatencyMs !== null && model.engineLatencyMs >= 0);

  // Verify instruction text
  assert.ok(
    model.instruction.includes('e2') && model.instruction.includes('e4'),
    `Unexpected instruction: "${model.instruction}"`,
  );
});

test('LiveGuidanceCoordinator reflects orientation flip in projected coordinates', async () => {
  const mockEngine = new MockEngine();
  const coordinatorWhite = new LiveGuidanceCoordinator({
    engine: mockEngine,
    orientation: 'white_bottom',
  });
  const coordinatorBlack = new LiveGuidanceCoordinator({
    engine: mockEngine,
    orientation: 'black_bottom',
  });

  const obs = createDummyObservations();
  await coordinatorWhite.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  await coordinatorBlack.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  await new Promise((resolve) => setTimeout(resolve, 5));

  const modelWhite = await coordinatorWhite.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  const modelBlack = await coordinatorBlack.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);

  // In white_bottom, e2 is in row 6 (near bottom, y > 200)
  // In black_bottom, e2 is in row 1 (near top, y < 150)
  const whiteSourceY = modelWhite.perspectiveGeometry.sourceQuad![0].y;
  const blackSourceY = modelBlack.perspectiveGeometry.sourceQuad![0].y;

  assert.ok(
    whiteSourceY > blackSourceY + 50,
    `whiteSourceY (${whiteSourceY}) should be significantly greater than blackSourceY (${blackSourceY})`,
  );
});

test('LiveGuidanceCoordinator handles lost tracking state gracefully', async () => {
  const coordinator = new LiveGuidanceCoordinator();
  // No corners passed
  const model = await coordinator.processFrame(null, null);

  assert.equal(model.trackingState, 'uncalibrated');
  assert.equal(model.perspectiveGeometry.corners, null);
  assert.equal(model.perspectiveGeometry.arrow, null);
  assert.ok(model.instruction.includes('Locating board') || model.instruction.includes('Point camera'));
});

test('LiveGuidanceCoordinator suppresses stale engine analysis from an earlier position', async () => {
  let resolveFirstSearch: (res: EngineAnalysisResult) => void;
  const delayedEngine: ChessEngine = {
    analyze: (fen: string) => {
      if (fen.includes('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR')) {
        // Initial position search: delay artificially
        return new Promise<EngineAnalysisResult>((resolve) => {
          resolveFirstSearch = resolve;
        });
      }
      // Subsequent position search: resolve immediately
      return Promise.resolve({
        fen,
        bestMove: { from: 'e7', to: 'e5', san: 'e5' },
        score: -20,
        perspective: 'side-to-move',
        terminal: 'none',
        pv: ['e7e5'],
        positionsEvaluated: 50,
        depth: 5,
        engine: 'chal-2.0.0',
        nodes: 50,
      });
    },
    stop: () => {},
  };

  const coordinator = new LiveGuidanceCoordinator({ engine: delayedEngine });
  const obs = createDummyObservations();

  // Position 1: start position triggers delayed search
  const model1 = await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  assert.equal(model1.bestMove, null); // search still in flight

  // Position 2: White plays e4, so position advances to Black to move
  coordinator.getRecognizer().confirmManualMove('e2', 'e4');
  await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  await new Promise((resolve) => setTimeout(resolve, 5));
  const model2 = await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  assert.equal(model2.bestMove?.san, 'e5'); // Position 2's immediate search accepted

  // Now, delayed search for Position 1 resolves late!
  resolveFirstSearch!({
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    bestMove: { from: 'd2', to: 'd4', san: 'd4' },
    score: 40,
    perspective: 'side-to-move',
    terminal: 'none',
    pv: ['d2d4'],
    positionsEvaluated: 100,
    depth: 6,
    engine: 'chal-2.0.0',
    nodes: 100,
  });

  // Verify next frame does NOT overwrite Position 2 with stale Position 1 result!
  const model3 = await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  assert.equal(model3.bestMove?.san, 'e5');
  assert.notEqual(model3.bestMove?.san, 'd4');
});

test('requirePositionEvidence keeps engine and arrows off until a classified position is adopted', async () => {
  const calls: string[] = [];
  const engine: ChessEngine = {
    async analyze(fen: string): Promise<EngineAnalysisResult> {
      calls.push(fen);
      return {
        fen,
        bestMove: { from: 'e2', to: 'e4', san: 'e4' },
        score: 35,
        perspective: 'side-to-move',
        terminal: 'none',
        pv: ['e2e4', 'e7e5'],
        positionsEvaluated: 100,
        depth: 6,
        engine: 'chal-2.0.0',
        nodes: 100,
      };
    },
    stop: () => {},
  };

  const coordinator = new LiveGuidanceCoordinator({
    engine,
    orientation: 'white_bottom',
    initialTurn: 'w',
    requirePositionEvidence: true,
    initializationMode: 'arbitrary_position',
  });

  // Occupancy-only observations must never confirm a position or start the
  // engine, even while the tracker has stable corners.
  const dummy = createDummyObservations();
  for (let frame = 0; frame < 4; frame += 1) {
    const model = await coordinator.processFrame(PERSPECTIVE_CORNERS, dummy, false, 0.9);
    assert.equal(model.trackingState, 'tracking');
    assert.equal(model.positionConfirmed, false);
    assert.equal(model.bestMove, null);
    assert.equal(model.evaluationScore, null);
    assert.equal(model.perspectiveGeometry.arrow, null);
  }
  assert.equal(calls.length, 0);

  // Four classified frames fill the averaging window. The first averaged
  // candidate establishes a streak; the next averaged frame confirms it.
  const start = classifiedObservations(STARTING_FEN);
  let model;
  for (let frame = 0; frame < 4; frame += 1) {
    model = await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
    assert.equal(model.positionConfirmed, false);
    assert.equal(model.recognition.adoptionFrames, frame + 1);
    assert.equal(model.bestMove, null);
  }
  assert.equal(model!.recognition.adoptionStreak, 1);

  model = await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  assert.equal(model.positionConfirmed, true);
  assert.equal(model.recognition.adoptionStreak, 0);
  assert.equal(model.recognition.adoptionFrames, 0);
  assert.equal(model.recognition.uncertainSquares.length, 0);
  assert.equal(model.bestMove, null, 'engine search should still be in flight');

  await sleep(10);
  model = await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  assert.equal(model.positionConfirmed, true);
  assert.equal(model.bestMove?.san, 'e4');
  assert.ok(model.perspectiveGeometry.arrow !== null);
  assert.equal(calls.length, 1);
});

test('a user-confirmed position is adopted without classifier evidence', async () => {
  const engine = new MockEngine();
  const coordinator = new LiveGuidanceCoordinator({
    engine,
    requirePositionEvidence: true,
    adoptionWindow: 4,
    adoptionConfirmations: 2,
  });

  assert.equal(coordinator.adoptUserPosition(STARTING_FEN), true);
  let model = await coordinator.processFrame(PERSPECTIVE_CORNERS, null, false, 0.9);
  assert.equal(model.positionConfirmed, true);
  assert.equal(model.fen, STARTING_FEN);
  assert.equal(model.recognition.adoptionFrames, 0);
  assert.equal(model.bestMove, null, 'search should still be in flight');

  await sleep(10);
  model = await coordinator.processFrame(PERSPECTIVE_CORNERS, null, false, 0.9);
  assert.equal(model.bestMove?.san, 'e4');
  assert.equal(model.recognition.reason, null);
});

test('rejects a FEN that does not round-trip exactly', () => {
  const coordinator = new LiveGuidanceCoordinator({
    engine: new MockEngine(),
    requirePositionEvidence: true,
  });
  assert.equal(coordinator.adoptUserPosition('not-a-fen'), false);
  assert.equal(coordinator.adoptUserPosition('8/8/8/8/8/8/8/8 w - - 0 1'), false);
});

test('adoption averaging absorbs a one-frame misclassification before confirming', async () => {
  const coordinator = new LiveGuidanceCoordinator({
    engine: new MockEngine(),
    requirePositionEvidence: true,
    adoptionWindow: 4,
    adoptionConfirmations: 2,
    initializationMode: 'arbitrary_position',
  });

  const start = classifiedObservations(STARTING_FEN);
  const misread = classifiedObservationsWithMistake(STARTING_FEN, 'e4', 'white_pawn');

  // The averaging window contains three correct frames and one misread pawn
  // on e4. The averaged evidence must still produce the legal start position.
  await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  await coordinator.processFrame(PERSPECTIVE_CORNERS, misread, false, 0.9);
  let model = await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  assert.equal(model.positionConfirmed, false);
  assert.equal(model.recognition.adoptionStreak, 1);

  model = await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  assert.equal(model.positionConfirmed, true, model.recognition.reason ?? '');
  assert.equal(model.fen, STARTING_FEN);
});

test('adoption evidence resets on occlusion, orientation, and turn changes', async () => {
  const coordinator = new LiveGuidanceCoordinator({
    engine: new MockEngine(),
    requirePositionEvidence: true,
    adoptionWindow: 4,
    adoptionConfirmations: 2,
    initializationMode: 'arbitrary_position',
  });
  const start = classifiedObservations(STARTING_FEN);

  await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  let model = await coordinator.processFrame(PERSPECTIVE_CORNERS, start, true, 0.9);
  assert.equal(model.positionConfirmed, false);
  assert.equal(model.recognition.adoptionFrames, 0);
  assert.match(model.instruction, /Hand over board/);

  // Refill the window, then flip orientation: stale evidence must not carry
  // across, because its algebraic squares change meaning.
  await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  coordinator.setOrientation('black_bottom');
  model = await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  assert.equal(model.recognition.adoptionFrames, 1);

  await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  assert.equal(
    (await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9)).recognition
      .adoptionFrames,
    3,
  );
  coordinator.setTurn('b');
  model = await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  assert.equal(model.recognition.adoptionFrames, 1);
  assert.equal(model.positionConfirmed, false);
});

test('classified move recognition never revives stale engine analysis', async () => {
  let resolveFirstSearch: ((result: EngineAnalysisResult) => void) | null = null;
  const engine: ChessEngine = {
    analyze(fen: string): Promise<EngineAnalysisResult> {
      if (fen === STARTING_FEN) {
        return new Promise<EngineAnalysisResult>((resolve) => {
          resolveFirstSearch = resolve;
        });
      }
      return Promise.resolve({
        fen,
        bestMove: { from: 'e7', to: 'e5', san: 'e5' },
        score: -20,
        perspective: 'side-to-move',
        terminal: 'none',
        pv: ['e7e5'],
        positionsEvaluated: 50,
        depth: 5,
        engine: 'chal-2.0.0',
        nodes: 50,
      });
    },
    stop: () => {},
  };

  const coordinator = new LiveGuidanceCoordinator({
    engine,
    initialTurn: 'w',
    requirePositionEvidence: true,
    initializationMode: 'arbitrary_position',
  });

  const start = classifiedObservations(STARTING_FEN);
  for (let frame = 0; frame < 5; frame += 1) {
    await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  }
  const pending = await coordinator.processFrame(PERSPECTIVE_CORNERS, start, false, 0.9);
  assert.equal(pending.positionConfirmed, true);
  assert.equal(pending.bestMove, null);
  assert.ok(resolveFirstSearch !== null);

  const afterE4 = classifiedObservations(AFTER_E4_FEN);
  let model = await coordinator.processFrame(PERSPECTIVE_CORNERS, afterE4, false, 0.9);
  for (let frame = 0; frame < 2; frame += 1) {
    model = await coordinator.processFrame(PERSPECTIVE_CORNERS, afterE4, false, 0.9);
  }
  assert.equal(model.recognizerStatus, 'move_detected');
  assert.equal(model.fen, AFTER_E4_FEN);

  await sleep(10);
  model = await coordinator.processFrame(PERSPECTIVE_CORNERS, afterE4, false, 0.9);
  assert.equal(model.bestMove?.san, 'e5');

  // The delayed starting-position search resolves after the move was
  // recognized. It must not replace the analysis for the new position.
  const resolveStart = resolveFirstSearch as unknown as (
    result: EngineAnalysisResult,
  ) => void;
  resolveStart({
    fen: STARTING_FEN,
    bestMove: { from: 'd2', to: 'd4', san: 'd4' },
    score: 40,
    perspective: 'side-to-move',
    terminal: 'none',
    pv: ['d2d4'],
    positionsEvaluated: 100,
    depth: 6,
    engine: 'chal-2.0.0',
    nodes: 100,
  });
  model = await coordinator.processFrame(PERSPECTIVE_CORNERS, afterE4, false, 0.9);
  assert.equal(model.bestMove?.san, 'e5');
  assert.notEqual(model.bestMove?.san, 'd4');
});
