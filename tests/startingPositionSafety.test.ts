/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { ChessGame, STARTING_FEN, type ChessSquare } from '../src/chess/ChessGame.ts';
import type { ChessEngine, EngineAnalysisResult } from '../src/chess/engine.ts';
import type { Point2D } from '../src/vision/perspectiveOverlay.ts';
import {
  algebraicToColRow,
  computeSquarePerspectiveCenter,
} from '../src/vision/perspectiveOverlay.ts';
import type { SquareObservation } from '../src/vision/moveRecognizer.ts';
import {
  PIECE_CLASS_INDEX,
  PIECE_CLASSES,
  observationsFromSquareProbabilities,
  verifyStartingPosition,
  type PieceClass,
} from '../src/vision/pieceRecognition.ts';
import { LiveGuidanceCoordinator } from '../src/vision/liveGuidance.ts';

const PERSPECTIVE_CORNERS: Point2D[] = [
  { x: 100, y: 50 }, // top-left
  { x: 300, y: 50 }, // top-right
  { x: 380, y: 350 }, // bottom-right
  { x: 20, y: 350 }, // bottom-left
];

const PIECE_NAMES: Record<string, string> = {
  p: 'pawn',
  n: 'knight',
  b: 'bishop',
  r: 'rook',
  q: 'queen',
  k: 'king',
};

function pieceClassAt(fen: string, square: ChessSquare): PieceClass {
  const placement = fen.split(' ')[0];
  const ranks = placement.split('/');
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]);
  const rankStr = ranks[8 - rank];
  if (!rankStr) return 'empty';

  let currentFile = 0;
  for (const char of rankStr) {
    if (char >= '1' && char <= '8') {
      const run = Number(char);
      if (file >= currentFile && file < currentFile + run) return 'empty';
      currentFile += run;
    } else {
      if (currentFile === file) {
        const color = char === char.toUpperCase() ? 'white' : 'black';
        return `${color}_${PIECE_NAMES[char.toLowerCase()]}` as PieceClass;
      }
      currentFile += 1;
    }
  }
  return 'empty';
}

function observationsForFen(fen: string, confidence = 0.97): SquareObservation[] {
  const remainder = (1 - confidence) / (PIECE_CLASSES.length - 1);
  const probabilities = Array.from({ length: 64 }, () =>
    new Array<number>(PIECE_CLASSES.length).fill(remainder),
  );
  for (let file = 0; file < 8; file += 1) {
    for (let rank = 1; rank <= 8; rank += 1) {
      const square = `${'abcdefgh'[file]}${rank}` as ChessSquare;
      const { col, row } = algebraicToColRow(square, 'white_bottom');
      probabilities[row * 8 + col][PIECE_CLASS_INDEX[pieceClassAt(fen, square)]] = confidence;
    }
  }
  return observationsFromSquareProbabilities(probabilities, 'white_bottom');
}

function fenAfter(moves: Array<[ChessSquare, ChessSquare]>): string {
  const game = new ChessGame();
  for (const [from, to] of moves) {
    const applied = game.tryMove({ from, to });
    assert.ok(applied, `opening replay rejected ${from}-${to}`);
  }
  return game.fen();
}

class MockEngine implements ChessEngine {
  async analyze(fen: string): Promise<EngineAnalysisResult> {
    return {
      fen,
      bestMove: { from: 'e2', to: 'e4', san: 'e4' },
      score: 20,
      perspective: 'side-to-move',
      terminal: 'none',
      pv: ['e2e4'],
      positionsEvaluated: 10,
      depth: 4,
      engine: 'chal-2.0.0',
      nodes: 10,
    };
  }

  stop(): void {}
}

function standardCoordinator(): LiveGuidanceCoordinator {
  return new LiveGuidanceCoordinator({
    engine: new MockEngine(),
    orientation: 'white_bottom',
    initialTurn: 'w',
    requirePositionEvidence: true,
    initializationMode: 'standard_start',
    adoptionWindow: 1,
    adoptionConfirmations: 1,
  });
}

test('accepts the exact untouched standard starting layout', () => {
  const result = verifyStartingPosition(observationsForFen(STARTING_FEN));
  assert.equal(result.ok, true, result.reason ?? '');
  assert.equal(result.fen, STARTING_FEN);
  assert.equal(result.matchingSquares, 64);
  assert.equal(result.whiteHomeCount, 16);
  assert.equal(result.blackHomeCount, 16);
  assert.equal(result.centerOccupiedCount, 0);
  assert.deepEqual(result.occupiedCenterSquares, []);
  assert.equal(result.totalOccupied, 32);
});

test('rejects every legal position immediately after one opening move', () => {
  const openingMoves = new ChessGame().allLegalMoves();
  assert.equal(openingMoves.length, 20, 'expected all 20 legal opening moves');

  for (const move of openingMoves) {
    const fen = fenAfter([[move.from as ChessSquare, move.to as ChessSquare]]);
    const result = verifyStartingPosition(observationsForFen(fen));
    assert.equal(result.ok, false, `accepted near-start position after ${move.san}`);
    assert.ok(result.reason, `no rejection reason after ${move.san}`);
    assert.ok(result.matchingSquares < 64, `template still complete after ${move.san}`);
  }
});

test('rejects representative two-ply near-start positions', () => {
  const lines: Array<Array<[ChessSquare, ChessSquare]>> = [
    [
      ['e2', 'e4'],
      ['e7', 'e5'],
    ],
    [
      ['e2', 'e4'],
      ['c7', 'c5'],
    ],
    [
      ['d2', 'd4'],
      ['d7', 'd5'],
    ],
    [
      ['g1', 'f3'],
      ['g8', 'f6'],
    ],
    [
      ['e2', 'e4'],
      ['d7', 'd5'],
    ],
  ];

  for (const line of lines) {
    const result = verifyStartingPosition(observationsForFen(fenAfter(line)));
    assert.equal(result.ok, false, `accepted near-start line ${line.map((m) => m.join('')).join(' ')}`);
    assert.ok(result.reason);
  }
});

test('rejects a pawn advance, a knight development, and a capture', () => {
  const cases: Array<[string, Array<[ChessSquare, ChessSquare]>]> = [
    ['e4 pawn advance', [['e2', 'e4']]],
    ['Nf3 knight development', [['g1', 'f3']]],
    [
      'capture after 1.e4 d5 2.exd5',
      [
        ['e2', 'e4'],
        ['d7', 'd5'],
        ['e4', 'd5'],
      ],
    ],
  ];

  for (const [label, line] of cases) {
    const result = verifyStartingPosition(observationsForFen(fenAfter(line)));
    assert.equal(result.ok, false, `accepted ${label}`);
    assert.ok(result.reason);
  }
});

test('rejects a board with a missing home-rank piece', () => {
  const missingRook = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w KQkq - 0 1';
  const result = verifyStartingPosition(observationsForFen(missingRook));
  assert.equal(result.ok, false);
  assert.equal(result.whiteHomeCount, 15);
  assert.ok(result.missingWhiteHomeSquares.includes('a1'));
  assert.match(result.reason ?? '', /White home square/i);
});

test('rejects swapped colours on a home rank', () => {
  const swappedKnight = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNn w KQkq - 0 1';
  const result = verifyStartingPosition(observationsForFen(swappedKnight));
  assert.equal(result.ok, false);
  assert.equal(result.wrongColorHomeErrors, 1);
  assert.match(result.reason ?? '', /Inverted piece color/i);
});

test('rejects an inverted board when White-bottom orientation is assumed', () => {
  // Physically inverted: White home ranks appear at the top of the grid and
  // Black home ranks at the bottom, which the verifier must not accept.
  const inverted = 'RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w - - 0 1';
  const result = verifyStartingPosition(observationsForFen(inverted));
  assert.equal(result.ok, false);
  assert.ok(result.wrongColorHomeErrors > 0);
});

test('standard-start layout is never auto-adopted, even when it verifies', async () => {
  const coordinator = standardCoordinator();
  const model = await coordinator.processFrame(
    PERSPECTIVE_CORNERS,
    observationsForFen(STARTING_FEN),
    false,
    0.9,
  );

  assert.equal(model.trackingState, 'tracking');
  assert.equal(model.positionConfirmed, false);
  assert.equal(model.startReady, true);
  assert.equal(model.bestMove, null);
  assert.equal(coordinator.isStartVerified(), true);

  // Only the explicit confirmation API may adopt the standard FEN.
  assert.equal(coordinator.confirmStartingPosition(), true);
  assert.equal(coordinator.isPositionConfirmed(), true);
  assert.equal(coordinator.isStartVerified(), false);
});

test('confirmation is refused when the current frame is not a verified start', async () => {
  const coordinator = standardCoordinator();
  const afterE4 = fenAfter([['e2', 'e4']]);
  const model = await coordinator.processFrame(
    PERSPECTIVE_CORNERS,
    observationsForFen(afterE4),
    false,
    0.9,
  );

  assert.equal(model.positionConfirmed, false);
  assert.equal(model.startReady, false);
  assert.equal(coordinator.confirmStartingPosition(), false);
  assert.equal(coordinator.isPositionConfirmed(), false);
});

test('a near-start position is never silently replaced by the starting FEN', async () => {
  const coordinator = standardCoordinator();
  const afterNf3 = fenAfter([['g1', 'f3']]);
  const model = await coordinator.processFrame(
    PERSPECTIVE_CORNERS,
    observationsForFen(afterNf3),
    false,
    0.9,
  );

  assert.equal(model.positionConfirmed, false);
  assert.equal(model.startReady, false);
  assert.match(model.recognition.reason ?? '', /not an untouched starting position|White home/i);
  assert.equal(coordinator.confirmStartingPosition(), false);
});

test('correct occupancy with swapped identities requires explicit confirmation', async () => {
  // The colour/occupancy verifier cannot distinguish a White rook from a
  // White knight on a home square. That class of error must therefore never
  // auto-adopt: the layout may verify, but the standard FEN is only adopted
  // through the explicit user-confirmation API.
  const swappedIdentity = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/NNBQKBNR w KQkq - 0 1';
  const result = verifyStartingPosition(observationsForFen(swappedIdentity));
  assert.equal(result.ok, true, 'colour/occupancy-only layout check');
  assert.equal(result.matchingSquares, 64);

  const coordinator = standardCoordinator();
  const model = await coordinator.processFrame(
    PERSPECTIVE_CORNERS,
    observationsForFen(swappedIdentity),
    false,
    0.9,
  );
  assert.equal(model.positionConfirmed, false);
  assert.equal(model.startReady, true);
  assert.equal(coordinator.confirmStartingPosition(), true);
});

test('confirmed start tracks one move and aligns the reply overlay', async () => {
  const engine: ChessEngine = {
    async analyze(fen: string): Promise<EngineAnalysisResult> {
      return {
        fen,
        bestMove: { from: 'e7', to: 'e5', san: 'e5' },
        score: -15,
        perspective: 'side-to-move',
        terminal: 'none',
        pv: ['e7e5'],
        positionsEvaluated: 25,
        depth: 5,
        engine: 'chal-2.0.0',
        nodes: 25,
      };
    },
    stop(): void {},
  };
  const coordinator = new LiveGuidanceCoordinator({
    engine,
    orientation: 'white_bottom',
    initialTurn: 'w',
    requirePositionEvidence: true,
    initializationMode: 'standard_start',
    adoptionWindow: 1,
    adoptionConfirmations: 1,
  });

  const startModel = await coordinator.processFrame(
    PERSPECTIVE_CORNERS,
    observationsForFen(STARTING_FEN),
    false,
    0.9,
  );
  assert.equal(startModel.startReady, true);
  assert.equal(coordinator.confirmStartingPosition(), true);

  const afterE4 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
  const afterE4Observations = observationsForFen(afterE4);
  let moveModel;
  for (let frame = 0; frame < 3; frame += 1) {
    moveModel = await coordinator.processFrame(
      PERSPECTIVE_CORNERS,
      afterE4Observations,
      false,
      0.9,
    );
  }
  assert.equal(moveModel!.positionConfirmed, true);
  assert.equal(moveModel!.recognizerStatus, 'move_detected');
  assert.equal(coordinator.getRecognizer().getLastMove()?.san, 'e4');
  assert.equal(moveModel!.fen, afterE4);
  assert.equal(moveModel!.turn, 'b');

  await new Promise((resolve) => setTimeout(resolve, 10));
  const replyModel = await coordinator.processFrame(
    PERSPECTIVE_CORNERS,
    afterE4Observations,
    false,
    0.9,
  );
  assert.equal(replyModel.bestMove?.san, 'e5');
  assert.equal(replyModel.fen, afterE4);
  assert.ok(replyModel.perspectiveGeometry.arrow !== null);
  assert.ok(replyModel.perspectiveGeometry.sourceQuad !== null);
  assert.ok(replyModel.perspectiveGeometry.destinationQuad !== null);

  const expectedFrom = computeSquarePerspectiveCenter(PERSPECTIVE_CORNERS, 'e7')!;
  const expectedTo = computeSquarePerspectiveCenter(PERSPECTIVE_CORNERS, 'e5')!;
  assert.ok(Math.abs(replyModel.perspectiveGeometry.arrow!.from.x - expectedFrom.x) < 1e-6);
  assert.ok(Math.abs(replyModel.perspectiveGeometry.arrow!.from.y - expectedFrom.y) < 1e-6);
  assert.ok(Math.abs(replyModel.perspectiveGeometry.arrow!.to.x - expectedTo.x) < 1e-6);
  assert.ok(Math.abs(replyModel.perspectiveGeometry.arrow!.to.y - expectedTo.y) < 1e-6);
});
