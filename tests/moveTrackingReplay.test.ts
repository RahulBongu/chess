/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { ChessGame, STARTING_FEN, type ChessSquare } from '../src/chess/ChessGame.ts';
import {
  PIECE_CLASS_INDEX,
  PIECE_CLASSES,
  observationsFromSquareProbabilities,
  verifyStartingPosition,
  type PieceClass,
} from '../src/vision/pieceRecognition.ts';
import { MoveRecognizer, type SquareObservation } from '../src/vision/moveRecognizer.ts';
import { algebraicToColRow, type BoardOrientation } from '../src/vision/perspectiveOverlay.ts';
import { LiveGuidanceCoordinator } from '../src/vision/liveGuidance.ts';

function pieceClassAt(fen: string, square: ChessSquare): PieceClass {
  const placement = fen.split(' ')[0];
  const ranks = placement.split('/');
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]);
  const row = 8 - rank;
  const rankStr = ranks[row];
  if (!rankStr) return 'empty';
  let curFile = 0;
  for (const char of rankStr) {
    if (char >= '1' && char <= '8') {
      const run = Number(char);
      if (file >= curFile && file < curFile + run) return 'empty';
      curFile += run;
    } else {
      if (curFile === file) {
        const color = char === char.toUpperCase() ? 'white' : 'black';
        const typeMap: Record<string, string> = {
          p: 'pawn',
          n: 'knight',
          b: 'bishop',
          r: 'rook',
          q: 'queen',
          k: 'king',
        };
        return `${color}_${typeMap[char.toLowerCase()]}` as PieceClass;
      }
      curFile += 1;
    }
  }
  return 'empty';
}

function probabilitiesForFen(
  fen: string,
  orientation: BoardOrientation = 'white_bottom',
  confidence = 0.95,
): number[][] {
  const remainder = (1 - confidence) / (PIECE_CLASSES.length - 1);
  const probabilities = Array.from({ length: 64 }, () =>
    new Array<number>(PIECE_CLASSES.length).fill(remainder),
  );
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  for (const file of files) {
    for (let rank = 1; rank <= 8; rank += 1) {
      const square = `${file}${rank}` as ChessSquare;
      const { col, row } = algebraicToColRow(square, orientation);
      probabilities[row * 8 + col][PIECE_CLASS_INDEX[pieceClassAt(fen, square)]] = confidence;
    }
  }
  return probabilities;
}

function observationsForFen(
  fen: string,
  orientation: BoardOrientation = 'white_bottom',
  confidence = 0.95,
): SquareObservation[] {
  return observationsFromSquareProbabilities(
    probabilitiesForFen(fen, orientation, confidence),
    orientation,
  );
}

function feedFrames(
  recognizer: MoveRecognizer,
  observations: SquareObservation[],
  count = 3,
): ReturnType<MoveRecognizer['processFrame']> {
  let last = recognizer.processFrame(observations);
  for (let i = 1; i < count; i += 1) {
    last = recognizer.processFrame(observations);
  }
  return last;
}

test('verifyStartingPosition confirms standard starting position', () => {
  const obs = observationsForFen(STARTING_FEN);
  const res = verifyStartingPosition(obs);
  assert.equal(res.ok, true, res.reason ?? '');
  assert.equal(res.fen, STARTING_FEN);
  assert.equal(res.whiteHomeCount, 16);
  assert.equal(res.blackHomeCount, 16);
  assert.equal(res.centerOccupiedCount, 0);
  assert.equal(res.matchingSquares, 64);
});

test('verifyStartingPosition rejects midgame with pawns in center', () => {
  const midgameFen = 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3';
  const obs = observationsForFen(midgameFen);
  const res = verifyStartingPosition(obs);
  assert.equal(res.ok, false);
  assert.match(res.reason ?? '', /center|not a starting/i);
});

test('verifyStartingPosition rejects endgame with missing home pieces', () => {
  const endgameFen = '8/5pk1/4k3/8/8/6P1/7P/8 w - - 0 1';
  const obs = observationsForFen(endgameFen);
  const res = verifyStartingPosition(obs);
  assert.equal(res.ok, false);
  assert.match(res.reason ?? '', /pieces|setup|ranks/i);
});

test('verifyStartingPosition rejects completely empty board', () => {
  const emptyFen = '8/8/8/8/8/8/8/8 w - - 0 1';
  const obs = observationsForFen(emptyFen);
  const res = verifyStartingPosition(obs);
  assert.equal(res.ok, false);
});

test('verifyStartingPosition warns on inverted orientation (Black at bottom)', () => {
  // If Black is physically at the camera bottom but orientation toggle is set to white_bottom,
  // black pieces appear on ranks 1-2 and white pieces appear on ranks 7-8:
  const invertedBoardFen = 'RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w - - 0 1';
  const obs = observationsForFen(invertedBoardFen, 'white_bottom');
  const res = verifyStartingPosition(obs, 'white_bottom');
  assert.equal(res.ok, false);
  assert.match(res.reason ?? '', /Inverted|toggle|ranks/i);
});

test('continuous move tracking from known starting position (Ruy Lopez sequence)', () => {
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  assert.equal(recognizer.adoptPosition(STARTING_FEN), true);
  assert.equal(recognizer.getFen(), STARTING_FEN);

  // Move 1: 1. e4
  const fen1 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
  const res1 = feedFrames(recognizer, observationsForFen(fen1));
  assert.equal(res1.status, 'move_detected');
  assert.equal(res1.move?.san, 'e4');
  assert.equal(res1.turn, 'b');

  // Move 2: 1... e5
  const fen2 = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
  const res2 = feedFrames(recognizer, observationsForFen(fen2));
  assert.equal(res2.status, 'move_detected');
  assert.equal(res2.move?.san, 'e5');
  assert.equal(res2.turn, 'w');

  // Move 3: 2. Nf3
  const fen3 = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2';
  const res3 = feedFrames(recognizer, observationsForFen(fen3));
  assert.equal(res3.status, 'move_detected');
  assert.equal(res3.move?.san, 'Nf3');
  assert.equal(res3.turn, 'b');

  // Move 4: 2... Nc6
  const fen4 = 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3';
  const res4 = feedFrames(recognizer, observationsForFen(fen4));
  assert.equal(res4.status, 'move_detected');
  assert.equal(res4.move?.san, 'Nc6');
  assert.equal(res4.turn, 'w');

  // Move 5: 3. Bb5
  const fen5 = 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3';
  const res5 = feedFrames(recognizer, observationsForFen(fen5));
  assert.equal(res5.status, 'move_detected');
  assert.equal(res5.move?.san, 'Bb5');
  assert.equal(res5.turn, 'b');

  // Verify game history length and legality
  const history = recognizer.getGame().history();
  assert.deepEqual(history, ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5']);
  assert.equal(recognizer.getFen(), fen5);
});

test('continuous move tracking from arbitrary adopted midgame position', () => {
  // Start from Sicilian Defense: 1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6
  const sicilianNajdorf = 'rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6';
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  assert.equal(recognizer.adoptPosition(sicilianNajdorf), true);
  assert.equal(recognizer.getFen(), sicilianNajdorf);

  // Next move: 6. Be3
  const fenAfterBe3 = 'rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQkq - 1 6';
  const res1 = feedFrames(recognizer, observationsForFen(fenAfterBe3));
  assert.equal(res1.status, 'move_detected');
  assert.equal(res1.move?.san, 'Be3');
  assert.equal(res1.turn, 'b');

  // Next move: 6... e5 (attacking d4 knight)
  const fenAfterE5 = 'rnbqkb1r/1p3ppp/p2p1n2/4p3/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 7';
  const res2 = feedFrames(recognizer, observationsForFen(fenAfterE5));
  assert.equal(res2.status, 'move_detected');
  assert.equal(res2.move?.san, 'e5');
  assert.equal(res2.turn, 'w');
});

test('tracking loss and recovery preserves current game position', () => {
  const recognizer = new MoveRecognizer();
  recognizer.adoptPosition(STARTING_FEN);

  // Tracking lost / occluded with hand over board
  const occludedResult = recognizer.processFrame(observationsForFen(STARTING_FEN), true);
  assert.equal(occludedResult.status, 'occluded');
  assert.equal(occludedResult.move, null);
  assert.equal(recognizer.getFen(), STARTING_FEN);

  // Tracking recovered with same board
  const recoveredResult = feedFrames(recognizer, observationsForFen(STARTING_FEN));
  assert.equal(recoveredResult.status, 'watching');
  assert.equal(recoveredResult.move, null);
  assert.equal(recognizer.getFen(), STARTING_FEN);
});
