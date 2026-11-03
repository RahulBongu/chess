/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { ChessGame, type ChessSquare } from '../src/chess/ChessGame.ts';
import {
  averageSquareObservations,
  PIECE_CLASS_INDEX,
  PIECE_CLASSES,
  observationsFromSquareProbabilities,
  reconcileLegalMove,
  recognizeBothOrientations,
  recognizePositionFromObservations,
  rotateSquareProbabilities,
  softmaxSquareScores,
  suggestPositionFromObservations,
  type PieceClass,
} from '../src/vision/pieceRecognition.ts';
import { MoveRecognizer, type SquareObservation } from '../src/vision/moveRecognizer.ts';
import { algebraicToColRow, type BoardOrientation } from '../src/vision/perspectiveOverlay.ts';
import { STARTING_FEN } from '../src/chess/ChessGame.ts';

function pieceClassAt(fen: string, square: ChessSquare): PieceClass {
  const game = new ChessGame(fen);
  const piece = game.pieceAt(square);
  if (!piece) return 'empty';
  return `${piece.color === 'w' ? 'white' : 'black'}_${piece.type === 'n' ? 'knight' : piece.type === 'b' ? 'bishop' : piece.type === 'r' ? 'rook' : piece.type === 'q' ? 'queen' : piece.type === 'k' ? 'king' : 'pawn'}` as PieceClass;
}

function probabilitiesForFen(
  fen: string,
  orientation: BoardOrientation = 'white_bottom',
  confidence = 0.94,
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
  confidence = 0.94,
): SquareObservation[] {
  return observationsFromSquareProbabilities(
    probabilitiesForFen(fen, orientation, confidence),
    orientation,
  );
}

function feedStableFrames(
  recognizer: MoveRecognizer,
  observations: SquareObservation[],
  frames = 3,
): ReturnType<MoveRecognizer['processFrame']> {
  let last = recognizer.processFrame(observations);
  for (let index = 1; index < frames; index += 1) {
    last = recognizer.processFrame(observations);
  }
  return last;
}

test('a FEN-seeded baseline treats the next matching frame as steady, not a move', () => {
  const recognizer = new MoveRecognizer();
  assert.equal(recognizer.adoptPosition(STARTING_FEN), true);
  assert.equal(recognizer.isPositionConfirmed(), true);

  const observations = observationsForFen(STARTING_FEN);
  const frame = recognizer.processFrame(observations, false);
  assert.equal(frame.move, null, 'no phantom move from baseline noise');
  assert.equal(frame.status, 'watching');
  assert.equal(frame.fen, STARTING_FEN);
});

test('suggestion offers a plausible board without claiming adoption', () => {
  const probabilities = probabilitiesForFen(STARTING_FEN);
  // Every occupied square keeps the correct top-1 class but loses the
  // confidence margin the strict gate requires.
  for (const entry of probabilities) {
    const top = entry.indexOf(Math.max(...entry));
    const rest = (1 - 0.2) / (PIECE_CLASSES.length - 1);
    entry.fill(rest);
    entry[top] = 0.2;
  }
  const observations = observationsFromSquareProbabilities(probabilities, 'white_bottom');
  const strict = recognizePositionFromObservations(observations);
  assert.equal(strict.ok, false, 'low-confidence evidence must not be auto-adopted');

  const suggestion = suggestPositionFromObservations(observations);
  assert.equal(suggestion.ok, true, suggestion.reason ?? '');
  assert.ok(
    suggestion.fen.startsWith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w'),
    suggestion.fen,
  );
});

test('sensor rotation is undone before grid rows are mapped to ranks', () => {
  const upright = probabilitiesForFen(STARTING_FEN);

  // The clockwise rotation moves the old bottom-left to the top-left.
  const clockwise = rotateSquareProbabilities(upright, 90);
  assert.ok(clockwise);
  assert.deepEqual(clockwise![0], upright[7 * 8]);
  assert.deepEqual(clockwise![7], upright[0]);
  assert.deepEqual(clockwise![7 * 8 + 7], upright[7]);
  assert.deepEqual(clockwise![7 * 8], upright[7 * 8 + 7]);

  // A 90-degree sensor image must be rotated back before rank mapping.
  for (const sensorRotation of [90, 180, 270]) {
    const sensorView = rotateSquareProbabilities(upright, sensorRotation);
    assert.ok(sensorView);
    const restored = rotateSquareProbabilities(sensorView!, (360 - sensorRotation) % 360);
    assert.ok(restored);
    const position = recognizePositionFromObservations(
      observationsFromSquareProbabilities(restored!, 'white_bottom'),
    );
    assert.equal(position.ok, true, `rotation ${sensorRotation}: ${position.reason ?? ''}`);
    assert.ok(position.fen.startsWith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w'));
  }

  assert.equal(rotateSquareProbabilities(upright, 45), null);
});

test('averaging cancels a one-frame square misread before adoption', () => {
  const start = observationsForFen(STARTING_FEN);
  const probabilities = probabilitiesForFen(STARTING_FEN);
  const { col, row } = algebraicToColRow('e4');
  const index = row * 8 + col;
  probabilities[index].fill(0.001);
  probabilities[index][PIECE_CLASS_INDEX.white_pawn] = 0.9;
  const misread = observationsFromSquareProbabilities(probabilities, 'white_bottom');

  const averaged = averageSquareObservations([start, start, start, misread]);
  assert.ok(averaged, 'four complete frames should average');
  const e4 = averaged!.find((observation) => observation.square === 'e4');
  assert.ok(e4);
  assert.equal(e4!.occupied, false, 'three empty frames must outweigh one misread pawn');

  const position = recognizePositionFromObservations(averaged!);
  assert.equal(position.ok, true, position.reason ?? '');
  assert.equal(position.occupied, 32);
  assert.ok(position.fen.startsWith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w'));
});

test('adopts a legal starting position from 13-class probabilities', () => {
  const observations = observationsForFen(STARTING_FEN);
  const position = recognizePositionFromObservations(observations);

  assert.equal(position.ok, true, position.reason ?? '');
  assert.equal(position.occupied, 32);
  assert.ok(position.averageConfidence > 0.9);
  assert.ok(position.fen.startsWith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w'));
});

test('rejects a board with no kings instead of accepting a screen or texture', () => {
  const probabilities = Array.from({ length: 64 }, () => {
    const vector = new Array<number>(PIECE_CLASSES.length).fill(0.005);
    vector[PIECE_CLASS_INDEX.empty] = 0.94;
    return vector;
  });
  const observations = observationsFromSquareProbabilities(probabilities, 'white_bottom');
  const position = recognizePositionFromObservations(observations);
  assert.equal(position.ok, false);
  assert.match(position.reason ?? '', /king|Implausible/i);
});

test('rejects an illegal pawn on the back rank', () => {
  const base = '7k/8/8/8/8/8/8/4K3 w - - 0 1';
  const probabilities = probabilitiesForFen(base);
  const e8 = algebraicToColRow('e8');
  const vector = probabilities[e8.row * 8 + e8.col];
  vector.fill(0.005);
  vector[PIECE_CLASS_INDEX.white_pawn] = 0.94;
  const observations = observationsFromSquareProbabilities(probabilities, 'white_bottom');

  const position = recognizePositionFromObservations(observations);
  assert.equal(position.ok, false);
  assert.match(position.reason ?? '', /pawn/i);
});

test('rejects more than sixteen pieces of one colour', () => {
  // Nine white pawns plus two kings: structurally impossible.
  const fen = '4k3/8/8/8/8/PPPPPPPP/PPPPPPPP/4K3 w - - 0 1';
  const observations = observationsForFen(fen);
  const position = recognizePositionFromObservations(observations);
  assert.equal(position.ok, false);
});

test('rejects a position with too many low-confidence squares', () => {
  const observations = observationsForFen(STARTING_FEN, 'white_bottom', 0.3);
  const position = recognizePositionFromObservations(observations);
  assert.equal(position.ok, false);
  assert.match(position.reason ?? '', /uncertain|confiden/i);
});

test('orientation stays the caller’s decision because rotation preserves legality', () => {
  const whiteBottom = probabilitiesForFen(STARTING_FEN, 'white_bottom');
  const fromWhite = recognizeBothOrientations(whiteBottom, 'w', 'white_bottom');
  assert.equal(fromWhite[0].orientation, 'white_bottom');
  assert.equal(fromWhite[0].position.ok, true);
  // A still image cannot tell which side is at the bottom: the 180-degree
  // rotation is also a structurally legal position.
  assert.equal(fromWhite[1].position.ok, true);

  const blackBottom = probabilitiesForFen(STARTING_FEN, 'black_bottom');
  const fromBlack = recognizeBothOrientations(blackBottom, 'w', 'black_bottom');
  assert.equal(fromBlack[0].orientation, 'black_bottom');
  assert.equal(fromBlack[0].position.ok, true);
});

test('softmax turns raw logits into normalised per-square probabilities', () => {
  const scores = new Array<number>(64 * PIECE_CLASSES.length).fill(-8);
  scores[PIECE_CLASS_INDEX.white_pawn] = 8;
  const probabilities = softmaxSquareScores(scores);
  assert.ok(probabilities);
  assert.equal(probabilities!.length, 64);
  const sum = probabilities![0].reduce((total, value) => total + value, 0);
  assert.ok(Math.abs(sum - 1) < 1e-6);
  assert.ok(probabilities![0][PIECE_CLASS_INDEX.white_pawn] > 0.99);
});

test('recognizer adopts a classified position and detects e4 without luminance', () => {
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  const start = observationsForFen(STARTING_FEN);
  assert.equal(recognizer.adoptPosition(STARTING_FEN, start), true);

  const afterE4 = observationsForFen(
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
  );
  const result = feedStableFrames(recognizer, afterE4);

  assert.equal(result.status, 'move_detected');
  assert.equal(result.move?.from, 'e2');
  assert.equal(result.move?.to, 'e4');
  assert.equal(result.move?.san, 'e4');
  assert.equal(result.turn, 'b');

  const duplicate = recognizer.processFrame(afterE4);
  assert.equal(duplicate.status, 'watching');
  assert.equal(duplicate.move, null);
});

test('recognizer detects a capture from destination class change', () => {
  const initial = 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  assert.equal(recognizer.adoptPosition(initial, observationsForFen(initial)), true);

  const captured = observationsForFen(
    'rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3',
  );
  const result = feedStableFrames(recognizer, captured);

  assert.equal(result.status, 'move_detected');
  assert.equal(result.move?.san, 'exd5');
  assert.equal(recognizer.getGame().pieceAt('d5')?.color, 'w');
});

test('recognizer detects castling as a four-square change', () => {
  const initial = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1';
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  assert.equal(recognizer.adoptPosition(initial, observationsForFen(initial)), true);

  const castled = observationsForFen(
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 b kq - 1 1',
  );
  const result = feedStableFrames(recognizer, castled);

  assert.equal(result.status, 'move_detected');
  assert.equal(result.move?.san, 'O-O');
});

test('recognizer reconstructs en passant when the adopted FEN has no ep field', () => {
  // A still image cannot supply en-passant rights, so adoption normalizes the
  // ep field to '-'. The observed board must still reconcile exd6.
  const initial = 'rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3';
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  assert.equal(recognizer.adoptPosition(initial, observationsForFen(initial)), true);

  const enPassant = observationsForFen(
    'rnbqkbnr/ppp1pppp/3P4/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 4',
  );
  const result = feedStableFrames(recognizer, enPassant);

  assert.equal(result.status, 'move_detected', result.message);
  assert.equal(result.move?.san, 'exd6');
  assert.equal(recognizer.getGame().pieceAt('d5'), undefined);
});

test('promotion requires explicit confirmation and never auto-queens', () => {
  const initial = '3k4/4P3/8/8/8/8/8/4K3 w - - 0 1';
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  assert.equal(recognizer.adoptPosition(initial, observationsForFen(initial)), true);

  const promoted = observationsForFen('3kQ3/8/8/8/8/8/8/4K3 b - - 0 1');
  const result = feedStableFrames(recognizer, promoted);

  assert.equal(result.status, 'uncertain');
  assert.equal(result.move, null);
  assert.match(result.message, /promotion/i);
  assert.equal(recognizer.getPendingPromotion()?.from, 'e7');
  assert.equal(recognizer.getPendingPromotion()?.to, 'e8');
  assert.ok(recognizer.confirmPromotion('q'));
  assert.equal(recognizer.getLastMove()?.san, 'e8=Q+');
});

test('rejects a capture when the source square is still occupied', () => {
  const initial = 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  assert.equal(recognizer.adoptPosition(initial, observationsForFen(initial)), true);

  // d5 changes colour, but e4 is still occupied: no legal move explains it.
  const probabilities = probabilitiesForFen(
    'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
  );
  const d5 = algebraicToColRow('d5');
  const before = probabilities[d5.row * 8 + d5.col][PIECE_CLASS_INDEX.black_pawn];
  probabilities[d5.row * 8 + d5.col][PIECE_CLASS_INDEX.black_pawn] = 0.02;
  probabilities[d5.row * 8 + d5.col][PIECE_CLASS_INDEX.white_pawn] = Math.max(0.9, before);
  const observations = observationsFromSquareProbabilities(probabilities, 'white_bottom');

  const result = feedStableFrames(recognizer, observations);
  assert.equal(result.move, null);
  assert.notEqual(result.status, 'move_detected');
  assert.ok(result.fen.includes('3p4'));
});

test('reports ambiguity instead of inventing a move', () => {
  const fen = '4k3/8/8/8/8/2N3N1/8/K7 w - - 0 1';
  const probabilities = probabilitiesForFen(fen);
  // Remove both knights from c3 and g3 and place one white knight on e4.
  for (const square of ['c3', 'g3'] as ChessSquare[]) {
    const { col, row } = algebraicToColRow(square);
    const vector = probabilities[row * 8 + col];
    vector.fill(0.005);
    vector[PIECE_CLASS_INDEX.empty] = 0.94;
  }
  const e4 = algebraicToColRow('e4');
  const destination = probabilities[e4.row * 8 + e4.col];
  destination.fill(0.005);
  destination[PIECE_CLASS_INDEX.white_knight] = 0.94;

  const observations = observationsFromSquareProbabilities(probabilities, 'white_bottom');
  const game = new ChessGame(fen);
  const reconciliation = reconcileLegalMove(game, observations);

  assert.equal(reconciliation.valid, false);
  assert.match(reconciliation.reason ?? '', /Ambiguous/i);
  assert.ok(reconciliation.best !== null);
  assert.ok(reconciliation.second !== null);
});
