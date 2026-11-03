/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  MoveRecognizer,
  type PieceColor,
  type SquareObservation,
} from '../src/vision/moveRecognizer.ts';
import type { ChessSquare } from '../src/chess/ChessGame.ts';

function createStartingPositionObservations(): SquareObservation[] {
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const obs: SquareObservation[] = [];

  for (const file of files) {
    for (let rank = 1; rank <= 8; rank += 1) {
      const sq = `${file}${rank}` as ChessSquare;
      const occupied = rank === 1 || rank === 2 || rank === 7 || rank === 8;
      let color: PieceColor = null;
      let luminance = 128;

      if (rank === 1 || rank === 2) {
        color = 'w';
        luminance = 180;
      } else if (rank === 7 || rank === 8) {
        color = 'b';
        luminance = 50;
      }

      obs.push({
        square: sq,
        occupied,
        color,
        luminance,
        confidence: 0.95,
      });
    }
  }

  return obs;
}

function cloneObservations(obs: SquareObservation[]): SquareObservation[] {
  return obs.map((o) => ({ ...o }));
}

function simulateMoveOnObservations(
  obs: SquareObservation[],
  from: ChessSquare,
  to: ChessSquare,
  movingColor: 'w' | 'b' = 'w',
): SquareObservation[] {
  const next = cloneObservations(obs);
  const fromItem = next.find((o) => o.square === from);
  const toItem = next.find((o) => o.square === to);

  if (fromItem) {
    fromItem.occupied = false;
    fromItem.color = null;
    fromItem.luminance = 128;
  }
  if (toItem) {
    toItem.occupied = true;
    toItem.color = movingColor;
    toItem.luminance = movingColor === 'w' ? 180 : 50;
  }
  return next;
}

test('MoveRecognizer initializes with standard starting position and baseline', () => {
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  assert.equal(recognizer.getStatus(), 'watching');
  assert.equal(recognizer.getTurn(), 'w');
  assert.equal(recognizer.getLastMove(), null);
  assert.ok(recognizer.getFen().startsWith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'));
});

test('detects pawn move 1. e4 after K stable frames', () => {
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  const startObs = createStartingPositionObservations();
  const movedObs = simulateMoveOnObservations(startObs, 'e2', 'e4', 'w');

  // Frame 1: movement seen, stabilizing
  let res = recognizer.processFrame(movedObs);
  assert.equal(res.status, 'stabilizing');
  assert.equal(res.move, null);
  assert.equal(res.stableFrames, 1);

  // Frame 2: stabilizing
  res = recognizer.processFrame(movedObs);
  assert.equal(res.status, 'stabilizing');
  assert.equal(res.move, null);
  assert.equal(res.stableFrames, 2);

  // Frame 3: reaches requiredStableFrames (3) -> move confirmed!
  res = recognizer.processFrame(movedObs);
  assert.equal(res.status, 'move_detected');
  assert.ok(res.move !== null);
  assert.equal(res.move?.from, 'e2');
  assert.equal(res.move?.to, 'e4');
  assert.equal(res.move?.san, 'e4');
  assert.equal(res.turn, 'b');
});

test('suppresses duplicate move events on subsequent identical stable frames', () => {
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  const startObs = createStartingPositionObservations();
  const movedObs = simulateMoveOnObservations(startObs, 'e2', 'e4', 'w');

  recognizer.processFrame(movedObs);
  recognizer.processFrame(movedObs);
  const moveRes = recognizer.processFrame(movedObs);
  assert.equal(moveRes.status, 'move_detected');

  // Frame 4 (same board position): must NOT re-emit move!
  const dupRes = recognizer.processFrame(movedObs);
  assert.equal(dupRes.status, 'watching');
  assert.equal(dupRes.move, null);
  assert.equal(dupRes.turn, 'b');
});

test('detects capture moves when destination square remains occupied (piece color inversion)', () => {
  // Start from position where White e4 can capture Black d5:
  // e4 pawn and d5 pawn already placed
  const game = new MoveRecognizer({
    requiredStableFrames: 2,
    initialFen: 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2',
  });

  const obs = createStartingPositionObservations();
  // Set up e4 (white pawn) and d5 (black pawn)
  const e4Item = obs.find((o) => o.square === 'e4')!;
  e4Item.occupied = true;
  e4Item.color = 'w';
  e4Item.luminance = 180;
  obs.find((o) => o.square === 'e2')!.occupied = false;

  const d5Item = obs.find((o) => o.square === 'd5')!;
  d5Item.occupied = true;
  d5Item.color = 'b';
  d5Item.luminance = 50;
  obs.find((o) => o.square === 'd7')!.occupied = false;

  game.setBaseline(obs);

  // Now White captures Black pawn on d5: exd5!
  // e4 becomes empty
  // d5 REMAINS OCCUPIED (true), but its piece color flips from 'b' (50) to 'w' (180)!
  const captureObs = cloneObservations(obs);
  const capFrom = captureObs.find((o) => o.square === 'e4')!;
  capFrom.occupied = false;
  capFrom.color = null;
  capFrom.luminance = 128;

  const capTo = captureObs.find((o) => o.square === 'd5')!;
  capTo.occupied = true; // Destination remains occupied!
  capTo.color = 'w';     // Color flipped to White
  capTo.luminance = 180; // Luminance shifted from 50 to 180

  game.processFrame(captureObs);
  const result = game.processFrame(captureObs);

  assert.equal(result.status, 'move_detected');
  assert.ok(result.move !== null);
  assert.equal(result.move?.from, 'e4');
  assert.equal(result.move?.to, 'd5');
  assert.equal(result.move?.san, 'exd5');
  assert.equal(result.turn, 'b');
});

test('detects castling (O-O) with King and Rook dual-square shifts', () => {
  // Position ready for White kingside castling: e1 King, h1 Rook, f1/g1 empty
  const recognizer = new MoveRecognizer({
    requiredStableFrames: 2,
    initialFen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1',
  });

  const obs = createStartingPositionObservations();
  obs.find((o) => o.square === 'f1')!.occupied = false;
  obs.find((o) => o.square === 'g1')!.occupied = false;
  recognizer.setBaseline(obs);

  // White castles: King moves e1 -> g1, Rook moves h1 -> f1
  // Emptied: e1, h1. Filled: g1, f1.
  const castleObs = cloneObservations(obs);
  castleObs.find((o) => o.square === 'e1')!.occupied = false;
  castleObs.find((o) => o.square === 'h1')!.occupied = false;
  castleObs.find((o) => o.square === 'g1')!.occupied = true;
  castleObs.find((o) => o.square === 'g1')!.color = 'w';
  castleObs.find((o) => o.square === 'f1')!.occupied = true;
  castleObs.find((o) => o.square === 'f1')!.color = 'w';

  // Frame 1 detects movement/fluctuations
  recognizer.processFrame(castleObs);
  // Frame 2 stabilizing (1/2)
  recognizer.processFrame(castleObs);
  // Frame 3 stable (2/2) -> confirmed
  const res = recognizer.processFrame(castleObs);

  assert.equal(res.status, 'move_detected');
  assert.equal(res.move?.san, 'O-O');
  assert.equal(res.move?.from, 'e1');
  assert.equal(res.move?.to, 'g1');
});

test('detects en passant captures correctly', () => {
  // White pawn on e5, Black pawn just moved d7-d5: e.p. square is d6
  const recognizer = new MoveRecognizer({
    requiredStableFrames: 2,
    initialFen: 'rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3',
  });

  const obs = createStartingPositionObservations();
  obs.find((o) => o.square === 'e2')!.occupied = false;
  const e5 = obs.find((o) => o.square === 'e5')!;
  e5.occupied = true;
  e5.color = 'w';

  obs.find((o) => o.square === 'd7')!.occupied = false;
  const d5 = obs.find((o) => o.square === 'd5')!;
  d5.occupied = true;
  d5.color = 'b';

  recognizer.setBaseline(obs);

  // White plays exd6 e.p.:
  // e5 is emptied
  // d6 is filled (with white pawn)
  // d5 is emptied (captured black pawn removed)
  const epObs = cloneObservations(obs);
  epObs.find((o) => o.square === 'e5')!.occupied = false;
  epObs.find((o) => o.square === 'd5')!.occupied = false;
  const d6 = epObs.find((o) => o.square === 'd6')!;
  d6.occupied = true;
  d6.color = 'w';

  // Frame 1: movement detected (3 squares changing)
  recognizer.processFrame(epObs);
  // Frame 2: stabilizing (1/2)
  recognizer.processFrame(epObs);
  // Frame 3: stable (2/2) -> confirmed
  const res = recognizer.processFrame(epObs);

  assert.equal(res.status, 'move_detected');
  assert.equal(res.move?.san, 'exd6');
  assert.equal(res.move?.from, 'e5');
  assert.equal(res.move?.to, 'd6');
});

test('handles pawn promotion requiring explicit piece confirmation (no automatic Queen promotion)', () => {
  // White pawn on e7, Black King on d8, e8 empty for promotion
  const recognizer = new MoveRecognizer({
    requiredStableFrames: 2,
    initialFen: '3k4/4P3/8/8/8/8/8/4K3 w - - 0 1',
  });

  const obs: SquareObservation[] = [];
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  for (const f of files) {
    for (let r = 1; r <= 8; r += 1) {
      const sq = `${f}${r}` as ChessSquare;
      const occupied = sq === 'e1' || sq === 'd8' || sq === 'e7';
      obs.push({
        square: sq,
        occupied,
        color: sq === 'd8' ? 'b' : 'w',
        confidence: 0.95,
      });
    }
  }

  recognizer.setBaseline(obs);

  // Move e7 -> e8 (promotion!)
  const promoObs = cloneObservations(obs);
  promoObs.find((o) => o.square === 'e7')!.occupied = false;
  const e8 = promoObs.find((o) => o.square === 'e8')!;
  e8.occupied = true;
  e8.color = 'w';

  recognizer.processFrame(promoObs);
  const res = recognizer.processFrame(promoObs);

  // Automatic Queen promotion must NOT happen silently!
  assert.equal(res.status, 'uncertain');
  assert.equal(res.move, null);
  assert.ok(res.message.includes('promotion detected'));
  assert.ok(recognizer.getPendingPromotion() !== null);
  assert.equal(recognizer.getPendingPromotion()?.from, 'e7');
  assert.equal(recognizer.getPendingPromotion()?.to, 'e8');

  // Explicit confirmation via confirmPromotion('q')
  const ok = recognizer.confirmPromotion('q');
  assert.ok(ok);
  assert.equal(recognizer.getStatus(), 'move_detected');
  assert.equal(recognizer.getLastMove()?.san, 'e8=Q+');
  assert.equal(recognizer.getPendingPromotion(), null);
});

test('captures with only luminance change and no color inversion or clearing are rejected', () => {
  const game = new MoveRecognizer({
    requiredStableFrames: 2,
    initialFen: 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2',
  });

  const obs = createStartingPositionObservations();
  obs.find((o) => o.square === 'e4')!.occupied = true;
  obs.find((o) => o.square === 'e4')!.color = 'w';
  obs.find((o) => o.square === 'e2')!.occupied = false;

  const d5Item = obs.find((o) => o.square === 'd5')!;
  d5Item.occupied = true;
  d5Item.color = 'b';
  d5Item.luminance = 50;
  obs.find((o) => o.square === 'd7')!.occupied = false;

  game.setBaseline(obs);

  // Shadow falls over d5 (luminance shifts from 50 to 90), but e4 is NOT emptied and piece color is still 'b'
  const shadowObs = cloneObservations(obs);
  shadowObs.find((o) => o.square === 'd5')!.luminance = 90;

  game.processFrame(shadowObs);
  const res = game.processFrame(shadowObs);

  // Must not detect a capture move
  assert.equal(res.status, 'watching');
  assert.equal(res.move, null);
});

test('widespread square changes (>4) trigger OCCLUDED and reset stability', () => {
  const recognizer = new MoveRecognizer({ requiredStableFrames: 2 });
  const startObs = createStartingPositionObservations();

  // Hand covers 6 squares simultaneously
  const occludedObs = cloneObservations(startObs);
  for (const sq of ['d1', 'd2', 'e1', 'e2', 'f1', 'f2'] as ChessSquare[]) {
    occludedObs.find((o) => o.square === sq)!.occupied = false;
  }

  recognizer.processFrame(occludedObs);
  const res = recognizer.processFrame(occludedObs);

  assert.equal(res.status, 'occluded');
  assert.equal(res.move, null);
  assert.equal(res.stableFrames, 0);
  // Game FEN must remain untouched
  assert.ok(recognizer.getFen().startsWith('rnbqkbnr/pppppppp'));
});

test('hand occlusion pauses evaluation and resets stable counter', () => {
  const recognizer = new MoveRecognizer({ requiredStableFrames: 3 });
  const startObs = createStartingPositionObservations();
  const movedObs = simulateMoveOnObservations(startObs, 'e2', 'e4', 'w');

  recognizer.processFrame(movedObs);
  recognizer.processFrame(movedObs);

  // Hand enters
  const occludedRes = recognizer.processFrame(movedObs, true);
  assert.equal(occludedRes.status, 'occluded');
  assert.equal(occludedRes.stableFrames, 0);

  // Hand cleared
  const resumeRes = recognizer.processFrame(movedObs, false);
  assert.equal(resumeRes.status, 'stabilizing');
  assert.equal(resumeRes.stableFrames, 1);
});

test('safe manual recovery via confirmManualMove from UNCERTAIN state', () => {
  const recognizer = new MoveRecognizer();
  const corruptObs = cloneObservations(createStartingPositionObservations());
  corruptObs.find((o) => o.square === 'h2')!.occupied = false;
  recognizer.processFrame(corruptObs);
  recognizer.processFrame(corruptObs);
  recognizer.processFrame(corruptObs);

  const ok = recognizer.confirmManualMove('e2', 'e4');
  assert.ok(ok);
  assert.equal(recognizer.getStatus(), 'watching');
  assert.equal(recognizer.getLastMove()?.san, 'e4');
  assert.equal(recognizer.getTurn(), 'b');
});
