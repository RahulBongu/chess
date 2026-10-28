/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { ChessGame } from '../src/chess/ChessGame.ts';

test('initial position has 20 legal moves and white to move', () => {
  const game = new ChessGame();
  assert.equal(game.turn(), 'w');
  assert.equal(game.allLegalMoves().length, 20);
  assert.equal(game.history().length, 0);
  assert.equal(game.isGameOver(), false);
});

test('a normal move changes turn and records history', () => {
  const game = new ChessGame();
  const move = game.tryMove({ from: 'e2', to: 'e4' });
  assert.ok(move);
  assert.equal(game.turn(), 'b');
  assert.deepEqual(game.history(), ['e4']);
  assert.equal(game.moveCount(), 1);
});

test('illegal moves are rejected', () => {
  const game = new ChessGame();
  const move = game.tryMove({ from: 'e2', to: 'e5' });
  assert.equal(move, null);
  assert.equal(game.turn(), 'w');
  assert.equal(game.history().length, 0);
});

test('undo restores the previous position and reset clears history', () => {
  const game = new ChessGame();
  game.tryMove({ from: 'e2', to: 'e4' });
  game.tryMove({ from: 'e7', to: 'e5' });

  const undone = game.undo();
  assert.ok(undone);
  assert.equal(game.turn(), 'b');
  assert.deepEqual(game.history(), ['e4']);

  game.reset();
  assert.equal(game.turn(), 'w');
  assert.equal(game.history().length, 0);
  assert.equal(game.allLegalMoves().length, 20);
});

test('checkmate is detected', () => {
  const game = new ChessGame();
  // Fool's mate.
  game.tryMove({ from: 'f2', to: 'f3' });
  game.tryMove({ from: 'e7', to: 'e5' });
  game.tryMove({ from: 'g2', to: 'g4' });
  game.tryMove({ from: 'd8', to: 'h4' });

  assert.equal(game.isCheckmate(), true);
  assert.equal(game.isGameOver(), true);
});

test('castling is supported', () => {
  const game = new ChessGame('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
  const move = game.tryMove({ from: 'e1', to: 'g1' });
  assert.ok(move);
  assert.equal(move?.isKingsideCastle(), true);
  assert.equal(game.pieceAt('g1')?.type, 'k');
  assert.equal(game.pieceAt('f1')?.type, 'r');
});

test('en passant is supported', () => {
  const game = new ChessGame();
  game.tryMove({ from: 'e2', to: 'e4' });
  game.tryMove({ from: 'a7', to: 'a6' });
  game.tryMove({ from: 'e4', to: 'e5' });
  game.tryMove({ from: 'd7', to: 'd5' });

  const move = game.tryMove({ from: 'e5', to: 'd6' });
  assert.ok(move);
  assert.equal(move?.isEnPassant(), true);
  assert.equal(game.pieceAt('d5'), undefined);
  assert.equal(game.pieceAt('d6')?.type, 'p');
});

test('promotion is supported', () => {
  const game = new ChessGame('4k3/P7/8/8/8/8/8/4K3 w - - 0 1');
  assert.equal(game.isPromotionMove('a7', 'a8'), true);

  const move = game.tryMove({ from: 'a7', to: 'a8', promotion: 'q' });
  assert.ok(move);
  assert.equal(move?.isPromotion(), true);
  assert.equal(game.pieceAt('a8')?.type, 'q');
});