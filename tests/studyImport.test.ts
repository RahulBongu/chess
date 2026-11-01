import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { importFen, importPgn, importStudyInput } from '../src/study/studyGame.ts';

const STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

describe('study import', () => {
  it('imports a valid FEN', () => {
    const result = importFen(STARTING_FEN);
    assert.ok(result);
    assert.equal(result.length, 1);
    assert.equal(result[0].fen, STARTING_FEN);
  });

  it('rejects an invalid FEN', () => {
    assert.equal(importFen('not a fen'), null);
    assert.equal(importFen(''), null);
  });

  it('imports a PGN and builds replay positions', () => {
    const pgn = '1. e4 e5 2. Nf3 Nc6';
    const result = importPgn(pgn);
    assert.ok(result);
    assert.deepEqual(result.moves, ['e4', 'e5', 'Nf3', 'Nc6']);
    assert.equal(result.positions.length, 5);
    assert.equal(result.positions[0].fen, STARTING_FEN);
  });

  it('rejects an invalid PGN', () => {
    assert.equal(importPgn('1. e4 e9'), null);
    assert.equal(importPgn(''), null);
  });

  it('auto-detects FEN vs PGN', () => {
    const fenResult = importStudyInput(STARTING_FEN);
    assert.ok(fenResult);
    assert.equal(fenResult.moves.length, 0);

    const pgnResult = importStudyInput('1. d4 d5');
    assert.ok(pgnResult);
    assert.equal(pgnResult.moves.length, 2);
  });

  it('rejects empty input', () => {
    assert.equal(importStudyInput('   '), null);
  });
});