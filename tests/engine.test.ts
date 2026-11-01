import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { ChessGame } from '../src/chess/ChessGame.ts';
import { createHeuristicEngine, EngineCancelledError } from '../src/chess/engine.ts';

const STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

describe('heuristic engine', () => {
  it('returns a legal suggested move for the starting position', async () => {
    const engine = createHeuristicEngine();
    const result = await engine.analyze(STARTING_FEN);
    assert.ok(result.bestMove);
    const game = new ChessGame();
    game.load(STARTING_FEN);
    const legal = game.allLegalMoves().some(
      (move) => move.from === result.bestMove?.from && move.to === result.bestMove?.to,
    );
    assert.equal(legal, true);
    assert.equal(result.engine, 'heuristic-v1');
    assert.equal(result.perspective, 'side-to-move');
  });

  it('is deterministic for the same FEN', async () => {
    const engine = createHeuristicEngine();
    const first = await engine.analyze(STARTING_FEN);
    const second = await engine.analyze(STARTING_FEN);
    assert.deepEqual(first.bestMove, second.bestMove);
    assert.equal(first.score, second.score);
  });

  it('handles terminal checkmate', async () => {
    const fen = 'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3';
    const engine = createHeuristicEngine();
    const result = await engine.analyze(fen);
    assert.equal(result.bestMove, null);
    assert.equal(result.terminal, 'checkmate');
    assert.ok(result.score < 0);
  });

  it('handles stalemate', async () => {
    const fen = '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1';
    const engine = createHeuristicEngine();
    const result = await engine.analyze(fen);
    assert.equal(result.bestMove, null);
    assert.equal(result.terminal, 'stalemate');
    assert.equal(result.score, 0);
  });

  it('rejects an invalid FEN', async () => {
    const engine = createHeuristicEngine();
    await assert.rejects(() => engine.analyze('not a fen'), /Invalid FEN/);
  });

  it('cancels an obsolete analysis', async () => {
    const engine = createHeuristicEngine();
    const first = engine.analyze(STARTING_FEN);
    engine.stop();
    await assert.rejects(() => first, EngineCancelledError);
  });

  it('does not let an older position overwrite a newer one', async () => {
    const engine = createHeuristicEngine();
    const firstFen = STARTING_FEN;
    const secondFen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1';

    const first = engine.analyze(firstFen);
    const second = engine.analyze(secondFen);

    await assert.rejects(() => first, EngineCancelledError);
    const result = await second;
    assert.equal(result.fen, new ChessGame().load(secondFen) ? new ChessGame(secondFen).fen() : secondFen);
  });

});