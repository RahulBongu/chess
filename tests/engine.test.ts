import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { ChessGame } from '../src/chess/ChessGame.ts';
import {
  createHeuristicEngine,
  EngineCancelledError,
  evaluateWhite,
  pieceSquareIndex,
} from '../src/chess/engine.ts';

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
    assert.equal(result.depth, 2);
    assert.ok(result.positionsEvaluated > 0);
  });

  it('is deterministic for the same FEN', async () => {
    const engine = createHeuristicEngine();
    const first = await engine.analyze(STARTING_FEN);
    const second = await engine.analyze(STARTING_FEN);
    assert.deepEqual(first.bestMove, second.bestMove);
    assert.equal(first.score, second.score);
    assert.deepEqual(first.pv, second.pv);
    assert.equal(first.positionsEvaluated, second.positionsEvaluated);
  });

  it('correctly maps piece-square indices vertically and preserves files', () => {
    // Row 0 in tables is Rank 8; Row 7 is Rank 1.
    // White king starting on e1 (file e=4, rank 1): row 7, col 4 => 60
    assert.equal(pieceSquareIndex(4, 1, 'w'), 60);
    // Black king starting on e8 (file e=4, rank 8): mirrored to row 7, col 4 => 60
    assert.equal(pieceSquareIndex(4, 8, 'b'), 60);

    // White pawn starting on a2 (file a=0, rank 2): row 6, col 0 => 48
    assert.equal(pieceSquareIndex(0, 2, 'w'), 48);
    // Black pawn starting on a7 (file a=0, rank 7): row 6, col 0 => 48
    assert.equal(pieceSquareIndex(0, 7, 'b'), 48);

    // Kingside file g=6 is preserved for both sides without horizontal flipping:
    assert.equal(pieceSquareIndex(6, 1, 'w') % 8, 6);
    assert.equal(pieceSquareIndex(6, 8, 'b') % 8, 6);

    // Starting board is perfectly symmetrical: evaluation must be 0
    const startBoard = new ChessGame();
    assert.equal(evaluateWhite(startBoard), 0);
  });

  it('prefers immediate checkmate over capturing free queen', async () => {
    // White can play Qh7# (checkmate) or Qxa2 (captures undefended queen).
    const fen = '7k/6p1/8/6N1/8/8/q7/1Q5K w - - 0 1';
    const engine = createHeuristicEngine();
    const result = await engine.analyze(fen);
    assert.ok(result.bestMove);
    assert.equal(result.bestMove.san, 'Qh7#');
    assert.equal(result.score, 100000);
    assert.deepEqual(result.pv, ['Qh7#']);
  });

  it('avoids a move that allows immediate checkmate for the opponent', async () => {
    // Black threatens Qg2# protected by h3 pawn.
    // b3 and b4 allow Qg2# (-100000). Ra8+ gives check and avoids immediate mate.
    const fen = '7k/8/8/8/8/6qp/1P6/R6K w - - 0 1';
    const engine = createHeuristicEngine();
    const result = await engine.analyze(fen);
    assert.ok(result.bestMove);
    assert.notEqual(result.bestMove.san, 'b3');
    assert.notEqual(result.bestMove.san, 'b4');
    assert.ok(result.score > -100000);
  });

  it('handles terminal checkmate at root', async () => {
    // Fool's mate: White is in checkmate.
    const fen = 'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3';
    const engine = createHeuristicEngine();
    const result = await engine.analyze(fen);
    assert.equal(result.bestMove, null);
    assert.equal(result.terminal, 'checkmate');
    assert.equal(result.score, -100000);
    assert.equal(result.depth, 0);
  });

  it('handles terminal stalemate and draws correctly', async () => {
    // Stalemate: Black to move has no legal moves and is not in check.
    const fenStalemate = '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1';
    const engine = createHeuristicEngine();
    const resultStalemate = await engine.analyze(fenStalemate);
    assert.equal(resultStalemate.bestMove, null);
    assert.equal(resultStalemate.terminal, 'stalemate');
    assert.equal(resultStalemate.score, 0);

    // Insufficient material draw: King + Bishop vs King.
    const fenDraw = '8/8/8/8/8/8/8/kb5K w - - 0 1';
    const resultDraw = await engine.analyze(fenDraw);
    assert.equal(resultDraw.bestMove, null);
    assert.equal(resultDraw.terminal, 'draw');
    assert.equal(resultDraw.score, 0);
  });

  it('addresses the Bxc6 horizon problem by evaluating Black recapture', async () => {
    // Ruy Lopez: 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6
    const fen = 'r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4';
    const engine = createHeuristicEngine();
    const result = await engine.analyze(fen);
    assert.ok(result.bestMove);
    assert.equal(result.depth, 2);
    assert.ok(result.positionsEvaluated > 32);
    // Score must reflect that Black can recapture, so White does not get a free +390 knight.
    assert.notEqual(result.score, 390);
    assert.ok(Math.abs(result.score) < 150);
    // If Bxc6 is played, the predicted continuation includes Black's recapture.
    if (result.bestMove.san === 'Bxc6') {
      assert.ok(result.pv.length >= 2);
      assert.match(result.pv[1], /^[bd]xc6$/);
    }
  });

  it('preserves evaluation perspective for White and Black side to move', async () => {
    const engine = createHeuristicEngine();

    // White has extra queen: White to move => positive score.
    const fenWhiteUpWhiteTurn = '7k/8/8/8/8/8/8/Q6K w - - 0 1';
    const res1 = await engine.analyze(fenWhiteUpWhiteTurn);
    assert.ok(res1.score > 800);
    assert.equal(res1.perspective, 'side-to-move');

    // White has extra queen: Black to move => negative score (from Black's perspective).
    const fenWhiteUpBlackTurn = '7k/8/8/8/8/8/8/Q6K b - - 0 1';
    const res2 = await engine.analyze(fenWhiteUpBlackTurn);
    assert.ok(res2.score < -800);
    assert.equal(res2.perspective, 'side-to-move');

    // Black has extra queen: Black to move => positive score (from Black's perspective).
    const fenBlackUpBlackTurn = '7k/8/8/8/8/8/8/q6K b - - 0 1';
    const res3 = await engine.analyze(fenBlackUpBlackTurn);
    assert.ok(res3.score > 800);
    assert.equal(res3.perspective, 'side-to-move');

    // Black is in checkmate => -MATE_SCORE for Black side to move.
    const fenBlackMated = 'rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3';
    const res4 = await engine.analyze(fenBlackMated);
    assert.equal(res4.bestMove, null);
    assert.equal(res4.terminal, 'checkmate');
    assert.equal(res4.score, -100000);
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
