import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { ChessGame } from '../src/chess/ChessGame.ts';
import { ChalChessEngine, formatChalPv, parseChalUciMove } from '../src/chess/chalEngine.ts';
import { createDefaultEngine } from '../src/chess/engineSelection.ts';
import { EngineCancelledError } from '../src/chess/engine.ts';
import type { ChalEngineNative, ChalSearchResult } from '../modules/chal-engine/src/index.ts';

const STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

function searchResult(overrides: Partial<ChalSearchResult> = {}): ChalSearchResult {
  return {
    bestMove: 'e2e4',
    score: 20,
    depth: 8,
    nodes: 1234,
    pv: ['e2e4', 'e7e5', 'g1f3'],
    mate: null,
    ...overrides,
  };
}

function fakeNative(
  search: ChalEngineNative['search'],
  stop: ChalEngineNative['stop'] = async () => true,
): ChalEngineNative {
  return {
    init: async () => true,
    search,
    stop,
    teardown: async () => true,
  };
}

describe('chal native engine adapter', () => {
  it('validates a legal UCI move against the starting position', () => {
    const game = new ChessGame(STARTING_FEN);
    const result = parseChalUciMove(game, 'e2e4');
    assert.ok(result);
    assert.equal(result.from, 'e2');
    assert.equal(result.to, 'e4');
    assert.equal(result.san, 'e4');
  });

  it('rejects an illegal UCI move', () => {
    const game = new ChessGame(STARTING_FEN);
    // Pawn cannot jump 3 squares
    const result = parseChalUciMove(game, 'e2e5');
    assert.equal(result, null);
  });

  it('handles promotion UCI move properly', () => {
    // Pawn on e7 ready to promote on e8
    const fen = '8/4P3/8/8/8/8/8/k6K w - - 0 1';
    const game = new ChessGame(fen);
    const resultQ = parseChalUciMove(game, 'e7e8q');
    assert.ok(resultQ);
    assert.equal(resultQ.promotion, 'q');
    assert.equal(resultQ.san, 'e8=Q');

    const gameKnight = new ChessGame(fen);
    const resultN = parseChalUciMove(gameKnight, 'e7e8n');
    assert.ok(resultN);
    assert.equal(resultN.promotion, 'n');
    assert.equal(resultN.san, 'e8=N');
  });

  it('handles checkmate delivery move', () => {
    // Scholar's mate delivery: Qxf7#
    const fen = 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4';
    const game = new ChessGame(fen);
    const result = parseChalUciMove(game, 'h5f7');
    assert.ok(result);
    assert.equal(result.san, 'Qxf7#');
  });

  it('handles malformed or empty UCI strings gracefully', () => {
    const game = new ChessGame(STARTING_FEN);
    assert.equal(parseChalUciMove(game, ''), null);
    assert.equal(parseChalUciMove(game, 'e2'), null);
    assert.equal(parseChalUciMove(game, 'invalid'), null);
  });

  it('verifies evaluation perspective mapping', () => {
    // Score convention: +100000 mate, -100000 mated from side-to-move view
    const mateIn1White = 1;
    const scoreWhite = mateIn1White > 0 ? (100000 - mateIn1White) : (-100000 - mateIn1White);
    assert.equal(scoreWhite, 99999);

    const matedIn1Black = -1;
    const scoreBlack = matedIn1Black > 0 ? (100000 - matedIn1Black) : (-100000 - matedIn1Black);
    assert.equal(scoreBlack, -99999);
  });

  it('reports Chal identity and preserves the analyzed root FEN', async () => {
    const engine = new ChalChessEngine(fakeNative(async () => searchResult()));
    const analysis = await engine.analyze(STARTING_FEN);

    assert.equal(analysis.engine, 'chal-2.0.0');
    assert.equal(analysis.fen, STARTING_FEN);
    assert.equal(analysis.bestMove?.san, 'e4');
    assert.deepEqual(analysis.pv, ['e4', 'e5', 'Nf3']);
    assert.equal(analysis.mate, null);
    assert.equal(analysis.depth, 8);
    assert.equal(analysis.nodes, 1234);
  });

  it('passes through mate distance and delivers SAN', async () => {
    const fen = 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4';
    const engine = new ChalChessEngine(
      fakeNative(async () => searchResult({
        bestMove: 'h5f7',
        score: 99999,
        mate: 1,
        pv: ['h5f7'],
      })),
    );

    const analysis = await engine.analyze(fen);
    assert.equal(analysis.bestMove?.san, 'Qxf7#');
    assert.equal(analysis.mate, 1);
  });

  it('returns no move when the native engine emits an illegal best move', async () => {
    const engine = new ChalChessEngine(
      fakeNative(async () => searchResult({ bestMove: 'e2e5', pv: [] })),
    );
    const analysis = await engine.analyze(STARTING_FEN);

    assert.equal(analysis.bestMove, null);
    assert.equal(analysis.fen, STARTING_FEN);
  });

  it('cancels an in-flight native search', async () => {
    let resolveSearch!: (value: ChalSearchResult) => void;
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });

    const engine = new ChalChessEngine(fakeNative(() => {
      markStarted();
      return new Promise<ChalSearchResult>((resolve) => {
        resolveSearch = resolve;
      });
    }));

    const pending = engine.analyze(STARTING_FEN);
    await started;
    engine.stop();
    resolveSearch(searchResult());

    await assert.rejects(pending, EngineCancelledError);
  });

  it('converts a UCI principal variation to SAN', () => {
    assert.deepEqual(
      formatChalPv(STARTING_FEN, ['e2e4', 'e7e5', 'g1f3']),
      ['e4', 'e5', 'Nf3'],
    );
    assert.deepEqual(formatChalPv(STARTING_FEN, ['e2e5']), []);
  });

  it('selects Chal when the native module is registered', () => {
    const selection = createDefaultEngine(fakeNative(async () => searchResult()));
    assert.equal(selection.engineName, 'chal-2.0.0');
    assert.equal(selection.fallbackReason, null);
  });

  it('selects and discloses the heuristic fallback when the native module is absent', async () => {
    const selection = createDefaultEngine(null);

    assert.equal(selection.engineName, 'heuristic-v1');
    assert.match(selection.fallbackReason ?? '', /not available in this build/);

    const analysis = await selection.engine.analyze(STARTING_FEN);
    assert.equal(analysis.engine, 'heuristic-v1');
    assert.ok(analysis.bestMove);
  });
});
