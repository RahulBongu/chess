/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  REACHABILITY_DISCLAIMER,
  buildPlacementFen,
  buildPositionFen,
  isValidSquareName,
  normalizeCastling,
  normalizePlacementInput,
} from '../src/vision/fenBuilder.ts';

// Helper to create starting position dictionary
function createStartingPositionMap(): Record<string, string> {
  return {
    a8: 'r', b8: 'n', c8: 'b', d8: 'q', e8: 'k', f8: 'b', g8: 'n', h8: 'r',
    a7: 'p', b7: 'p', c7: 'p', d7: 'p', e7: 'p', f7: 'p', g7: 'p', h7: 'p',
    a2: 'P', b2: 'P', c2: 'P', d2: 'P', e2: 'P', f2: 'P', g2: 'P', h2: 'P',
    a1: 'R', b1: 'N', c1: 'B', d1: 'Q', e1: 'K', f1: 'B', g1: 'N', h1: 'R',
  };
}

// Minimal legal king vs king map
function createKingsOnlyMap(whiteKing = 'e1', blackKing = 'e8'): Record<string, string> {
  return {
    [whiteKing]: 'K',
    [blackKing]: 'k',
  };
}

describe('FEN builder and position constraint validation', () => {
  it('correctly builds standard starting FEN with empty-square compression and rank ordering', () => {
    const startingMap = createStartingPositionMap();
    const result = buildPositionFen(startingMap, {
      metadata: {
        turn: 'w',
        castling: 'KQkq',
        enPassant: '-',
        halfmoveClock: 0,
        fullmoveNumber: 1,
      },
    });

    assert.equal(result.valid, true);
    assert.equal(result.errors.length, 0);
    assert.equal(result.placementFen, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR');
    assert.equal(result.fen, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    assert.equal(result.details?.whitePieceCount, 16);
    assert.equal(result.details?.blackPieceCount, 16);
    assert.equal(result.details?.isCheck, false);
    assert.equal(result.details?.legalMoveCount, 20);
    assert.equal(result.reachabilityDisclaimer, REACHABILITY_DISCLAIMER);
  });

  it('correctly maps 64-element array under white-bottom and black-bottom orientations', () => {
    // Array with White king at display index 0 (top-left) and Black king at display index 63 (bottom-right)
    const array64: (string | null)[] = new Array(64).fill(null);
    array64[0] = 'K';
    array64[63] = 'k';

    // In white-bottom view:
    // index 0 is a8, index 63 is h1
    const whiteBottomResult = buildPositionFen(array64, {
      orientation: 'white-bottom',
    });
    assert.equal(whiteBottomResult.valid, true);
    assert.equal(whiteBottomResult.details?.whiteKingSquare, 'a8');
    assert.equal(whiteBottomResult.details?.blackKingSquare, 'h1');
    assert.equal(whiteBottomResult.placementFen, 'K7/8/8/8/8/8/8/7k');

    // In black-bottom view:
    // index 0 is h1, index 63 is a8
    const blackBottomResult = buildPositionFen(array64, {
      orientation: 'black-bottom',
    });
    assert.equal(blackBottomResult.valid, true);
    assert.equal(blackBottomResult.details?.whiteKingSquare, 'h1');
    assert.equal(blackBottomResult.details?.blackKingSquare, 'a8');
    assert.equal(blackBottomResult.placementFen, 'k7/8/8/8/8/8/8/7K');
  });

  it('rejects missing or multiple kings for each color', () => {
    // Missing White king
    const noWhiteKing = buildPositionFen({ e8: 'k' });
    assert.equal(noWhiteKing.valid, false);
    assert.match(noWhiteKing.errors.join('; '), /Missing White king/);

    // Missing Black king
    const noBlackKing = buildPositionFen({ e1: 'K' });
    assert.equal(noBlackKing.valid, false);
    assert.match(noBlackKing.errors.join('; '), /Missing Black king/);

    // Two White kings
    const twoWhiteKings = buildPositionFen({ e1: 'K', d1: 'K', e8: 'k' });
    assert.equal(twoWhiteKings.valid, false);
    assert.match(twoWhiteKings.errors.join('; '), /Found 2 White kings/);

    // Two Black kings
    const twoBlackKings = buildPositionFen({ e1: 'K', e8: 'k', d8: 'k' });
    assert.equal(twoBlackKings.valid, false);
    assert.match(twoBlackKings.errors.join('; '), /Found 2 Black kings/);
  });

  it('rejects adjacent kings (kings touching horizontally, vertically, or diagonally)', () => {
    // Vertically adjacent: e1 and e2
    const verticalTouch = buildPositionFen({ e1: 'K', e2: 'k' });
    assert.equal(verticalTouch.valid, false);
    assert.match(verticalTouch.errors.join('; '), /adjacent \(kings cannot touch\)/);

    // Diagonally adjacent: e4 and d5
    const diagonalTouch = buildPositionFen({ e4: 'K', d5: 'k' });
    assert.equal(diagonalTouch.valid, false);
    assert.match(diagonalTouch.errors.join('; '), /adjacent \(kings cannot touch\)/);

    // Horizontally adjacent: e4 and f4
    const horizontalTouch = buildPositionFen({ e4: 'K', f4: 'k' });
    assert.equal(horizontalTouch.valid, false);
    assert.match(horizontalTouch.errors.join('; '), /adjacent \(kings cannot touch\)/);

    // Separated by 2 squares: e1 and e3 -> Valid!
    const separated = buildPositionFen({ e1: 'K', e3: 'k' });
    assert.equal(separated.valid, true);
    assert.equal(separated.errors.length, 0);
  });

  it('rejects pawns on the first or eighth rank', () => {
    const whitePawnRank1 = buildPositionFen({ e1: 'K', e8: 'k', a1: 'P' });
    assert.equal(whitePawnRank1.valid, false);
    assert.match(whitePawnRank1.errors.join('; '), /White pawn on a1 is illegal/);

    const whitePawnRank8 = buildPositionFen({ e1: 'K', e8: 'k', h8: 'P' });
    assert.equal(whitePawnRank8.valid, false);
    assert.match(whitePawnRank8.errors.join('; '), /White pawn on h8 is illegal/);

    const blackPawnRank1 = buildPositionFen({ e1: 'K', e8: 'k', c1: 'p' });
    assert.equal(blackPawnRank1.valid, false);
    assert.match(blackPawnRank1.errors.join('; '), /Black pawn on c1 is illegal/);

    const blackPawnRank8 = buildPositionFen({ e1: 'K', e8: 'k', f8: 'p' });
    assert.equal(blackPawnRank8.valid, false);
    assert.match(blackPawnRank8.errors.join('; '), /Black pawn on f8 is illegal/);
  });

  it('enforces maximum piece and pawn counts and promoted piece limits', () => {
    // 9 White pawns
    const tooManyPawns = createKingsOnlyMap();
    for (let f = 0; f < 8; f += 1) {
      tooManyPawns[`${FILES[f]}2`] = 'P';
    }
    tooManyPawns['b3'] = 'P'; // 9th pawn
    const pawnOvercount = buildPositionFen(tooManyPawns);
    assert.equal(pawnOvercount.valid, false);
    assert.match(pawnOvercount.errors.join('; '), /White has 9 pawns/);

    // 2 Queens with 8 pawns (impossible: 2 queens requires at least 1 promoted pawn)
    const impossibleQueens = createKingsOnlyMap();
    for (let f = 0; f < 8; f += 1) {
      impossibleQueens[`${FILES[f]}2`] = 'P';
    }
    impossibleQueens['d1'] = 'Q';
    impossibleQueens['d3'] = 'Q';
    const promotionOvercount = buildPositionFen(impossibleQueens);
    assert.equal(promotionOvercount.valid, false);
    assert.match(promotionOvercount.errors.join('; '), /require.* more than 0 promoted pawns/);

    // 2 Queens with 7 pawns -> Permissible (1 promoted pawn)
    delete impossibleQueens['h2']; // now 7 pawns
    const legalPromotedQueen = buildPositionFen(impossibleQueens);
    assert.equal(legalPromotedQueen.valid, true);
    assert.equal(legalPromotedQueen.errors.length, 0);
  });

  it('validates side-to-move consistency with check state (opponent king cannot be in check)', () => {
    // White rook on e1, Black king on e8.
    // If it is White to move (w), this is illegal because Black king is in check!
    const opponentInCheck = createKingsOnlyMap('a1', 'e8');
    opponentInCheck['e1'] = 'R';

    const whiteToMoveIllegal = buildPositionFen(opponentInCheck, {
      metadata: { turn: 'w' },
    });
    assert.equal(whiteToMoveIllegal.valid, false);
    assert.match(whiteToMoveIllegal.errors.join('; '), /Black king is in check, but it is White to move/);

    // If it is Black to move (b), Black is currently in check -> Legal check state!
    const blackToMoveLegal = buildPositionFen(opponentInCheck, {
      metadata: { turn: 'b' },
    });
    assert.equal(blackToMoveLegal.valid, true);
    assert.equal(blackToMoveLegal.details?.isCheck, true);
  });

  it('validates castling rights against king and rook placement and includes reachability warning', () => {
    // White kingside (K) requested, but White king is on e2 instead of e1
    const invalidKingPlacement = createKingsOnlyMap('e2', 'e8');
    invalidKingPlacement['h1'] = 'R';
    const castlingResult = buildPositionFen(invalidKingPlacement, {
      metadata: { castling: 'K' },
    });
    assert.equal(castlingResult.valid, false);
    assert.match(castlingResult.errors.join('; '), /White kingside castling \("K"\) is invalid.*King must be on e1/);

    // King on e1 and Rook on h1 -> Placement satisfies castling prerequisite
    const validKingPlacement = createKingsOnlyMap('e1', 'e8');
    validKingPlacement['h1'] = 'R';
    const validCastling = buildPositionFen(validKingPlacement, {
      metadata: { castling: 'K' },
    });
    assert.equal(validCastling.valid, true);
    assert.equal(validCastling.metadata.castling, 'K');
    assert.ok(
      validCastling.warnings.some((w) => w.includes('never moved historically')),
      'Must surface historical reachability disclaimer for castling rights',
    );
  });

  it('validates en-passant target rank, passed pawn placement, and empty destination square', () => {
    // White to move: Black just played e7-e5 -> En passant target is e6 (rank 6)
    const validEpPosition = createKingsOnlyMap('a1', 'h8');
    validEpPosition['e5'] = 'p'; // Black pawn on e5

    const validEp = buildPositionFen(validEpPosition, {
      metadata: { turn: 'w', enPassant: 'e6' },
    });
    assert.equal(validEp.valid, true);
    assert.equal(validEp.metadata.enPassant, 'e6');

    // Wrong rank for White to move (e3 is rank 3, only valid when Black is to move)
    const wrongRankEp = buildPositionFen(validEpPosition, {
      metadata: { turn: 'w', enPassant: 'e3' },
    });
    assert.equal(wrongRankEp.valid, false);
    assert.match(wrongRankEp.errors.join('; '), /En-passant target for White to move must be on rank 6/);

    // Missing passed pawn on e5
    const missingPawnEpPosition = createKingsOnlyMap('a1', 'h8');
    const missingPawnEp = buildPositionFen(missingPawnEpPosition, {
      metadata: { turn: 'w', enPassant: 'e6' },
    });
    assert.equal(missingPawnEp.valid, false);
    assert.match(missingPawnEp.errors.join('; '), /expected Black pawn on e5/);

    // En-passant target square is not empty
    const occupiedEpPosition = createKingsOnlyMap('a1', 'h8');
    occupiedEpPosition['e5'] = 'p';
    occupiedEpPosition['e6'] = 'N'; // Blocked by a knight
    const occupiedEp = buildPositionFen(occupiedEpPosition, {
      metadata: { turn: 'w', enPassant: 'e6' },
    });
    assert.equal(occupiedEp.valid, false);
    assert.match(occupiedEp.errors.join('; '), /target square e6 must be empty/);
  });

  it('validates halfmove clock and fullmove number format', () => {
    const kings = createKingsOnlyMap();

    const negativeHalfmove = buildPositionFen(kings, {
      metadata: { halfmoveClock: -1 },
    });
    assert.equal(negativeHalfmove.valid, false);
    assert.match(negativeHalfmove.errors.join('; '), /Halfmove clock must be a non-negative integer/);

    const zeroFullmove = buildPositionFen(kings, {
      metadata: { fullmoveNumber: 0 },
    });
    assert.equal(zeroFullmove.valid, false);
    assert.match(zeroFullmove.errors.join('; '), /Fullmove number must be a positive integer/);

    const validCounters = buildPositionFen(kings, {
      metadata: { halfmoveClock: 49, fullmoveNumber: 42 },
    });
    assert.equal(validCounters.valid, true);
    assert.equal(validCounters.metadata.halfmoveClock, 49);
    assert.equal(validCounters.metadata.fullmoveNumber, 42);
  });

  it('rejects invalid piece symbols and malformed array lengths without silent correction', () => {
    const invalidSymbol = buildPositionFen({ e1: 'K', e8: 'k', d4: 'X' });
    assert.equal(invalidSymbol.valid, false);
    assert.match(invalidSymbol.errors.join('; '), /Invalid piece symbol "X" at square d4/);

    // 63-element array
    const wrongLength = new Array(63).fill(null);
    const lengthResult = buildPositionFen(wrongLength);
    assert.equal(lengthResult.valid, false);
    assert.match(lengthResult.errors.join('; '), /Expected array of exactly 64 squares, received 63/);
  });

  it('correctly reports terminal checkmate and stalemate states', () => {
    // Scholar's Mate position after Qxf7# (Black king on e8, White queen on f7, White bishop on c4, White king on e1)
    const checkmatePosition: Record<string, string> = {
      e8: 'k',
      d8: 'q',
      f8: 'b',
      f7: 'Q',
      c4: 'B',
      e1: 'K',
      a7: 'p', b7: 'p', c7: 'p', d7: 'p',
    };
    const mateResult = buildPositionFen(checkmatePosition, {
      metadata: { turn: 'b' },
    });
    assert.equal(mateResult.valid, true);
    assert.equal(mateResult.details?.isCheck, true);
    assert.equal(mateResult.details?.isCheckmate, true);
    assert.equal(mateResult.details?.legalMoveCount, 0);

    // Stalemate: Black king on a8, White king on a6, White queen on c7 (Black to move, no legal moves, not in check)
    const stalematePosition: Record<string, string> = {
      a8: 'k',
      a6: 'K',
      c7: 'Q',
    };
    const stalemateResult = buildPositionFen(stalematePosition, {
      metadata: { turn: 'b' },
    });
    assert.equal(stalemateResult.valid, true);
    assert.equal(stalemateResult.details?.isCheck, false);
    assert.equal(stalemateResult.details?.isStalemate, true);
    assert.equal(stalemateResult.details?.legalMoveCount, 0);
  });
});

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;
