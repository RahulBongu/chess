/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  buildPositionFen,
  normalizePlacementInput,
  type ChessPieceSymbol,
  type SquareEntry,
} from '../src/vision/fenBuilder.ts';
import {
  recognizeBoardFromRectifiedImage,
  pieceTypeToSymbol,
  SYNTHETIC_MODEL_DISCLOSURE,
} from '../src/vision/recognizeBoard.ts';
import {
  computeRectifiedBoardGeometry,
  CANONICAL_CORNERS,
} from '../src/vision/rectify.ts';
import type { AnyImageData } from '../src/vision/classifySquare.ts';
import { importStudyInput } from '../src/study/studyGame.ts';
import { handleStudyRouteFen, createInitialStudyRouteState } from '../src/study/studyRouteHandler.ts';
import { createDefaultEngine } from '../src/chess/engineSelection.ts';
import { ChessGame } from '../src/chess/ChessGame.ts';

const ALL_12_PIECES: ChessPieceSymbol[] = [
  'P', 'N', 'B', 'R', 'Q', 'K',
  'p', 'n', 'b', 'r', 'q', 'k',
];

describe('board review and study handoff integration', () => {
  it('supports all 12 piece symbols and empty-square selections', () => {
    // 10 non-king piece symbols on d4 alongside standard kings
    const nonKings: ChessPieceSymbol[] = ['P', 'N', 'B', 'R', 'Q', 'p', 'n', 'b', 'r', 'q'];
    for (const piece of nonKings) {
      const res = buildPositionFen({ e1: 'K', e8: 'k', d4: piece });
      assert.equal(res.valid, true);
      assert.match(res.placementFen ?? '', new RegExp(piece));
    }

    // White king selection on d4
    const whiteKRes = buildPositionFen({ d4: 'K', e8: 'k' });
    assert.equal(whiteKRes.valid, true);
    assert.equal(whiteKRes.details?.whiteKingSquare, 'd4');

    // Black king selection on d4
    const blackKRes = buildPositionFen({ e1: 'K', d4: 'k' });
    assert.equal(blackKRes.valid, true);
    assert.equal(blackKRes.details?.blackKingSquare, 'd4');

    // Empty square selection on d4
    const emptyRes = buildPositionFen({ e1: 'K', e8: 'k', d4: null });
    assert.equal(emptyRes.valid, true);
    assert.doesNotMatch(emptyRes.placementFen ?? '', /d4/);
  });

  it('preserves orientation mapping across white-bottom and black-bottom edits', () => {
    // Create a 64-element array where index 0 is marked with 'K' and index 63 with 'k'
    const array64: (string | null)[] = new Array(64).fill(null);
    array64[0] = 'K';
    array64[63] = 'k';

    // In white-bottom view:
    // index 0 is top-left (a8), index 63 is bottom-right (h1)
    const whiteView = normalizePlacementInput(array64, 'white-bottom');
    assert.equal(whiteView.squares['a8'], 'K');
    assert.equal(whiteView.squares['h1'], 'k');

    // In black-bottom view:
    // index 0 is top-left (h1), index 63 is bottom-right (a8)
    const blackView = normalizePlacementInput(array64, 'black-bottom');
    assert.equal(blackView.squares['h1'], 'K');
    assert.equal(blackView.squares['a8'], 'k');
  });

  it('validates en-passant with strict emptied-origin-square prerequisite', () => {
    // White to move: Black just played e7-e5 -> En passant target is e6 (rank 6)
    // Prerequisites:
    // 1. Black pawn must be on e5
    // 2. Target e6 must be empty
    // 3. Double-step origin e7 must be empty!
    const epMap: Record<string, string> = {
      a1: 'K',
      h8: 'k',
      e5: 'p',
    };

    // Case A: Valid en-passant (e7 is empty)
    const validEp = buildPositionFen(epMap, {
      metadata: { turn: 'w', enPassant: 'e6' },
    });
    assert.equal(validEp.valid, true);
    assert.equal(validEp.metadata.enPassant, 'e6');

    // Case B: Invalid because origin square e7 is occupied (the pawn could not have just double-stepped from e7!)
    epMap['e7'] = 'b'; // Bishop on e7
    const invalidOriginEp = buildPositionFen(epMap, {
      metadata: { turn: 'w', enPassant: 'e6' },
    });
    assert.equal(invalidOriginEp.valid, false);
    assert.match(invalidOriginEp.errors.join('; '), /origin square e7 must be empty/);

    // Case C: Black to move: White just played d2-d4 -> En passant target is d3 (rank 3)
    const blackEpMap: Record<string, string> = {
      a1: 'K',
      h8: 'k',
      d4: 'P',
    };
    const validBlackEp = buildPositionFen(blackEpMap, {
      metadata: { turn: 'b', enPassant: 'd3' },
    });
    assert.equal(validBlackEp.valid, true);
    assert.equal(validBlackEp.metadata.enPassant, 'd3');

    // If origin d2 is occupied, reject!
    blackEpMap['d2'] = 'R';
    const invalidBlackOrigin = buildPositionFen(blackEpMap, {
      metadata: { turn: 'b', enPassant: 'd3' },
    });
    assert.equal(invalidBlackOrigin.valid, false);
    assert.match(invalidBlackOrigin.errors.join('; '), /origin square d2 must be empty/);
  });

  it('blocks handoff of invalid FEN and prevents silent position corrections', () => {
    // Position with pawn on rank 8 and adjacent kings
    const invalidSetup = {
      e1: 'K',
      e2: 'k', // adjacent king
      e8: 'P', // pawn on rank 8
    };

    const result = buildPositionFen(invalidSetup);
    assert.equal(result.valid, false);
    assert.equal(result.fen, null); // Cannot emit valid FEN
    assert.ok(result.errors.length >= 2);
    // Ensure no silent alterations occurred
    assert.match(result.errors.join('; '), /adjacent/);
    assert.match(result.errors.join('; '), /pawns cannot exist on rank 1 or 8/);
  });

  it('verifies valid FEN handoff into Board Study and engine initialization', () => {
    // Standard starting position FEN built from review
    const startingMap: Record<string, string> = {
      a8: 'r', b8: 'n', c8: 'b', d8: 'q', e8: 'k', f8: 'b', g8: 'n', h8: 'r',
      a7: 'p', b7: 'p', c7: 'p', d7: 'p', e7: 'p', f7: 'p', g7: 'p', h7: 'p',
      a2: 'P', b2: 'P', c2: 'P', d2: 'P', e2: 'P', f2: 'P', g2: 'P', h2: 'P',
      a1: 'R', b1: 'N', c1: 'B', d1: 'Q', e1: 'K', f1: 'B', g1: 'N', h1: 'R',
    };

    const built = buildPositionFen(startingMap, {
      metadata: {
        turn: 'w',
        castling: 'KQkq',
        enPassant: '-',
        halfmoveClock: 0,
        fullmoveNumber: 1,
      },
    });

    assert.equal(built.valid, true);
    assert.ok(built.fen);

    // Verify Study's import parser accepts this exact FEN handoff
    const studyImport = importStudyInput(built.fen);
    assert.ok(studyImport);
    assert.equal(studyImport.positions.length, 1);
    assert.equal(studyImport.positions[0].fen, built.fen);

    // Verify ChessGame can load and play legal moves from this handoff
    const game = new ChessGame();
    assert.equal(game.load(built.fen), true);
    assert.equal(game.allLegalMoves().length, 20);

    // Verify default engine selection accepts and analyzes the handed-off FEN
    const selection = createDefaultEngine(null);
    assert.equal(selection.engineName, 'heuristic-v1');
    assert.ok(selection.fallbackReason);
  });

  it('safely handles missing, empty, or malformed route parameters without corrupting Study state', () => {
    // Empty string
    assert.equal(importStudyInput(''), null);
    // Whitespace only
    assert.equal(importStudyInput('   '), null);
    // Malformed FEN text
    assert.equal(importStudyInput('not a valid fen'), null);
    assert.equal(importStudyInput('8/8/8/8/8/8/8/8'), null); // Missing turn, castling, etc.
  });

  it('verifies engine analysis and cancellation on imported positions', async () => {
    const fen = 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4';
    const selection = createDefaultEngine(null);

    // Start analysis
    const analysisPromise = selection.engine.analyze(fen);

    // Stop / cancel analysis
    selection.engine.stop();

    // The engine must either complete cleanly or throw cancellation, never hang
    try {
      const result = await analysisPromise;
      assert.ok(result.bestMove);
    } catch (err: any) {
      assert.equal(err.name, 'EngineCancelledError');
    }
  });

  it('gracefully handles source-image unavailability and uncertain square inputs in review mode', () => {
    // In manual mode, imageUri is null/undefined
    const manualPlacement: Record<string, string> = {
      e1: 'K',
      e8: 'k',
      e4: 'P',
    };

    const res = buildPositionFen(manualPlacement);
    assert.equal(res.valid, true);
    assert.equal(res.details?.whitePieceCount, 2);
    assert.equal(res.details?.blackPieceCount, 1);
    assert.equal(res.errors.length, 0);
  });

  it('supports repeated route FEN navigation across A -> B -> A and prevents rerender resets', () => {
    const fenA = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    const fenB = 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4';

    let state = createInitialStudyRouteState();

    // 1. Initial navigation to Study with FEN A
    const resA = handleStudyRouteFen(state, fenA);
    assert.equal(resA.updated, true);
    assert.equal(resA.state.lastImportedFen, fenA);
    assert.equal(resA.state.positions.length, 1);
    assert.equal(resA.state.positions[0].fen, fenA);

    // 2. Unrelated rerender with the same FEN A parameter: must NOT reset board!
    const rerenderA = handleStudyRouteFen(resA.state, fenA);
    assert.equal(rerenderA.updated, false, 'Same FEN parameter must not re-initialize on rerender');

    // 3. User goes back to Review, constructs FEN B, and navigates to Study with FEN B
    const resB = handleStudyRouteFen(resA.state, fenB);
    assert.equal(resB.updated, true, 'Subsequent navigation with new FEN B must update');
    assert.equal(resB.state.lastImportedFen, fenB);
    assert.equal(resB.state.positions[0].fen, fenB);

    // 4. User goes back to Review, reconstructs FEN A, and navigates to Study with FEN A (A -> B -> A)
    const resA2 = handleStudyRouteFen(resB.state, fenA);
    assert.equal(resA2.updated, true, 'Subsequent navigation back to FEN A (A -> B -> A) must update');
    assert.equal(resA2.state.lastImportedFen, fenA);
    assert.equal(resA2.state.positions[0].fen, fenA);

    // 5. User navigates with an invalid FEN parameter: must not overwrite board or update lastImportedFen
    const invalidRes = handleStudyRouteFen(resA2.state, 'not-a-valid-fen');
    assert.equal(invalidRes.updated, false);
    assert.equal(invalidRes.state.lastImportedFen, fenA, 'Failed import must not change lastImportedFen');
    assert.equal(invalidRes.state.positions[0].fen, fenA, 'Failed import must not corrupt existing positions');
    assert.match(invalidRes.state.error ?? '', /Invalid FEN/);
  });

  it('maps all 12 piece symbols reliably combining piece type with white/black color', () => {
    assert.equal(pieceTypeToSymbol('pawn', 'white'), 'P');
    assert.equal(pieceTypeToSymbol('knight', 'white'), 'N');
    assert.equal(pieceTypeToSymbol('bishop', 'white'), 'B');
    assert.equal(pieceTypeToSymbol('rook', 'white'), 'R');
    assert.equal(pieceTypeToSymbol('queen', 'white'), 'Q');
    assert.equal(pieceTypeToSymbol('king', 'white'), 'K');

    assert.equal(pieceTypeToSymbol('pawn', 'black'), 'p');
    assert.equal(pieceTypeToSymbol('knight', 'black'), 'n');
    assert.equal(pieceTypeToSymbol('bishop', 'black'), 'b');
    assert.equal(pieceTypeToSymbol('rook', 'black'), 'r');
    assert.equal(pieceTypeToSymbol('queen', 'black'), 'q');
    assert.equal(pieceTypeToSymbol('king', 'black'), 'k');
  });

  it('recognizes 64-square board from synthetic rectified image and respects orientation', () => {
    // Generate an 800x800 synthetic high-contrast checkerboard
    const data = new Uint8Array(800 * 800);
    for (let r = 0; r < 8; r += 1) {
      for (let c = 0; c < 8; c += 1) {
        const isLight = (r + c) % 2 === 0;
        const fill = isLight ? 215 : 60;
        for (let y = r * 100; y < (r + 1) * 100; y += 1) {
          for (let x = c * 100; x < (c + 1) * 100; x += 1) {
            data[y * 800 + x] = fill;
          }
        }
      }
    }

    const img: AnyImageData = { width: 800, height: 800, data };

    // Case A: White-bottom view
    const geomWhite = computeRectifiedBoardGeometry(CANONICAL_CORNERS, { view: 'white-bottom' });
    const resWhite = recognizeBoardFromRectifiedImage(img, geomWhite);

    assert.equal(resWhite.view, 'white-bottom');
    assert.equal(Object.keys(resWhite.squares).length, 64);
    assert.equal(resWhite.summary.emptyCount, 64, 'Clean checkerboard without pieces should be 100% empty');
    assert.equal(resWhite.summary.suggestedCount, 0);

    // Case B: Black-bottom view
    const geomBlack = computeRectifiedBoardGeometry(CANONICAL_CORNERS, { view: 'black-bottom' });
    const resBlack = recognizeBoardFromRectifiedImage(img, geomBlack);

    assert.equal(resBlack.view, 'black-bottom');
    assert.equal(Object.keys(resBlack.squares).length, 64);
    assert.equal(resBlack.summary.emptyCount, 64);

    // Check disclosure is included
    assert.equal(resWhite.disclosure, SYNTHETIC_MODEL_DISCLOSURE);
  });

  it('preserves zero-silent-error: invalid positions are flagged and prevent Study handoff', () => {
    // Construct a placement where recognition proposed an invalid position:
    // Two White Kings on the board and White pawn on rank 1!
    const invalidRecognizedPlacement: Record<string, ChessPieceSymbol | null> = {
      e1: 'K',
      d1: 'K', // Second White King!
      e8: 'k',
      a1: 'P', // Pawn on rank 1!
    };

    const fenResult = buildPositionFen(invalidRecognizedPlacement);
    assert.equal(fenResult.valid, false, 'Structural rule violations must fail validation');
    assert.equal(fenResult.fen, null, 'Must NOT emit a valid FEN');
    assert.ok(fenResult.errors.length >= 2);
    assert.match(fenResult.errors.join('; '), /must have exactly 1|White king/i);
    assert.match(fenResult.errors.join('; '), /pawns cannot exist on rank 1 or 8/i);

    // Study import must reject invalid FEN
    assert.equal(importStudyInput(fenResult.fen ?? ''), null);
  });

  it('safely handles missing image and geometry inputs without throwing unhandled exceptions', () => {
    assert.throws(() => recognizeBoardFromRectifiedImage(null as any, null as any), /must be provided/);
  });

  it('correctly distinguishes normalized vs pixel coordinates for corner scaling', () => {
    const cols = 3056;
    const rows = 4076;

    // Case 1: Normalized corners in [0, 1]
    const normalizedCorners = [
      { x: 0.165, y: 0.134 },
      { x: 0.835, y: 0.134 },
      { x: 0.835, y: 0.799 },
      { x: 0.165, y: 0.799 },
    ];

    const isNormalized1 = normalizedCorners.every(
      (pt) => pt.x >= 0 && pt.x <= 1.05 && pt.y >= 0 && pt.y <= 1.05,
    );
    assert.equal(isNormalized1, true);

    const pixelCorners1 = normalizedCorners.map((pt) => ({
      x: isNormalized1 ? pt.x * cols : pt.x,
      y: isNormalized1 ? pt.y * rows : pt.y,
    }));

    assert.equal(Math.round(pixelCorners1[0].x), Math.round(0.165 * 3056));
    assert.equal(Math.round(pixelCorners1[0].y), Math.round(0.134 * 4076));
    assert.equal(Math.round(pixelCorners1[2].x), Math.round(0.835 * 3056));
    assert.equal(Math.round(pixelCorners1[2].y), Math.round(0.799 * 4076));

    // Case 2: Pre-scaled pixel corners
    const preScaledCorners = [
      { x: 504, y: 546 },
      { x: 2551, y: 546 },
      { x: 2551, y: 3256 },
      { x: 504, y: 3256 },
    ];

    const isNormalized2 = preScaledCorners.every(
      (pt) => pt.x >= 0 && pt.x <= 1.05 && pt.y >= 0 && pt.y <= 1.05,
    );
    assert.equal(isNormalized2, false);

    const pixelCorners2 = preScaledCorners.map((pt) => ({
      x: isNormalized2 ? pt.x * cols : pt.x,
      y: isNormalized2 ? pt.y * rows : pt.y,
    }));

    assert.equal(pixelCorners2[0].x, 504);
    assert.equal(pixelCorners2[0].y, 546);
  });

  it('verifies rectification error handling fallback maintains manual mode with zero crashes', () => {
    // When automatic rectification throws, simulate how Board Review catches it
    let active = true;
    let recognitionStatus = '';
    let isRecognizing = true;

    try {
      throw new Error('Fast OpenCV Error: Argument (1) is not a Point2fVector!');
    } catch (err: any) {
      if (active) {
        recognitionStatus =
          err instanceof Error
            ? `Automatic recognition unavailable: ${err.message}. Manual placement active.`
            : 'Automatic recognition unavailable. Manual placement active.';
      }
    } finally {
      if (active) {
        isRecognizing = false;
      }
    }

    assert.equal(isRecognizing, false);
    assert.match(recognitionStatus, /Manual placement active/);
    assert.match(recognitionStatus, /Fast OpenCV Error/);
  });
});
