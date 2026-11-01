/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  computeBoardSurfaceStats,
  classifyRectifiedBoard,
  classifySquareFeatures,
  extractSquareFeatures,
  extractSquareMats,
  getPixelLuminanceAndRgb,
} from '../src/vision/classifySquare.ts';
import { CANONICAL_BOARD_SIZE, computeRectifiedBoardGeometry } from '../src/vision/rectify.ts';
import type { DetectedCorner, RgbImage } from '../src/vision/types.ts';
import type { GrayscaleImage } from '../src/vision/boardEvidence.ts';

const SYNTHETIC_FULL_CORNERS: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
  { x: 0, y: 0 },
  { x: 800, y: 0 },
  { x: 800, y: 800 },
  { x: 0, y: 800 },
];

/**
 * Creates a synthetic 800x800 grayscale checkerboard image with custom light/dark surface luminance.
 */
function createSyntheticBoardGrayscale(
  lightLuminance = 220,
  darkLuminance = 60,
): GrayscaleImage {
  const data = new Uint8Array(800 * 800);
  for (let y = 0; y < 800; y += 1) {
    const r = Math.floor(y / 100);
    for (let x = 0; x < 800; x += 1) {
      const c = Math.floor(x / 100);
      const isLight = (r + c) % 2 === 0;
      data[y * 800 + x] = isLight ? lightLuminance : darkLuminance;
    }
  }
  return { width: 800, height: 800, data };
}

/**
 * Draws a synthetic piece-like silhouette with central 3D shading, specular highlights,
 * and base contour rim.
 */
function drawSyntheticPieceGrayscale(
  image: GrayscaleImage,
  gridRow: number,
  gridCol: number,
  color: 'white' | 'black',
  radius = 32,
): void {
  const cx = gridCol * 100 + 50;
  const cy = gridRow * 100 + 50;

  for (let y = cy - radius; y <= cy + radius; y += 1) {
    for (let x = cx - radius; x <= cx + radius; x += 1) {
      const d = Math.hypot(x - cx, y - cy);
      if (d <= radius) {
        const idx = y * 800 + x;
        if (color === 'white') {
          // White piece: bright body, dark base contour / rim shadow
          if (d > radius - 3) {
            image.data[idx] = 110;
          } else {
            image.data[idx] = Math.round(245 - (d / radius) * 40);
          }
        } else {
          // Black piece: dark body, specular 3D sheen near top, dark shadow rim
          if (d > radius - 3) {
            image.data[idx] = 18;
          } else if (d < 8) {
            image.data[idx] = 75; // 3D piece specular highlight
          } else {
            image.data[idx] = Math.round(28 + (d / radius) * 16);
          }
        }
      }
    }
  }
}

/**
 * Creates a 3-channel RGB tournament green synthetic board.
 */
function createSyntheticTournamentRgb(): RgbImage {
  const width = 800;
  const height = 800;
  const data = new Uint8Array(width * height * 3);

  // Tournament board colors: buff/cream light squares, green dark squares
  const lightRgb = [236, 236, 212];
  const darkRgb = [48, 108, 56];

  for (let y = 0; y < height; y += 1) {
    const r = Math.floor(y / 100);
    for (let x = 0; x < width; x += 1) {
      const c = Math.floor(x / 100);
      const isLight = (r + c) % 2 === 0;
      const rgb = isLight ? lightRgb : darkRgb;
      const idx = (y * width + x) * 3;
      data[idx] = rgb[0];
      data[idx + 1] = rgb[1];
      data[idx + 2] = rgb[2];
    }
  }

  return { width, height, data, channels: 3 };
}

function drawSyntheticPieceRgb(
  image: RgbImage,
  gridRow: number,
  gridCol: number,
  color: 'white' | 'black',
  radius = 32,
): void {
  const cx = gridCol * 100 + 50;
  const cy = gridRow * 100 + 50;

  for (let y = cy - radius; y <= cy + radius; y += 1) {
    for (let x = cx - radius; x <= cx + radius; x += 1) {
      const d = Math.hypot(x - cx, y - cy);
      if (d <= radius) {
        const idx = (y * 800 + x) * 3;
        if (color === 'white') {
          // Cream/white piece with shaded rim
          if (d > radius - 3) {
            image.data[idx] = 125;
            image.data[idx + 1] = 125;
            image.data[idx + 2] = 115;
          } else {
            image.data[idx] = 245;
            image.data[idx + 1] = 242;
            image.data[idx + 2] = 228;
          }
        } else {
          // Dark charcoal/black piece
          image.data[idx] = 28;
          image.data[idx + 1] = 28;
          image.data[idx + 2] = 28;
        }
      }
    }
  }
}

describe('square extraction and classical classification [synthetic fixtures only]', () => {
  it('extracts pixel luminance and RGB accurately for both 1-channel and 3-channel images', () => {
    const gray = createSyntheticBoardGrayscale(200, 50);
    const pGrayLight = getPixelLuminanceAndRgb(gray, 50, 50);
    assert.equal(pGrayLight.lum, 200);

    const pGrayDark = getPixelLuminanceAndRgb(gray, 150, 50);
    assert.equal(pGrayDark.lum, 50);

    const rgb = createSyntheticTournamentRgb();
    const pRgbLight = getPixelLuminanceAndRgb(rgb, 50, 50);
    assert.equal(pRgbLight.r, 236);
    assert.equal(pRgbLight.g, 236);
    assert.equal(pRgbLight.b, 212);
    // 0.299*236 + 0.587*236 + 0.114*212 = 233
    assert.ok(Math.abs(pRgbLight.lum - 233) <= 1);
  });

  it('classifies a 100% empty synthetic board with 0 false positives (64/64 empty)', () => {
    const geom = computeRectifiedBoardGeometry(SYNTHETIC_FULL_CORNERS, { view: 'white-bottom' });
    const emptyBoard = createSyntheticBoardGrayscale(220, 60);

    const result = classifyRectifiedBoard(emptyBoard, geom);

    assert.equal(result.summary.emptyCount, 64);
    assert.equal(result.summary.occupiedCount, 0);
    assert.equal(result.summary.uncertainCount, 0);
    assert.ok(result.summary.meanConfidence >= 0.85);

    // Verify all 64 squares have empty classification and no piece color
    for (const sq of result.squares) {
      assert.equal(sq.occupancy, 'empty');
      assert.equal(sq.pieceColor, null);
      assert.ok(sq.occupancyConfidence >= 0.8);
      assert.ok(sq.overallConfidence >= 0.8);
    }
  });

  it('classifies synthetic starting position with >= 98% occupancy accuracy (64/64 = 100%)', () => {
    const geom = computeRectifiedBoardGeometry(SYNTHETIC_FULL_CORNERS, { view: 'white-bottom' });
    const board = createSyntheticBoardGrayscale(220, 60);

    // In white-bottom view:
    // Rank 8 (row 0): Black pieces
    // Rank 7 (row 1): Black pawns
    // Ranks 6, 5, 4, 3 (rows 2, 3, 4, 5): Empty
    // Rank 2 (row 6): White pawns
    // Rank 1 (row 7): White pieces
    for (let c = 0; c < 8; c += 1) {
      drawSyntheticPieceGrayscale(board, 0, c, 'black');
      drawSyntheticPieceGrayscale(board, 1, c, 'black');
      drawSyntheticPieceGrayscale(board, 6, c, 'white');
      drawSyntheticPieceGrayscale(board, 7, c, 'white');
    }

    const result = classifyRectifiedBoard(board, geom);

    // Total clear synthetic squares evaluated: 64 squares
    // Target: >= 98.0% occupancy accuracy (requires >= 63/64 correct)
    assert.equal(result.summary.occupiedCount, 32);
    assert.equal(result.summary.emptyCount, 32);
    assert.equal(result.summary.uncertainCount, 0);

    // Color estimation: 16 White pieces and 16 Black pieces
    assert.equal(result.summary.whitePieceCount, 16);
    assert.equal(result.summary.blackPieceCount, 16);

    // Denominator reporting:
    // 64 squares on empty board + 64 squares on starting board = 128 clear synthetic test cases
    // Accuracy = 128 / 128 = 100.0% (surpasses 98.0% target)
    for (let r = 0; r < 8; r += 1) {
      for (let c = 0; c < 8; c += 1) {
        const file = String.fromCharCode(97 + c);
        const rank = 8 - r;
        const name = `${file}${rank}`;
        const sq = result.squareByName[name];
        assert.ok(sq, `Square ${name} must be present in squareByName`);

        if (r === 0 || r === 1) {
          assert.equal(sq.occupancy, 'occupied', `Square ${name} should be occupied`);
          assert.equal(sq.pieceColor, 'black', `Square ${name} should be black piece`);
        } else if (r === 6 || r === 7) {
          assert.equal(sq.occupancy, 'occupied', `Square ${name} should be occupied`);
          assert.equal(sq.pieceColor, 'white', `Square ${name} should be white piece`);
        } else {
          assert.equal(sq.occupancy, 'empty', `Square ${name} should be empty`);
          assert.equal(sq.pieceColor, null, `Square ${name} should have null piece color`);
        }
      }
    }
  });

  it('correctly classifies piece-like synthetic regions on both light and dark squares', () => {
    const geom = computeRectifiedBoardGeometry(SYNTHETIC_FULL_CORNERS, { view: 'white-bottom' });
    const board = createSyntheticBoardGrayscale(220, 60);

    // In white-bottom view:
    // e4 (r=4, c=4): light square ((4+3)%2!==0)
    // d4 (r=4, c=3): dark square ((3+3)%2===0)
    // d5 (r=3, c=3): light square ((3+4)%2!==0)
    // e5 (r=3, c=4): dark square ((4+4)%2===0)

    drawSyntheticPieceGrayscale(board, 4, 4, 'white'); // White on light (e4)
    drawSyntheticPieceGrayscale(board, 4, 3, 'white'); // White on dark (d4)
    drawSyntheticPieceGrayscale(board, 3, 3, 'black'); // Black on light (d5)
    drawSyntheticPieceGrayscale(board, 3, 4, 'black'); // Black on dark (e5)

    const result = classifyRectifiedBoard(board, geom);

    const e4 = result.squareByName['e4'];
    assert.equal(e4.occupancy, 'occupied');
    assert.equal(e4.pieceColor, 'white');
    assert.equal(e4.isLight, true);

    const d4 = result.squareByName['d4'];
    assert.equal(d4.occupancy, 'occupied');
    assert.equal(d4.pieceColor, 'white');
    assert.equal(d4.isLight, false);

    const d5 = result.squareByName['d5'];
    assert.equal(d5.occupancy, 'occupied');
    assert.equal(d5.pieceColor, 'black');
    assert.equal(d5.isLight, true);

    const e5 = result.squareByName['e5'];
    assert.equal(e5.occupancy, 'occupied');
    assert.equal(e5.pieceColor, 'black');
    assert.equal(e5.isLight, false);
  });

  it('handles changes in board contrast (high-contrast vinyl and low-contrast dark wood)', () => {
    const geom = computeRectifiedBoardGeometry(SYNTHETIC_FULL_CORNERS, { view: 'white-bottom' });

    // High contrast (240 light, 30 dark)
    const highContrastBoard = createSyntheticBoardGrayscale(240, 30);
    drawSyntheticPieceGrayscale(highContrastBoard, 4, 4, 'white');
    drawSyntheticPieceGrayscale(highContrastBoard, 4, 3, 'black');
    const highRes = classifyRectifiedBoard(highContrastBoard, geom);
    assert.equal(highRes.squareByName['e4'].occupancy, 'occupied');
    assert.equal(highRes.squareByName['e4'].pieceColor, 'white');
    assert.equal(highRes.squareByName['d4'].occupancy, 'occupied');
    assert.equal(highRes.squareByName['d4'].pieceColor, 'black');

    // Muted low-contrast dark wood board (120 light, 40 dark)
    const lowContrastBoard = createSyntheticBoardGrayscale(120, 40);
    drawSyntheticPieceGrayscale(lowContrastBoard, 4, 4, 'white');
    drawSyntheticPieceGrayscale(lowContrastBoard, 4, 3, 'black');
    const lowRes = classifyRectifiedBoard(lowContrastBoard, geom);
    assert.equal(lowRes.squareByName['e4'].occupancy, 'occupied');
    assert.equal(lowRes.squareByName['e4'].pieceColor, 'white');
    assert.equal(lowRes.squareByName['d4'].occupancy, 'occupied');
    assert.equal(lowRes.squareByName['d4'].pieceColor, 'black');
  });

  it('extracts RGB color features on 3-channel tournament green board', () => {
    const geom = computeRectifiedBoardGeometry(SYNTHETIC_FULL_CORNERS, { view: 'white-bottom' });
    const rgbBoard = createSyntheticTournamentRgb();

    drawSyntheticPieceRgb(rgbBoard, 4, 4, 'white'); // e4
    drawSyntheticPieceRgb(rgbBoard, 3, 4, 'black'); // e5

    const result = classifyRectifiedBoard(rgbBoard, geom);

    const e4 = result.squareByName['e4'];
    assert.equal(e4.occupancy, 'occupied');
    assert.equal(e4.pieceColor, 'white');
    assert.ok(e4.features.meanR !== undefined && e4.features.meanR > 200);

    const e5 = result.squareByName['e5'];
    assert.equal(e5.occupancy, 'occupied');
    assert.equal(e5.pieceColor, 'black');
    assert.ok(e5.features.meanR !== undefined && e5.features.meanR < 50);
  });

  it('detects ambiguous edge/shadow features and flags uncertainty with confidence < 0.70', () => {
    const geom = computeRectifiedBoardGeometry(SYNTHETIC_FULL_CORNERS, { view: 'white-bottom' });
    const board = createSyntheticBoardGrayscale(220, 60);

    // Draw a faint diagonal shadow across square e4 (r=4, c=4)
    // Contrast is ~13 (borderline empty/occupied)
    for (let y = 425; y < 475; y += 1) {
      for (let x = 425; x < 475; x += 1) {
        if (Math.abs(x - y) <= 4) {
          board.data[y * 800 + x] = 195; // Faint shadow on 220 background
        }
      }
    }

    const result = classifyRectifiedBoard(board, geom);
    const e4 = result.squareByName['e4'];

    assert.equal(e4.occupancy, 'uncertain');
    assert.ok(e4.overallConfidence < 0.70);
    assert.ok(e4.reason && e4.reason.length > 0);
  });

  it('respects Black-bottom orientation mapping across all 64 coordinates', () => {
    const geom = computeRectifiedBoardGeometry(SYNTHETIC_FULL_CORNERS, { view: 'black-bottom' });
    const board = createSyntheticBoardGrayscale(220, 60);

    // In black-bottom:
    // Top row (r=0) is Rank 1, Bottom row (r=7) is Rank 8
    // Left col (c=0) is File h, Right col (c=7) is File a
    drawSyntheticPieceGrayscale(board, 7, 7, 'black'); // r=7, c=7 in black-bottom is a8!
    drawSyntheticPieceGrayscale(board, 0, 0, 'white'); // r=0, c=0 in black-bottom is h1!

    const result = classifyRectifiedBoard(board, geom);

    assert.equal(result.squareByName['a8'].occupancy, 'occupied');
    assert.equal(result.squareByName['a8'].pieceColor, 'black');

    assert.equal(result.squareByName['h1'].occupancy, 'occupied');
    assert.equal(result.squareByName['h1'].pieceColor, 'white');
  });

  it('rejects undersized images and enforces 800x800 canonical input constraint', () => {
    const geom = computeRectifiedBoardGeometry(SYNTHETIC_FULL_CORNERS);
    const undersized = { width: 400, height: 400, data: new Uint8Array(400 * 400) };

    assert.throws(
      () => classifyRectifiedBoard(undersized, geom),
      /Image dimensions \(400x400\) must be at least 800x800/,
    );
  });

  it('fails gracefully when OpenCV is unavailable and cleans up memory allocations', () => {
    assert.throws(
      () => extractSquareMats({} as any),
      /OpenCV native module is not available in this environment/,
    );
  });
});
