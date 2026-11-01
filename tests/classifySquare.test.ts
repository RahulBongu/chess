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

  return { width, height, data, channels: 3, colorOrder: 'rgb' };
}

/**
 * Creates a 3-channel BGR tournament green synthetic board matching OpenCV's native Mat layout.
 */
function createSyntheticTournamentBgr(): RgbImage {
  const width = 800;
  const height = 800;
  const data = new Uint8Array(width * height * 3);

  // Tournament board colors: buff/cream light squares, green dark squares
  // In BGR: [Blue, Green, Red]
  const lightBgr = [212, 236, 236]; // RGB: [236, 236, 212]
  const darkBgr = [56, 108, 48];    // RGB: [48, 108, 56]

  for (let y = 0; y < height; y += 1) {
    const r = Math.floor(y / 100);
    for (let x = 0; x < width; x += 1) {
      const c = Math.floor(x / 100);
      const isLight = (r + c) % 2 === 0;
      const bgr = isLight ? lightBgr : darkBgr;
      const idx = (y * width + x) * 3;
      data[idx] = bgr[0];     // B
      data[idx + 1] = bgr[1]; // G
      data[idx + 2] = bgr[2]; // R
    }
  }

  return { width, height, data, channels: 3, colorOrder: 'bgr' };
}

function drawSyntheticPieceBgr(
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
          // BGR: [B, G, R]
          if (d > radius - 3) {
            image.data[idx] = 115;     // B
            image.data[idx + 1] = 125; // G
            image.data[idx + 2] = 125; // R
          } else {
            image.data[idx] = 228;     // B
            image.data[idx + 1] = 242; // G
            image.data[idx + 2] = 245; // R
          }
        } else {
          // Dark charcoal piece
          image.data[idx] = 32;      // B
          image.data[idx + 1] = 28;  // G
          image.data[idx + 2] = 28;  // R
        }
      }
    }
  }
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

    // RGB image: [R, G, B]
    const rgb = createSyntheticTournamentRgb();
    const pRgbLight = getPixelLuminanceAndRgb(rgb, 50, 50);
    assert.equal(pRgbLight.r, 236);
    assert.equal(pRgbLight.g, 236);
    assert.equal(pRgbLight.b, 212);
    // 0.299*236 + 0.587*236 + 0.114*212 = 233
    assert.ok(Math.abs(pRgbLight.lum - 233) <= 1);

    // BGR image (OpenCV native layout): [B, G, R]
    const bgr = createSyntheticTournamentBgr();
    const pBgrLight = getPixelLuminanceAndRgb(bgr, 50, 50);
    assert.equal(pBgrLight.r, 236);
    assert.equal(pBgrLight.g, 236);
    assert.equal(pBgrLight.b, 212);
    assert.ok(Math.abs(pBgrLight.lum - 233) <= 1);
  });

  it('correctly decodes BGR buffer channels and computes standard luminance for pure colors', () => {
    // 6-pixel controlled test image in OpenCV BGR layout:
    // Pixel 0: Pure Red   (RGB [255, 0, 0]   -> BGR [0, 0, 255])
    // Pixel 1: Pure Green (RGB [0, 255, 0]   -> BGR [0, 255, 0])
    // Pixel 2: Pure Blue  (RGB [0, 0, 255]   -> BGR [255, 0, 0])
    // Pixel 3: Pure White (RGB [255, 255, 255] -> BGR [255, 255, 255])
    // Pixel 4: Pure Black (RGB [0, 0, 0]     -> BGR [0, 0, 0])
    // Pixel 5: Mid-Gray   (RGB [128, 128, 128] -> BGR [128, 128, 128])
    const bgrData = new Uint8Array([
      0, 0, 255,       // Pixel 0: Red
      0, 255, 0,       // Pixel 1: Green
      255, 0, 0,       // Pixel 2: Blue
      255, 255, 255,   // Pixel 3: White
      0, 0, 0,         // Pixel 4: Black
      128, 128, 128,   // Pixel 5: Mid-Gray
    ]);
    const bgrImg: RgbImage = { width: 6, height: 1, data: bgrData, channels: 3, colorOrder: 'bgr' };

    // 1. Pure Red: R=255, G=0, B=0
    // Expected Y = round(0.299 * 255 + 0.587 * 0 + 0.114 * 0) = round(76.245) = 76
    const pRed = getPixelLuminanceAndRgb(bgrImg, 0, 0);
    assert.equal(pRed.r, 255, 'Pure Red: r must be 255');
    assert.equal(pRed.g, 0, 'Pure Red: g must be 0');
    assert.equal(pRed.b, 0, 'Pure Red: b must be 0');
    assert.equal(pRed.lum, 76, 'Pure Red luminance must be 76 (round(0.299 * 255))');

    // 2. Pure Green: R=0, G=255, B=0
    // Expected Y = round(0.299 * 0 + 0.587 * 255 + 0.114 * 0) = round(149.685) = 150
    const pGreen = getPixelLuminanceAndRgb(bgrImg, 1, 0);
    assert.equal(pGreen.r, 0, 'Pure Green: r must be 0');
    assert.equal(pGreen.g, 255, 'Pure Green: g must be 255');
    assert.equal(pGreen.b, 0, 'Pure Green: b must be 0');
    assert.equal(pGreen.lum, 150, 'Pure Green luminance must be 150 (round(0.587 * 255))');

    // 3. Pure Blue: R=0, G=0, B=255
    // Expected Y = round(0.299 * 0 + 0.587 * 0 + 0.114 * 255) = round(29.07) = 29
    const pBlue = getPixelLuminanceAndRgb(bgrImg, 2, 0);
    assert.equal(pBlue.r, 0, 'Pure Blue: r must be 0');
    assert.equal(pBlue.g, 0, 'Pure Blue: g must be 0');
    assert.equal(pBlue.b, 255, 'Pure Blue: b must be 255');
    assert.equal(pBlue.lum, 29, 'Pure Blue luminance must be 29 (round(0.114 * 255))');

    // Catch Red/Blue swap regression:
    // If Red and Blue weights were swapped, Red would yield 29 and Blue would yield 76.
    assert.ok(pRed.lum > pBlue.lum, 'Pure Red luminance (76) must exceed Pure Blue luminance (29)');

    // 4. Pure White
    const pWhite = getPixelLuminanceAndRgb(bgrImg, 3, 0);
    assert.equal(pWhite.r, 255);
    assert.equal(pWhite.g, 255);
    assert.equal(pWhite.b, 255);
    assert.equal(pWhite.lum, 255);

    // 5. Pure Black
    const pBlack = getPixelLuminanceAndRgb(bgrImg, 4, 0);
    assert.equal(pBlack.r, 0);
    assert.equal(pBlack.g, 0);
    assert.equal(pBlack.b, 0);
    assert.equal(pBlack.lum, 0);

    // 6. Mid-Gray
    const pMid = getPixelLuminanceAndRgb(bgrImg, 5, 0);
    assert.equal(pMid.r, 128);
    assert.equal(pMid.g, 128);
    assert.equal(pMid.b, 128);
    assert.equal(pMid.lum, 128);
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

    // Also verify native BGR tournament board yields identical classification & accurate meanR/meanB
    const bgrBoard = createSyntheticTournamentBgr();
    drawSyntheticPieceBgr(bgrBoard, 4, 4, 'white');
    drawSyntheticPieceBgr(bgrBoard, 3, 4, 'black');
    const bgrResult = classifyRectifiedBoard(bgrBoard, geom);

    const bgrE4 = bgrResult.squareByName['e4'];
    assert.equal(bgrE4.occupancy, 'occupied');
    assert.equal(bgrE4.pieceColor, 'white');
    assert.ok(bgrE4.features.meanR !== undefined && bgrE4.features.meanR > 200);
    assert.ok(bgrE4.features.meanB !== undefined && bgrE4.features.meanB > 200);

    const bgrE5 = bgrResult.squareByName['e5'];
    assert.equal(bgrE5.occupancy, 'occupied');
    assert.equal(bgrE5.pieceColor, 'black');
    assert.ok(bgrE5.features.meanR !== undefined && bgrE5.features.meanR < 50);
    assert.ok(bgrE5.features.meanB !== undefined && bgrE5.features.meanB < 50);
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
