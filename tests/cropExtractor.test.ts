/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  extractSquareCropsFromBoard,
  parsePlacementToSquareMap,
} from '../src/vision/cropExtractor.ts';
import type { GrayscaleImage } from '../src/vision/boardEvidence.ts';
import type { RgbImage } from '../src/vision/types.ts';

describe('offline square crop extractor engine [Milestone 5.2G]', () => {
  it('1. correctly parses FEN placement to square-by-square piece occupant map', () => {
    const placement = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
    const map = parsePlacementToSquareMap(placement);

    assert.equal(map['a1'].pieceChar, 'R');
    assert.equal(map['a1'].pieceType, 'rook');
    assert.equal(map['a1'].pieceColor, 'white');

    assert.equal(map['e1'].pieceChar, 'K');
    assert.equal(map['e1'].pieceType, 'king');
    assert.equal(map['e1'].pieceColor, 'white');

    assert.equal(map['e8'].pieceChar, 'k');
    assert.equal(map['e8'].pieceType, 'king');
    assert.equal(map['e8'].pieceColor, 'black');

    assert.equal(map['d8'].pieceChar, 'q');
    assert.equal(map['d8'].pieceType, 'queen');
    assert.equal(map['d8'].pieceColor, 'black');

    assert.equal(map['e4'], undefined); // Empty square not in map
  });

  it('2. extracts exactly 64 100x100 crops from 800x800 grayscale image', () => {
    const data = new Uint8Array(800 * 800);
    // Fill square a8 (row 0, col 0: x in 0..99, y in 0..99) with value 200
    for (let y = 0; y < 100; y++) {
      for (let x = 0; x < 100; x++) {
        data[y * 800 + x] = 200;
      }
    }

    const grayImage: GrayscaleImage = {
      width: 800,
      height: 800,
      data,
    };

    const crops = extractSquareCropsFromBoard(grayImage, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR');
    assert.equal(crops.length, 64);

    // Verify first crop is a8
    const a8 = crops[0];
    assert.equal(a8.square, 'a8');
    assert.equal(a8.width, 100);
    assert.equal(a8.height, 100);
    assert.equal(a8.channels, 1);
    assert.equal(a8.data.length, 10000);
    assert.equal(a8.data[0], 200); // Has the pixel value
    assert.equal(a8.pieceChar, 'r');
    assert.equal(a8.pieceType, 'rook');
    assert.equal(a8.pieceColor, 'black');

    // Verify a1 (row 7, col 0)
    const a1 = crops.find((c) => c.square === 'a1')!;
    assert.ok(a1);
    assert.equal(a1.col, 0);
    assert.equal(a1.row, 7);
    assert.equal(a1.bounds.x, 0);
    assert.equal(a1.bounds.y, 700);
    assert.equal(a1.pieceChar, 'R');
    assert.equal(a1.pieceType, 'rook');
    assert.equal(a1.pieceColor, 'white');

    // Verify h1 (row 7, col 7)
    const h1 = crops.find((c) => c.square === 'h1')!;
    assert.ok(h1);
    assert.equal(h1.col, 7);
    assert.equal(h1.row, 7);
    assert.equal(h1.bounds.x, 700);
    assert.equal(h1.bounds.y, 700);

    // Verify e4 (empty square in starting position)
    const e4 = crops.find((c) => c.square === 'e4')!;
    assert.ok(e4);
    assert.equal(e4.pieceType, 'empty');
    assert.equal(e4.pieceColor, 'none');
    assert.equal(e4.pieceChar, null);
  });

  it('3. extracts 3-channel RGB/BGR crops with correct buffer stride', () => {
    const data = new Uint8Array(800 * 800 * 3);
    const rgbImage: RgbImage = {
      width: 800,
      height: 800,
      data,
      channels: 3,
      colorOrder: 'bgr',
    };

    const crops = extractSquareCropsFromBoard(rgbImage);
    assert.equal(crops.length, 64);
    assert.equal(crops[0].channels, 3);
    assert.equal(crops[0].data.length, 30000); // 100 * 100 * 3
  });

  it('4. rejects non-canonical input image dimensions', () => {
    const badImage: GrayscaleImage = {
      width: 600,
      height: 600,
      data: new Uint8Array(600 * 600),
    };

    assert.throws(
      () => extractSquareCropsFromBoard(badImage),
      /Cannot extract crops from non-canonical image size/
    );
  });

  it('5. correctly reverses coordinates under black-bottom orientation', () => {
    const data = new Uint8Array(800 * 800);
    const grayImage: GrayscaleImage = {
      width: 800,
      height: 800,
      data,
    };

    const crops = extractSquareCropsFromBoard(grayImage, null, 'black-bottom');
    // In black-bottom: row 0 col 0 is h1
    assert.equal(crops[0].square, 'h1');
    assert.equal(crops[0].fileIndex, 7);
    assert.equal(crops[0].rankIndex, 0);

    // Last crop (row 7 col 7) is a8
    const last = crops[63];
    assert.equal(last.square, 'a8');
    assert.equal(last.fileIndex, 0);
    assert.equal(last.rankIndex, 7);
  });
});
