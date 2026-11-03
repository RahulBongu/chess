/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { sampleSquareObservations } from '../src/vision/squareSampler.ts';
import type { GrayscaleImage } from '../src/vision/boardEvidence.ts';
import type { Point2D } from '../src/vision/perspectiveOverlay.ts';

function createSyntheticBoardImage(width = 400, height = 400): GrayscaleImage {
  const data = new Uint8Array(width * height);
  const squareSize = width / 8;

  for (let y = 0; y < height; y += 1) {
    const row = Math.floor(y / squareSize);
    for (let x = 0; x < width; x += 1) {
      const col = Math.floor(x / squareSize);
      // Alternating checkerboard background: 160 (light) and 100 (dark)
      const isLightSquare = (row + col) % 2 === 0;
      let pixelVal = isLightSquare ? 160 : 100;

      // Add a bright white piece in the center of e4 (col 4, row 4)
      if (col === 4 && row === 4) {
        const dx = (x % squareSize) - squareSize / 2;
        const dy = (y % squareSize) - squareSize / 2;
        if (dx * dx + dy * dy < (squareSize * 0.28) ** 2) {
          pixelVal = 230; // White piece
        }
      }

      // Add a dark black piece in the center of d5 (col 3, row 3)
      if (col === 3 && row === 3) {
        const dx = (x % squareSize) - squareSize / 2;
        const dy = (y % squareSize) - squareSize / 2;
        if (dx * dx + dy * dy < (squareSize * 0.28) ** 2) {
          pixelVal = 40; // Black piece
        }
      }

      data[y * width + x] = pixelVal;
    }
  }

  return { width, height, data };
}

const CORNERS: Point2D[] = [
  { x: 0, y: 0 },
  { x: 400, y: 0 },
  { x: 400, y: 400 },
  { x: 0, y: 400 },
];

test('sampleSquareObservations rejects null, small or invalid images', () => {
  assert.equal(sampleSquareObservations({ width: 0, height: 0, data: new Uint8Array() }, CORNERS), null);
  assert.equal(sampleSquareObservations(createSyntheticBoardImage(), []), null);
});

test('sampleSquareObservations emits exactly 64 square observations', () => {
  const img = createSyntheticBoardImage();
  const obs = sampleSquareObservations(img, CORNERS, 'white_bottom');
  assert.ok(obs !== null);
  assert.equal(obs.length, 64);
});

test('sampleSquareObservations detects white and black pieces on synthetic board', () => {
  const img = createSyntheticBoardImage();
  const obs = sampleSquareObservations(img, CORNERS, 'white_bottom')!;
  assert.ok(obs !== null);

  // In white_bottom, col 4, row 4 is 'e4'
  const e4 = obs.find((o) => o.square === 'e4');
  assert.ok(e4 !== undefined);
  assert.equal(e4.occupied, true);
  assert.equal(e4.color, 'w');
  assert.ok(e4.luminance !== undefined && e4.luminance > 180);

  // In white_bottom, col 3, row 3 is 'd5'
  const d5 = obs.find((o) => o.square === 'd5');
  assert.ok(d5 !== undefined);
  assert.equal(d5.occupied, true);
  assert.equal(d5.color, 'b');
  assert.ok(d5.luminance !== undefined && d5.luminance < 110);

  // An empty square like a1 should be detected as unoccupied
  const a1 = obs.find((o) => o.square === 'a1');
  assert.ok(a1 !== undefined);
  assert.equal(a1.occupied, false);
});

test('sampleSquareObservations respects black_bottom orientation mapping', () => {
  const img = createSyntheticBoardImage();
  // Under black_bottom, col 4, row 4 is 'd5' (7 - 4 = 3 -> 'd', 4 + 1 = 5)
  const obsBlack = sampleSquareObservations(img, CORNERS, 'black_bottom')!;
  assert.ok(obsBlack !== null);

  const d5 = obsBlack.find((o) => o.square === 'd5');
  assert.ok(d5 !== undefined);
  // The piece at col 4, row 4 was bright white (230), now mapped to d5 under black_bottom
  assert.equal(d5.occupied, true);
  assert.equal(d5.color, 'w');
});
