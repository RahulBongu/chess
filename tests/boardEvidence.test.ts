/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  MIN_PATTERN_ALTERNATION,
  MIN_PATTERN_CONTRAST,
  MIN_PATTERN_FIT,
  PATTERN_GRID_SIZE,
  measureBoardPattern,
  projectSquarePoint,
  squareToQuadrilateral,
} from '../src/vision/boardEvidence.ts';
import type { GrayscaleImage, SquareHomography } from '../src/vision/boardEvidence.ts';
import {
  MIN_DETECTION_CONFIDENCE,
  combineDetectionConfidence,
  insetCorners,
  reorderCorners,
  scoreCandidate,
} from '../src/vision/boardDetection.ts';
import type { DetectedCorner, VisionSize } from '../src/vision/types.ts';

interface InverseHomography {
  m00: number;
  m01: number;
  m02: number;
  m10: number;
  m11: number;
  m12: number;
  m20: number;
  m21: number;
  m22: number;
}

function invertSquareHomography(homography: SquareHomography): InverseHomography {
  const { a, b, c, d, e, f, g, h } = homography;
  const m00 = e - f * h;
  const m01 = c * h - b;
  const m02 = b * f - c * e;
  const m10 = f * g - d;
  const m11 = a - c * g;
  const m12 = c * d - a * f;
  const m20 = d * h - e * g;
  const m21 = b * g - a * h;
  const m22 = a * e - b * d;
  const determinant = a * m00 + b * m10 + c * m20;

  assert.ok(Math.abs(determinant) > 1e-9, 'fixture homography must be invertible');
  return {
    m00: m00 / determinant,
    m01: m01 / determinant,
    m02: m02 / determinant,
    m10: m10 / determinant,
    m11: m11 / determinant,
    m12: m12 / determinant,
    m20: m20 / determinant,
    m21: m21 / determinant,
    m22: m22 / determinant,
  };
}

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

interface SceneOptions {
  width: number;
  height: number;
  corners: DetectedCorner[];
  valueAt: (u: number, v: number) => number;
  outside?: number;
  noise?: number;
  seed?: number;
}

/** Draws a quad by inverse-projecting every pixel into unit-square coordinates. */
function createSceneImage(options: SceneOptions): GrayscaleImage {
  const { width, height, corners, valueAt, outside = 0, noise = 0, seed = 1 } = options;
  const homography = squareToQuadrilateral(corners);
  assert.ok(homography, 'fixture corners must form a valid quadrilateral');
  const inverse = invertSquareHomography(homography);
  const random = createRandom(seed);
  const data = new Uint8Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pixelX = x + 0.5;
      const pixelY = y + 0.5;
      const denominator =
        inverse.m20 * pixelX + inverse.m21 * pixelY + inverse.m22;
      let value = outside;

      if (Math.abs(denominator) > 1e-9) {
        const u = (inverse.m00 * pixelX + inverse.m01 * pixelY + inverse.m02) / denominator;
        const v = (inverse.m10 * pixelX + inverse.m11 * pixelY + inverse.m12) / denominator;
        if (u >= 0 && u < 1 && v >= 0 && v < 1) {
          value = valueAt(u, v);
        }
      }

      if (noise > 0) {
        value += (random() * 2 - 1) * noise;
      }

      data[y * width + x] = Math.max(0, Math.min(255, Math.round(value)));
    }
  }

  return { width, height, data };
}

interface PieceFixture {
  row: number;
  column: number;
  shade: number;
  radius?: number;
}

interface BoardSceneOptions {
  width: number;
  height: number;
  corners: DetectedCorner[];
  light?: number;
  dark?: number;
  shadeGradient?: number;
  noise?: number;
  seed?: number;
  gridSize?: number;
  pieces?: PieceFixture[];
  insideOverride?: (u: number, v: number) => number | null;
}

function createBoardScene(options: BoardSceneOptions): GrayscaleImage {
  const {
    width,
    height,
    corners,
    light = 235,
    dark = 45,
    shadeGradient = 0,
    noise = 0,
    seed = 1,
    gridSize = PATTERN_GRID_SIZE,
    pieces = [],
    insideOverride,
  } = options;

  return createSceneImage({
    width,
    height,
    corners,
    outside: 8,
    noise,
    seed,
    valueAt: (u, v) => {
      const override = insideOverride?.(u, v);
      if (override !== null && override !== undefined) {
        return override;
      }
      const row = Math.min(gridSize - 1, Math.floor(v * gridSize));
      const column = Math.min(gridSize - 1, Math.floor(u * gridSize));
      const base = (row + column) % 2 === 0 ? light : dark;

      for (const piece of pieces) {
        const du = u * gridSize - (piece.column + 0.5);
        const dv = v * gridSize - (piece.row + 0.5);
        if (Math.hypot(du, dv) < (piece.radius ?? 0.38)) {
          return piece.shade;
        }
      }

      return base * (1 - shadeGradient * ((u + v) / 2));
    },
  });
}

function axisAlignedSquare(centerX: number, centerY: number, side: number): DetectedCorner[] {
  const half = side / 2;
  return [
    { x: centerX - half, y: centerY - half },
    { x: centerX + half, y: centerY - half },
    { x: centerX + half, y: centerY + half },
    { x: centerX - half, y: centerY + half },
  ];
}

function rotatedSquare(
  centerX: number,
  centerY: number,
  side: number,
  degrees: number,
): DetectedCorner[] {
  const half = side / 2;
  const radians = (degrees * Math.PI) / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  return [
    { x: -half, y: -half },
    { x: half, y: -half },
    { x: half, y: half },
    { x: -half, y: half },
  ].map((point) => ({
    x: centerX + point.x * cosine - point.y * sine,
    y: centerY + point.x * sine + point.y * cosine,
  }));
}

function evidenceFor(image: GrayscaleImage, corners: DetectedCorner[]) {
  return measureBoardPattern(image, reorderCorners(corners));
}

function assertBoardRejected(
  image: GrayscaleImage,
  corners: DetectedCorner[],
  label: string,
) {
  const ordered = reorderCorners(corners);
  const evidence = measureBoardPattern(image, ordered);
  const boundaryScore = scoreCandidate(ordered, {
    width: image.width,
    height: image.height,
  });
  const confidence = combineDetectionConfidence(boundaryScore, evidence);

  assert.equal(evidence.valid, false, `${label}: ${evidence.reason ?? 'accepted'}`);
  assert.equal(confidence, 0, `${label}: combined confidence must be zero`);
  return { evidence, boundaryScore, confidence };
}

function confidenceFor(image: GrayscaleImage, corners: DetectedCorner[]): number {
  const ordered = reorderCorners(corners);
  const size: VisionSize = { width: image.width, height: image.height };
  return combineDetectionConfidence(
    scoreCandidate(ordered, size),
    measureBoardPattern(image, ordered),
  );
}

test('square-to-quadrilateral mapping handles axis-aligned and rotated boards', () => {
  const identity = squareToQuadrilateral([
    { x: 0, y: 0 },
    { x: 8, y: 0 },
    { x: 8, y: 8 },
    { x: 0, y: 8 },
  ]);
  assert.ok(identity);
  assert.deepEqual(projectSquarePoint(identity, 0, 0), { x: 0, y: 0 });
  assert.deepEqual(projectSquarePoint(identity, 1, 1), { x: 8, y: 8 });
  assert.deepEqual(projectSquarePoint(identity, 0.5, 0.5), { x: 4, y: 4 });

  const rectangle = squareToQuadrilateral([
    { x: 10, y: 20 },
    { x: 110, y: 20 },
    { x: 110, y: 220 },
    { x: 10, y: 220 },
  ]);
  assert.ok(rectangle);
  assert.deepEqual(projectSquarePoint(rectangle, 0.5, 0), { x: 60, y: 20 });
  assert.deepEqual(projectSquarePoint(rectangle, 0, 1), { x: 10, y: 220 });
  assert.deepEqual(projectSquarePoint(rectangle, 0.5, 0.5), { x: 60, y: 120 });

  const diamond = squareToQuadrilateral([
    { x: 100, y: 0 },
    { x: 200, y: 100 },
    { x: 100, y: 200 },
    { x: 0, y: 100 },
  ]);
  assert.ok(diamond);
  assert.deepEqual(projectSquarePoint(diamond, 0.5, 0.5), { x: 100, y: 100 });
  assert.deepEqual(projectSquarePoint(diamond, 0.25, 0), { x: 125, y: 25 });

  // Recover a known projective transform from its four unit-square corner images.
  const known: SquareHomography = {
    a: 260,
    b: 35,
    c: 40,
    d: 15,
    e: 210,
    f: 30,
    g: 0.0009,
    h: 0.0004,
  };
  const projectiveCorners = [
    projectSquarePoint(known, 0, 0),
    projectSquarePoint(known, 1, 0),
    projectSquarePoint(known, 1, 1),
    projectSquarePoint(known, 0, 1),
  ];
  const recovered = squareToQuadrilateral(projectiveCorners);
  assert.ok(recovered);
  for (const [u, v] of [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]] as const) {
    const expected = projectSquarePoint(known, u, v);
    const actual = projectSquarePoint(recovered, u, v);
    assert.ok(Math.abs(actual.x - expected.x) < 1e-9);
    assert.ok(Math.abs(actual.y - expected.y) < 1e-9);
  }

  assert.equal(
    squareToQuadrilateral([
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 0, y: 0 },
    ]),
    null,
  );
});

test('rotated square boundary scores are comparable at 0 and 45 degrees', () => {
  const size = { width: 400, height: 400 };
  const axisAligned = axisAlignedSquare(200, 200, 200);
  const diamond = rotatedSquare(200, 200, 200, 45);

  const axisScore = scoreCandidate(axisAligned, size);
  const diamondScore = scoreCandidate(diamond, size);

  assert.ok(axisScore > 0.39);
  assert.ok(diamondScore > 0.39);
  assert.ok(Math.abs(axisScore - diamondScore) < 0.02);
});

test('dark-frame regression: a large bright quadrilateral cannot be a board', () => {
  const corners = [
    { x: 60, y: 40 },
    { x: 580, y: 50 },
    { x: 570, y: 430 },
    { x: 70, y: 420 },
  ];
  const image = createSceneImage({
    width: 640,
    height: 480,
    corners,
    outside: 4,
    noise: 2,
    seed: 7,
    // A bright, smoothly lit rectangle with no alternating squares.
    valueAt: (u, v) => 205 - 85 * u - 20 * v,
  });

  const ordered = reorderCorners(corners);
  const boundaryScore = scoreCandidate(ordered, { width: image.width, height: image.height });
  const evidence = measureBoardPattern(image, ordered);
  const confidence = combineDetectionConfidence(boundaryScore, evidence);

  assert.ok(
    boundaryScore > MIN_DETECTION_CONFIDENCE,
    `expected a strong boundary score, received ${boundaryScore.toFixed(3)}`,
  );
  assert.equal(evidence.valid, false);
  assert.equal(confidence, 0);
});

test('nearly uniform and blank scenes contain no chessboard evidence', () => {
  const uniform = createSceneImage({
    width: 320,
    height: 320,
    corners: axisAlignedSquare(160, 160, 300),
    outside: 128,
    noise: 2,
    seed: 11,
    valueAt: () => 128,
  });
  assertBoardRejected(uniform, axisAlignedSquare(160, 160, 300), 'uniform frame');

  const blank = createSceneImage({
    width: 320,
    height: 320,
    corners: axisAlignedSquare(160, 160, 300),
    outside: 24,
    noise: 1,
    seed: 12,
    valueAt: (u, v) => 40 + 30 * u + 20 * v,
  });
  assertBoardRejected(blank, axisAlignedSquare(160, 160, 300), 'blank gradient');
});

test('non-board rectangular objects are rejected despite a strong boundary', () => {
  const corners = axisAlignedSquare(160, 160, 280);
  const image = createSceneImage({
    width: 320,
    height: 320,
    corners,
    outside: 10,
    noise: 2,
    seed: 13,
    valueAt: (u, v) => 180 - 30 * v,
  });

  const result = assertBoardRejected(image, corners, 'non-board rectangle');
  assert.ok(result.boundaryScore > MIN_DETECTION_CONFIDENCE);
});

test('low-contrast 8 x 8 patterns are not accepted as boards', () => {
  const corners = axisAlignedSquare(160, 160, 280);
  const image = createBoardScene({
    width: 320,
    height: 320,
    corners,
    light: 136,
    dark: 124,
    noise: 1.5,
    seed: 17,
  });

  const result = assertBoardRejected(image, corners, 'low-contrast board');
  assert.ok(result.evidence.contrast < 20);
});

test('striped, coarse, and half-complete grids are rejected as misleading', () => {
  const corners = axisAlignedSquare(160, 160, 280);

  const stripes = createBoardScene({
    width: 320,
    height: 320,
    corners,
    seed: 19,
    insideOverride: (u) => (Math.floor(u * PATTERN_GRID_SIZE) % 2 === 0 ? 220 : 45),
  });
  assertBoardRejected(stripes, corners, 'vertical stripes');

  const coarse = createBoardScene({
    width: 320,
    height: 320,
    corners,
    light: 225,
    dark: 45,
    gridSize: 4,
    seed: 20,
  });
  assertBoardRejected(coarse, corners, '4 x 4 checkerboard');

  const sevenBySeven = createBoardScene({
    width: 320,
    height: 320,
    corners,
    light: 225,
    dark: 45,
    gridSize: 7,
    seed: 22,
  });
  assertBoardRejected(sevenBySeven, corners, '7 x 7 checkerboard');

  const halfComplete = createBoardScene({
    width: 320,
    height: 320,
    corners,
    light: 225,
    dark: 45,
    seed: 21,
    insideOverride: (u, v) => {
      if (u >= 0.5) {
        return 130;
      }
      const row = Math.floor(v * PATTERN_GRID_SIZE);
      const column = Math.floor(u * PATTERN_GRID_SIZE);
      return (row + column) % 2 === 0 ? 225 : 45;
    },
  });
  assertBoardRejected(halfComplete, corners, 'half-complete checkerboard');
});

test('valid boards are accepted across rotations', () => {
  for (const degrees of [0, 15, 45, 90, 135]) {
    const corners = rotatedSquare(180, 180, 240, degrees);
    const image = createBoardScene({
      width: 360,
      height: 360,
      corners,
      noise: 1.5,
      seed: 30 + degrees,
    });

    const ordered = reorderCorners(corners);
    const evidence = measureBoardPattern(image, ordered);
    assert.equal(evidence.valid, true, `${degrees} degrees: ${evidence.reason ?? 'accepted'}`);
    assert.ok(evidence.alternation >= MIN_PATTERN_ALTERNATION);
    assert.ok(evidence.fit >= MIN_PATTERN_FIT);
    assert.ok(confidenceFor(image, corners) > 0.7, `${degrees} degrees confidence too low`);
  }
});

test('valid boards are accepted under perspective and uneven lighting', () => {
  const corners = [
    { x: 80, y: 50 },
    { x: 560, y: 80 },
    { x: 520, y: 430 },
    { x: 110, y: 400 },
  ];
  const image = createBoardScene({
    width: 640,
    height: 480,
    corners,
    light: 235,
    dark: 50,
    shadeGradient: 0.3,
    noise: 2,
    seed: 41,
  });

  const ordered = reorderCorners(corners);
  const evidence = measureBoardPattern(image, ordered);
  assert.equal(evidence.valid, true, evidence.reason ?? 'accepted');
  assert.ok(evidence.contrast > MIN_PATTERN_CONTRAST);
  assert.ok(confidenceFor(image, corners) > 0.7);
});

test('a dim but high-contrast board is still accepted', () => {
  const corners = axisAlignedSquare(180, 180, 260);
  const image = createBoardScene({
    width: 360,
    height: 360,
    corners,
    light: 95,
    dark: 30,
    noise: 3,
    seed: 44,
  });

  const evidence = evidenceFor(image, corners);
  assert.equal(evidence.valid, true, evidence.reason ?? 'accepted');
  assert.ok(evidence.contrast > MIN_PATTERN_CONTRAST);
  assert.ok(confidenceFor(image, corners) > 0.7);
});

test('a board with pieces on the back ranks is still accepted', () => {
  const corners = axisAlignedSquare(320, 320, 560);
  const pieces: PieceFixture[] = [];
  for (let column = 0; column < 8; column += 1) {
    pieces.push({ row: 0, column, shade: 30 });
    pieces.push({ row: 1, column, shade: 30 });
    pieces.push({ row: 6, column, shade: 225 });
    pieces.push({ row: 7, column, shade: 225 });
  }
  const image = createBoardScene({
    width: 640,
    height: 640,
    corners,
    noise: 2,
    seed: 47,
    pieces,
  });

  const evidence = evidenceFor(image, corners);
  assert.equal(evidence.valid, true, evidence.reason ?? 'accepted');
  assert.ok(confidenceFor(image, corners) > 0.8);
});

test('a board with an outer border is rejected at the outer corners but accepted through inset sampling', () => {
  const corners = axisAlignedSquare(320, 320, 560);
  const image = createSceneImage({
    width: 640,
    height: 640,
    corners,
    outside: 100,
    noise: 2,
    seed: 61,
    valueAt: (u, v) => {
      if (u < 0.05 || u > 0.95 || v < 0.05 || v > 0.95) {
        return 110;
      }
      const innerU = (u - 0.05) / 0.9;
      const innerV = (v - 0.05) / 0.9;
      const row = Math.min(PATTERN_GRID_SIZE - 1, Math.floor(innerV * PATTERN_GRID_SIZE));
      const column = Math.min(PATTERN_GRID_SIZE - 1, Math.floor(innerU * PATTERN_GRID_SIZE));
      return (row + column) % 2 === 0 ? 225 : 45;
    },
  });

  const ordered = reorderCorners(corners);
  const outerEvidence = measureBoardPattern(image, ordered);
  assert.equal(outerEvidence.valid, false, 'outer border candidate must reproduce the device failure');

  const samplingCorners = insetCorners(ordered, 0.9);
  const samplingEvidence = measureBoardPattern(image, samplingCorners);
  assert.equal(samplingEvidence.valid, true, samplingEvidence.reason ?? 'accepted');
  assert.ok(samplingEvidence.fit >= MIN_PATTERN_FIT);
  assert.ok(samplingEvidence.alternation >= MIN_PATTERN_ALTERNATION);

  const size: VisionSize = { width: image.width, height: image.height };
  const confidence = combineDetectionConfidence(
    scoreCandidate(ordered, size),
    samplingEvidence,
  );
  assert.ok(confidence >= MIN_DETECTION_CONFIDENCE, `bordered board confidence ${confidence}`);
});

test('a bordered rectangle with a uniform centre is rejected at every sampling scale', () => {
  const corners = axisAlignedSquare(320, 320, 560);
  const image = createSceneImage({
    width: 640,
    height: 640,
    corners,
    outside: 100,
    noise: 1,
    seed: 62,
    valueAt: (u, v) => {
      if (u < 0.05 || u > 0.95 || v < 0.05 || v > 0.95) {
        return 110;
      }
      return 128;
    },
  });

  const ordered = reorderCorners(corners);
  for (const scale of [1, 0.97, 0.94, 0.9, 0.86, 0.82]) {
    const evidence = measureBoardPattern(image, insetCorners(ordered, scale));
    assert.equal(evidence.valid, false, `scale ${scale}: ${evidence.reason ?? 'accepted'}`);
  }
});

test('valid boards are accepted at different image scales', () => {
  for (const side of [200, 280, 320]) {
    const corners = axisAlignedSquare(200, 200, side);
    const image = createBoardScene({
      width: 400,
      height: 400,
      corners,
      noise: 1,
      seed: 50 + side,
    });

    const confidence = confidenceFor(image, corners);
    assert.ok(confidence >= MIN_DETECTION_CONFIDENCE, `${side}px board confidence ${confidence}`);
  }
});

test('a large uniform quadrilateral cannot reach high confidence even with a perfect boundary', () => {
  const corners = axisAlignedSquare(160, 160, 300);
  const image = createSceneImage({
    width: 320,
    height: 320,
    corners,
    outside: 0,
    valueAt: () => 245,
  });

  const ordered = reorderCorners(corners);
  const boundaryScore = scoreCandidate(ordered, { width: 320, height: 320 });
  const evidence = measureBoardPattern(image, ordered);

  assert.equal(boundaryScore, 1);
  assert.equal(evidence.valid, false);
  assert.equal(combineDetectionConfidence(boundaryScore, evidence), 0);
});
