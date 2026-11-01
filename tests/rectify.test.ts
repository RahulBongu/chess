/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  CANONICAL_BOARD_SIZE,
  CANONICAL_SQUARE_SIZE,
  GRID_DIMENSION,
  assessOrientation,
  computeForwardAndInverseHomography,
  computeRectifiedBoardGeometry,
  computeReprojectionError,
  computeSquareRegions,
  enumerateCandidateCornerOrderings,
  invertSquareHomography,
  projectPointWithHomography,
  rectifyGrayscaleBuffer,
  rectifyMat,
  validateAndOrderCorners,
} from '../src/vision/rectify.ts';
import { squareToQuadrilateral } from '../src/vision/boardEvidence.ts';
import type { DetectedCorner } from '../src/vision/types.ts';
import type { GrayscaleImage } from '../src/vision/boardEvidence.ts';

// Synthetic ground-truth fixtures (labeled as synthetic tests)
const SYNTHETIC_AXIS_ALIGNED: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
  { x: 100, y: 100 },
  { x: 700, y: 100 },
  { x: 700, y: 700 },
  { x: 100, y: 700 },
];

const SYNTHETIC_FULL_BOARD: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
  { x: 0, y: 0 },
  { x: 800, y: 0 },
  { x: 800, y: 800 },
  { x: 0, y: 800 },
];

const SYNTHETIC_TRAPEZOID_TILT: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
  { x: 200, y: 100 },
  { x: 600, y: 100 },
  { x: 750, y: 700 },
  { x: 50, y: 700 },
];

const SYNTHETIC_DIAMOND_ROTATED: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
  { x: 400, y: 50 },
  { x: 750, y: 400 },
  { x: 400, y: 750 },
  { x: 50, y: 400 },
];

const SYNTHETIC_ASYMMETRIC_SHEAR: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
  { x: 120, y: 80 },
  { x: 680, y: 140 },
  { x: 620, y: 720 },
  { x: 90, y: 650 },
];

function createSyntheticCheckerboardImage(width: number, height: number): GrayscaleImage {
  const data = new Uint8Array(width * height);
  const cellW = width / 8;
  const cellH = height / 8;

  for (let y = 0; y < height; y += 1) {
    const r = Math.floor(y / cellH);
    for (let x = 0; x < width; x += 1) {
      const c = Math.floor(x / cellW);
      // Top-left (0,0) is light (220), top-right (0,7) is dark (40)
      const isLight = (r + c) % 2 === 0;
      data[y * width + x] = isLight ? 220 : 40;
    }
  }

  return { width, height, data };
}

function createWarpedSyntheticCheckerboardImage(
  width: number,
  height: number,
  quad: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner],
): GrayscaleImage {
  const hUnitToQuad = squareToQuadrilateral(quad);
  if (!hUnitToQuad) {
    throw new Error('Invalid quad in synthetic test image generator');
  }
  const invH = invertSquareHomography(hUnitToQuad);
  const data = new Uint8Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pt = projectPointWithHomography(invH, { x, y });
      if (pt.x >= 0 && pt.x < 1 && pt.y >= 0 && pt.y < 1) {
        const c = Math.floor(pt.x * 8);
        const r = Math.floor(pt.y * 8);
        const isLight = (r + c) % 2 === 0;
        data[y * width + x] = isLight ? 220 : 40;
      }
    }
  }

  return { width, height, data };
}

describe('perspective rectification and alignment [synthetic fixtures only]', () => {
  it('validates canonical output dimensions (800x800) and square partitioning (64 cells)', () => {
    const geometry = computeRectifiedBoardGeometry(SYNTHETIC_AXIS_ALIGNED);

    assert.equal(geometry.canonicalSize.width, CANONICAL_BOARD_SIZE);
    assert.equal(geometry.canonicalSize.height, CANONICAL_BOARD_SIZE);
    assert.equal(geometry.squares.length, 64);

    // Each cell must be exactly 100x100
    for (const sq of geometry.squares) {
      assert.equal(sq.bounds.width, CANONICAL_SQUARE_SIZE);
      assert.equal(sq.bounds.height, CANONICAL_SQUARE_SIZE);
      assert.equal(sq.bounds.x, sq.gridCol * CANONICAL_SQUARE_SIZE);
      assert.equal(sq.bounds.y, sq.gridRow * CANONICAL_SQUARE_SIZE);
      assert.equal(sq.center.x, sq.bounds.x + 50);
      assert.equal(sq.center.y, sq.bounds.y + 50);
    }
  });

  it('computes forward and inverse homographies with reprojection error < 3.0 px across synthetic cases', () => {
    const testCases = [
      { name: 'axis-aligned', corners: SYNTHETIC_AXIS_ALIGNED },
      { name: 'trapezoid-tilt', corners: SYNTHETIC_TRAPEZOID_TILT },
      { name: 'diamond-rotated', corners: SYNTHETIC_DIAMOND_ROTATED },
      { name: 'asymmetric-shear', corners: SYNTHETIC_ASYMMETRIC_SHEAR },
    ];

    for (const { name, corners } of testCases) {
      const geometry = computeRectifiedBoardGeometry(corners);
      assert.ok(
        geometry.reprojectionErrorPx < 3.0,
        `Reprojection error for ${name} must be < 3.0 px, got ${geometry.reprojectionErrorPx} px`,
      );
      // In exact synthetic cases, error should be near numerical epsilon
      assert.ok(
        geometry.reprojectionErrorPx < 1e-4,
        `Synthetic exact case ${name} should have near-zero reprojection error, got ${geometry.reprojectionErrorPx}`,
      );

      // Verify each source corner projects to the exact matching canonical corner
      for (let i = 0; i < 4; i += 1) {
        const canonical = projectPointWithHomography(geometry.forwardHomography, geometry.sourceCorners[i]);
        assert.ok(Math.abs(canonical.x - geometry.canonicalCorners[i].x) < 0.1);
        assert.ok(Math.abs(canonical.y - geometry.canonicalCorners[i].y) < 0.1);

        const source = projectPointWithHomography(geometry.inverseHomography, geometry.canonicalCorners[i]);
        assert.ok(Math.abs(source.x - geometry.sourceCorners[i].x) < 0.1);
        assert.ok(Math.abs(source.y - geometry.sourceCorners[i].y) < 0.1);
      }
    }
  });

  it('normalizes corner orderings regardless of cyclic permutation or input ordering', () => {
    const p0 = { x: 100, y: 100 };
    const p1 = { x: 700, y: 100 };
    const p2 = { x: 700, y: 700 };
    const p3 = { x: 100, y: 700 };

    // Pass in scrambled order
    const scrambled = [p2, p0, p3, p1];
    const ordered = validateAndOrderCorners(scrambled);

    assert.deepEqual(ordered, [p0, p1, p2, p3]);

    // Enumerate candidate cyclic orderings
    const candidates = enumerateCandidateCornerOrderings(scrambled);
    assert.equal(candidates.length, 4);
    assert.deepEqual(candidates[0], [p0, p1, p2, p3]);
    assert.deepEqual(candidates[1], [p1, p2, p3, p0]);
    assert.deepEqual(candidates[2], [p2, p3, p0, p1]);
    assert.deepEqual(candidates[3], [p3, p0, p1, p2]);
  });

  it('rejects invalid, collinear, or non-convex candidate corners', () => {
    assert.throws(() => validateAndOrderCorners([]), /Expected exactly 4 corners/);
    assert.throws(
      () =>
        validateAndOrderCorners([
          { x: 10, y: 10 },
          { x: 20, y: 20 },
          { x: 30, y: 30 },
          { x: 40, y: 40 },
        ]),
      /do not form a strictly convex quadrilateral/,
    );
    // Concave arrowhead: point (30, 30) indents into the triangle (0,0)-(100,0)-(0,100)
    assert.throws(
      () =>
        validateAndOrderCorners([
          { x: 0, y: 0 },
          { x: 100, y: 0 },
          { x: 30, y: 30 },
          { x: 0, y: 100 },
        ]),
      /do not form a strictly convex quadrilateral/,
    );
  });

  it('correctly maps 8x8 algebraic coordinates under White-bottom and Black-bottom orientations', () => {
    const whiteGeometry = computeRectifiedBoardGeometry(SYNTHETIC_AXIS_ALIGNED, { view: 'white-bottom' });

    // Under White-bottom:
    // a8 = top-left (col 0, row 0)
    // h8 = top-right (col 7, row 0)
    // a1 = bottom-left (col 0, row 7)
    // h1 = bottom-right (col 7, row 7)
    assert.equal(whiteGeometry.squareByName['a8'].gridCol, 0);
    assert.equal(whiteGeometry.squareByName['a8'].gridRow, 0);
    assert.equal(whiteGeometry.squareByName['h8'].gridCol, 7);
    assert.equal(whiteGeometry.squareByName['h8'].gridRow, 0);
    assert.equal(whiteGeometry.squareByName['a1'].gridCol, 0);
    assert.equal(whiteGeometry.squareByName['a1'].gridRow, 7);
    assert.equal(whiteGeometry.squareByName['h1'].gridCol, 7);
    assert.equal(whiteGeometry.squareByName['h1'].gridRow, 7);

    // e4 and d5
    assert.equal(whiteGeometry.squareByName['e4'].gridCol, 4);
    assert.equal(whiteGeometry.squareByName['e4'].gridRow, 4);
    assert.equal(whiteGeometry.squareByName['d5'].gridCol, 3);
    assert.equal(whiteGeometry.squareByName['d5'].gridRow, 3);

    // Parity: h1 is light, a8 is light, a1 is dark, h8 is dark
    assert.equal(whiteGeometry.squareByName['h1'].isLight, true);
    assert.equal(whiteGeometry.squareByName['a8'].isLight, true);
    assert.equal(whiteGeometry.squareByName['a1'].isLight, false);
    assert.equal(whiteGeometry.squareByName['h8'].isLight, false);

    const blackGeometry = computeRectifiedBoardGeometry(SYNTHETIC_AXIS_ALIGNED, { view: 'black-bottom' });

    // Under Black-bottom:
    // h1 = top-left (col 0, row 0)
    // a1 = top-right (col 7, row 0)
    // h8 = bottom-left (col 0, row 7)
    // a8 = bottom-right (col 7, row 7)
    assert.equal(blackGeometry.squareByName['h1'].gridCol, 0);
    assert.equal(blackGeometry.squareByName['h1'].gridRow, 0);
    assert.equal(blackGeometry.squareByName['a1'].gridCol, 7);
    assert.equal(blackGeometry.squareByName['a1'].gridRow, 0);
    assert.equal(blackGeometry.squareByName['h8'].gridCol, 0);
    assert.equal(blackGeometry.squareByName['h8'].gridRow, 7);
    assert.equal(blackGeometry.squareByName['a8'].gridCol, 7);
    assert.equal(blackGeometry.squareByName['a8'].gridRow, 7);

    // In both views, bottom-right square is light (h1 in White view, a8 in Black view)
    assert.equal(blackGeometry.squareByName['a8'].isLight, true);
    assert.equal(blackGeometry.squareByName['h1'].isLight, true);
  });

  it('surfaces orientation ambiguity and notes that parity cannot distinguish White vs Black perspective', () => {
    const geometry = computeRectifiedBoardGeometry(SYNTHETIC_AXIS_ALIGNED, { view: 'white-bottom' });

    assert.equal(geometry.orientation.view, 'white-bottom');
    assert.match(
      geometry.orientation.ambiguityWarning,
      /Square parity confirms bottom-right is a light square, but does NOT distinguish whether White or Black is at the bottom/,
    );
  });

  it('evaluates parity consistency on synthetic checkerboard image [synthetic validation]', () => {
    const syntheticImg = createSyntheticCheckerboardImage(800, 800);
    const assessment = assessOrientation(SYNTHETIC_FULL_BOARD, syntheticImg, 'white-bottom');

    assert.equal(assessment.parityConsistency, 'consistent');

    // If rotated 90 degrees, bottom-right square parity flips to inconsistent
    const rotated90: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
      SYNTHETIC_FULL_BOARD[1],
      SYNTHETIC_FULL_BOARD[2],
      SYNTHETIC_FULL_BOARD[3],
      SYNTHETIC_FULL_BOARD[0],
    ];
    const assessmentRotated = assessOrientation(rotated90, syntheticImg, 'white-bottom');
    assert.equal(assessmentRotated.parityConsistency, 'inconsistent');
    assert.match(assessmentRotated.ambiguityWarning, /inconsistent with standard chess board orientation/);
  });

  it('performs pure TypeScript pixel-level backward perspective warp to 800x800 [synthetic image]', () => {
    // Generate synthetic image with a trapezoid-tilted 8x8 checkerboard
    const warpedSynthetic = createWarpedSyntheticCheckerboardImage(800, 800, SYNTHETIC_TRAPEZOID_TILT);
    const rectified = rectifyGrayscaleBuffer(warpedSynthetic, SYNTHETIC_TRAPEZOID_TILT);

    assert.equal(rectified.width, 800);
    assert.equal(rectified.height, 800);
    assert.equal(rectified.data.length, 800 * 800);

    // In the rectified image, sample the center of the 4 corner squares (each cell is 100x100)
    // Top-left square center (x=50, y=50) -> should be light (> 150)
    const tlSample = rectified.data[50 * 800 + 50];
    assert.ok(tlSample > 150, `Top-left square should be light, got brightness ${tlSample}`);

    // Top-right square center (x=750, y=50) -> should be dark (< 100)
    const trSample = rectified.data[50 * 800 + 750];
    assert.ok(trSample < 100, `Top-right square should be dark, got brightness ${trSample}`);

    // Bottom-right square center (x=750, y=750) -> should be light (> 150)
    const brSample = rectified.data[750 * 800 + 750];
    assert.ok(brSample > 150, `Bottom-right square should be light, got brightness ${brSample}`);

    // Bottom-left square center (x=50, y=750) -> should be dark (< 100)
    const blSample = rectified.data[750 * 800 + 50];
    assert.ok(blSample < 100, `Bottom-left square should be dark, got brightness ${blSample}`);
  });

  it('handles OpenCV Mat lifecycle cleanly and fails gracefully when OpenCV is unavailable', () => {
    assert.throws(
      () => rectifyMat({} as any, SYNTHETIC_AXIS_ALIGNED),
      /OpenCV native module is not available in this environment/,
    );
  });
});
