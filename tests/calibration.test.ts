/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  addCalibrationPoint,
  canAddCalibrationPoint,
  isConvexQuadrilateral,
  isValidCalibration,
  reorderCalibrationPoints,
  resetCalibrationPoints,
  signedArea,
  undoCalibrationPoint,
} from '../src/calibration/geometry.ts';
import {
  computeAspectFitDisplaySize,
  normalizePreviewTouch,
  normalizedToPixelCorners,
} from '../src/calibration/coordinates.ts';

type Point = { x: number; y: number };

function getPermutations<T>(arr: T[]): T[][] {
  if (arr.length <= 1) return [arr];
  const result: T[][] = [];
  for (let index = 0; index < arr.length; index += 1) {
    const rest = [...arr.slice(0, index), ...arr.slice(index + 1)];
    for (const permutation of getPermutations(rest)) {
      result.push([arr[index], ...permutation]);
    }
  }
  return result;
}

function pointKey(point: Point): string {
  return `${point.x},${point.y}`;
}

function assertSamePointSet(expected: Point[], actual: Point[]): void {
  assert.deepEqual(
    expected.map(pointKey).sort(),
    actual.map(pointKey).sort(),
    'Reordering must preserve the exact original coordinates.',
  );
}

function assertClockwiseFromTopLeft(ordered: Point[]): void {
  assert.equal(ordered.length, 4);
  assert.ok(signedArea(ordered) > 0, 'Ordered corners must be clockwise in screen space.');

  // Deterministic start: minimum x + y, then minimum y, then minimum x.
  const expectedStart = ordered.reduce((bestIndex, point, index) => {
    const best = ordered[bestIndex];
    const pointSum = point.x + point.y;
    const bestSum = best.x + best.y;
    if (pointSum < bestSum) return index;
    if (pointSum === bestSum && point.y < best.y) return index;
    if (pointSum === bestSum && point.y === best.y && point.x < best.x) return index;
    return bestIndex;
  }, 0);

  const rotated = [...ordered.slice(expectedStart), ...ordered.slice(0, expectedStart)];
  assert.deepEqual(ordered, rotated, 'Ordered corners must start at the screen-space top-left.');

  for (let index = 0; index < 4; index += 1) {
    const a = ordered[index];
    const b = ordered[(index + 1) % 4];
    const c = ordered[(index + 2) % 4];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    assert.ok(cross > 0, `Turn ${index} must be strictly clockwise.`);
  }
}

function assertPermutationInvariance(base: Point[]): Point[] {
  assert.equal(base.length, 4);
  const permutations = getPermutations(base);
  assert.equal(permutations.length, 24);

  const reference = reorderCalibrationPoints(base);
  assertSamePointSet(base, reference);
  assertClockwiseFromTopLeft(reference);
  assert.equal(isValidCalibration(reference).valid, true);

  for (const permutation of permutations) {
    const reordered = reorderCalibrationPoints(permutation);
    assertSamePointSet(base, reordered);
    assert.deepEqual(
      reordered,
      reference,
      `Reordering must be independent of input order: ${JSON.stringify(permutation)}`,
    );
    assertClockwiseFromTopLeft(reordered);
    assert.equal(
      isValidCalibration(reordered).valid,
      true,
      `Permutation failed validation: ${JSON.stringify(permutation)}`,
    );
  }

  return reference;
}

test('empty calibration state has no points and is invalid', () => {
  const points = resetCalibrationPoints();
  assert.equal(points.length, 0);
  assert.equal(isValidCalibration(points).valid, false);
});

test('four points can be added in the expected order', () => {
  let points = resetCalibrationPoints();
  points = addCalibrationPoint(points, { x: 0.1, y: 0.1 }) ?? points;
  points = addCalibrationPoint(points, { x: 0.9, y: 0.1 }) ?? points;
  points = addCalibrationPoint(points, { x: 0.9, y: 0.9 }) ?? points;
  points = addCalibrationPoint(points, { x: 0.1, y: 0.9 }) ?? points;

  assert.equal(points.length, 4);
  assert.equal(isValidCalibration(points).valid, true);
});

test('undo removes the most recent point', () => {
  const points = [
    { x: 0.1, y: 0.1 },
    { x: 0.9, y: 0.1 },
  ];
  const undone = undoCalibrationPoint(points);
  assert.equal(undone.length, 1);
  assert.deepEqual(undone[0], points[0]);
});

test('reset removes all points', () => {
  const points = resetCalibrationPoints();
  assert.equal(points.length, 0);
});

test('duplicate points are rejected', () => {
  const points = [{ x: 0.5, y: 0.5 }];
  const result = canAddCalibrationPoint(points, { x: 0.51, y: 0.51 });
  assert.equal(result.valid, false);
  assert.equal(addCalibrationPoint(points, { x: 0.51, y: 0.51 }), null);
});

test('out-of-bounds points are rejected', () => {
  const points: { x: number; y: number }[] = [];
  const result = canAddCalibrationPoint(points, { x: 1.2, y: 0.5 });
  assert.equal(result.valid, false);
  assert.equal(canAddCalibrationPoint(points, { x: 0.5, y: -0.1 }).valid, false);
});

test('self-intersecting quadrilaterals are rejected', () => {
  const points = [
    { x: 0.1, y: 0.1 },
    { x: 0.9, y: 0.9 },
    { x: 0.9, y: 0.1 },
    { x: 0.1, y: 0.9 },
  ];
  assert.equal(isValidCalibration(points).valid, false);
});

test('a valid four-corner quadrilateral is accepted', () => {
  const points = [
    { x: 0.15, y: 0.2 },
    { x: 0.85, y: 0.18 },
    { x: 0.82, y: 0.82 },
    { x: 0.18, y: 0.84 },
  ];
  const result = isValidCalibration(points);
  assert.equal(result.valid, true);
});

test('reorderCalibrationPoints normalizes counter-clockwise or unaligned corners to valid clockwise order', () => {
  // Counter-clockwise points: top-left, bottom-left, bottom-right, top-right
  const ccwPoints = [
    { x: 0.1, y: 0.1 },
    { x: 0.1, y: 0.9 },
    { x: 0.9, y: 0.9 },
    { x: 0.9, y: 0.1 },
  ];
  // Initial CCW order fails because signedArea is negative
  assert.equal(isValidCalibration(ccwPoints).valid, false);

  // After reordering clockwise starting at top-left
  const reordered = reorderCalibrationPoints(ccwPoints);
  assert.equal(reordered.length, 4);
  assert.deepEqual(reordered[0], { x: 0.1, y: 0.1 }); // Top-left
  assert.deepEqual(reordered[1], { x: 0.9, y: 0.1 }); // Top-right
  assert.deepEqual(reordered[2], { x: 0.9, y: 0.9 }); // Bottom-right
  assert.deepEqual(reordered[3], { x: 0.1, y: 0.9 }); // Bottom-left
  assert.equal(isValidCalibration(reordered).valid, true);
});

test('all 24 permutations of a normal convex quadrilateral reorder deterministically', () => {
  assertPermutationInvariance([
    { x: 0.15, y: 0.2 },
    { x: 0.85, y: 0.18 },
    { x: 0.82, y: 0.82 },
    { x: 0.18, y: 0.84 },
  ]);
});

test('all 24 permutations of a steep perspective trapezoid reorder deterministically', () => {
  assertPermutationInvariance([
    { x: 0.35, y: 0.15 },
    { x: 0.65, y: 0.15 },
    { x: 0.95, y: 0.85 },
    { x: 0.05, y: 0.85 },
  ]);
});

test('all 24 permutations of a rotated diamond reorder deterministically', () => {
  assertPermutationInvariance([
    { x: 0.5, y: 0.1 },
    { x: 0.9, y: 0.5 },
    { x: 0.5, y: 0.9 },
    { x: 0.1, y: 0.5 },
  ]);
});

test('all 24 permutations of a sheared perspective quadrilateral reorder deterministically', () => {
  assertPermutationInvariance([
    { x: 0.12, y: 0.08 },
    { x: 0.72, y: 0.22 },
    { x: 0.86, y: 0.78 },
    { x: 0.2, y: 0.92 },
  ]);
});

test('convexity is orientation-agnostic but clockwise winding is required for confirmation', () => {
  const clockwise = [
    { x: 0.15, y: 0.2 },
    { x: 0.85, y: 0.18 },
    { x: 0.82, y: 0.82 },
    { x: 0.18, y: 0.84 },
  ];
  const counterClockwise = [...clockwise].reverse();

  assert.equal(isConvexQuadrilateral(clockwise), true);
  assert.equal(isConvexQuadrilateral(counterClockwise), true);
  assert.equal(isValidCalibration(clockwise).valid, true);

  const ccwValidation = isValidCalibration(counterClockwise);
  assert.equal(ccwValidation.valid, false);
  assert.match(ccwValidation.reason ?? '', /top-left, top-right, bottom-right, bottom-left/);

  assert.equal(isValidCalibration(reorderCalibrationPoints(counterClockwise)).valid, true);
});

test('a simple concave quadrilateral remains invalid for every permutation and after reordering', () => {
  const concave = [
    { x: 0.05, y: 0.05 },
    { x: 0.95, y: 0.05 },
    { x: 0.3, y: 0.3 },
    { x: 0.05, y: 0.95 },
  ];

  assert.ok(signedArea(concave) > 0);
  assert.equal(isConvexQuadrilateral(concave), false);
  assert.equal(isValidCalibration(concave).valid, false);

  for (const permutation of getPermutations(concave)) {
    assert.equal(isConvexQuadrilateral(permutation), false);
    const reordered = reorderCalibrationPoints(permutation);
    assertSamePointSet(concave, reordered);
    const validation = isValidCalibration(reordered);
    assert.equal(
      validation.valid,
      false,
      `Concave geometry must not become valid after reordering: ${JSON.stringify(permutation)}`,
    );
  }
});

test('nearly collinear, collinear, coincident, and tiny quadrilaterals stay invalid after reordering', () => {
  const nearlyCollinear = [
    { x: 0.1, y: 0.1 },
    { x: 0.45, y: 0.1 },
    { x: 0.8, y: 0.1 + 1e-8 },
    { x: 0.5, y: 0.8 },
  ];
  const collinear = [
    { x: 0.1, y: 0.5 },
    { x: 0.4, y: 0.5 },
    { x: 0.7, y: 0.5 },
    { x: 0.9, y: 0.5 },
  ];
  const coincident = [
    { x: 0.2, y: 0.2 },
    { x: 0.21, y: 0.21 },
    { x: 0.8, y: 0.2 },
    { x: 0.8, y: 0.8 },
  ];
  const tiny = [
    { x: 0.4, y: 0.4 },
    { x: 0.45, y: 0.4 },
    { x: 0.45, y: 0.44 },
    { x: 0.4, y: 0.44 },
  ];

  for (const geometry of [nearlyCollinear, collinear, coincident, tiny]) {
    assert.equal(isValidCalibration(geometry).valid, false);
    for (const permutation of getPermutations(geometry)) {
      assert.equal(
        isValidCalibration(reorderCalibrationPoints(permutation)).valid,
        false,
        `Invalid geometry became valid after reordering: ${JSON.stringify(permutation)}`,
      );
    }
  }
});

test('isValidCalibration detects collinear, degenerate, and coincident points correctly', () => {
  // Collinear points along horizontal line (encloses 0 area and edges overlap)
  const collinear = [
    { x: 0.1, y: 0.5 },
    { x: 0.4, y: 0.5 },
    { x: 0.7, y: 0.5 },
    { x: 0.9, y: 0.5 },
  ];
  const resCollinear = isValidCalibration(collinear);
  assert.equal(resCollinear.valid, false);
  assert.ok(resCollinear.reason && resCollinear.reason.length > 0);

  // Coincident points (duplicate distance < 0.05)
  const coincident = [
    { x: 0.2, y: 0.2 },
    { x: 0.21, y: 0.21 }, // too close
    { x: 0.8, y: 0.2 },
    { x: 0.8, y: 0.8 },
  ];
  const resCoincident = isValidCalibration(coincident);
  assert.equal(resCoincident.valid, false);
  assert.match(resCoincident.reason ?? '', /too close together/);
});

test('display normalization and aspect-ratio quantization error is strictly bounded below 0.35%', () => {
  const formats = [
    { name: '4:3 landscape', W: 4000, H: 3000 },
    { name: '3:4 portrait', W: 3000, H: 4000 },
    { name: '16:9 landscape', W: 3840, H: 2160 },
    { name: '9:16 portrait', W: 2160, H: 3840 },
    { name: '1:1 square', W: 3000, H: 3000 },
    { name: 'FENiT high-res portrait', W: 4284, H: 5712 },
  ];

  for (const { W, H } of formats) {
    for (const w_disp of [320, 360, 390, 420, 480]) {
      const { width, height } = computeAspectFitDisplaySize(W, H, w_disp);
      assert.equal(width, w_disp);
      assert.equal(height, Math.round(w_disp * (H / W)));

      const trueRatio = W / H;
      const dispRatio = width / height;
      const relError = Math.abs(trueRatio - dispRatio) / trueRatio;

      // Bound: relative aspect-ratio quantization error caused by integer pixel display rounding is strictly < 0.35%
      assert.ok(
        relError < 0.0035,
        `Quantization error ${relError} exceeded 0.35% for ${W}x${H} at display width ${w_disp}`,
      );
    }
  }
});

test('boundary touch coordinates at preview edges map to exact original sensor pixels', () => {
  const W_orig = 4000;
  const H_orig = 3000;
  const preview = computeAspectFitDisplaySize(W_orig, H_orig, 480);

  const boundaryPoints = [
    normalizePreviewTouch(0, 0, preview.width, preview.height),
    normalizePreviewTouch(preview.width, 0, preview.width, preview.height),
    normalizePreviewTouch(preview.width, preview.height, preview.width, preview.height),
    normalizePreviewTouch(0, preview.height, preview.width, preview.height),
  ];

  const validation = isValidCalibration(boundaryPoints);
  assert.equal(validation.valid, true);

  const pixelCorners = normalizedToPixelCorners(boundaryPoints, W_orig, H_orig);

  assert.deepEqual(pixelCorners[0], { x: 0, y: 0 });
  assert.deepEqual(pixelCorners[1], { x: 4000, y: 0 });
  assert.deepEqual(pixelCorners[2], { x: 4000, y: 3000 });
  assert.deepEqual(pixelCorners[3], { x: 0, y: 3000 });
});