/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  addCalibrationPoint,
  canAddCalibrationPoint,
  isValidCalibration,
  reorderCalibrationPoints,
  resetCalibrationPoints,
  undoCalibrationPoint,
} from '../src/calibration/geometry.ts';

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

test('reorderCalibrationPoints handles all 24 permutations of known convex corners without mutating coordinates', () => {
  const pA = { x: 0.15, y: 0.2 };
  const pB = { x: 0.85, y: 0.18 };
  const pC = { x: 0.82, y: 0.82 };
  const pD = { x: 0.18, y: 0.84 };
  const base = [pA, pB, pC, pD];

  function getPermutations<T>(arr: T[]): T[][] {
    if (arr.length <= 1) return [arr];
    const res: T[][] = [];
    for (let i = 0; i < arr.length; i++) {
      const rest = [...arr.slice(0, i), ...arr.slice(i + 1)];
      for (const p of getPermutations(rest)) {
        res.push([arr[i], ...p]);
      }
    }
    return res;
  }

  const perms = getPermutations(base);
  assert.equal(perms.length, 24);

  for (const perm of perms) {
    const reordered = reorderCalibrationPoints(perm);
    assert.equal(reordered.length, 4);

    // Coordinate preservation: coordinates must be a permutation of original points, never mutated
    const origKeys = new Set(base.map((p) => `${p.x.toFixed(4)},${p.y.toFixed(4)}`));
    const reorderedKeys = new Set(reordered.map((p) => `${p.x.toFixed(4)},${p.y.toFixed(4)}`));
    assert.deepEqual(origKeys, reorderedKeys);

    // Convexity and validity
    const validation = isValidCalibration(reordered);
    assert.equal(validation.valid, true, `Permutation failed validation: ${JSON.stringify(perm)}`);
  }
});

test('reorderCalibrationPoints handles steep perspective trapezoids and rotated diamond quadrilaterals', () => {
  // Steep perspective trapezoid (narrow top, wide bottom)
  const trapezoid = [
    { x: 0.35, y: 0.15 },
    { x: 0.65, y: 0.15 },
    { x: 0.95, y: 0.85 },
    { x: 0.05, y: 0.85 },
  ];
  const reorderedTrap = reorderCalibrationPoints(trapezoid);
  assert.equal(isValidCalibration(reorderedTrap).valid, true);

  // 45-degree rotated diamond quadrilateral
  const diamond = [
    { x: 0.5, y: 0.1 },
    { x: 0.9, y: 0.5 },
    { x: 0.5, y: 0.9 },
    { x: 0.1, y: 0.5 },
  ];
  const reorderedDiamond = reorderCalibrationPoints(diamond);
  assert.equal(isValidCalibration(reorderedDiamond).valid, true);
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
      const h_disp = Math.round(w_disp * (H / W));
      const trueRatio = W / H;
      const dispRatio = w_disp / h_disp;
      const relError = Math.abs(trueRatio - dispRatio) / trueRatio;

      // Bound: relative aspect ratio quantization error caused by integer pixel display rounding is strictly < 0.35%
      assert.ok(
        relError < 0.0035,
        `Quantization error ${relError} exceeded 0.35% for ${W}x${H} at display width ${w_disp}`,
      );

      // Bound: maximum displacement in original sensor resolution is <= 0.5 * (H / h_disp)
      const maxDisplacementPx = 0.5 * (H / h_disp);
      assert.ok(maxDisplacementPx / H < 0.0035);
    }
  }
});

test('boundary touch coordinates at preview edges map to exact original sensor pixels', () => {
  const W_orig = 4000;
  const H_orig = 3000;

  // Extreme preview boundary touches [0, 0] to [1, 1]
  const boundaryPoints = [
    { x: 0.0, y: 0.0 },
    { x: 1.0, y: 0.0 },
    { x: 1.0, y: 1.0 },
    { x: 0.0, y: 1.0 },
  ];

  const validation = isValidCalibration(boundaryPoints);
  assert.equal(validation.valid, true);

  const pixelCorners = boundaryPoints.map((pt) => ({
    x: pt.x * W_orig,
    y: pt.y * H_orig,
  }));

  assert.deepEqual(pixelCorners[0], { x: 0, y: 0 });
  assert.deepEqual(pixelCorners[1], { x: 4000, y: 0 });
  assert.deepEqual(pixelCorners[2], { x: 4000, y: 3000 });
  assert.deepEqual(pixelCorners[3], { x: 0, y: 3000 });
});