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