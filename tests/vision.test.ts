/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  MIN_DETECTION_CONFIDENCE,
  confidenceFromScore,
  isInBounds,
  isSelfIntersecting,
  isValidDetection,
  normalizePoints,
  quadrilateralArea,
  reorderCorners,
  scoreCandidate,
  validateCandidateGeometry,
} from '../src/vision/boardDetection.ts';

test('reorderCorners produces top-left, top-right, bottom-right, bottom-left', () => {
  const points = [
    { x: 10, y: 90 },
    { x: 90, y: 10 },
    { x: 10, y: 10 },
    { x: 90, y: 90 },
  ];
  const ordered = reorderCorners(points);
  assert.deepEqual(ordered, [
    { x: 10, y: 10 },
    { x: 90, y: 10 },
    { x: 90, y: 90 },
    { x: 10, y: 90 },
  ]);
});

test('invalid and degenerate quadrilaterals are rejected', () => {
  const size = { width: 100, height: 100 };
  const degenerate = [
    { x: 10, y: 10 },
    { x: 20, y: 20 },
    { x: 30, y: 30 },
    { x: 40, y: 40 },
  ];
  assert.equal(validateCandidateGeometry(degenerate, size).valid, false);

  const selfIntersecting = [
    { x: 10, y: 10 },
    { x: 90, y: 90 },
    { x: 10, y: 90 },
    { x: 90, y: 10 },
  ];
  assert.equal(isSelfIntersecting(selfIntersecting), true);
  assert.equal(validateCandidateGeometry(selfIntersecting, size).valid, false);
});

test('out-of-bounds corners are rejected', () => {
  const size = { width: 100, height: 100 };
  const points = [
    { x: -1, y: 10 },
    { x: 90, y: 10 },
    { x: 90, y: 90 },
    { x: 10, y: 90 },
  ];
  assert.equal(isInBounds(points, size), false);
  assert.equal(validateCandidateGeometry(points, size).valid, false);
});

test('incorrect corner ordering is normalized before validation', () => {
  const size = { width: 100, height: 100 };
  const points = [
    { x: 90, y: 90 },
    { x: 10, y: 90 },
    { x: 10, y: 10 },
    { x: 90, y: 10 },
  ];
  const ordered = reorderCorners(points);
  assert.equal(validateCandidateGeometry(ordered, size).valid, true);
  assert.ok(quadrilateralArea(ordered) > 0);
});

test('confidence threshold rejects weak detections', () => {
  assert.equal(isValidDetection({ confidence: 0.1, corners: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }] }, 0.32), false);
  assert.equal(isValidDetection({ confidence: 0.8, corners: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }] }, 0.32), true);
  assert.equal(confidenceFromScore(2), 1);
});

test('a valid detection is converted into normalized preview coordinates', () => {
  const points = [
    { x: 10, y: 10 },
    { x: 90, y: 10 },
    { x: 90, y: 90 },
    { x: 10, y: 90 },
  ];
  const normalized = normalizePoints(points, { width: 100, height: 100 });
  assert.deepEqual(normalized, [
    { x: 0.1, y: 0.1 },
    { x: 0.9, y: 0.1 },
    { x: 0.9, y: 0.9 },
    { x: 0.1, y: 0.9 },
  ]);
});

test('candidate ranking favors larger, more rectangular boards', () => {
  const size = { width: 100, height: 100 };
  const strong = [
    { x: 10, y: 10 },
    { x: 90, y: 10 },
    { x: 90, y: 90 },
    { x: 10, y: 90 },
  ];
  const weak = [
    { x: 40, y: 40 },
    { x: 60, y: 40 },
    { x: 60, y: 60 },
    { x: 40, y: 60 },
  ];
  assert.ok(scoreCandidate(strong, size) > scoreCandidate(weak, size));
});

test('manual refinement keeps valid geometry after a corner move', () => {
  const size = { width: 100, height: 100 };
  const points = [
    { x: 10, y: 10 },
    { x: 90, y: 10 },
    { x: 90, y: 90 },
    { x: 10, y: 90 },
  ];
  points[1] = { x: 85, y: 12 };
  assert.equal(validateCandidateGeometry(points, size).valid, true);
});