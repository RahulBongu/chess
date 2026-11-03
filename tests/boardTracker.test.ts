/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { BoardTracker } from '../src/vision/boardTracker.ts';
import type { Point2D } from '../src/vision/perspectiveOverlay.ts';

const BASE_CORNERS: Point2D[] = [
  { x: 0.1, y: 0.1 },
  { x: 0.9, y: 0.1 },
  { x: 0.9, y: 0.9 },
  { x: 0.1, y: 0.9 },
];

test('BoardTracker initializes in uncalibrated state', () => {
  const tracker = new BoardTracker();
  assert.equal(tracker.getState(), 'uncalibrated');
  assert.equal(tracker.getCorners(), null);
  assert.equal(tracker.getConfidence(), 0);
});

test('seed sets corners and transitions to tracking state', () => {
  const tracker = new BoardTracker();
  const ok = tracker.seed(BASE_CORNERS);
  assert.ok(ok);
  assert.equal(tracker.getState(), 'tracking');
  assert.deepEqual(tracker.getCorners(), BASE_CORNERS);
});

test('seed rejects non-convex corners', () => {
  const tracker = new BoardTracker();
  const nonConvex: Point2D[] = [
    { x: 0.1, y: 0.1 },
    { x: 0.5, y: 0.5 }, // concave
    { x: 0.9, y: 0.1 },
    { x: 0.9, y: 0.9 },
  ];
  assert.equal(tracker.seed(nonConvex), false);
  assert.equal(tracker.getState(), 'uncalibrated');
});

test('update with small jitter applies EMA smoothing smoothly', () => {
  const tracker = new BoardTracker({ smoothingFactor: 0.4 });
  tracker.seed(BASE_CORNERS);

  // Tiny perturbation (1% jitter)
  const jittered: Point2D[] = [
    { x: 0.11, y: 0.10 },
    { x: 0.90, y: 0.11 },
    { x: 0.89, y: 0.90 },
    { x: 0.10, y: 0.89 },
  ];

  const state = tracker.update(jittered, 0.9);
  assert.equal(state, 'tracking');

  const smoothed = tracker.getCorners()!;
  assert.ok(smoothed !== null);

  // EMA expected: (1 - 0.4) * 0.1 + 0.4 * 0.11 = 0.06 + 0.044 = 0.104
  assert.ok(Math.abs(smoothed[0].x - 0.104) < 1e-4);
});

test('outlier detection holds current corners on single-frame sudden jump', () => {
  const tracker = new BoardTracker({ maxOutlierDistance: 0.1 });
  tracker.seed(BASE_CORNERS);

  // Large single-frame jump (e.g. 0.3 shift due to hand obstruction)
  const outlierJump: Point2D[] = [
    { x: 0.4, y: 0.4 },
    { x: 0.9, y: 0.1 },
    { x: 0.9, y: 0.9 },
    { x: 0.1, y: 0.9 },
  ];

  const state = tracker.update(outlierJump, 0.8);
  // Must hold previous corners rather than instantly warping to the outlier
  assert.equal(state, 'holding');
  assert.deepEqual(tracker.getCorners(), BASE_CORNERS);
});

test('occlusion grace period maintains holding state for up to N frames before losing tracking', () => {
  const tracker = new BoardTracker({ gracePeriodFrames: 3 });
  tracker.seed(BASE_CORNERS);

  // Frame 1: dropped detection (hand crossing board)
  assert.equal(tracker.update(null), 'holding');
  assert.deepEqual(tracker.getCorners(), BASE_CORNERS);

  // Frame 2: dropped detection
  assert.equal(tracker.update(null), 'holding');
  assert.deepEqual(tracker.getCorners(), BASE_CORNERS);

  // Frame 3: dropped detection (still within grace period of 3)
  assert.equal(tracker.update(null), 'holding');
  assert.deepEqual(tracker.getCorners(), BASE_CORNERS);

  // Frame 4: exceeds grace period -> tracking lost
  assert.equal(tracker.update(null), 'lost');
  assert.equal(tracker.getCorners(), null);
});

test('seamless recovery when detection returns within grace period', () => {
  const tracker = new BoardTracker({ gracePeriodFrames: 4, smoothingFactor: 0.5 });
  tracker.seed(BASE_CORNERS);

  // Drop 2 frames
  tracker.update(null);
  tracker.update(null);
  assert.equal(tracker.getState(), 'holding');

  // Detection returns
  const state = tracker.update(BASE_CORNERS, 0.9);
  assert.equal(state, 'tracking');
  assert.ok(tracker.getCorners() !== null);
  assert.equal(tracker.getConsecutiveLostFrames(), 0);
});

test('setLocked freezes tracking adjustments', () => {
  const tracker = new BoardTracker();
  tracker.seed(BASE_CORNERS);
  tracker.setLocked(true);

  // Even if null or new corners arrive, locked corners remain unchanged
  tracker.update(null);
  assert.deepEqual(tracker.getCorners(), BASE_CORNERS);
  assert.equal(tracker.getIsLocked(), true);
});

test('occlusion grace period expires based on elapsed time (gracePeriodMs)', () => {
  const tracker = new BoardTracker({ gracePeriodMs: 1500, gracePeriodFrames: 100 });
  const t0 = 10000;
  tracker.seed(BASE_CORNERS, t0);

  // At t0 + 500ms: within gracePeriodMs (1500)
  assert.equal(tracker.update(null, 0, t0 + 500), 'holding');
  assert.deepEqual(tracker.getCorners(), BASE_CORNERS);

  // At t0 + 1200ms: still within gracePeriodMs
  assert.equal(tracker.update(null, 0, t0 + 1200), 'holding');

  // At t0 + 1600ms: exceeds 1500ms -> tracking lost!
  assert.equal(tracker.update(null, 0, t0 + 1600), 'lost');
  assert.equal(tracker.getCorners(), null);
});

test('outlier shift adapts after sustained duration (outlierSustainMs)', () => {
  const tracker = new BoardTracker({ maxOutlierDistance: 0.1, outlierSustainMs: 500 });
  const t0 = 10000;
  tracker.seed(BASE_CORNERS, t0);

  const shiftedCorners: Point2D[] = [
    { x: 0.3, y: 0.3 },
    { x: 0.9, y: 0.1 },
    { x: 0.9, y: 0.9 },
    { x: 0.1, y: 0.9 },
  ];

  // At t0 + 100ms: outlier detected, holding
  assert.equal(tracker.update(shiftedCorners, 0.9, t0 + 100), 'holding');
  assert.deepEqual(tracker.getCorners(), BASE_CORNERS);

  // At t0 + 600ms: sustained for 500ms -> adapts to shifted position!
  assert.equal(tracker.update(shiftedCorners, 0.9, t0 + 600), 'tracking');
  assert.deepEqual(tracker.getCorners(), shiftedCorners);
});
