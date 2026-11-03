/**
 * Detector bounds / clamp-tolerance tests (Milestone 5.4D).
 *
 * Covers `clampCornersToBounds`, `measureCornerClampDistance` and
 * `maxClampTolerancePx` from `src/vision/boardDetection.ts`, plus the
 * `validateCandidateGeometry` structural gate for board quadrilaterals that
 * sit near each of the four image boundaries.
 *
 * The 5.4C fix clamps marginally out-of-frame proposals instead of rejecting
 * them; the 5.4D fix adds a maximum clamp tolerance so a clamp can never
 * invent geometry. These are unit tests of that logic with synthetic
 * coordinates - not measurements on real camera frames.
 */

import { test } from 'node:test';
import nodeAssert from 'node:assert/strict';

import {
  MAX_CLAMP_RATIO,
  MIN_CLAMP_TOLERANCE_PX,
  clampCornersToBounds,
  isConvexQuadrilateral,
  maxClampTolerancePx,
  measureCornerClampDistance,
  validateCandidateGeometry,
} from '../src/vision/boardDetection.ts';
import type { DetectedCorner, VisionSize } from '../src/vision/types.ts';

const SIZE: VisionSize = { width: 640, height: 480 };

function corner(x: number, y: number): DetectedCorner {
  return { x, y };
}

function quad(
  tl: DetectedCorner,
  tr: DetectedCorner,
  br: DetectedCorner,
  bl: DetectedCorner,
): DetectedCorner[] {
  return [tl, tr, br, bl];
}

// ---------------------------------------------------------------------------
// clampCornersToBounds
// ---------------------------------------------------------------------------

test('clampCornersToBounds leaves in-frame corners untouched', () => {
  const points = quad(
    corner(10, 10),
    corner(630, 12),
    corner(628, 470),
    corner(12, 468),
  );
  const clamped = clampCornersToBounds(points, SIZE);
  nodeAssert.deepEqual(clamped, points);
});

test('clampCornersToBounds clamps each boundary independently', () => {
  const points = quad(
    corner(-3, -5), // past left + top
    corner(647, -2), // past right + top
    corner(650, 490), // past right + bottom
    corner(-4, 486), // past left + bottom
  );
  const clamped = clampCornersToBounds(points, SIZE);
  nodeAssert.deepEqual(clamped[0], corner(0, 0));
  nodeAssert.deepEqual(clamped[1], corner(640, 0));
  nodeAssert.deepEqual(clamped[2], corner(640, 480));
  nodeAssert.deepEqual(clamped[3], corner(0, 480));
});

test('clampCornersToBounds is monotone (order along an edge preserved)', () => {
  const points = [corner(-10, 5), corner(-4, 5), corner(3, 5), corner(9, 5)];
  const clamped = clampCornersToBounds(points, SIZE);
  const xs = clamped.map((p) => p.x);
  for (let i = 1; i < xs.length; i += 1) {
    nodeAssert.ok(xs[i] >= xs[i - 1], `x order preserved: ${xs.join(', ')}`);
  }
});

// ---------------------------------------------------------------------------
// measureCornerClampDistance / maxClampTolerancePx
// ---------------------------------------------------------------------------

test('measureCornerClampDistance is 0 for fully in-frame quads', () => {
  const points = quad(
    corner(5, 5),
    corner(635, 5),
    corner(635, 475),
    corner(5, 475),
  );
  nodeAssert.equal(measureCornerClampDistance(points, SIZE), 0);
});

test('measureCornerClampDistance reports the worst corner movement', () => {
  const points = quad(
    corner(-6, 10), // 6 px past left edge
    corner(630, 10),
    corner(630, 470),
    corner(10, 470),
  );
  nodeAssert.ok(Math.abs(measureCornerClampDistance(points, SIZE) - 6) < 1e-9);
});

test('measureCornerClampDistance handles diagonal overshoot (hypotenuse)', () => {
  const points = [corner(-3, -4), corner(100, 100), corner(200, 200), corner(300, 300)];
  // dx=3, dy=4 -> 5 px
  nodeAssert.ok(Math.abs(measureCornerClampDistance(points, SIZE) - 5) < 1e-9);
});

test('maxClampTolerancePx uses the 2% rule with a 4 px floor', () => {
  // 640x480 -> shorter side 480 -> 2% = 9.6 px > 4 px floor
  nodeAssert.ok(Math.abs(maxClampTolerancePx(SIZE) - 0.02 * 480) < 1e-9);
  // Tiny frame -> floor applies
  const tiny: VisionSize = { width: 80, height: 60 };
  nodeAssert.equal(maxClampTolerancePx(tiny), MIN_CLAMP_TOLERANCE_PX);
  nodeAssert.ok(MIN_CLAMP_TOLERANCE_PX === 4);
  nodeAssert.ok(Math.abs(MAX_CLAMP_RATIO - 0.02) < 1e-12);
});

test('tolerance is always at least 4 px and never more than 2% of shorter side', () => {
  const sizes: VisionSize[] = [
    { width: 160, height: 120 },
    { width: 320, height: 240 },
    { width: 640, height: 480 },
    { width: 1280, height: 720 },
    { width: 1080, height: 1920 },
  ];
  for (const size of sizes) {
    const tolerance = maxClampTolerancePx(size);
    const shorter = Math.min(size.width, size.height);
    nodeAssert.ok(tolerance >= MIN_CLAMP_TOLERANCE_PX, `${size.width}x${size.height}: >= floor`);
    nodeAssert.ok(
      tolerance >= MAX_CLAMP_RATIO * shorter - 1e-9,
      `${size.width}x${size.height}: >= 2% of shorter side`,
    );
    // The 2% rationale: a tolerated clamp is well under half a board square.
    const squarePx = shorter / 8; // 8 squares across the board
    nodeAssert.ok(
      tolerance < squarePx / 2,
      `${size.width}x${size.height}: tolerance ${tolerance}px < half a square ${squarePx / 2}px`,
    );
  }
});

// ---------------------------------------------------------------------------
// validateCandidateGeometry near all four boundaries
// ---------------------------------------------------------------------------

test('board quad touching the top boundary validates', () => {
  const points = quad(
    corner(60, 0),
    corner(580, 0),
    corner(600, 460),
    corner(40, 460),
  );
  const result = validateCandidateGeometry(points, SIZE);
  nodeAssert.equal(result.valid, true, result.reason ?? '');
});

test('board quad touching the bottom boundary validates', () => {
  const points = quad(
    corner(60, 20),
    corner(580, 20),
    corner(600, 480),
    corner(40, 480),
  );
  const result = validateCandidateGeometry(points, SIZE);
  nodeAssert.equal(result.valid, true, result.reason ?? '');
});

test('board quad touching the left boundary validates', () => {
  const points = quad(
    corner(0, 30),
    corner(560, 10),
    corner(580, 450),
    corner(0, 470),
  );
  const result = validateCandidateGeometry(points, SIZE);
  nodeAssert.equal(result.valid, true, result.reason ?? '');
});

test('board quad touching the right boundary validates', () => {
  const points = quad(
    corner(60, 10),
    corner(640, 30),
    corner(640, 470),
    corner(40, 450),
  );
  const result = validateCandidateGeometry(points, SIZE);
  nodeAssert.equal(result.valid, true, result.reason ?? '');
});

test('board quad spanning all four boundaries (full-frame board) validates', () => {
  const points = quad(
    corner(0, 0),
    corner(640, 0),
    corner(640, 480),
    corner(0, 480),
  );
  const result = validateCandidateGeometry(points, SIZE);
  nodeAssert.equal(result.valid, true, result.reason ?? '');
});

test('marginally out-of-frame proposal clamps to a valid quad (5.4C behaviour)', () => {
  // 2 px past the top-left corner, as a rotated minAreaRect may propose.
  const raw = quad(
    corner(-2, -2),
    corner(642, -1),
    corner(640, 481),
    corner(-1, 480),
  );
  const clampPx = measureCornerClampDistance(raw, SIZE);
  nodeAssert.ok(clampPx <= maxClampTolerancePx(SIZE), `clamp ${clampPx}px within tolerance`);
  const clamped = clampCornersToBounds(raw, SIZE);
  nodeAssert.equal(isConvexQuadrilateral(clamped), true);
  const result = validateCandidateGeometry(clamped, SIZE);
  nodeAssert.equal(result.valid, true, result.reason ?? '');
});

test('proposal needing excessive clamping exceeds the 5.4D tolerance', () => {
  // A candidate whose corners are 60 px outside the frame is not a rounding
  // overshoot - clamping it would invent geometry.
  const raw = quad(
    corner(-60, 10),
    corner(700, 10),
    corner(700, 470),
    corner(-60, 470),
  );
  const clampPx = measureCornerClampDistance(raw, SIZE);
  nodeAssert.ok(
    clampPx > maxClampTolerancePx(SIZE),
    `clamp ${clampPx}px exceeds tolerance ${maxClampTolerancePx(SIZE)}px`,
  );
});

test('quads far outside all four boundaries each exceed tolerance', () => {
  const boundaryCases: { label: string; points: DetectedCorner[] }[] = [
    {
      label: 'past top',
      points: quad(corner(60, -40), corner(580, -40), corner(600, 400), corner(40, 400)),
    },
    {
      label: 'past bottom',
      points: quad(corner(60, 80), corner(580, 80), corner(600, 520), corner(40, 520)),
    },
    {
      label: 'past left',
      points: quad(corner(-50, 30), corner(560, 10), corner(580, 450), corner(-50, 470)),
    },
    {
      label: 'past right',
      points: quad(corner(60, 10), corner(690, 30), corner(690, 470), corner(40, 450)),
    },
  ];
  for (const c of boundaryCases) {
    const clampPx = measureCornerClampDistance(c.points, SIZE);
    nodeAssert.ok(
      clampPx > maxClampTolerancePx(SIZE),
      `${c.label}: clamp ${clampPx}px exceeds tolerance`,
    );
  }
});

test('in-frame but self-intersecting quad still rejected (clamp not involved)', () => {
  const bowtie = quad(
    corner(100, 100),
    corner(500, 400),
    corner(500, 100),
    corner(100, 400),
  );
  nodeAssert.equal(measureCornerClampDistance(bowtie, SIZE), 0, 'no clamping needed');
  const result = validateCandidateGeometry(bowtie, SIZE);
  nodeAssert.equal(result.valid, false);
});
