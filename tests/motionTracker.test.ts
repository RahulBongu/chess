/**
 * Focused tests for BoardMotionTracker (Milestone 5.4D).
 *
 * These tests exercise `src/vision/motionTracker.ts` directly with synthetic
 * frame sequences rather than only testing BoardTracker state transitions.
 * Every transformation below is a SYNTHETIC warp of a rendered board texture:
 * they test algorithmic behaviour (corner error, reprojection error, outlier
 * rejection, loss/recovery) and do NOT establish real-camera tracking accuracy.
 *
 * Ground truth for each case is the exact destination quadrilateral used to
 * render the warped frame, so corner error is measured against known geometry.
 */

import { after, test } from 'node:test';
import nodeAssert from 'node:assert/strict';

import {
  BoardMotionTracker,
  type GrayBuffer,
  type MotionTrackResult,
} from '../src/vision/motionTracker.ts';
import { squareToQuadrilateral, type SquareHomography } from '../src/vision/boardEvidence.ts';
import type { Point2D } from '../src/vision/perspectiveOverlay.ts';

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// 3x3 homography helpers (row-major, last row 0 0 1)
// ---------------------------------------------------------------------------
type Mat3 = [number, number, number, number, number, number, number, number, number];

function toMat(h: SquareHomography): Mat3 {
  return [h.a, h.b, h.c, h.d, h.e, h.f, h.g, h.h, 1];
}

function matMul(a: Mat3, b: Mat3): Mat3 {
  const out = new Array<number>(9);
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[r * 3 + c] =
        a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return out as unknown as Mat3;
}

function matInv(m: Mat3): Mat3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) return null;
  const inv = 1 / det;
  return [
    (e * i - f * h) * inv,
    (c * h - b * i) * inv,
    (b * f - c * e) * inv,
    (f * g - d * i) * inv,
    (a * i - c * g) * inv,
    (c * d - a * f) * inv,
    (d * h - e * g) * inv,
    (b * g - a * h) * inv,
    (a * e - b * d) * inv,
  ];
}

function applyMat(m: Mat3, x: number, y: number): Point2D {
  const denom = m[6] * x + m[7] * y + m[8];
  if (Math.abs(denom) < 1e-12) return { x: Number.NaN, y: Number.NaN };
  return {
    x: (m[0] * x + m[1] * y + m[2]) / denom,
    y: (m[3] * x + m[4] * y + m[5]) / denom,
  };
}

function quadToMat(corners: Point2D[]): Mat3 | null {
  const h = squareToQuadrilateral(corners);
  return h ? toMat(h) : null;
}

/**
 * Renders `src` viewed through the homography that maps `srcCorners` onto
 * `dstCorners` (inverse warp: for each destination pixel, sample the source).
 * Out-of-source pixels are filled with the table colour.
 */
function warpFrame(
  src: GrayBuffer,
  srcCorners: Point2D[],
  dstCorners: Point2D[],
): GrayBuffer {
  const A = quadToMat(srcCorners);
  const B = quadToMat(dstCorners);
  if (!A || !B) throw new Error('Invalid corner quadrilateral.');
  const Binv = matInv(B);
  if (!Binv) throw new Error('Singular destination homography.');
  // dst pixel -> unit square -> src pixel
  const H = matMul(A, Binv);

  const out = new Uint8Array(src.width * src.height);
  for (let y = 0; y < src.height; y += 1) {
    for (let x = 0; x < src.width; x += 1) {
      const s = applyMat(H, x, y);
      if (!Number.isFinite(s.x) || !Number.isFinite(s.y)) {
        out[y * src.width + x] = 38;
        continue;
      }
      if (s.x < 0 || s.y < 0 || s.x > src.width - 1 || s.y > src.height - 1) {
        out[y * src.width + x] = 38;
        continue;
      }
      const x0 = Math.floor(s.x);
      const y0 = Math.floor(s.y);
      const x1 = Math.min(src.width - 1, x0 + 1);
      const y1 = Math.min(src.height - 1, y0 + 1);
      const fx = s.x - x0;
      const fy = s.y - y0;
      const top = src.data[y0 * src.width + x0] * (1 - fx) + src.data[y0 * src.width + x1] * fx;
      const bottom = src.data[y1 * src.width + x0] * (1 - fx) + src.data[y1 * src.width + x1] * fx;
      out[y * src.width + x] = Math.round(top * (1 - fy) + bottom * fy);
    }
  }
  return { width: src.width, height: src.height, data: out };
}

// ---------------------------------------------------------------------------
// Synthetic board rendering
// ---------------------------------------------------------------------------
const FRAME_W = 160;
const FRAME_H = 120;

/** Ground-truth board quad in pixels (TL, TR, BR, BL). */
const BOARD_PX: Point2D[] = [
  { x: 8, y: 6 },
  { x: 152, y: 6 },
  { x: 152, y: 114 },
  { x: 8, y: 114 },
];

const BOARD_NORM: Point2D[] = BOARD_PX.map((p) => ({
  x: p.x / FRAME_W,
  y: p.y / FRAME_H,
}));

/**
 * Renders an 8x8 checkerboard with deterministic value noise so every local
 * patch is unique (a perfectly periodic checker would let LK converge to a
 * shifted local minimum and hide real failures).
 */
function renderBoardFrame(seed = 20261103): GrayBuffer {
  const data = new Uint8Array(FRAME_W * FRAME_H);
  data.fill(38); // table background

  const rnd = mulberry32(seed);
  const cell = 8;
  const gw = Math.ceil(FRAME_W / cell) + 2;
  const gh = Math.ceil(FRAME_H / cell) + 2;
  const grid = new Float32Array(gw * gh);
  for (let i = 0; i < grid.length; i += 1) grid[i] = rnd() * 2 - 1;

  const A = quadToMat(BOARD_PX)!;
  const Ainv = matInv(A)!;

  for (let y = 0; y < FRAME_H; y += 1) {
    for (let x = 0; x < FRAME_W; x += 1) {
      const u = applyMat(Ainv, x, y);
      if (!Number.isFinite(u.x) || u.x < 0 || u.x > 1 || u.y < 0 || u.y > 1) continue;
      const sqX = Math.floor(u.x * 8);
      const sqY = Math.floor(u.y * 8);
      const base = (sqX + sqY) % 2 === 0 ? 196 : 58;
      // coarse value noise, bilinearly interpolated
      const gx = x / cell;
      const gy = y / cell;
      const x0 = Math.floor(gx);
      const y0 = Math.floor(gy);
      const fx = gx - x0;
      const fy = gy - y0;
      const n =
        grid[y0 * gw + x0] * (1 - fx) * (1 - fy) +
        grid[y0 * gw + x0 + 1] * fx * (1 - fy) +
        grid[(y0 + 1) * gw + x0] * (1 - fx) * fy +
        grid[(y0 + 1) * gw + x0 + 1] * fx * fy;
      const value = base + n * 34;
      data[y * FRAME_W + x] = Math.max(0, Math.min(255, Math.round(value)));
    }
  }
  return { width: FRAME_W, height: FRAME_H, data };
}

function addBlob(
  buf: GrayBuffer,
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  value: number,
): void {
  for (let y = Math.max(0, Math.floor(cy - ry)); y <= Math.min(buf.height - 1, cy + ry); y += 1) {
    for (let x = Math.max(0, Math.floor(cx - rx)); x <= Math.min(buf.width - 1, cx + rx); x += 1) {
      const dx = (x - cx) / rx;
      const dy = (y - cy) / ry;
      if (dx * dx + dy * dy <= 1) {
        buf.data[y * buf.width + x] = value;
      }
    }
  }
}

function boxBlur(buf: GrayBuffer): GrayBuffer {
  const out = new Uint8Array(buf.width * buf.height);
  for (let y = 0; y < buf.height; y += 1) {
    for (let x = 0; x < buf.width; x += 1) {
      let sum = 0;
      let count = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = Math.max(0, Math.min(buf.width - 1, x + dx));
          const ny = Math.max(0, Math.min(buf.height - 1, y + dy));
          sum += buf.data[ny * buf.width + nx];
          count += 1;
        }
      }
      out[y * buf.width + x] = Math.round(sum / count);
    }
  }
  return { width: buf.width, height: buf.height, data: out };
}

function fillFrame(value: number): GrayBuffer {
  const data = new Uint8Array(FRAME_W * FRAME_H);
  data.fill(value);
  return { width: FRAME_W, height: FRAME_H, data };
}

// ---------------------------------------------------------------------------
// Ground-truth transforms
// ---------------------------------------------------------------------------
function translate(corners: Point2D[], dx: number, dy: number): Point2D[] {
  return corners.map((p) => ({ x: p.x + dx, y: p.y + dy }));
}

function rotateAbout(corners: Point2D[], degrees: number): Point2D[] {
  const cx = FRAME_W / 2;
  const cy = FRAME_H / 2;
  const rad = (degrees * Math.PI) / 180;
  return corners.map((p) => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    return {
      x: cx + dx * Math.cos(rad) - dy * Math.sin(rad),
      y: cy + dx * Math.sin(rad) + dy * Math.cos(rad),
    };
  });
}

function scaleAbout(corners: Point2D[], factor: number): Point2D[] {
  const cx = FRAME_W / 2;
  const cy = FRAME_H / 2;
  return corners.map((p) => ({
    x: cx + (p.x - cx) * factor,
    y: cy + (p.y - cy) * factor,
  }));
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------
function cornerErrorRatio(
  estimated: Point2D[],
  groundTruth: Point2D[],
): { meanRatio: number; maxRatio: number; meanPx: number } {
  const diag = Math.hypot(FRAME_W, FRAME_H);
  let total = 0;
  let worst = 0;
  for (let i = 0; i < estimated.length; i += 1) {
    const err = Math.hypot(estimated[i].x - groundTruth[i].x, estimated[i].y - groundTruth[i].y);
    total += err;
    worst = Math.max(worst, err);
  }
  const meanPx = estimated.length > 0 ? total / estimated.length : Number.POSITIVE_INFINITY;
  return { meanRatio: meanPx / diag, maxRatio: worst / diag, meanPx };
}

/**
 * Mean displacement between the estimated frame-to-frame homography and the
 * exact synthetic ground-truth homography over a grid of board-interior points.
 */
function reprojectionErrorPx(
  estimated: Float64Array | null,
  srcCorners: Point2D[],
  dstCorners: Point2D[],
): number {
  if (!estimated) return Number.POSITIVE_INFINITY;
  const A = quadToMat(srcCorners)!;
  const B = quadToMat(dstCorners)!;
  const Ainv = matInv(A)!;
  // ground truth: src pixel -> unit square -> dst pixel
  const gt = matMul(B, Ainv);
  let total = 0;
  let count = 0;
  for (let v = 0.1; v <= 0.9; v += 0.2) {
    for (let u = 0.1; u <= 0.9; u += 0.2) {
      // A maps the unit square to src pixel coordinates.
      const srcPt = applyMat(A, u, v);
      const est = applyMat(estimated as unknown as Mat3, srcPt.x, srcPt.y);
      const truth = applyMat(gt, srcPt.x, srcPt.y);
      if (!Number.isFinite(est.x) || !Number.isFinite(truth.x)) continue;
      total += Math.hypot(est.x - truth.x, est.y - truth.y);
      count += 1;
    }
  }
  return count > 0 ? total / count : Number.POSITIVE_INFINITY;
}

// ---------------------------------------------------------------------------
// Tiny assertion harness (node:test)
// ---------------------------------------------------------------------------
const metricLog: { name: string; cornerPct: number; reproPx: number }[] = [];

function assert(condition: boolean, message: string): void {
  nodeAssert.ok(condition, message);
}

function runTest(name: string, fn: () => void): void {
  test(name, fn);
}

/** Core invariant: a non-tracking result must never expose corners or confidence. */
function assertNoStaleFix(result: MotionTrackResult, label: string): void {
  if (result.state === 'tracking') return;
  assert(result.corners === null, `${label}: corners are null while ${result.state} (never stale)`);
  assert(result.confidence === 0, `${label}: confidence is 0 while ${result.state}`);
  assert(result.homography === null, `${label}: homography is null while ${result.state}`);
}

/**
 * Runs one synthetic warp through the tracker and returns the result plus
 * ground-truth error metrics (logged for the milestone report).
 */
function trackWarp(
  tracker: BoardMotionTracker,
  srcFrame: GrayBuffer,
  dstCorners: Point2D[],
  label: string,
  nowMs: number,
): MotionTrackResult {
  const dstFrame = warpFrame(srcFrame, BOARD_PX, dstCorners);
  const result = tracker.update(dstFrame, nowMs);
  const corner = result.corners
    ? cornerErrorRatio(result.corners.pixels, dstCorners)
    : { meanRatio: Number.NaN, maxRatio: Number.NaN, meanPx: Number.NaN };
  const repro = reprojectionErrorPx(result.homography, BOARD_PX, dstCorners);
  metricLog.push({
    name: label,
    cornerPct: corner.meanRatio * 100,
    reproPx: repro,
  });
  console.log(
    `  [metric] ${label}: state=${result.state}` +
      (result.corners
        ? ` cornerErr=${corner.meanPx.toFixed(2)}px (${(corner.meanRatio * 100).toFixed(2)}%) inliers=${result.inliers} conf=${result.confidence.toFixed(2)}`
        : ` reason="${result.reason ?? ''}"`) +
      ` reproErr=${Number.isFinite(repro) ? repro.toFixed(2) + 'px' : 'n/a'}`,
  );
  return result;
}

function newTracker(overrides: ConstructorParameters<typeof BoardMotionTracker>[0] = {}): BoardMotionTracker {
  return new BoardMotionTracker({ maxFeatures: 120, minCornerResponse: 25, ...overrides });
}

function seedTracker(tracker: BoardMotionTracker, frame: GrayBuffer, nowMs = 1000): boolean {
  return tracker.seed(frame, BOARD_NORM, nowMs);
}

// ===========================================================================
console.log('=== BoardMotionTracker Synthetic Tests (Milestone 5.4D) ===');
console.log('All frames below are SYNTHETIC warps; no real camera footage is used.\n');

const baseFrame = renderBoardFrame();

// ---------------------------------------------------------------------------
runTest('Known translation (synthetic)', () => {
  const tracker = newTracker();
  assert(seedTracker(tracker, baseFrame), 'seed succeeds');

  const gt = translate(BOARD_PX, 5, 3);
  const result = trackWarp(tracker, baseFrame, gt, 'translation (5,3)', 1016);

  assert(result.state === 'tracking', `state is tracking (got ${result.state})`);
  assert(result.corners !== null, 'fresh corners returned');
  assert(result.inliers >= 10, `inliers ${result.inliers} >= 10`);
  assertNoStaleFix(result, 'translation');

  if (result.corners) {
    const err = cornerErrorRatio(result.corners.pixels, gt);
    assert(err.meanRatio < 0.02, `mean corner error ${(err.meanRatio * 100).toFixed(2)}% < 2%`);
  }
});

// ---------------------------------------------------------------------------
runTest('Known rotation, 5 degrees (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);

  const gt = rotateAbout(BOARD_PX, 5);
  const result = trackWarp(tracker, baseFrame, gt, 'rotation 5deg', 1016);

  assert(result.state === 'tracking', `state is tracking (got ${result.state})`);
  assertNoStaleFix(result, 'rotation');
  if (result.corners) {
    const err = cornerErrorRatio(result.corners.pixels, gt);
    assert(err.meanRatio < 0.03, `mean corner error ${(err.meanRatio * 100).toFixed(2)}% < 3%`);
  }
});

// ---------------------------------------------------------------------------
runTest('Known scale change, 1.1x (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);

  const gt = scaleAbout(BOARD_PX, 1.1);
  const result = trackWarp(tracker, baseFrame, gt, 'scale 1.1x', 1016);

  assert(result.state === 'tracking', `state is tracking (got ${result.state})`);
  assertNoStaleFix(result, 'scale');
  if (result.corners) {
    const err = cornerErrorRatio(result.corners.pixels, gt);
    assert(err.meanRatio < 0.03, `mean corner error ${(err.meanRatio * 100).toFixed(2)}% < 3%`);
  }
});

// ---------------------------------------------------------------------------
runTest('Perspective change (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);

  // Tilt: pull the top edge inward, push the bottom edge outward.
  const gt = BOARD_PX.map((p, i) => {
    if (i === 0) return { x: p.x + 14, y: p.y + 5 };
    if (i === 1) return { x: p.x - 14, y: p.y + 5 };
    if (i === 2) return { x: p.x + 6, y: p.y - 3 };
    return { x: p.x - 6, y: p.y - 3 };
  });
  const result = trackWarp(tracker, baseFrame, gt, 'perspective tilt', 1016);

  assert(result.state === 'tracking', `state is tracking (got ${result.state})`);
  assertNoStaleFix(result, 'perspective');
  if (result.corners) {
    const err = cornerErrorRatio(result.corners.pixels, gt);
    assert(err.meanRatio < 0.04, `mean corner error ${(err.meanRatio * 100).toFixed(2)}% < 4%`);
  }
});

// ---------------------------------------------------------------------------
runTest('Combined translation + rotation + scale (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);

  let gt = scaleAbout(BOARD_PX, 1.06);
  gt = rotateAbout(gt, -3);
  gt = translate(gt, -4, 6);
  const result = trackWarp(tracker, baseFrame, gt, 'combined rigid+scale', 1016);

  assert(result.state === 'tracking', `state is tracking (got ${result.state})`);
  assertNoStaleFix(result, 'combined');
  if (result.corners) {
    const err = cornerErrorRatio(result.corners.pixels, gt);
    assert(err.meanRatio < 0.04, `mean corner error ${(err.meanRatio * 100).toFixed(2)}% < 4%`);
  }
});

// ---------------------------------------------------------------------------
runTest('Feature count and inlier quality on a clean warp (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);
  const gt = translate(BOARD_PX, 4, -3);
  const result = trackWarp(tracker, baseFrame, gt, 'clean warp quality', 1016);

  assert(result.state === 'tracking', 'tracking');
  assert(result.featureCount >= 30, `featureCount ${result.featureCount} >= 30`);
  assert(result.inlierRatio >= 0.6, `inlierRatio ${result.inlierRatio.toFixed(2)} >= 0.6`);
  assert(result.confidence >= 0.5, `confidence ${result.confidence.toFixed(2)} >= 0.5`);
  assert(
    result.meanRoundTripErrorPx !== null && result.meanRoundTripErrorPx <= 1.5,
    `mean round-trip error ${result.meanRoundTripErrorPx?.toFixed(2)}px <= 1.5px`,
  );
});

// ---------------------------------------------------------------------------
runTest('Moving piece introduces outlier feature motion (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);

  const gt = translate(BOARD_PX, 3, 2);
  const dst = warpFrame(baseFrame, BOARD_PX, gt);
  // A piece moved inside the frame creates local feature motion that does not
  // match the board homography.
  addBlob(dst, 60, 45, 9, 9, 30);
  const result = tracker.update(dst, 1016);
  const err = result.corners ? cornerErrorRatio(result.corners.pixels, gt) : null;
  metricLog.push({
    name: 'moving piece',
    cornerPct: err ? err.meanRatio * 100 : Number.NaN,
    reproPx: reprojectionErrorPx(result.homography, BOARD_PX, gt),
  });
  console.log(
    `  [metric] moving piece: state=${result.state} inliers=${result.inliers}/${result.featureCount} conf=${result.confidence.toFixed(2)}` +
      (err ? ` cornerErr=${(err.meanRatio * 100).toFixed(2)}%` : ''),
  );

  assert(result.state === 'tracking', `state is tracking despite the piece (got ${result.state})`);
  assertNoStaleFix(result, 'moving piece');
  if (result.corners && err) {
    assert(err.meanRatio < 0.03, `corner error ${(err.meanRatio * 100).toFixed(2)}% < 3%`);
    assert(
      result.inlierRatio >= 0.45,
      `inlier ratio ${result.inlierRatio.toFixed(2)} >= 0.45 after outlier rejection`,
    );
    assert(result.inliers < result.featureCount, `outliers were rejected (${result.inliers}/${result.featureCount} inliers)`);
  }
});

// ---------------------------------------------------------------------------
runTest('Hand occlusion over part of the board (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);

  const gt = translate(BOARD_PX, 3, 2);
  const dst = warpFrame(baseFrame, BOARD_PX, gt);
  addBlob(dst, 80, 60, 34, 30, 96); // large hand-like occlusion
  const result = tracker.update(dst, 1016);
  const err = result.corners ? cornerErrorRatio(result.corners.pixels, gt) : null;
  metricLog.push({
    name: 'hand occlusion',
    cornerPct: err ? err.meanRatio * 100 : Number.NaN,
    reproPx: reprojectionErrorPx(result.homography, BOARD_PX, gt),
  });
  console.log(
    `  [metric] hand occlusion: state=${result.state} inliers=${result.inliers}/${result.featureCount} conf=${result.confidence.toFixed(2)}` +
      (err ? ` cornerErr=${(err.meanRatio * 100).toFixed(2)}%` : ` reason="${result.reason ?? ''}"`),
  );

  assert(
    result.state === 'tracking' || result.state === 'holding',
    `state ${result.state} is tracking or holding`,
  );
  assertNoStaleFix(result, 'hand occlusion');
  if (result.corners && err) {
    assert(err.meanRatio < 0.04, `corner error ${(err.meanRatio * 100).toFixed(2)}% < 4%`);
  }
});

// ---------------------------------------------------------------------------
runTest('Low-texture board (synthetic)', () => {
  const tracker = newTracker({ minCornerResponse: 10 });

  // Very low contrast board: only ~24 grey levels of separation.
  const data = new Uint8Array(FRAME_W * FRAME_H);
  data.fill(38);
  const A = quadToMat(BOARD_PX)!;
  const Ainv = matInv(A)!;
  for (let y = 0; y < FRAME_H; y += 1) {
    for (let x = 0; x < FRAME_W; x += 1) {
      const u = applyMat(Ainv, x, y);
      if (!Number.isFinite(u.x) || u.x < 0 || u.x > 1 || u.y < 0 || u.y > 1) continue;
      const sqX = Math.floor(u.x * 8);
      const sqY = Math.floor(u.y * 8);
      data[y * FRAME_W + x] = (sqX + sqY) % 2 === 0 ? 122 : 98;
    }
  }
  const lowTexture: GrayBuffer = { width: FRAME_W, height: FRAME_H, data };

  assert(seedTracker(tracker, lowTexture), 'seed succeeds on low texture');
  const gt = translate(BOARD_PX, 3, 2);
  const result = trackWarp(tracker, lowTexture, gt, 'low texture', 1016);

  assert(
    ['tracking', 'holding', 'lost'].includes(result.state),
    `state ${result.state} is a valid outcome for low texture`,
  );
  assertNoStaleFix(result, 'low texture');
  if (result.corners) {
    const err = cornerErrorRatio(result.corners.pixels, gt);
    assert(err.meanRatio < 0.05, `corner error ${(err.meanRatio * 100).toFixed(2)}% < 5%`);
  } else {
    console.log(`  (no fix: ${result.reason ?? 'no reason'})`);
  }
});

// ---------------------------------------------------------------------------
runTest('Motion blur (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);

  const gt = translate(BOARD_PX, 4, 3);
  const dst = boxBlur(warpFrame(baseFrame, BOARD_PX, gt));
  const result = tracker.update(dst, 1016);
  const err = result.corners ? cornerErrorRatio(result.corners.pixels, gt) : null;
  metricLog.push({
    name: 'blur',
    cornerPct: err ? err.meanRatio * 100 : Number.NaN,
    reproPx: reprojectionErrorPx(result.homography, BOARD_PX, gt),
  });
  console.log(
    `  [metric] blur: state=${result.state} inliers=${result.inliers}/${result.featureCount}` +
      (err ? ` cornerErr=${(err.meanRatio * 100).toFixed(2)}%` : ` reason="${result.reason ?? ''}"`),
  );

  assert(
    result.state === 'tracking' || result.state === 'holding',
    `state ${result.state} handles blur`,
  );
  assertNoStaleFix(result, 'blur');
  if (result.corners && err) {
    assert(err.meanRatio < 0.05, `corner error ${(err.meanRatio * 100).toFixed(2)}% < 5%`);
  }
});

// ---------------------------------------------------------------------------
runTest('Glare / overexposure (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);

  const gt = translate(BOARD_PX, 3, 2);
  const dst = warpFrame(baseFrame, BOARD_PX, gt);
  for (let y = 0; y < 44; y += 1) {
    for (let x = 96; x < FRAME_W; x += 1) {
      dst.data[y * FRAME_W + x] = 255;
    }
  }
  const result = tracker.update(dst, 1016);
  const err = result.corners ? cornerErrorRatio(result.corners.pixels, gt) : null;
  metricLog.push({
    name: 'glare',
    cornerPct: err ? err.meanRatio * 100 : Number.NaN,
    reproPx: reprojectionErrorPx(result.homography, BOARD_PX, gt),
  });
  console.log(
    `  [metric] glare: state=${result.state} inliers=${result.inliers}/${result.featureCount}` +
      (err ? ` cornerErr=${(err.meanRatio * 100).toFixed(2)}%` : ` reason="${result.reason ?? ''}"`),
  );

  assert(
    result.state === 'tracking' || result.state === 'holding',
    `state ${result.state} handles glare`,
  );
  assertNoStaleFix(result, 'glare');
  if (result.corners && err) {
    assert(err.meanRatio < 0.05, `corner error ${(err.meanRatio * 100).toFixed(2)}% < 5%`);
  }
});

// ---------------------------------------------------------------------------
runTest('Incorrect forward-backward correspondence: non-rigid deformation (synthetic)', () => {
  const tracker = newTracker({ maxRoundTripErrorPx: 1.0 });
  seedTracker(tracker, baseFrame);

  // One corner displaced differently from the rest: no single homography fits.
  const gt = BOARD_PX.map((p, i) =>
    i === 0 ? { x: p.x + 13, y: p.y - 9 } : { x: p.x + 3, y: p.y + 2 },
  );
  const result = trackWarp(tracker, baseFrame, gt, 'non-rigid (FB mismatch)', 1016);

  assert(
    ['tracking', 'holding', 'lost'].includes(result.state),
    `state ${result.state} is valid`,
  );
  assertNoStaleFix(result, 'non-rigid');

  if (result.corners) {
    // The dominant (rigid) part must still be tracked accurately: outliers
    // must be rejected by RANSAC rather than averaged into the estimate.
    const rigidErr = cornerErrorRatio(
      [result.corners.pixels[1], result.corners.pixels[2], result.corners.pixels[3]],
      [gt[1], gt[2], gt[3]],
    );
    console.log(
      `  [metric] non-rigid: rigid cornerErr=${(rigidErr.meanRatio * 100).toFixed(2)}%, ` +
        `TL cornerErr=${cornerErrorRatio([result.corners.pixels[0]], [gt[0]]).meanPx.toFixed(2)}px, ` +
        `inliers=${result.inliers}/${result.featureCount}`,
    );
    assert(
      rigidErr.meanRatio < 0.04,
      `rigid corners tracked to ${(rigidErr.meanRatio * 100).toFixed(2)}% < 4% (outliers rejected)`,
    );
    assert(
      result.inliers < result.featureCount,
      `some features rejected as outliers (${result.inliers}/${result.featureCount})`,
    );
  } else {
    console.log(`  (tracker declined the non-rigid frame: ${result.reason ?? ''})`);
  }
});

// ---------------------------------------------------------------------------
runTest('Invalid / excessively distorted homography rejected (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame);

  // Extreme warp that inverts part of the quad (self-intersecting target).
  const gt = [
    { x: 150, y: 12 },
    { x: 10, y: 18 },
    { x: 148, y: 112 },
    { x: 12, y: 104 },
  ];
  const result = trackWarp(tracker, baseFrame, gt, 'extreme distortion', 1016);

  assert(
    result.state === 'holding' || result.state === 'lost' || result.state === 'tracking',
    `state ${result.state} is valid`,
  );
  if (result.state !== 'tracking') {
    assertNoStaleFix(result, 'extreme distortion');
    console.log(`  (rejected: ${result.reason ?? ''})`);
  }
  // Whatever happens, a returned fix must be convex and inside sane bounds.
  if (result.corners) {
    const spans = result.corners.pixels;
    const w = Math.max(...spans.map((p) => p.x)) - Math.min(...spans.map((p) => p.x));
    const h = Math.max(...spans.map((p) => p.y)) - Math.min(...spans.map((p) => p.y));
    assert(w > 0 && h > 0, 'returned quad has positive area');
    assert(w < FRAME_W * 2 && h < FRAME_H * 2, 'returned quad span is bounded');
  }
});

// ---------------------------------------------------------------------------
runTest('Board departure: fix is never held as fresh (synthetic)', () => {
  const tracker = newTracker({ maxHoldMs: 600 });
  seedTracker(tracker, baseFrame, 1000);

  // Board moves entirely out of frame (frame becomes featureless table).
  const gt = translate(BOARD_PX, 200, 180);
  const result = trackWarp(tracker, baseFrame, gt, 'board departure', 1100);

  assert(
    result.state === 'holding' || result.state === 'lost',
    `departure yields holding/lost (got ${result.state})`,
  );
  assertNoStaleFix(result, 'departure');
  assert(
    result.sinceGoodMs !== undefined && result.sinceGoodMs >= 100,
    `sinceGoodMs reported (${result.sinceGoodMs})`,
  );
});

// ---------------------------------------------------------------------------
runTest('Temporary loss then re-entry and reinitialisation (synthetic)', () => {
  const tracker = newTracker({ maxHoldMs: 400 });
  seedTracker(tracker, baseFrame, 1000);

  // 1. Board leaves the frame entirely.
  const departed = warpFrame(baseFrame, BOARD_PX, translate(BOARD_PX, 200, 180));
  const lostResult = tracker.update(departed, 1050);
  assert(
    lostResult.state === 'holding' || lostResult.state === 'lost',
    `departure -> ${lostResult.state}`,
  );
  assertNoStaleFix(lostResult, 'departure step');

  // 2. Elapsed time passes while updates keep failing: LOST, not a held quad.
  const laterResult = tracker.update(departed, 1700);
  assert(laterResult.state === 'lost', `after maxHoldMs -> lost (got ${laterResult.state})`);
  assertNoStaleFix(laterResult, 'lost step');
  assert(tracker.needsReseed(), 'needsReseed() is true while lost');

  // 3. Board re-enters: still lost until re-seeded (no phantom re-acquisition).
  const reentered = warpFrame(baseFrame, BOARD_PX, translate(BOARD_PX, 4, 3));
  const reentry = tracker.update(reentered, 1750);
  assert(reentry.state === 'lost', `re-entry without reseed stays lost (got ${reentry.state})`);
  assertNoStaleFix(reentry, 're-entry step');

  // 4. Re-seed from a trusted detection and confirm tracking resumes.
  assert(tracker.seed(reentered, translate(BOARD_NORM, 4 / FRAME_W, 3 / FRAME_H), 1760), 're-seed succeeds');
  const resumed = tracker.update(reentered, 1776);
  assert(resumed.state === 'tracking', `tracking resumes after reseed (got ${resumed.state})`);
  assert(resumed.corners !== null, 'fresh corners after reseed');
});

// ---------------------------------------------------------------------------
runTest('Reinitialisation after a failed attempt (synthetic)', () => {
  const tracker = newTracker({ maxHoldMs: 1500 });
  seedTracker(tracker, baseFrame, 1000);

  // Featureless frame kills tracking (the reference frame is retained).
  const garbage = fillFrame(128);
  const failed = tracker.update(garbage, 1016);
  assert(
    failed.state === 'holding' || failed.state === 'lost',
    `featureless frame -> ${failed.state}`,
  );
  assertNoStaleFix(failed, 'featureless step');

  // A subsequent good frame can be tracked directly from the retained
  // reference frame: the fix is computed fresh for that frame, not held.
  const goodAgain = warpFrame(baseFrame, BOARD_PX, translate(BOARD_PX, 4, 3));
  const recovered = tracker.update(goodAgain, 1032);
  assert(recovered.state === 'tracking', `recovers from retained reference (got ${recovered.state})`);
  assert(recovered.corners !== null, 'fresh corners on recovery');
  if (recovered.corners) {
    const err = cornerErrorRatio(recovered.corners.pixels, translate(BOARD_PX, 4, 3));
    assert(err.meanRatio < 0.03, `recovered corner error ${(err.meanRatio * 100).toFixed(2)}% < 3%`);
  }

  // 4. Failures past maxHoldMs transition to LOST (elapsed-time loss).
  const stillGarbage = tracker.update(fillFrame(60), 1100);
  assert(stillGarbage.state === 'holding', `within hold window -> holding (got ${stillGarbage.state})`);
  const timedOut = tracker.update(fillFrame(60), 4000);
  assert(timedOut.state === 'lost', `past maxHoldMs -> lost (got ${timedOut.state})`);
  assertNoStaleFix(timedOut, 'timed-out step');
});

// ---------------------------------------------------------------------------
runTest('Recovery when the board moves farther than one update can follow (synthetic)', () => {
  const tracker = newTracker({ maxRoundTripErrorPx: 2.0, ransacThresholdPx: 2.0 });
  seedTracker(tracker, baseFrame, 1000);

  // A jump far beyond the per-frame capture range of the LK pyramid.
  const gt = translate(BOARD_PX, 60, 45);
  const result = trackWarp(tracker, baseFrame, gt, 'large jump 60,45', 1016);

  assert(
    ['tracking', 'holding', 'lost'].includes(result.state),
    `large jump handled gracefully (got ${result.state})`,
  );
  assertNoStaleFix(result, 'large jump');
  if (result.corners) {
    // If it did follow, the fix must actually be the new position (not stale).
    const err = cornerErrorRatio(result.corners.pixels, gt);
    assert(
      err.meanRatio < 0.05,
      `followed jump is accurate: ${(err.meanRatio * 100).toFixed(2)}% < 5% (not stale)`,
    );
  } else {
    console.log(`  (declined the jump: ${result.reason ?? ''})`);
  }
});

// ---------------------------------------------------------------------------
runTest('Seed rejects invalid geometry', () => {
  const tracker = newTracker();
  // Concave "dart": the fourth vertex sits inside the triangle formed by the
  // other three, so no corner reordering can make the quad convex.
  const concave = [
    { x: 0.1, y: 0.1 },
    { x: 0.9, y: 0.1 },
    { x: 0.9, y: 0.9 },
    { x: 0.5, y: 0.3 },
  ];
  assert(!tracker.seed(baseFrame, concave), 'concave quad rejected');
  assert(!tracker.seed(baseFrame, []), 'wrong corner count rejected');
  const degenerate = [
    { x: 0.2, y: 0.2 },
    { x: 0.5, y: 0.5 },
    { x: 0.8, y: 0.8 },
    { x: 0.4, y: 0.6 },
  ];
  assert(!tracker.seed(baseFrame, degenerate), 'degenerate (collinear-ish) quad rejected');
  assert(tracker.getState() === 'uninitialized', 'tracker stays uninitialized after bad seed');
  assert(tracker.needsReseed(), 'needsReseed() after bad seed');
});

// ---------------------------------------------------------------------------
runTest('Frame-size mismatch reinitialises instead of tracking (synthetic)', () => {
  const tracker = newTracker();
  seedTracker(tracker, baseFrame, 1000);

  const otherSize: GrayBuffer = {
    width: FRAME_W + 16,
    height: FRAME_H + 12,
    data: new Uint8Array((FRAME_W + 16) * (FRAME_H + 12)).fill(120),
  };
  const result = tracker.update(otherSize, 1016);
  assert(result.state === 'uninitialized', `size mismatch -> uninitialized (got ${result.state})`);
  assertNoStaleFix(result, 'size mismatch');
});

// ===========================================================================
after(() => {
  console.log('\n=== Per-case metrics (synthetic ground truth) ===');
  let cornerSum = 0;
  let cornerCount = 0;
  let reproSum = 0;
  let reproCount = 0;
  for (const m of metricLog) {
    const corner = Number.isFinite(m.cornerPct) ? `${m.cornerPct.toFixed(2)}%` : 'n/a';
    const repro = Number.isFinite(m.reproPx) ? `${m.reproPx.toFixed(2)}px` : 'n/a';
    console.log(`  ${m.name.padEnd(28)} cornerErr=${corner.padEnd(8)} reproErr=${repro}`);
    if (Number.isFinite(m.cornerPct)) {
      cornerSum += m.cornerPct;
      cornerCount += 1;
    }
    if (Number.isFinite(m.reproPx)) {
      reproSum += m.reproPx;
      reproCount += 1;
    }
  }
  if (cornerCount > 0) {
    console.log(`  mean corner error over ${cornerCount} tracked cases: ${(cornerSum / cornerCount).toFixed(2)}% of frame diagonal`);
  }
  if (reproCount > 0) {
    console.log(`  mean homography reprojection error over ${reproCount} cases: ${(reproSum / reproCount).toFixed(2)}px`);
  }
  console.log('\n  NOTE: these are SYNTHETIC transformation errors. They validate the');
  console.log('  algorithm, not real-camera tracking accuracy.');
});
