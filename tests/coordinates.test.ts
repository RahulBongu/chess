/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  clampUnit,
  computeAspectFitDisplaySize,
  isNormalizedCornerSet,
  normalizePreviewTouch,
  normalizedToPixelCorners,
  normalizedToPixelPoint,
  pixelToNormalizedPoint,
} from '../src/calibration/coordinates.ts';

test('preview-local touches normalize against the exact preview box', () => {
  assert.deepEqual(normalizePreviewTouch(0, 0, 400, 300), { x: 0, y: 0 });
  assert.deepEqual(normalizePreviewTouch(200, 150, 400, 300), { x: 0.5, y: 0.5 });
  assert.deepEqual(normalizePreviewTouch(400, 300, 400, 300), { x: 1, y: 1 });
});

test('touches at or just outside preview boundaries are clamped, not wrapped or negative', () => {
  assert.deepEqual(normalizePreviewTouch(-3, -2, 400, 300), { x: 0, y: 0 });
  assert.deepEqual(normalizePreviewTouch(403, 302, 400, 300), { x: 1, y: 1 });
  assert.deepEqual(normalizePreviewTouch(Number.NaN, 10, 400, 300), { x: 0, y: 10 / 300 });
  assert.equal(clampUnit(-0.5), 0);
  assert.equal(clampUnit(1.5), 1);
});

test('zero or negative preview dimensions are rejected instead of producing NaN coordinates', () => {
  assert.throws(() => normalizePreviewTouch(10, 10, 0, 300), /finite and positive/);
  assert.throws(() => normalizePreviewTouch(10, 10, 400, -1), /finite and positive/);
  assert.throws(() => computeAspectFitDisplaySize(0, 300, 400), /finite and positive/);
});

test('landscape and portrait images produce the expected display boxes', () => {
  const landscape = computeAspectFitDisplaySize(4000, 3000, 480);
  assert.deepEqual(landscape, { width: 480, height: 360 });

  const portrait = computeAspectFitDisplaySize(3000, 4000, 480);
  assert.deepEqual(portrait, { width: 480, height: 640 });

  const square = computeAspectFitDisplaySize(3000, 3000, 390);
  assert.deepEqual(square, { width: 390, height: 390 });
});

test('aspect-ratio rounding error is bounded by half a display pixel', () => {
  const cases = [
    { W: 4000, H: 3000 },
    { W: 3000, H: 4000 },
    { W: 3840, H: 2160 },
    { W: 2160, H: 3840 },
    { W: 4284, H: 5712 },
  ];

  for (const { W, H } of cases) {
    for (const maxWidth of [320, 360, 390, 420, 480]) {
      const display = computeAspectFitDisplaySize(W, H, maxWidth);
      const exactHeight = maxWidth * (H / W);
      assert.ok(Math.abs(display.height - exactHeight) <= 0.5 + 1e-12);
      const relativeError = Math.abs(W / H - display.width / display.height) / (W / H);
      assert.ok(relativeError < 0.0035, `${W}x${H} @ ${maxWidth}: ${relativeError}`);
    }
  }
});

test('normalized coordinates map to continuous image-boundary coordinates', () => {
  const topLeft = normalizedToPixelPoint({ x: 0, y: 0 }, 4000, 3000);
  const bottomRight = normalizedToPixelPoint({ x: 1, y: 1 }, 4000, 3000);

  assert.deepEqual(topLeft, { x: 0, y: 0 });
  // x = 1 is the right image boundary, not the last pixel center W - 1.
  assert.deepEqual(bottomRight, { x: 4000, y: 3000 });

  const pixelCorners = normalizedToPixelCorners(
    [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ],
    4000,
    3000,
  );
  assert.deepEqual(pixelCorners, [
    { x: 0, y: 0 },
    { x: 4000, y: 0 },
    { x: 4000, y: 3000 },
    { x: 0, y: 3000 },
  ]);
});

test('normalized and pixel conversions round-trip for portrait and landscape inputs', () => {
  const points = [
    { x: 0.0125, y: 0.9875 },
    { x: 0.5, y: 0.5 },
    { x: 0.999, y: 0.001 },
    { x: 0.3333, y: 0.6667 },
  ];

  for (const { width, height } of [
    { width: 4000, height: 3000 },
    { width: 3000, height: 4000 },
    { width: 4284, height: 5712 },
  ]) {
    for (const point of points) {
      const pixel = normalizedToPixelPoint(point, width, height);
      const roundTrip = pixelToNormalizedPoint(pixel, width, height);
      assert.ok(Math.abs(roundTrip.x - point.x) < 1e-12);
      assert.ok(Math.abs(roundTrip.y - point.y) < 1e-12);
    }
  }
});

test('normalized corner payload recognition distinguishes normalized from pixel payloads', () => {
  assert.equal(
    isNormalizedCornerSet([
      { x: 0.1, y: 0.1 },
      { x: 0.9, y: 0.1 },
      { x: 0.9, y: 0.9 },
      { x: 0.1, y: 0.9 },
    ]),
    true,
  );
  assert.equal(
    isNormalizedCornerSet([
      { x: 100, y: 100 },
      { x: 700, y: 100 },
      { x: 700, y: 700 },
      { x: 100, y: 700 },
    ]),
    false,
  );
  assert.equal(isNormalizedCornerSet([{ x: 0.5, y: 0.5 }]), false);
});
