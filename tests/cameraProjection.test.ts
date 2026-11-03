/**
 * Dedicated tests for `src/vision/cameraProjection.ts` (Milestone 5.4D).
 *
 * Covers the full sensor -> view mapping used by the live overlay:
 *   - portrait/landscape rotations (0/90/180/270)
 *   - varied frame and view aspect ratios (FILL_CENTER crop offsets)
 *   - mirrored and unmirrored views
 *   - points at and near the board/frame edges
 *   - exact forward/inverse round-trip error
 *   - quantitative registration error of the naive `u * viewWidth` mapping
 *     that the composed transform exists to replace
 *
 * All numbers here are transform unit-test errors computed in Node. They say
 * nothing about real-camera overlay registration, which still requires
 * measured footage from the target phone.
 */

import { test } from 'node:test';
import nodeAssert from 'node:assert/strict';

import {
  createFrameToViewTransform,
  mapNormalizedCornersToView,
  type FrameToViewOptions,
} from '../src/vision/cameraProjection.ts';

function assert(condition: boolean, message: string): void {
  nodeAssert.ok(condition, message);
}

function runTest(name: string, fn: () => void): void {
  test(name, fn);
}

function maxRoundTripError(options: FrameToViewOptions): number {
  const transform = createFrameToViewTransform(options);
  let worst = 0;
  for (let v = 0; v <= 1.0001; v += 0.1) {
    for (let u = 0; u <= 1.0001; u += 0.1) {
      const view = transform.mapNormalized(u, v);
      const back = transform.unmapView(view.x, view.y);
      worst = Math.max(worst, Math.abs(back.u - u), Math.abs(back.v - v));
    }
  }
  return worst;
}

function maxInverseRoundTripError(options: FrameToViewOptions): number {
  const transform = createFrameToViewTransform(options);
  let worst = 0;
  for (let y = 0; y <= options.viewHeight; y += options.viewHeight / 8) {
    for (let x = 0; x <= options.viewWidth; x += options.viewWidth / 8) {
      const norm = transform.unmapView(x, y);
      const view = transform.mapNormalized(norm.u, norm.v);
      worst = Math.max(worst, Math.abs(view.x - x), Math.abs(view.y - y));
    }
  }
  return worst;
}

const CONFIGS: { label: string; options: FrameToViewOptions }[] = [
  {
    label: '320x240 frame, 420x315 view, rot 0 (4:3 match)',
    options: { frameWidth: 320, frameHeight: 240, rotationDegrees: 0, viewWidth: 420, viewHeight: 315 },
  },
  {
    label: '320x240 frame, 420x315 view, rot 90',
    options: { frameWidth: 320, frameHeight: 240, rotationDegrees: 90, viewWidth: 420, viewHeight: 315 },
  },
  {
    label: '320x240 frame, 420x315 view, rot 180',
    options: { frameWidth: 320, frameHeight: 240, rotationDegrees: 180, viewWidth: 420, viewHeight: 315 },
  },
  {
    label: '320x240 frame, 420x315 view, rot 270',
    options: { frameWidth: 320, frameHeight: 240, rotationDegrees: 270, viewWidth: 420, viewHeight: 315 },
  },
  {
    label: '160x120 tracking frame, 420x400 view (aspect mismatch)',
    options: { frameWidth: 160, frameHeight: 120, rotationDegrees: 0, viewWidth: 420, viewHeight: 400 },
  },
  {
    label: '640x480 frame, 400x700 portrait view (heavy crop)',
    options: { frameWidth: 640, frameHeight: 480, rotationDegrees: 0, viewWidth: 400, viewHeight: 700 },
  },
  {
    label: '320x240 frame, 420x315 view, rot 90, mirrored',
    options: {
      frameWidth: 320,
      frameHeight: 240,
      rotationDegrees: 90,
      viewWidth: 420,
      viewHeight: 315,
      mirror: 'horizontal',
    },
  },
  {
    label: '160x120 tracking frame, 420x400 view, rot 270, mirrored',
    options: {
      frameWidth: 160,
      frameHeight: 120,
      rotationDegrees: 270,
      viewWidth: 420,
      viewHeight: 400,
      mirror: 'horizontal',
    },
  },
];

// ---------------------------------------------------------------------------
console.log('=== cameraProjection Tests (Milestone 5.4D) ===\n');

runTest('Forward/inverse round-trip error is float-rounding only', () => {
  let worstForward = 0;
  let worstInverse = 0;
  for (const config of CONFIGS) {
    const f = maxRoundTripError(config.options);
    const b = maxInverseRoundTripError(config.options);
    worstForward = Math.max(worstForward, f);
    worstInverse = Math.max(worstInverse, b);
    console.log(`  [metric] ${config.label}: fwd=${f.toExponential(2)} inv=${b.toExponential(2)}`);
  }
  assert(worstForward < 1e-9, `max forward round-trip error ${worstForward.toExponential(2)} < 1e-9`);
  assert(worstInverse < 1e-6, `max inverse round-trip error ${worstInverse.toExponential(2)} < 1e-6`);
});

runTest('FILL_CENTER always covers the view (content never letterboxed)', () => {
  for (const config of CONFIGS) {
    const t = createFrameToViewTransform(config.options);
    assert(
      t.displayWidth * t.scale >= config.options.viewWidth - 1e-9 &&
        t.displayHeight * t.scale >= config.options.viewHeight - 1e-9,
      `${config.label}: content covers view (scale=${t.scale.toFixed(3)})`,
    );
    // At least one axis fits exactly (the other may overflow and be cropped).
    const fitsWidth = Math.abs(t.displayWidth * t.scale - config.options.viewWidth) < 1e-9;
    const fitsHeight = Math.abs(t.displayHeight * t.scale - config.options.viewHeight) < 1e-9;
    assert(fitsWidth || fitsHeight, `${config.label}: one axis fits exactly`);
  }
});

runTest('Frame centre maps to view centre for every rotation and mirror', () => {
  for (const config of CONFIGS) {
    const t = createFrameToViewTransform(config.options);
    const c = t.mapNormalized(0.5, 0.5);
    const err = Math.hypot(c.x - config.options.viewWidth / 2, c.y - config.options.viewHeight / 2);
    assert(err < 1e-9, `${config.label}: centre error ${err.toExponential(2)}px`);
  }
});

runTest('Frame corners map to the rotated content corners', () => {
  for (const config of CONFIGS) {
    const t = createFrameToViewTransform(config.options);
    const corners = [
      t.mapNormalized(0, 0),
      t.mapNormalized(1, 0),
      t.mapNormalized(1, 1),
      t.mapNormalized(0, 1),
    ];
    const left = Math.min(...corners.map((p) => p.x));
    const right = Math.max(...corners.map((p) => p.x));
    const top = Math.min(...corners.map((p) => p.y));
    const bottom = Math.max(...corners.map((p) => p.y));
    const expected = {
      left: t.offsetX,
      right: t.offsetX + t.displayWidth * t.scale,
      top: t.offsetY,
      bottom: t.offsetY + t.displayHeight * t.scale,
    };
    const err = Math.max(
      Math.abs(left - expected.left),
      Math.abs(right - expected.right),
      Math.abs(top - expected.top),
      Math.abs(bottom - expected.bottom),
    );
    assert(err < 1e-9, `${config.label}: corner span error ${err.toExponential(2)}px`);
  }
});

runTest('Rotation 90 maps the frame top-left to the view top-right', () => {
  const t = createFrameToViewTransform({
    frameWidth: 320,
    frameHeight: 240,
    rotationDegrees: 90,
    viewWidth: 360,
    viewHeight: 480,
  });
  // CW 90: display is 240x320, scale = max(360/240, 480/320) = 1.5, offsets 0.
  assert(Math.abs(t.scale - 1.5) < 1e-12, `scale ${t.scale} == 1.5`);
  const tl = t.mapNormalized(0, 0);
  assert(Math.abs(tl.x - 360) < 1e-9 && Math.abs(tl.y - 0) < 1e-9, `TL -> (360, 0), got (${tl.x.toFixed(2)}, ${tl.y.toFixed(2)})`);
  const br = t.mapNormalized(1, 1);
  assert(Math.abs(br.x - 0) < 1e-9 && Math.abs(br.y - 480) < 1e-9, `BR -> (0, 480), got (${br.x.toFixed(2)}, ${br.y.toFixed(2)})`);
});

runTest('Rotation 180 maps the frame top-left to the view bottom-right', () => {
  const t = createFrameToViewTransform({
    frameWidth: 400,
    frameHeight: 300,
    rotationDegrees: 180,
    viewWidth: 400,
    viewHeight: 300,
  });
  const tl = t.mapNormalized(0, 0);
  assert(Math.abs(tl.x - 400) < 1e-9 && Math.abs(tl.y - 300) < 1e-9, `TL -> (400, 300), got (${tl.x.toFixed(2)}, ${tl.y.toFixed(2)})`);
});

runTest('Rotation 270 maps the frame top-left to the view bottom-left', () => {
  const t = createFrameToViewTransform({
    frameWidth: 400,
    frameHeight: 300,
    rotationDegrees: 270,
    viewWidth: 300,
    viewHeight: 400,
  });
  const tl = t.mapNormalized(0, 0);
  assert(Math.abs(tl.x - 0) < 1e-9 && Math.abs(tl.y - 400) < 1e-9, `TL -> (0, 400), got (${tl.x.toFixed(2)}, ${tl.y.toFixed(2)})`);
});

runTest('Crop offsets: overflow is cropped symmetrically about the centre', () => {
  const t = createFrameToViewTransform({
    frameWidth: 640,
    frameHeight: 480,
    rotationDegrees: 0,
    viewWidth: 400,
    viewHeight: 700,
  });
  // scale = max(400/640, 700/480) = 1.4583... -> width overflows
  assert(t.offsetX < 0, `offsetX ${t.offsetX.toFixed(2)} < 0 (horizontal crop)`);
  assert(Math.abs(t.offsetY) < 1e-9, `offsetY ${t.offsetY} == 0 (vertical fits)`);
  const dispW = t.displayWidth * t.scale;
  const dispH = t.displayHeight * t.scale;
  assert(dispW >= 400 - 1e-9, `content width ${dispW.toFixed(1)} >= view width 400`);
  assert(Math.abs(dispH - 700) < 1e-9, `content height ${dispH.toFixed(1)} == view height 700`);
  // Symmetry: left crop equals right crop.
  const leftCrop = -t.offsetX;
  const rightCrop = dispW - 400 - leftCrop;
  assert(Math.abs(leftCrop - rightCrop) < 1e-9, `left crop ${leftCrop.toFixed(2)} == right crop ${rightCrop.toFixed(2)}`);
});

runTest('Mirroring is an exact horizontal reflection about the view centre', () => {
  for (const rotation of [0, 90, 180, 270]) {
    const base: FrameToViewOptions = {
      frameWidth: 320,
      frameHeight: 240,
      rotationDegrees: rotation,
      viewWidth: 420,
      viewHeight: 315,
    };
    const plain = createFrameToViewTransform(base);
    const mirrored = createFrameToViewTransform({ ...base, mirror: 'horizontal' });
    let worst = 0;
    for (let v = 0; v <= 1.0001; v += 0.25) {
      for (let u = 0; u <= 1.0001; u += 0.25) {
        const a = plain.mapNormalized(u, v);
        const b = mirrored.mapNormalized(u, v);
        worst = Math.max(worst, Math.abs(b.x - (base.viewWidth - a.x)), Math.abs(b.y - a.y));
      }
    }
    assert(worst < 1e-9, `rot ${rotation}: mirror reflection error ${worst.toExponential(2)}px`);
    // And the mirror inverse still round-trips.
    assert(
      maxRoundTripError({ ...base, mirror: 'horizontal' }) < 1e-9,
      `rot ${rotation}: mirrored round-trip error < 1e-9`,
    );
  }
});

runTest('mapPixels agrees with mapNormalized on the same physical point', () => {
  for (const config of CONFIGS) {
    const t = createFrameToViewTransform(config.options);
    let worst = 0;
    for (let y = 0; y <= config.options.frameHeight; y += config.options.frameHeight / 5) {
      for (let x = 0; x <= config.options.frameWidth; x += config.options.frameWidth / 5) {
        const a = t.mapPixels(x, y);
        const b = t.mapNormalized(x / config.options.frameWidth, y / config.options.frameHeight);
        worst = Math.max(worst, Math.abs(a.x - b.x), Math.abs(a.y - b.y));
      }
    }
    assert(worst < 1e-9, `${config.label}: mapPixels/mapNormalized agree (${worst.toExponential(2)}px)`);
  }
});

runTest('Board-edge points stay inside the displayed content rect', () => {
  for (const config of CONFIGS) {
    const t = createFrameToViewTransform(config.options);
    const edges = [
      { u: 0, v: 0 },
      { u: 1, v: 0 },
      { u: 1, v: 1 },
      { u: 0, v: 1 },
      { u: 0.5, v: 0 },
      { u: 0.5, v: 1 },
      { u: 0, v: 0.5 },
      { u: 1, v: 0.5 },
    ];
    let ok = true;
    for (const e of edges) {
      const p = t.mapNormalized(e.u, e.v);
      const eps = 1e-9;
      if (
        p.x < t.offsetX - eps ||
        p.x > t.offsetX + t.displayWidth * t.scale + eps ||
        p.y < t.offsetY - eps ||
        p.y > t.offsetY + t.displayHeight * t.scale + eps ||
        !Number.isFinite(p.x) ||
        !Number.isFinite(p.y)
      ) {
        ok = false;
      }
    }
    assert(ok, `${config.label}: all 8 edge points map inside the content rect`);
  }
});

runTest('Overlay registration: convex board quad stays convex and oriented', () => {
  const board = [
    { x: 0.1, y: 0.15 },
    { x: 0.9, y: 0.12 },
    { x: 0.88, y: 0.85 },
    { x: 0.08, y: 0.88 },
  ];
  const signedArea = (pts: { x: number; y: number }[]) => {
    let area = 0;
    for (let i = 0; i < pts.length; i += 1) {
      const p = pts[i];
      const q = pts[(i + 1) % pts.length];
      area += p.x * q.y - q.x * p.y;
    }
    return area / 2;
  };
  const reference = signedArea(board);
  for (const config of CONFIGS) {
    const mapped = mapNormalizedCornersToView(board, config.options);
    const area = signedArea(mapped);
    // Horizontal mirroring legitimately flips the winding; rotations do not.
    const mirrored = config.options.mirror === 'horizontal';
    const expectedSign = mirrored ? -Math.sign(reference) : Math.sign(reference);
    assert(
      Math.sign(area) === expectedSign,
      `${config.label}: quad winding preserved (area ${area.toFixed(0)})`,
    );
    let convex = true;
    for (let i = 0; i < 4; i += 1) {
      const a = mapped[i];
      const b = mapped[(i + 1) % 4];
      const c = mapped[(i + 2) % 4];
      const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
      if (cross === 0 || Math.sign(cross) !== expectedSign) convex = false;
    }
    assert(convex, `${config.label}: mapped quad remains strictly convex`);
  }
});

runTest('Naive u*viewWidth mapping would misregister the overlay', () => {
  // This documents WHY the composed transform exists: for rotated frames and
  // aspect-mismatched FILL_CENTER crops the naive mapping is wrong by a
  // large, easily visible amount.
  const cases: { options: FrameToViewOptions; label: string }[] = [
    {
      label: 'rot 90, 4:3 frame into portrait view',
      options: { frameWidth: 320, frameHeight: 240, rotationDegrees: 90, viewWidth: 360, viewHeight: 480 },
    },
    {
      label: 'aspect mismatch (160x120 into 420x400)',
      options: { frameWidth: 160, frameHeight: 120, rotationDegrees: 0, viewWidth: 420, viewHeight: 400 },
    },
    {
      label: 'heavy vertical crop (640x480 into 400x700)',
      options: { frameWidth: 640, frameHeight: 480, rotationDegrees: 0, viewWidth: 400, viewHeight: 700 },
    },
  ];
  const board = [
    { u: 0.1, v: 0.15 },
    { u: 0.9, v: 0.12 },
    { u: 0.88, v: 0.85 },
    { u: 0.08, v: 0.88 },
  ];
  for (const c of cases) {
    const t = createFrameToViewTransform(c.options);
    let worst = 0;
    for (const p of board) {
      const correct = t.mapNormalized(p.u, p.v);
      const naive = { x: p.u * c.options.viewWidth, y: p.v * c.options.viewHeight };
      worst = Math.max(worst, Math.hypot(correct.x - naive.x, correct.y - naive.y));
    }
    console.log(`  [metric] ${c.label}: naive-mapping registration error = ${worst.toFixed(1)}px`);
    assert(worst > 10, `${c.label}: naive mapping is off by ${worst.toFixed(1)}px (> 10px)`);
  }
});

runTest('Invalid inputs throw instead of silently misregistering', () => {
  let threw = false;
  try {
    createFrameToViewTransform({
      frameWidth: 320,
      frameHeight: 240,
      rotationDegrees: 45,
      viewWidth: 400,
      viewHeight: 300,
    });
  } catch {
    threw = true;
  }
  assert(threw, 'rotation 45 throws');

  threw = false;
  try {
    createFrameToViewTransform({
      frameWidth: 0,
      frameHeight: 240,
      rotationDegrees: 0,
      viewWidth: 400,
      viewHeight: 300,
    });
  } catch {
    threw = true;
  }
  assert(threw, 'zero frame width throws');

  threw = false;
  try {
    createFrameToViewTransform({
      frameWidth: 320,
      frameHeight: 240,
      rotationDegrees: 0,
      viewWidth: -10,
      viewHeight: 300,
    });
  } catch {
    threw = true;
  }
  assert(threw, 'negative view width throws');
});

runTest('mapNormalizedCornersToView matches the transform exactly', () => {
  const board = [
    { x: 0.1, y: 0.15 },
    { x: 0.9, y: 0.12 },
    { x: 0.88, y: 0.85 },
    { x: 0.08, y: 0.88 },
  ];
  for (const config of CONFIGS) {
    const t = createFrameToViewTransform(config.options);
    const viaHelper = mapNormalizedCornersToView(board, config.options);
    const viaTransform = board.map((p) => t.mapNormalized(p.x, p.y));
    let worst = 0;
    for (let i = 0; i < 4; i += 1) {
      worst = Math.max(worst, Math.hypot(viaHelper[i].x - viaTransform[i].x, viaHelper[i].y - viaTransform[i].y));
    }
    assert(worst < 1e-12, `${config.label}: helper agrees (${worst.toExponential(2)}px)`);
  }
});

console.log('\n  NOTE: these are transform unit-test errors measured in Node.');
console.log('  They are NOT measured overlay registration on camera footage;');
