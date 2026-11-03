/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  algebraicToColRow,
  colRowToAlgebraic,
  computeMoveArrowGeometry,
  computePerspectiveGridLines,
  computeSquarePerspectiveCenter,
  computeSquarePerspectiveQuad,
  getBoardHomography,
  type Point2D,
} from '../src/vision/perspectiveOverlay.ts';
import { isConvexQuadrilateral } from '../src/vision/boardDetection.ts';

const UNIT_SQUARE_CORNERS: Point2D[] = [
  { x: 0, y: 0 },
  { x: 400, y: 0 },
  { x: 400, y: 400 },
  { x: 0, y: 400 },
];

// Angled trapezoid representing steep perspective looking from bottom to top
const PERSPECTIVE_TRAPEZOID: Point2D[] = [
  { x: 100, y: 50 },  // top-left (narrow, distant)
  { x: 300, y: 50 },  // top-right (narrow, distant)
  { x: 380, y: 350 }, // bottom-right (wide, close)
  { x: 20, y: 350 },  // bottom-left (wide, close)
];

test('algebraicToColRow maps standard files and ranks correctly under white_bottom', () => {
  // a8 is top-left in screen space (col 0, row 0)
  assert.deepEqual(algebraicToColRow('a8', 'white_bottom'), { col: 0, row: 0 });
  // h8 is top-right (col 7, row 0)
  assert.deepEqual(algebraicToColRow('h8', 'white_bottom'), { col: 7, row: 0 });
  // a1 is bottom-left (col 0, row 7)
  assert.deepEqual(algebraicToColRow('a1', 'white_bottom'), { col: 0, row: 7 });
  // h1 is bottom-right (col 7, row 7)
  assert.deepEqual(algebraicToColRow('h1', 'white_bottom'), { col: 7, row: 7 });
  // e4 is col 4, row 4
  assert.deepEqual(algebraicToColRow('e4', 'white_bottom'), { col: 4, row: 4 });
});

test('algebraicToColRow maps correctly under black_bottom', () => {
  // From Black perspective:
  // h1 is top-left in screen space (col 0, row 0)
  assert.deepEqual(algebraicToColRow('h1', 'black_bottom'), { col: 0, row: 0 });
  // a1 is top-right (col 7, row 0)
  assert.deepEqual(algebraicToColRow('a1', 'black_bottom'), { col: 7, row: 0 });
  // h8 is bottom-left (col 0, row 7)
  assert.deepEqual(algebraicToColRow('h8', 'black_bottom'), { col: 0, row: 7 });
  // a8 is bottom-right (col 7, row 7)
  assert.deepEqual(algebraicToColRow('a8', 'black_bottom'), { col: 7, row: 7 });
});

test('colRowToAlgebraic round-trips correctly for all 64 squares', () => {
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  for (const orientation of ['white_bottom', 'black_bottom'] as const) {
    for (const file of files) {
      for (let rank = 1; rank <= 8; rank += 1) {
        const sq = `${file}${rank}`;
        const { col, row } = algebraicToColRow(sq, orientation);
        const reconstructed = colRowToAlgebraic(col, row, orientation);
        assert.equal(reconstructed, sq, `Failed round-trip for ${sq} under ${orientation}`);
      }
    }
  }
});

test('getBoardHomography returns valid homography for strictly convex quads, null for invalid', () => {
  const hSquare = getBoardHomography(UNIT_SQUARE_CORNERS);
  assert.ok(hSquare !== null);

  const hTrapezoid = getBoardHomography(PERSPECTIVE_TRAPEZOID);
  assert.ok(hTrapezoid !== null);

  // Self-intersecting / concave quad must fail
  const concave: Point2D[] = [
    { x: 0, y: 0 },
    { x: 200, y: 150 }, // indented
    { x: 400, y: 0 },
    { x: 400, y: 400 },
  ];
  assert.equal(getBoardHomography(concave), null);

  // Less than 4 points must fail
  assert.equal(getBoardHomography(UNIT_SQUARE_CORNERS.slice(0, 3)), null);
});

test('computeSquarePerspectiveQuad projects strictly convex quads on perspective board', () => {
  const e4Quad = computeSquarePerspectiveQuad(PERSPECTIVE_TRAPEZOID, 'e4', 'white_bottom');
  assert.ok(e4Quad !== null);
  assert.equal(e4Quad.length, 4);
  assert.ok(isConvexQuadrilateral(e4Quad));

  // Verify corners are within bounding hull of the trapezoid
  for (const pt of e4Quad) {
    assert.ok(pt.x >= 20 && pt.x <= 380);
    assert.ok(pt.y >= 50 && pt.y <= 350);
  }
});

test('perspective projection exhibits true foreshortening unlike linear interpolation', () => {
  // Under perspective, a rank near the bottom (rank 1, row 7) is physically closer and
  // subtends more vertical screen pixels than a distant rank near the top (rank 8, row 0).
  const distantQuad = computeSquarePerspectiveQuad(PERSPECTIVE_TRAPEZOID, 'e8', 'white_bottom')!;
  const nearQuad = computeSquarePerspectiveQuad(PERSPECTIVE_TRAPEZOID, 'e1', 'white_bottom')!;

  const distantHeight = Math.abs(distantQuad[2].y - distantQuad[1].y);
  const nearHeight = Math.abs(nearQuad[2].y - nearQuad[1].y);

  // In perspective trapezoid, nearer squares MUST be vertically taller than distant squares
  assert.ok(
    nearHeight > distantHeight * 1.5,
    `Expected nearHeight (${nearHeight.toFixed(2)}) > 1.5 * distantHeight (${distantHeight.toFixed(2)})`,
  );
});

test('computeMoveArrowGeometry creates forward pointing arrow with correctly angled wings', () => {
  const arrow = computeMoveArrowGeometry(UNIT_SQUARE_CORNERS, 'e2', 'e4', 'white_bottom', 15);
  assert.ok(arrow !== null);

  // e2 to e4 goes upward in screen space: y decreases
  assert.ok(arrow.to.y < arrow.from.y);
  // e-file maintains same x
  assert.ok(Math.abs(arrow.to.x - arrow.from.x) < 1e-4);

  // Arrowhead wings should extend below the destination point (backwards along vector)
  assert.ok(arrow.headLeft.y > arrow.to.y);
  assert.ok(arrow.headRight.y > arrow.to.y);
  // One wing to the left, one wing to the right of destination center
  const minX = Math.min(arrow.headLeft.x, arrow.headRight.x);
  const maxX = Math.max(arrow.headLeft.x, arrow.headRight.x);
  assert.ok(minX < arrow.to.x);
  assert.ok(maxX > arrow.to.x);
});

test('computePerspectiveGridLines generates 14 internal grid lines spanning the board', () => {
  const lines = computePerspectiveGridLines(PERSPECTIVE_TRAPEZOID);
  assert.ok(lines !== null);
  assert.equal(lines.length, 14); // 7 vertical + 7 horizontal

  // Check vertical lines span from top edge to bottom edge
  const vertical = lines.filter((l) => l.key.startsWith('v-'));
  assert.equal(vertical.length, 7);
  for (const vLine of vertical) {
    assert.ok(Math.abs(vLine.from.y - 50) < 1e-3); // top y
    assert.ok(Math.abs(vLine.to.y - 350) < 1e-3); // bottom y
  }

  // Check horizontal lines span from left edge to right edge
  const horizontal = lines.filter((l) => l.key.startsWith('h-'));
  assert.equal(horizontal.length, 7);
  for (const hLine of horizontal) {
    assert.ok(hLine.from.x < hLine.to.x);
  }
});
