/**
 * Perspective Overlay & Projective Geometry Engine for Handheld Chess Navigation.
 *
 * Maps chessboard coordinates and move vectors to the live camera preview using
 * the projective homography from unit-square space [0, 1]^2 to the detected four corners.
 */

import {
  projectSquarePoint,
  squareToQuadrilateral,
  type SquareHomography,
} from './boardEvidence.ts';
import { isConvexQuadrilateral } from './boardDetection.ts';
import type { DetectedCorner } from './types.ts';

export type BoardOrientation = 'white_bottom' | 'black_bottom';

export interface Point2D {
  x: number;
  y: number;
}

export type PerspectiveQuad = [Point2D, Point2D, Point2D, Point2D];

export interface GridLine {
  key: string;
  from: Point2D;
  to: Point2D;
}

export interface ArrowGeometry {
  from: Point2D;
  to: Point2D;
  headLeft: Point2D;
  headRight: Point2D;
}

/**
 * Converts algebraic chess notation (e.g. 'e4') to unit-square grid indices
 * [0..7, 0..7] based on board orientation.
 */
export function algebraicToColRow(
  square: string,
  orientation: BoardOrientation = 'white_bottom',
): { col: number; row: number } {
  if (!square || square.length < 2) {
    throw new Error(`Invalid algebraic square notation: "${square}"`);
  }

  const fileChar = square[0].toLowerCase();
  const rankChar = square[1];
  const fileIndex = fileChar.charCodeAt(0) - 97; // 'a' -> 0, 'h' -> 7
  const rank = parseInt(rankChar, 10); // 1..8

  if (fileIndex < 0 || fileIndex > 7 || Number.isNaN(rank) || rank < 1 || rank > 8) {
    throw new Error(`Invalid algebraic square coordinate: "${square}"`);
  }

  if (orientation === 'white_bottom') {
    return {
      col: fileIndex,
      row: 8 - rank,
    };
  }

  return {
    col: 7 - fileIndex,
    row: rank - 1,
  };
}

/**
 * Converts unit-square grid indices [0..7, 0..7] back to algebraic chess notation.
 */
export function colRowToAlgebraic(
  col: number,
  row: number,
  orientation: BoardOrientation = 'white_bottom',
): string {
  if (col < 0 || col > 7 || row < 0 || row > 7) {
    throw new Error(`Grid index out of bounds: col=${col}, row=${row}`);
  }

  if (orientation === 'white_bottom') {
    const file = String.fromCharCode(97 + col);
    const rank = 8 - row;
    return `${file}${rank}`;
  }

  const file = String.fromCharCode(97 + (7 - col));
  const rank = row + 1;
  return `${file}${rank}`;
}

/**
 * Helper to compute the homography from board corners.
 * Requires four finite corners forming a strictly convex quadrilateral.
 */
export function getBoardHomography(corners: Point2D[]): SquareHomography | null {
  if (!corners || corners.length !== 4) {
    return null;
  }

  const detected: DetectedCorner[] = corners.map((pt) => ({ x: pt.x, y: pt.y }));
  if (!isConvexQuadrilateral(detected)) {
    return null;
  }

  return squareToQuadrilateral(detected);
}

/**
 * Computes the perspective-projected quadrilateral for a single square in preview space.
 * Returned in clockwise order: [topLeft, topRight, bottomRight, bottomLeft].
 */
export function computeSquarePerspectiveQuad(
  corners: Point2D[],
  square: string,
  orientation: BoardOrientation = 'white_bottom',
): PerspectiveQuad | null {
  const homography = getBoardHomography(corners);
  if (!homography) {
    return null;
  }

  const { col, row } = algebraicToColRow(square, orientation);
  const u0 = col / 8;
  const u1 = (col + 1) / 8;
  const v0 = row / 8;
  const v1 = (row + 1) / 8;

  const p0 = projectSquarePoint(homography, u0, v0);
  const p1 = projectSquarePoint(homography, u1, v0);
  const p2 = projectSquarePoint(homography, u1, v1);
  const p3 = projectSquarePoint(homography, u0, v1);

  return [p0, p1, p2, p3];
}

/**
 * Computes the perspective-projected center point of a square.
 */
export function computeSquarePerspectiveCenter(
  corners: Point2D[],
  square: string,
  orientation: BoardOrientation = 'white_bottom',
): Point2D | null {
  const homography = getBoardHomography(corners);
  if (!homography) {
    return null;
  }

  const { col, row } = algebraicToColRow(square, orientation);
  const uCenter = (col + 0.5) / 8;
  const vCenter = (row + 0.5) / 8;

  return projectSquarePoint(homography, uCenter, vCenter);
}

/**
 * Computes the perspective move arrow from source square to destination square,
 * including perspective-aligned arrowhead wings.
 */
export function computeMoveArrowGeometry(
  corners: Point2D[],
  fromSquare: string,
  toSquare: string,
  orientation: BoardOrientation = 'white_bottom',
  headLength: number = 14,
): ArrowGeometry | null {
  const fromCenter = computeSquarePerspectiveCenter(corners, fromSquare, orientation);
  const toCenter = computeSquarePerspectiveCenter(corners, toSquare, orientation);

  if (!fromCenter || !toCenter) {
    return null;
  }

  const dx = toCenter.x - fromCenter.x;
  const dy = toCenter.y - fromCenter.y;
  const distance = Math.hypot(dx, dy);

  if (distance < 1e-4) {
    return null;
  }

  const angle = Math.atan2(dy, dx);
  // Wings point back at 150 degrees and -150 degrees from forward vector
  const wingAngleOffset = (150 * Math.PI) / 180;
  const clampedHeadLength = Math.min(headLength, distance * 0.4);

  const headLeft: Point2D = {
    x: toCenter.x + clampedHeadLength * Math.cos(angle + wingAngleOffset),
    y: toCenter.y + clampedHeadLength * Math.sin(angle + wingAngleOffset),
  };

  const headRight: Point2D = {
    x: toCenter.x + clampedHeadLength * Math.cos(angle - wingAngleOffset),
    y: toCenter.y + clampedHeadLength * Math.sin(angle - wingAngleOffset),
  };

  return {
    from: fromCenter,
    to: toCenter,
    headLeft,
    headRight,
  };
}

/**
 * Computes the perspective-correct internal 8x8 grid lines.
 * Because straight lines map to straight lines under projective homography,
 * projecting the endpoints of each grid line on the square boundary
 * guarantees exact perspective foreshortening.
 */
export function computePerspectiveGridLines(corners: Point2D[]): GridLine[] | null {
  const homography = getBoardHomography(corners);
  if (!homography) {
    return null;
  }

  const lines: GridLine[] = [];

  // 7 vertical internal lines (indices 1..7)
  for (let i = 1; i < 8; i += 1) {
    const u = i / 8;
    const top = projectSquarePoint(homography, u, 0);
    const bottom = projectSquarePoint(homography, u, 1);
    lines.push({
      key: `v-${i}`,
      from: top,
      to: bottom,
    });
  }

  // 7 horizontal internal lines (indices 1..7)
  for (let j = 1; j < 8; j += 1) {
    const v = j / 8;
    const left = projectSquarePoint(homography, 0, v);
    const right = projectSquarePoint(homography, 1, v);
    lines.push({
      key: `h-${j}`,
      from: left,
      to: right,
    });
  }

  return lines;
}
