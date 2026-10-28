/**
 * Calibration geometry helpers.
 *
 * Coordinate convention for normalized preview points:
 * - Origin (0, 0) is the top-left corner of the displayed camera preview.
 * - x increases to the right and ranges from 0 to 1.
 * - y increases downward and ranges from 0 to 1.
 * - Values are fractions of the preview's displayed width and height, so they
 *   remain stable when the preview's pixel dimensions change.
 */

export interface NormalizedPoint {
  x: number;
  y: number;
}

export interface CalibrationValidation {
  valid: boolean;
  reason?: string;
}

export const MIN_POINT_SEPARATION = 0.05;
export const MIN_POINT_SEPARATION_SQUARED = MIN_POINT_SEPARATION * MIN_POINT_SEPARATION;
export const MIN_QUADRILATERAL_AREA = 0.02;

function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

export function isWithinBounds(point: NormalizedPoint): boolean {
  return (
    isFiniteNumber(point.x) &&
    isFiniteNumber(point.y) &&
    point.x >= 0 &&
    point.x <= 1 &&
    point.y >= 0 &&
    point.y <= 1
  );
}

export function squaredDistance(a: NormalizedPoint, b: NormalizedPoint): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

export function isTooClose(points: NormalizedPoint[], point: NormalizedPoint): boolean {
  return points.some((existing) => squaredDistance(existing, point) < MIN_POINT_SEPARATION_SQUARED);
}

export function segmentsIntersect(
  a: NormalizedPoint,
  b: NormalizedPoint,
  c: NormalizedPoint,
  d: NormalizedPoint,
): boolean {
  const cross = (p: NormalizedPoint, q: NormalizedPoint, r: NormalizedPoint) =>
    (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);

  const d1 = cross(c, d, a);
  const d2 = cross(c, d, b);
  const d3 = cross(a, b, c);
  const d4 = cross(a, b, d);

  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    return true;
  }

  const onSegment = (p: NormalizedPoint, q: NormalizedPoint, r: NormalizedPoint) =>
    Math.min(p.x, q.x) - 1e-12 <= r.x &&
    r.x <= Math.max(p.x, q.x) + 1e-12 &&
    Math.min(p.y, q.y) - 1e-12 <= r.y &&
    r.y <= Math.max(p.y, q.y) + 1e-12;

  if (d1 === 0 && onSegment(c, d, a)) return true;
  if (d2 === 0 && onSegment(c, d, b)) return true;
  if (d3 === 0 && onSegment(a, b, c)) return true;
  if (d4 === 0 && onSegment(a, b, d)) return true;

  return false;
}

export function signedArea(points: NormalizedPoint[]): number {
  if (points.length < 3) {
    return 0;
  }

  let sum = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    sum += current.x * next.y - next.x * current.y;
  }
  return sum / 2;
}

export function canAddCalibrationPoint(
  points: NormalizedPoint[],
  point: NormalizedPoint,
): CalibrationValidation {
  if (points.length >= 4) {
    return { valid: false, reason: 'All four corners have already been selected.' };
  }

  if (!isWithinBounds(point)) {
    return { valid: false, reason: 'Tap inside the camera preview to select a corner.' };
  }

  if (isTooClose(points, point)) {
    return { valid: false, reason: 'Choose a point farther from the previous corners.' };
  }

  return { valid: true };
}

export function addCalibrationPoint(
  points: NormalizedPoint[],
  point: NormalizedPoint,
): NormalizedPoint[] | null {
  if (!canAddCalibrationPoint(points, point).valid) {
    return null;
  }
  return [...points, point];
}

export function undoCalibrationPoint(points: NormalizedPoint[]): NormalizedPoint[] {
  return points.slice(0, -1);
}

export function resetCalibrationPoints(): NormalizedPoint[] {
  return [];
}

export function isValidCalibration(points: NormalizedPoint[]): CalibrationValidation {
  if (points.length !== 4) {
    return { valid: false, reason: 'Select all four corners before confirming.' };
  }

  for (const point of points) {
    if (!isWithinBounds(point)) {
      return { valid: false, reason: 'A selected corner is outside the preview.' };
    }
  }

  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      if (squaredDistance(points[i], points[j]) < MIN_POINT_SEPARATION_SQUARED) {
        return { valid: false, reason: 'Two corners are too close together.' };
      }
    }
  }

  // Opposite edges of the quadrilateral are (0,1) vs (2,3) and (1,2) vs (3,0).
  if (segmentsIntersect(points[0], points[1], points[2], points[3])) {
    return { valid: false, reason: 'The selected corners form a self-intersecting quadrilateral.' };
  }
  if (segmentsIntersect(points[1], points[2], points[3], points[0])) {
    return { valid: false, reason: 'The selected corners form a self-intersecting quadrilateral.' };
  }

  const area = Math.abs(signedArea(points));
  if (area < MIN_QUADRILATERAL_AREA) {
    return { valid: false, reason: 'The selected corners do not enclose enough of the preview.' };
  }

  if (signedArea(points) <= 0) {
    return { valid: false, reason: 'Select the corners in top-left, top-right, bottom-right, bottom-left order.' };
  }

  return { valid: true };
}

export function describeCalibration(points: NormalizedPoint[]): string {
  if (points.length === 0) {
    return 'No corners selected.';
  }
  return points.map((point, index) => `${index + 1}: (${point.x.toFixed(3)}, ${point.y.toFixed(3)})`).join('; ');
}