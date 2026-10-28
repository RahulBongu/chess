import type { DetectedCorner, VisionSize } from './types';

export const MIN_DETECTION_CONFIDENCE = 0.32;
export const MIN_CANDIDATE_AREA_RATIO = 0.08;
export const MIN_QUADRILATERAL_AREA_RATIO = 0.05;

export function signedArea(points: DetectedCorner[]): number {
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

export function reorderCorners(points: DetectedCorner[]): DetectedCorner[] {
  if (points.length !== 4) {
    return [...points];
  }

  // These extrema identify the four displayed corners for an arbitrarily
  // rotated convex quadrilateral.
  const topLeft = points.reduce((best, point) =>
    point.x + point.y < best.x + best.y ? point : best,
  );
  const topRight = points.reduce((best, point) =>
    point.x - point.y > best.x - best.y ? point : best,
  );
  const bottomRight = points.reduce((best, point) =>
    point.x + point.y > best.x + best.y ? point : best,
  );
  const bottomLeft = points.reduce((best, point) =>
    point.x - point.y < best.x - best.y ? point : best,
  );

  return [topLeft, topRight, bottomRight, bottomLeft];
}

export function isConvexQuadrilateral(points: DetectedCorner[]): boolean {
  if (points.length !== 4) {
    return false;
  }

  const ordered = reorderCorners(points);
  const crosses: number[] = [];
  for (let index = 0; index < 4; index += 1) {
    const a = ordered[index];
    const b = ordered[(index + 1) % 4];
    const c = ordered[(index + 2) % 4];
    const cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    crosses.push(cross);
  }

  const hasPositive = crosses.some((value) => value > 1e-9);
  const hasNegative = crosses.some((value) => value < -1e-9);
  return hasPositive !== hasNegative;
}

export function quadrilateralArea(points: DetectedCorner[]): number {
  return Math.abs(signedArea(points));
}

export function segmentsIntersect(
  a: DetectedCorner,
  b: DetectedCorner,
  c: DetectedCorner,
  d: DetectedCorner,
): boolean {
  const cross = (p: DetectedCorner, q: DetectedCorner, r: DetectedCorner) =>
    (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);

  const d1 = cross(c, d, a);
  const d2 = cross(c, d, b);
  const d3 = cross(a, b, c);
  const d4 = cross(a, b, d);

  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    return true;
  }

  const onSegment = (p: DetectedCorner, q: DetectedCorner, r: DetectedCorner) =>
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

export function isSelfIntersecting(points: DetectedCorner[]): boolean {
  if (points.length !== 4) {
    return false;
  }

  return (
    segmentsIntersect(points[0], points[1], points[2], points[3]) ||
    segmentsIntersect(points[1], points[2], points[3], points[0])
  );
}

export function isInBounds(points: DetectedCorner[], size: VisionSize): boolean {
  return points.every(
    (point) =>
      Number.isFinite(point.x) &&
      Number.isFinite(point.y) &&
      point.x >= 0 &&
      point.x <= size.width &&
      point.y >= 0 &&
      point.y <= size.height,
  );
}

export function validateCandidateGeometry(
  points: DetectedCorner[],
  size: VisionSize,
): { valid: boolean; reason?: string } {
  if (points.length !== 4) {
    return { valid: false, reason: 'Detection did not produce four corners.' };
  }

  if (!isInBounds(points, size)) {
    return { valid: false, reason: 'A detected corner is outside the image.' };
  }

  if (isSelfIntersecting(points)) {
    return { valid: false, reason: 'The detected corners form a self-intersecting quadrilateral.' };
  }

  const ordered = reorderCorners(points);
  if (!isConvexQuadrilateral(ordered)) {
    return { valid: false, reason: 'The detected corners are not convex.' };
  }

  if (isSelfIntersecting(ordered)) {
    return { valid: false, reason: 'The detected corners form a self-intersecting quadrilateral.' };
  }

  const area = quadrilateralArea(ordered);
  const imageArea = size.width * size.height;
  if (imageArea <= 0 || area / imageArea < MIN_QUADRILATERAL_AREA_RATIO) {
    return { valid: false, reason: 'The detected quadrilateral is too small.' };
  }

  const xs = ordered.map((point) => point.x);
  const ys = ordered.map((point) => point.y);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  if (width < size.width * 0.15 || height < size.height * 0.15) {
    return { valid: false, reason: 'The detected corners do not span enough of the image.' };
  }

  return { valid: true };
}

export function scoreCandidate(points: DetectedCorner[], size: VisionSize): number {
  const ordered = reorderCorners(points);
  const area = quadrilateralArea(ordered);
  const imageArea = size.width * size.height;
  const areaRatio = imageArea > 0 ? area / imageArea : 0;

  const xs = ordered.map((point) => point.x);
  const ys = ordered.map((point) => point.y);
  const boundingWidth = Math.max(...xs) - Math.min(...xs);
  const boundingHeight = Math.max(...ys) - Math.min(...ys);
  const boundingArea = boundingWidth * boundingHeight;
  const rectangularity = boundingArea > 0 ? area / boundingArea : 0;

  return Math.max(0, Math.min(1, areaRatio * 1.6 * rectangularity));
}

export function confidenceFromScore(score: number): number {
  return Math.max(0, Math.min(1, score));
}

export function normalizePoints(points: DetectedCorner[], size: VisionSize): DetectedCorner[] {
  if (size.width <= 0 || size.height <= 0) {
    return points.map((point) => ({ x: 0, y: 0 }));
  }

  return points.map((point) => ({
    x: Math.min(1, Math.max(0, point.x / size.width)),
    y: Math.min(1, Math.max(0, point.y / size.height)),
  }));
}

export function isValidDetection(result: { confidence: number; corners: DetectedCorner[] }, threshold = MIN_DETECTION_CONFIDENCE): boolean {
  return result.confidence >= threshold && result.corners.length === 4;
}