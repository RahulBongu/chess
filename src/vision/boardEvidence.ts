import type { DetectedCorner } from './types';

/**
 * The detector samples an 8 x 8 grid inside a candidate quadrilateral and
 * checks whether the cells behave like alternating chessboard squares.
 *
 * These are evidence thresholds, not probabilities. They are calibrated by
 * the deterministic synthetic fixtures in tests/boardEvidence.test.ts:
 * - a real-looking checkerboard must clearly pass;
 * - uniform, low-contrast, striped, partial, and non-board rectangles must fail.
 */
export const PATTERN_GRID_SIZE = 8;
export const PATTERN_CELL_INSET = 0.18;
export const PATTERN_CORNER_SPREAD = 0.02;
export const PATTERN_CORNER_SAMPLES = 3;

/** Checkerboard parity must explain at least this share of cell-mean variance. */
export const MIN_PATTERN_FIT = 0.55;
/** Share of adjacent cell pairs whose brightness difference follows the expected parity. */
export const MIN_PATTERN_ALTERNATION = 0.8;
/** Adjacent squares must differ by at least this many gray levels on average. */
export const MIN_PATTERN_CONTRAST = 8;
/** Adjacent-square contrast at which the contrast component is fully satisfied. */
export const FULL_PATTERN_CONTRAST = 48;
/** Below this combined pattern quality, a candidate is not a plausible board. */
export const MIN_PATTERN_SCORE = 0.45;

export interface GrayscaleImage {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
}

export interface SquareHomography {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
  g: number;
  h: number;
}

export interface PatternEvidence {
  valid: boolean;
  score: number;
  fit: number;
  alternation: number;
  contrast: number;
  noise: number;
  orientation: 0 | 1 | null;
  reason?: string;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function isFiniteCorner(corner: DetectedCorner): boolean {
  return Number.isFinite(corner.x) && Number.isFinite(corner.y);
}

/**
 * Builds the projective mapping from the unit square to an ordered
 * top-left, top-right, bottom-right, bottom-left quadrilateral.
 */
export function squareToQuadrilateral(corners: DetectedCorner[]): SquareHomography | null {
  if (corners.length !== 4 || !corners.every(isFiniteCorner)) {
    return null;
  }

  const [topLeft, topRight, bottomRight, bottomLeft] = corners;
  const twiceArea = Math.abs(
    topLeft.x * topRight.y -
      topRight.x * topLeft.y +
      topRight.x * bottomRight.y -
      bottomRight.x * topRight.y +
      bottomRight.x * bottomLeft.y -
      bottomLeft.x * bottomRight.y +
      bottomLeft.x * topLeft.y -
      topLeft.x * bottomLeft.y,
  );
  if (twiceArea < 1e-9) {
    return null;
  }

  const dx1 = topRight.x - bottomRight.x;
  const dx2 = bottomLeft.x - bottomRight.x;
  const dy1 = topRight.y - bottomRight.y;
  const dy2 = bottomLeft.y - bottomRight.y;
  const sx = topLeft.x - topRight.x + bottomRight.x - bottomLeft.x;
  const sy = topLeft.y - topRight.y + bottomRight.y - bottomLeft.y;

  if (Math.abs(sx) < 1e-9 && Math.abs(sy) < 1e-9) {
    return {
      a: topRight.x - topLeft.x,
      b: bottomLeft.x - topLeft.x,
      c: topLeft.x,
      d: topRight.y - topLeft.y,
      e: bottomLeft.y - topLeft.y,
      f: topLeft.y,
      g: 0,
      h: 0,
    };
  }

  const denominator = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(denominator) < 1e-9) {
    return null;
  }

  const g = (sx * dy2 - dx2 * sy) / denominator;
  const h = (dx1 * sy - sx * dy1) / denominator;

  return {
    a: topRight.x - topLeft.x + g * topRight.x,
    b: bottomLeft.x - topLeft.x + h * bottomLeft.x,
    c: topLeft.x,
    d: topRight.y - topLeft.y + g * topRight.y,
    e: bottomLeft.y - topLeft.y + h * bottomLeft.y,
    f: topLeft.y,
    g,
    h,
  };
}

/** Projects normalized unit-square coordinates (0..1) through the homography. */
export function projectSquarePoint(
  homography: SquareHomography,
  u: number,
  v: number,
): DetectedCorner {
  const denominator = homography.g * u + homography.h * v + 1;
  return {
    x: (homography.a * u + homography.b * v + homography.c) / denominator,
    y: (homography.d * u + homography.e * v + homography.f) / denominator,
  };
}

function sampleBilinear(image: GrayscaleImage, x: number, y: number): number {
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return Number.NaN;
  }

  const clampedX = Math.max(0, Math.min(image.width - 1, x));
  const clampedY = Math.max(0, Math.min(image.height - 1, y));
  const x0 = Math.floor(clampedX);
  const y0 = Math.floor(clampedY);
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const fx = clampedX - x0;
  const fy = clampedY - y0;

  const top = image.data[y0 * image.width + x0] * (1 - fx) + image.data[y0 * image.width + x1] * fx;
  const bottom =
    image.data[y1 * image.width + x0] * (1 - fx) + image.data[y1 * image.width + x1] * fx;
  return top * (1 - fy) + bottom * fy;
}

interface CellStats {
  mean: number;
  standardDeviation: number;
}

function measureCells(
  image: GrayscaleImage,
  homography: SquareHomography,
): CellStats[] | null {
  const cells: CellStats[] = [];
  const anchors = [PATTERN_CELL_INSET, 1 - PATTERN_CELL_INSET];
  const halfSpread = (PATTERN_CORNER_SAMPLES - 1) / 2;

  for (let row = 0; row < PATTERN_GRID_SIZE; row += 1) {
    for (let column = 0; column < PATTERN_GRID_SIZE; column += 1) {
      const values: number[] = [];

      // Sample near the four corners of each cell rather than its center.
      // Chess pieces and their shadows occupy the middle of a square, while
      // the corner patches still show the square colour needed for parity.
      for (const anchorV of anchors) {
        for (const anchorU of anchors) {
          for (let sampleRow = 0; sampleRow < PATTERN_CORNER_SAMPLES; sampleRow += 1) {
            const cellV = anchorV + (sampleRow - halfSpread) * PATTERN_CORNER_SPREAD;
            for (
              let sampleColumn = 0;
              sampleColumn < PATTERN_CORNER_SAMPLES;
              sampleColumn += 1
            ) {
              const cellU =
                anchorU + (sampleColumn - halfSpread) * PATTERN_CORNER_SPREAD;
              const point = projectSquarePoint(
                homography,
                (column + cellU) / PATTERN_GRID_SIZE,
                (row + cellV) / PATTERN_GRID_SIZE,
              );
              const value = sampleBilinear(image, point.x, point.y);
              if (!Number.isFinite(value)) {
                return null;
              }
              values.push(value);
            }
          }
        }
      }

      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      const variance =
        values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
      cells.push({ mean, standardDeviation: Math.sqrt(variance) });
    }
  }

  return cells;
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function invalidEvidence(
  reason: string,
  metrics: Omit<PatternEvidence, 'valid' | 'reason'>,
): PatternEvidence {
  return { valid: false, reason, ...metrics };
}

/**
 * Measures whether the area inside `corners` contains credible evidence of an
 * 8 x 8 alternating-square pattern. The result is a quality score, not a
 * probability that the candidate is a chessboard.
 */
export function measureBoardPattern(
  image: GrayscaleImage,
  corners: DetectedCorner[],
): PatternEvidence {
  const emptyMetrics = {
    score: 0,
    fit: 0,
    alternation: 0,
    contrast: 0,
    noise: 0,
    orientation: null as 0 | 1 | null,
  };

  if (
    image.width < 2 ||
    image.height < 2 ||
    image.data.length < image.width * image.height
  ) {
    return invalidEvidence('The grayscale image buffer is invalid.', emptyMetrics);
  }

  if (corners.length !== 4 || !corners.every(isFiniteCorner)) {
    return invalidEvidence('Pattern evidence requires four finite corners.', emptyMetrics);
  }

  const homography = squareToQuadrilateral(corners);
  if (!homography) {
    return invalidEvidence('The candidate quadrilateral cannot be rectified.', emptyMetrics);
  }

  const cells = measureCells(image, homography);
  if (!cells) {
    return invalidEvidence('The candidate interior could not be sampled.', emptyMetrics);
  }

  const means = cells.map((cell) => cell.mean);
  const baseMean = means.reduce((sum, value) => sum + value, 0) / means.length;
  const totalVariance = means.reduce((sum, value) => sum + (value - baseMean) ** 2, 0);

  let paritySum = 0;
  for (let index = 0; index < means.length; index += 1) {
    const row = Math.floor(index / PATTERN_GRID_SIZE);
    const column = index % PATTERN_GRID_SIZE;
    const parity = (row + column) % 2 === 0 ? 1 : -1;
    paritySum += (means[index] - baseMean) * parity;
  }

  const amplitude = paritySum / means.length;
  let residualVariance = 0;
  for (let index = 0; index < means.length; index += 1) {
    const row = Math.floor(index / PATTERN_GRID_SIZE);
    const column = index % PATTERN_GRID_SIZE;
    const parity = (row + column) % 2 === 0 ? 1 : -1;
    residualVariance += (means[index] - baseMean - amplitude * parity) ** 2;
  }

  const fit =
    totalVariance > 1e-6 ? clamp01(1 - residualVariance / totalVariance) : 0;

  const adjacentPairs: [number, number][] = [];
  for (let row = 0; row < PATTERN_GRID_SIZE; row += 1) {
    for (let column = 0; column < PATTERN_GRID_SIZE; column += 1) {
      const index = row * PATTERN_GRID_SIZE + column;
      if (column + 1 < PATTERN_GRID_SIZE) {
        adjacentPairs.push([index, index + 1]);
      }
      if (row + 1 < PATTERN_GRID_SIZE) {
        adjacentPairs.push([index, index + PATTERN_GRID_SIZE]);
      }
    }
  }

  let contrastSum = 0;
  let agreeingPairsForOrientation: [number, number] = [0, 0];
  for (const [first, second] of adjacentPairs) {
    const difference = means[first] - means[second];
    contrastSum += Math.abs(difference);

    const firstRow = Math.floor(first / PATTERN_GRID_SIZE);
    const firstColumn = first % PATTERN_GRID_SIZE;
    const secondRow = Math.floor(second / PATTERN_GRID_SIZE);
    const secondColumn = second % PATTERN_GRID_SIZE;
    const firstParity = (firstRow + firstColumn) % 2 === 0 ? 1 : -1;
    const secondParity = (secondRow + secondColumn) % 2 === 0 ? 1 : -1;
    const parityStep = firstParity - secondParity;

    // A board can be rotated by 90/180/270 degrees, which flips which parity
    // is light. Evaluate both orientations and keep the stronger one.
    for (const orientation of [0, 1] as const) {
      const direction = orientation === 0 ? 1 : -1;
      if (difference * parityStep * direction > 0) {
        agreeingPairsForOrientation[orientation] += 1;
      }
    }
  }

  const contrast = contrastSum / adjacentPairs.length;
  const orientation: 0 | 1 =
    agreeingPairsForOrientation[0] >= agreeingPairsForOrientation[1] ? 0 : 1;
  const alternation =
    agreeingPairsForOrientation[orientation] / adjacentPairs.length;
  const noise = median(cells.map((cell) => cell.standardDeviation));

  const contrastFactor = clamp01(
    (contrast - MIN_PATTERN_CONTRAST) / (FULL_PATTERN_CONTRAST - MIN_PATTERN_CONTRAST),
  );
  const noiseFactor = clamp01((contrast / (noise + 1) - 1) / 2);
  const score = clamp01(fit * alternation * contrastFactor * noiseFactor);

  const metrics = {
    score,
    fit,
    alternation,
    contrast,
    noise,
    orientation,
  };

  if (contrast < MIN_PATTERN_CONTRAST) {
    return invalidEvidence(
      'Adjacent cells do not have enough brightness contrast for a chessboard.',
      metrics,
    );
  }

  if (fit < MIN_PATTERN_FIT) {
    return invalidEvidence(
      'The candidate interior is not explained by an alternating 8 x 8 pattern.',
      metrics,
    );
  }

  if (alternation < MIN_PATTERN_ALTERNATION) {
    return invalidEvidence(
      'Too few adjacent cells alternate in brightness like chessboard squares.',
      metrics,
    );
  }

  if (contrast <= noise) {
    return invalidEvidence(
      'Variation inside cells is as strong as the variation between squares.',
      metrics,
    );
  }

  if (score < MIN_PATTERN_SCORE) {
    return invalidEvidence(
      'The interior pattern is too weak to confirm an 8 x 8 chessboard.',
      metrics,
    );
  }

  return { valid: true, ...metrics };
}
