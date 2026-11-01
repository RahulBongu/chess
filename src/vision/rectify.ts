import type { Mat } from 'react-native-fast-opencv';

import { isConvexQuadrilateral, reorderCorners } from './boardDetection.ts';
import {
  measureBoardPattern,
  projectSquarePoint,
  squareToQuadrilateral,
  type GrayscaleImage,
  type SquareHomography,
} from './boardEvidence.ts';
import type {
  BoardViewOrientation,
  DetectedCorner,
  Homography3x3,
  OrientationAssessment,
  ParityConsistency,
  RectifiedBoardGeometry,
  SquareRegion,
  VisionSize,
} from './types.ts';

export const CANONICAL_BOARD_SIZE = 800;
export const CANONICAL_SQUARE_SIZE = 100;
export const GRID_DIMENSION = 8;

export const CANONICAL_CORNERS: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
  { x: 0, y: 0 },
  { x: CANONICAL_BOARD_SIZE, y: 0 },
  { x: CANONICAL_BOARD_SIZE, y: CANONICAL_BOARD_SIZE },
  { x: 0, y: CANONICAL_BOARD_SIZE },
];

export interface RectificationOptions {
  view?: BoardViewOrientation;
  image?: GrayscaleImage;
  canonicalSize?: number;
}

function getOpenCV(): any | null {
  try {
    return require('react-native-fast-opencv');
  } catch {
    return null;
  }
}

/**
 * Validates that four corners form a non-degenerate convex quadrilateral,
 * then normalizes them to canonical angular order starting at top-left.
 */
export function validateAndOrderCorners(
  corners: DetectedCorner[],
): [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] {
  if (!corners || corners.length !== 4) {
    throw new Error(`Expected exactly 4 corners, received ${corners ? corners.length : 0}.`);
  }
  for (let i = 0; i < 4; i += 1) {
    const pt = corners[i];
    if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) {
      throw new Error(`Corner at index ${i} has invalid or non-finite coordinates.`);
    }
  }

  const ordered = reorderCorners(corners);
  if (!isConvexQuadrilateral(ordered)) {
    throw new Error('The specified 4 corners do not form a strictly convex quadrilateral.');
  }

  return [ordered[0], ordered[1], ordered[2], ordered[3]];
}

/**
 * Computes the exact 3x3 matrix inverse of a SquareHomography.
 */
export function invertSquareHomography(homography: SquareHomography): Homography3x3 {
  const { a, b, c, d, e, f, g, h } = homography;
  const m00 = e - f * h;
  const m01 = c * h - b;
  const m02 = b * f - c * e;
  const m10 = f * g - d;
  const m11 = a - c * g;
  const m12 = c * d - a * f;
  const m20 = d * h - e * g;
  const m21 = b * g - a * h;
  const m22 = a * e - b * d;
  const determinant = a * m00 + b * m10 + c * m20;

  if (Math.abs(determinant) < 1e-12) {
    throw new Error('Singular homography matrix cannot be inverted.');
  }

  return {
    m00: m00 / determinant,
    m01: m01 / determinant,
    m02: m02 / determinant,
    m10: m10 / determinant,
    m11: m11 / determinant,
    m12: m12 / determinant,
    m20: m20 / determinant,
    m21: m21 / determinant,
    m22: m22 / determinant,
  };
}

/**
 * Projects a 2D point through a 3x3 planar homography matrix.
 */
export function projectPointWithHomography(
  h: Homography3x3,
  point: DetectedCorner,
): DetectedCorner {
  const denom = h.m20 * point.x + h.m21 * point.y + h.m22;
  if (Math.abs(denom) < 1e-12) {
    throw new Error('Projected point has near-zero homogeneous denominator.');
  }
  return {
    x: (h.m00 * point.x + h.m01 * point.y + h.m02) / denom,
    y: (h.m10 * point.x + h.m11 * point.y + h.m12) / denom,
  };
}

/**
 * Computes direct forward (source -> canonical 800x800) and inverse
 * (canonical 800x800 -> source) homography matrices.
 */
export function computeForwardAndInverseHomography(
  sourceCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner],
  canonicalSize = CANONICAL_BOARD_SIZE,
): {
  forwardHomography: Homography3x3;
  inverseHomography: Homography3x3;
  canonicalCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner];
} {
  const hUnitToSrc = squareToQuadrilateral(sourceCorners);
  if (!hUnitToSrc) {
    throw new Error('Failed to compute unit-square to source homography.');
  }

  const hSrcToUnit = invertSquareHomography(hUnitToSrc);

  // Forward: source (x, y) -> canonical (X, Y) = 800 * (u, v)
  const forwardHomography: Homography3x3 = {
    m00: canonicalSize * hSrcToUnit.m00,
    m01: canonicalSize * hSrcToUnit.m01,
    m02: canonicalSize * hSrcToUnit.m02,
    m10: canonicalSize * hSrcToUnit.m10,
    m11: canonicalSize * hSrcToUnit.m11,
    m12: canonicalSize * hSrcToUnit.m12,
    m20: hSrcToUnit.m20,
    m21: hSrcToUnit.m21,
    m22: hSrcToUnit.m22,
  };

  // Inverse: canonical (X, Y) -> source (x, y), where (u, v) = (X/800, Y/800)
  const inverseHomography: Homography3x3 = {
    m00: hUnitToSrc.a / canonicalSize,
    m01: hUnitToSrc.b / canonicalSize,
    m02: hUnitToSrc.c,
    m10: hUnitToSrc.d / canonicalSize,
    m11: hUnitToSrc.e / canonicalSize,
    m12: hUnitToSrc.f,
    m20: hUnitToSrc.g / canonicalSize,
    m21: hUnitToSrc.h / canonicalSize,
    m22: 1,
  };

  const canonicalCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
    { x: 0, y: 0 },
    { x: canonicalSize, y: 0 },
    { x: canonicalSize, y: canonicalSize },
    { x: 0, y: canonicalSize },
  ];

  return { forwardHomography, inverseHomography, canonicalCorners };
}

/**
 * Computes mean bidirectional reprojection error in pixels between
 * source quadrilateral corners and canonical corners.
 */
export function computeReprojectionError(
  sourceCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner],
  forwardH: Homography3x3,
  inverseH: Homography3x3,
  canonicalCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner],
): number {
  let totalErrorSq = 0;

  for (let i = 0; i < 4; i += 1) {
    const projectedCanonical = projectPointWithHomography(forwardH, sourceCorners[i]);
    const dX1 = projectedCanonical.x - canonicalCorners[i].x;
    const dY1 = projectedCanonical.y - canonicalCorners[i].y;
    totalErrorSq += dX1 * dX1 + dY1 * dY1;

    const projectedSource = projectPointWithHomography(inverseH, canonicalCorners[i]);
    const dX2 = projectedSource.x - sourceCorners[i].x;
    const dY2 = projectedSource.y - sourceCorners[i].y;
    totalErrorSq += dX2 * dX2 + dY2 * dY2;
  }

  return Math.sqrt(totalErrorSq / 8);
}

/**
 * Partitions the canonical 800x800 plane into 64 uniform 100x100 square regions,
 * projects their bounds back to source coordinates, and assigns algebraic names.
 */
export function computeSquareRegions(
  forwardH: Homography3x3,
  inverseH: Homography3x3,
  view: BoardViewOrientation = 'white-bottom',
  canonicalSize = CANONICAL_BOARD_SIZE,
): { squares: SquareRegion[]; squareByName: Record<string, SquareRegion> } {
  const cellSize = canonicalSize / GRID_DIMENSION;
  const squares: SquareRegion[] = [];
  const squareByName: Record<string, SquareRegion> = {};

  for (let r = 0; r < GRID_DIMENSION; r += 1) {
    for (let c = 0; c < GRID_DIMENSION; c += 1) {
      const bounds = {
        x: c * cellSize,
        y: r * cellSize,
        width: cellSize,
        height: cellSize,
      };

      const centerCanonical: DetectedCorner = {
        x: bounds.x + cellSize / 2,
        y: bounds.y + cellSize / 2,
      };

      const canonicalCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner] = [
        { x: bounds.x, y: bounds.y },
        { x: bounds.x + cellSize, y: bounds.y },
        { x: bounds.x + cellSize, y: bounds.y + cellSize },
        { x: bounds.x, y: bounds.y + cellSize },
      ];

      const sourceCorners = canonicalCorners.map((pt) =>
        projectPointWithHomography(inverseH, pt),
      ) as [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner];

      // Algebraic mapping based on orientation:
      // White-bottom: file a=col 0, rank 1=row 7
      // Black-bottom: file h=col 0, rank 8=row 7
      const file = view === 'white-bottom' ? c : 7 - c;
      const rank = view === 'white-bottom' ? 7 - r : r;
      const name = `${String.fromCharCode(97 + file)}${rank + 1}`;

      // Standard chess square parity: (file + rank) is odd for light squares
      const isLight = (file + rank) % 2 !== 0;

      const region: SquareRegion = {
        name,
        file,
        rank,
        gridRow: r,
        gridCol: c,
        bounds,
        center: centerCanonical,
        canonicalCorners,
        sourceCorners,
        isLight,
      };

      squares.push(region);
      squareByName[name] = region;
    }
  }

  return { squares, squareByName };
}

/**
 * Enumerates all 4 cyclic candidate rotations of the four corners (0°, 90°, 180°, 270°).
 */
export function enumerateCandidateCornerOrderings(
  corners: DetectedCorner[],
): [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner][] {
  const ordered = validateAndOrderCorners(corners);
  const candidates: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner][] = [];

  for (let shift = 0; shift < 4; shift += 1) {
    candidates.push([
      ordered[shift % 4],
      ordered[(shift + 1) % 4],
      ordered[(shift + 2) % 4],
      ordered[(shift + 3) % 4],
    ]);
  }

  return candidates;
}

/**
 * Assesses orientation consistency against checkerboard parity rules.
 *
 * In standard chess, the bottom-right corner square (h1 from White's perspective,
 * a8 from Black's perspective) MUST be a light square.
 *
 * Square parity acts as a consistency check to reject 90°/270° sideways rotations,
 * but CANNOT prove whether White or Black is at the bottom because both views
 * share light-square bottom-right corners.
 */
export function assessOrientation(
  corners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner],
  image?: GrayscaleImage,
  view: BoardViewOrientation = 'white-bottom',
): OrientationAssessment {
  let parityConsistency: ParityConsistency = 'unknown';

  if (image) {
    const pattern = measureBoardPattern(image, corners);
    if (pattern.valid && pattern.orientation !== null) {
      // In measureBoardPattern: orientation 0 means (row+col)%2===0 is lighter.
      // In canonical grid: (r=7, c=7) sum is 14 (even), which corresponds to
      // the light square (h1 or a8).
      // If pattern.orientation === 0, bottom-right is light -> consistent.
      // If pattern.orientation === 1, bottom-right is dark -> inconsistent.
      parityConsistency = pattern.orientation === 0 ? 'consistent' : 'inconsistent';
    }
  }

  const ambiguityWarning =
    parityConsistency === 'inconsistent'
      ? 'Bottom-right corner appears to be a dark square, which is inconsistent with standard chess board orientation (h1 and a8 must be light). The board may be rotated by 90 or 270 degrees.'
      : 'Square parity confirms bottom-right is a light square, but does NOT distinguish whether White or Black is at the bottom (both perspectives share light bottom-right corners). User confirmation or piece distribution is required.';

  return {
    view,
    parityConsistency,
    ambiguityWarning,
  };
}

/**
 * Computes the complete perspective rectification geometry for a board candidate.
 */
export function computeRectifiedBoardGeometry(
  sourceCorners: DetectedCorner[],
  options: RectificationOptions = {},
): RectifiedBoardGeometry {
  const ordered = validateAndOrderCorners(sourceCorners);
  const canonicalSize = options.canonicalSize ?? CANONICAL_BOARD_SIZE;
  const view = options.view ?? 'white-bottom';

  const { forwardHomography, inverseHomography, canonicalCorners } =
    computeForwardAndInverseHomography(ordered, canonicalSize);

  const reprojectionErrorPx = computeReprojectionError(
    ordered,
    forwardHomography,
    inverseHomography,
    canonicalCorners,
  );

  const orientation = assessOrientation(ordered, options.image, view);
  const { squares, squareByName } = computeSquareRegions(
    forwardHomography,
    inverseHomography,
    view,
    canonicalSize,
  );

  return {
    canonicalSize: { width: canonicalSize, height: canonicalSize },
    sourceCorners: ordered,
    canonicalCorners,
    forwardHomography,
    inverseHomography,
    reprojectionErrorPx,
    orientation,
    squares,
    squareByName,
  };
}

function sampleBilinearClamped(image: GrayscaleImage, x: number, y: number): number {
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return 0;
  }
  const clampedX = Math.max(0, Math.min(image.width - 1, x));
  const clampedY = Math.max(0, Math.min(image.height - 1, y));
  const x0 = Math.floor(clampedX);
  const y0 = Math.floor(clampedY);
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const fx = clampedX - x0;
  const fy = clampedY - y0;

  const top =
    image.data[y0 * image.width + x0] * (1 - fx) + image.data[y0 * image.width + x1] * fx;
  const bottom =
    image.data[y1 * image.width + x0] * (1 - fx) + image.data[y1 * image.width + x1] * fx;
  return Math.round(top * (1 - fy) + bottom * fy);
}

/**
 * Pure TypeScript backward-mapping perspective warp. Produces an exact
 * canonical 800x800 GrayscaleImage buffer without requiring native OpenCV.
 */
export function rectifyGrayscaleBuffer(
  image: GrayscaleImage,
  corners: DetectedCorner[],
  targetSize = CANONICAL_BOARD_SIZE,
): GrayscaleImage {
  const ordered = validateAndOrderCorners(corners);
  const hUnitToSrc = squareToQuadrilateral(ordered);
  if (!hUnitToSrc) {
    throw new Error('Candidate corners cannot be mapped to projective homography.');
  }

  const outData = new Uint8Array(targetSize * targetSize);
  for (let y = 0; y < targetSize; y += 1) {
    const v = (y + 0.5) / targetSize;
    for (let x = 0; x < targetSize; x += 1) {
      const u = (x + 0.5) / targetSize;
      const pt = projectSquarePoint(hUnitToSrc, u, v);
      outData[y * targetSize + x] = sampleBilinearClamped(image, pt.x, pt.y);
    }
  }

  return {
    width: targetSize,
    height: targetSize,
    data: outData,
  };
}

/**
 * Hardware-accelerated OpenCV Mat perspective rectification.
 * Uses Point2fVector and Point2f for getPerspectiveTransform as required by
 * react-native-fast-opencv, and provides all 7 parameters to warpPerspective.
 * Guarantees release of all temporary intermediate OpenCV allocations.
 */
export function rectifyMat(
  src: Mat,
  corners: DetectedCorner[],
  targetSize = CANONICAL_BOARD_SIZE,
  opencvModule?: any,
): { rectified: Mat; release: () => void } {
  const opencv = opencvModule ?? getOpenCV();
  if (!opencv) {
    throw new Error('OpenCV native module is not available in this environment.');
  }
  const {
    OpenCV,
    Mat,
    DataTypes,
    Size,
    InterpolationFlags,
    Point2f,
    Point2fVector,
    DecompTypes,
    BorderTypes,
    Scalar,
  } = opencv;

  const ordered = validateAndOrderCorners(corners);

  let srcVector: any = null;
  let dstVector: any = null;
  let transformMat: any = null;
  let dstMat: any = null;
  let sizeObj: any = null;
  let borderScalar: any = null;
  const srcPoints: any[] = [];
  const dstPoints: any[] = [];

  try {
    const Point2fClass = Point2f ?? OpenCV?.Point2f;
    const Point2fVectorClass = Point2fVector ?? OpenCV?.Point2fVector;
    const SizeClass = Size ?? OpenCV?.Size;
    const ScalarClass = Scalar ?? OpenCV?.Scalar;

    if (!Point2fClass || !Point2fVectorClass) {
      throw new Error('Point2f or Point2fVector is not available in OpenCV module.');
    }

    srcVector = Point2fVectorClass.create();
    dstVector = Point2fVectorClass.create();

    srcPoints.push(
      Point2fClass.create(ordered[0].x, ordered[0].y),
      Point2fClass.create(ordered[1].x, ordered[1].y),
      Point2fClass.create(ordered[2].x, ordered[2].y),
      Point2fClass.create(ordered[3].x, ordered[3].y),
    );

    dstPoints.push(
      Point2fClass.create(0, 0),
      Point2fClass.create(targetSize, 0),
      Point2fClass.create(targetSize, targetSize),
      Point2fClass.create(0, targetSize),
    );

    for (const pt of srcPoints) {
      srcVector.push(pt);
    }
    for (const pt of dstPoints) {
      dstVector.push(pt);
    }

    const solveMethod = DecompTypes?.DECOMP_LU ?? 0;
    transformMat = OpenCV.getPerspectiveTransform(srcVector, dstVector, solveMethod);

    const channels = (src as any)?.channels ?? 3;
    const dataType =
      channels === 1
        ? (DataTypes?.CV_8UC1 ?? 0)
        : channels === 4
        ? (DataTypes?.CV_8UC4 ?? 24)
        : (DataTypes?.CV_8UC3 ?? 16);

    dstMat = Mat.create(targetSize, targetSize, dataType);
    sizeObj = SizeClass.create(targetSize, targetSize);
    borderScalar = ScalarClass?.create ? ScalarClass.create(0, 0, 0, 0) : null;
    const borderMode = BorderTypes?.BORDER_CONSTANT ?? 0;
    const interpFlag = InterpolationFlags?.INTER_LINEAR ?? 1;

    OpenCV.warpPerspective(
      src,
      dstMat,
      transformMat,
      sizeObj,
      interpFlag,
      borderMode,
      borderScalar,
    );

    const resultMat = dstMat;
    return {
      rectified: resultMat,
      release: () => {
        resultMat.release();
      },
    };
  } catch (error) {
    dstMat?.release?.();
    throw error;
  } finally {
    for (const pt of srcPoints) {
      try {
        pt?.release?.();
      } catch {
        // Safe release
      }
    }
    for (const pt of dstPoints) {
      try {
        pt?.release?.();
      } catch {
        // Safe release
      }
    }
    try {
      srcVector?.release?.();
    } catch {
      // Safe release
    }
    try {
      dstVector?.release?.();
    } catch {
      // Safe release
    }
    try {
      transformMat?.release?.();
    } catch {
      // Safe release
    }
    try {
      sizeObj?.release?.();
    } catch {
      // Safe release
    }
    try {
      borderScalar?.release?.();
    } catch {
      // Safe release
    }
  }
}
