import type { Mat } from 'react-native-fast-opencv';

import type { GrayscaleImage } from './boardEvidence.ts';
import type {
  BoardClassificationResult,
  BoardViewOrientation,
  ClassifiedSquare,
  PieceColorEstimate,
  RectifiedBoardGeometry,
  RgbImage,
  SquareDiagnosticFeatures,
  SquareOccupancy,
  SquareRegion,
} from './types.ts';
import { CANONICAL_BOARD_SIZE, CANONICAL_SQUARE_SIZE } from './rectify.ts';

export type AnyImageData = GrayscaleImage | RgbImage;

function getOpenCV(): any | null {
  try {
    return require('react-native-fast-opencv');
  } catch {
    return null;
  }
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function stdDev(values: number[], meanVal?: number): number {
  if (values.length < 2) return 0;
  const mean = meanVal ?? values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function isRgbImage(image: AnyImageData): image is RgbImage {
  return (
    ('channels' in image && image.channels === 3) ||
    image.data.length === image.width * image.height * 3
  );
}

export function getPixelLuminanceAndRgb(
  image: AnyImageData,
  x: number,
  y: number,
): { lum: number; r: number; g: number; b: number } {
  const clampedX = Math.max(0, Math.min(image.width - 1, Math.round(x)));
  const clampedY = Math.max(0, Math.min(image.height - 1, Math.round(y)));

  if (isRgbImage(image)) {
    const idx = (clampedY * image.width + clampedX) * 3;
    const isRgb = image.colorOrder === 'rgb';
    const r = isRgb ? image.data[idx] : image.data[idx + 2];
    const g = image.data[idx + 1];
    const b = isRgb ? image.data[idx + 2] : image.data[idx];
    const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    return { lum, r, g, b };
  }

  const idx = clampedY * image.width + clampedX;
  const lum = image.data[idx];
  return { lum, r: lum, g: lum, b: lum };
}

/**
 * Extracts diagnostic surface and foreground features from an 800x800 canonical board
 * for a specific square region.
 */
export function extractSquareFeatures(
  image: AnyImageData,
  square: SquareRegion,
): SquareDiagnosticFeatures {
  const { x: x0, y: y0 } = square.bounds;

  // 1. Sample surface baseline from four corner patches
  // Insets: 12% to 22% from each edge, staying clear of borders and piece body
  const surfaceValues: number[] = [];
  const patchOffsets = [
    { startX: 12, startY: 12 },
    { startX: 78, startY: 12 },
    { startX: 78, startY: 78 },
    { startX: 12, startY: 78 },
  ];

  for (const patch of patchOffsets) {
    for (let dy = 0; dy < 10; dy += 2) {
      for (let dx = 0; dx < 10; dx += 2) {
        const px = x0 + patch.startX + dx;
        const py = y0 + patch.startY + dy;
        const { lum } = getPixelLuminanceAndRgb(image, px, py);
        surfaceValues.push(lum);
      }
    }
  }

  const surfaceLuminance = median(surfaceValues);
  const surfaceStdDev = stdDev(surfaceValues, surfaceLuminance);

  // 2. Sample center foreground region (where the piece base & body sit)
  // Window: 25% to 75% inside the cell (50x50 area centered at 50,50)
  const centerValues: number[] = [];
  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  const hasColor = isRgbImage(image);

  let minLuminance = 255;
  let maxLuminance = 0;

  for (let dy = 25; dy < 75; dy += 2) {
    for (let dx = 25; dx < 75; dx += 2) {
      const px = x0 + dx;
      const py = y0 + dy;
      const { lum, r, g, b } = getPixelLuminanceAndRgb(image, px, py);
      centerValues.push(lum);
      if (lum < minLuminance) minLuminance = lum;
      if (lum > maxLuminance) maxLuminance = lum;
      if (hasColor) {
        rSum += r;
        gSum += g;
        bSum += b;
      }
    }
  }

  const centerLuminance = median(centerValues);
  const centerMean = centerValues.reduce((s, v) => s + v, 0) / centerValues.length;
  const centerStdDev = stdDev(centerValues, centerMean);
  const contrastWithSurface = Math.abs(centerLuminance - surfaceLuminance);
  const luminanceRange = maxLuminance - minLuminance;

  // 3. Compute edge intensity via finite difference gradients inside center window
  let gradientSum = 0;
  let gradientCount = 0;

  for (let dy = 28; dy < 72; dy += 3) {
    for (let dx = 28; dx < 72; dx += 3) {
      const px = x0 + dx;
      const py = y0 + dy;
      const pRight = getPixelLuminanceAndRgb(image, px + 2, py).lum;
      const pLeft = getPixelLuminanceAndRgb(image, px - 2, py).lum;
      const pDown = getPixelLuminanceAndRgb(image, px, py + 2).lum;
      const pUp = getPixelLuminanceAndRgb(image, px, py - 2).lum;

      const gradX = Math.abs(pRight - pLeft) / 2;
      const gradY = Math.abs(pDown - pUp) / 2;
      gradientSum += gradX + gradY;
      gradientCount += 1;
    }
  }

  const edgeIntensity = gradientCount > 0 ? gradientSum / gradientCount : 0;

  const features: SquareDiagnosticFeatures = {
    surfaceLuminance,
    surfaceStdDev,
    centerLuminance,
    centerStdDev,
    contrastWithSurface,
    edgeIntensity,
    minLuminance,
    maxLuminance,
    luminanceRange,
  };

  if (hasColor && centerValues.length > 0) {
    features.meanR = Math.round(rSum / centerValues.length);
    features.meanG = Math.round(gSum / centerValues.length);
    features.meanB = Math.round(bSum / centerValues.length);
  }

  return features;
}

export interface BoardSurfaceStats {
  medianLightSurface: number;
  medianDarkSurface: number;
  boardContrast: number;
}

/**
 * Computes global board surface baseline luminance across all light and dark squares.
 */
export function computeBoardSurfaceStats(
  allFeatures: { square: SquareRegion; features: SquareDiagnosticFeatures }[],
): BoardSurfaceStats {
  const lightSurfaces: number[] = [];
  const darkSurfaces: number[] = [];

  for (const { square, features } of allFeatures) {
    if (square.isLight) {
      lightSurfaces.push(features.surfaceLuminance);
    } else {
      darkSurfaces.push(features.surfaceLuminance);
    }
  }

  const medianLightSurface = median(lightSurfaces);
  const medianDarkSurface = median(darkSurfaces);
  const boardContrast = Math.max(0, medianLightSurface - medianDarkSurface);

  return {
    medianLightSurface,
    medianDarkSurface,
    boardContrast,
  };
}

/**
 * Classifies a single square into occupancy and piece color using local features
 * and adaptive board surface statistics.
 */
export function classifySquareFeatures(
  square: SquareRegion,
  features: SquareDiagnosticFeatures,
  boardStats: BoardSurfaceStats,
): {
  occupancy: SquareOccupancy;
  occupancyConfidence: number;
  pieceColor: PieceColorEstimate | null;
  colorConfidence: number;
  overallConfidence: number;
  reason?: string;
} {
  // If the board has virtually zero contrast, surface detection is uncalibrated
  if (boardStats.boardContrast < 12) {
    return {
      occupancy: 'uncertain',
      occupancyConfidence: 0.4,
      pieceColor: null,
      colorConfidence: 0,
      overallConfidence: 0.4,
      reason: 'Board surface has insufficient contrast between light and dark squares.',
    };
  }

  const {
    surfaceLuminance,
    centerLuminance,
    centerStdDev,
    contrastWithSurface,
    edgeIntensity,
    minLuminance,
    maxLuminance,
    luminanceRange,
  } = features;

  // 1. Clear Empty Square:
  // Negligible contrast with square corners, low variance, low edge density
  if (
    contrastWithSurface <= 10 &&
    edgeIntensity <= 6 &&
    centerStdDev <= 8 &&
    luminanceRange <= 22
  ) {
    const conf = clamp01(0.88 + (10 - contrastWithSurface) * 0.012);
    return {
      occupancy: 'empty',
      occupancyConfidence: conf,
      pieceColor: null,
      colorConfidence: 0,
      overallConfidence: conf,
    };
  }

  // 2. Clear Occupied Square:
  // Condition A: Strong surface contrast (>= 16)
  const isContrastOccupied = contrastWithSurface >= 16;
  // Condition B: Strong internal piece texture/edges (for piece matching square, e.g. white on light)
  const isTextureOccupied =
    (centerStdDev >= 12 && luminanceRange >= 28) || edgeIntensity >= 10;
  // Condition C: Dark piece on dark square (very dark minimum luminance + moderate contrast/edges)
  const isDarkPieceOnDarkSq =
    !square.isLight &&
    minLuminance <= 48 &&
    (contrastWithSurface >= 14 || edgeIntensity >= 5 || centerLuminance <= 55);

  let occupancy: SquareOccupancy;
  let occupancyConfidence: number;
  let reason: string | undefined;

  if (isContrastOccupied || isTextureOccupied || isDarkPieceOnDarkSq) {
    occupancy = 'occupied';
    const strength = Math.max(
      contrastWithSurface / 30,
      edgeIntensity / 18,
      centerStdDev / 20,
    );
    occupancyConfidence = clamp01(0.75 + Math.min(0.24, strength * 0.2));
  } else if (
    contrastWithSurface <= 11 &&
    edgeIntensity <= 7 &&
    centerStdDev <= 9
  ) {
    occupancy = 'empty';
    occupancyConfidence = 0.82;
  } else {
    // Borderline contrast (12..15) with low edge intensity -> shadow or ambiguous artifact
    occupancy = 'uncertain';
    occupancyConfidence = 0.5;
    reason = 'Ambiguous contrast and edge density relative to square surface.';
  }

  // 3. Piece Color Estimation (for likely occupied squares)
  let pieceColor: PieceColorEstimate | null = null;
  let colorConfidence = 0;

  if (occupancy === 'occupied') {
    const midpoint = (boardStats.medianLightSurface + boardStats.medianDarkSurface) / 2;

    // Piece luminance thresholds relative to dynamic board range
    const whiteThreshold = midpoint + Math.min(15, boardStats.boardContrast * 0.18);
    const blackThreshold = midpoint - Math.min(15, boardStats.boardContrast * 0.18);

    if (centerLuminance >= whiteThreshold || (centerLuminance >= 135 && minLuminance >= 75)) {
      pieceColor = 'white';
      colorConfidence = clamp01(
        0.75 + (centerLuminance - whiteThreshold) / Math.max(1, boardStats.medianLightSurface - whiteThreshold),
      );
    } else if (centerLuminance <= blackThreshold || (centerLuminance <= 85 && minLuminance <= 50)) {
      pieceColor = 'black';
      colorConfidence = clamp01(
        0.75 + (blackThreshold - centerLuminance) / Math.max(1, blackThreshold - boardStats.medianDarkSurface),
      );
    } else {
      pieceColor = 'uncertain';
      colorConfidence = 0.5;
      reason = reason ?? 'Piece luminance sits midway between light and dark piece thresholds.';
    }
  }

  const overallConfidence =
    occupancy === 'occupied'
      ? Math.min(occupancyConfidence, colorConfidence > 0 ? colorConfidence : occupancyConfidence)
      : occupancyConfidence;

  return {
    occupancy,
    occupancyConfidence,
    pieceColor,
    colorConfidence,
    overallConfidence,
    reason,
  };
}

/**
 * Classifies an entire 800x800 canonical board into 64 occupied/empty squares and piece colors.
 */
export function classifyRectifiedBoard(
  image: AnyImageData,
  geometry: RectifiedBoardGeometry,
): BoardClassificationResult {
  if (image.width < CANONICAL_BOARD_SIZE || image.height < CANONICAL_BOARD_SIZE) {
    throw new Error(
      `Image dimensions (${image.width}x${image.height}) must be at least ${CANONICAL_BOARD_SIZE}x${CANONICAL_BOARD_SIZE}.`,
    );
  }

  // Pass 1: Extract features for all 64 squares
  const extracted = geometry.squares.map((square) => ({
    square,
    features: extractSquareFeatures(image, square),
  }));

  // Pass 2: Compute board-wide adaptive surface statistics
  const boardStats = computeBoardSurfaceStats(extracted);

  // Pass 3: Classify each square
  const classifiedSquares: ClassifiedSquare[] = [];
  const squareByName: Record<string, ClassifiedSquare> = {};

  let emptyCount = 0;
  let occupiedCount = 0;
  let uncertainCount = 0;
  let whitePieceCount = 0;
  let blackPieceCount = 0;
  let confidenceSum = 0;

  for (const { square, features } of extracted) {
    const classification = classifySquareFeatures(square, features, boardStats);

    const classified: ClassifiedSquare = {
      name: square.name,
      file: square.file,
      rank: square.rank,
      gridRow: square.gridRow,
      gridCol: square.gridCol,
      isLight: square.isLight,
      occupancy: classification.occupancy,
      occupancyConfidence: classification.occupancyConfidence,
      pieceColor: classification.pieceColor,
      colorConfidence: classification.colorConfidence,
      overallConfidence: classification.overallConfidence,
      features,
      reason: classification.reason,
    };

    classifiedSquares.push(classified);
    squareByName[square.name] = classified;

    if (classified.occupancy === 'empty') emptyCount += 1;
    else if (classified.occupancy === 'occupied') occupiedCount += 1;
    else uncertainCount += 1;

    if (classified.pieceColor === 'white') whitePieceCount += 1;
    else if (classified.pieceColor === 'black') blackPieceCount += 1;

    confidenceSum += classified.overallConfidence;
  }

  return {
    view: geometry.orientation.view,
    squares: classifiedSquares,
    squareByName,
    summary: {
      emptyCount,
      occupiedCount,
      uncertainCount,
      whitePieceCount,
      blackPieceCount,
      meanConfidence: confidenceSum / 64,
    },
  };
}

/**
 * Extracts 64 native OpenCV sub-mats from an 800x800 rectified Mat with guaranteed
 * cleanup of all allocated sub-matrices.
 */
export function extractSquareMats(
  rectifiedMat: Mat,
  canonicalSize = CANONICAL_BOARD_SIZE,
): {
  squareMats: Record<string, Mat>;
  releaseAll: () => void;
} {
  const opencv = getOpenCV();
  if (!opencv) {
    throw new Error('OpenCV native module is not available in this environment.');
  }
  const { OpenCV, Rect } = opencv;

  const cellSize = canonicalSize / 8;
  const squareMats: Record<string, any> = {};
  const allocated: any[] = [];

  const releaseAll = () => {
    for (const mat of allocated) {
      try {
        mat.release();
      } catch {
        // Ignore already-released mats
      }
    }
    allocated.length = 0;
  };

  try {
    for (let r = 0; r < 8; r += 1) {
      for (let c = 0; c < 8; c += 1) {
        const file = c;
        const rank = 7 - r;
        const name = `${String.fromCharCode(97 + file)}${rank + 1}`;
        const rect = Rect.create(c * cellSize, r * cellSize, cellSize, cellSize);

        // Submat slice
        const cellMat = OpenCV.submat(rectifiedMat, rect);
        allocated.push(cellMat);
        squareMats[name] = cellMat;
      }
    }

    return { squareMats, releaseAll };
  } catch (error) {
    releaseAll();
    throw error;
  }
}
