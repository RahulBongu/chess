import {
  BorderTypes,
  ColorConversionCodes,
  ContourApproximationModes,
  DataTypes,
  InterpolationFlags,
  Mat,
  MorphShapes,
  OpenCV,
  Point,
  PointVector,
  PointVectorOfVectors,
  RetrievalModes,
  Scalar,
  Size,
} from 'react-native-fast-opencv';

import {
  MIN_DETECTION_CONFIDENCE,
  isConvexQuadrilateral,
  normalizePoints,
  reorderCorners,
  scoreCandidate,
  validateCandidateGeometry,
} from './boardDetection';
import type { BoardDetectionResult, VisionSize } from './types';

function emptyResult(size: VisionSize, message: string, candidateCount = 0): BoardDetectionResult {
  return {
    success: false,
    imageSize: size,
    processingSize: size,
    corners: [],
    confidence: 0,
    candidateCount,
    message,
  };
}

export async function detectBoardFromBase64(base64: string): Promise<BoardDetectionResult> {
  const src = Mat.createFromBase64(base64);
  const originalSize: VisionSize = { width: src.cols, height: src.rows };

  let gray: Mat | null = null;
  let blurred: Mat | null = null;
  let edges: Mat | null = null;
  let dilated: Mat | null = null;
  let resized: Mat | null = null;
  const contours = PointVectorOfVectors.create();

  try {
    const targetWidth = Math.min(720, Math.max(1, src.cols));
    const scale = targetWidth / src.cols;
    const targetHeight = Math.max(1, Math.round(src.rows * scale));
    const processingSize: VisionSize = { width: targetWidth, height: targetHeight };

    gray = Mat.create(targetHeight, targetWidth, DataTypes.CV_8UC1);
    blurred = Mat.create(targetHeight, targetWidth, DataTypes.CV_8UC1);
    edges = Mat.create(targetHeight, targetWidth, DataTypes.CV_8UC1);
    dilated = Mat.create(targetHeight, targetWidth, DataTypes.CV_8UC1);
    resized = Mat.create(targetHeight, targetWidth, DataTypes.CV_8UC3);

    OpenCV.resize(src, resized, Size.create(targetWidth, targetHeight), 0, 0, InterpolationFlags.INTER_AREA);
    OpenCV.cvtColor(resized, gray, ColorConversionCodes.COLOR_BGR2GRAY);
    OpenCV.GaussianBlur(gray, blurred, Size.create(5, 5), 0);
    OpenCV.Canny(blurred, edges, 60, 180);

    const kernel = OpenCV.getStructuringElement(MorphShapes.MORPH_RECT, Size.create(3, 3));
    OpenCV.dilate(edges, dilated, kernel, Point.create(-1, -1), 1, BorderTypes.BORDER_CONSTANT, Scalar.create(1));

    OpenCV.findContours(dilated, contours, RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_SIMPLE);

    const candidates: { corners: { x: number; y: number }[]; score: number }[] = [];
    const imageArea = originalSize.width * originalSize.height;

    for (let index = 0; index < contours.length; index += 1) {
      const contour = contours.get(index);
      const perimeter = OpenCV.arcLength(contour, true).value;
      if (!Number.isFinite(perimeter) || perimeter <= 0) {
        continue;
      }

      const epsilon = 0.02 * perimeter;
      const approx = PointVector.create();
      try {
        OpenCV.approxPolyDP(contour, approx, epsilon, true);
        if (approx.length !== 4) {
          continue;
        }

        const points = approx.getAll().map((point) => ({
          x: point.x / scale,
          y: point.y / scale,
        }));

        const ordered = reorderCorners(points);
        if (!isConvexQuadrilateral(ordered)) {
          continue;
        }

        const geometry = validateCandidateGeometry(ordered, originalSize);
        if (!geometry.valid) {
          continue;
        }

        const score = scoreCandidate(ordered, originalSize);
        if (score < MIN_DETECTION_CONFIDENCE * 0.5) {
          continue;
        }

        candidates.push({ corners: ordered, score });
      } finally {
        approx.release();
      }
    }

    if (candidates.length === 0) {
      return {
        ...emptyResult(originalSize, 'Board not detected. Try a clearer, top-down view with the whole board visible.', 0),
        processingSize,
      };
    }

    candidates.sort((a, b) => b.score - a.score);
    const best = candidates[0];

    if (best.score < MIN_DETECTION_CONFIDENCE) {
      return {
        ...emptyResult(
          originalSize,
          'Board not detected with enough confidence. Improve lighting and framing, then retry.',
          candidates.length,
        ),
        processingSize,
        confidence: best.score,
      };
    }

    const normalized = normalizePoints(best.corners, originalSize);
    return {
      success: true,
      imageSize: originalSize,
      processingSize,
      corners: normalized,
      confidence: best.score,
      candidateCount: candidates.length,
      message: 'Board candidate detected. Review and adjust the corners before confirming.',
    };
  } catch (error) {
    return {
      ...emptyResult(originalSize, error instanceof Error ? error.message : 'Board detection failed.'),
      processingSize: originalSize,
    };
  } finally {
    src.release();
    gray?.release();
    blurred?.release();
    edges?.release();
    dilated?.release();
    resized?.release();
    contours.release?.();
  }
}