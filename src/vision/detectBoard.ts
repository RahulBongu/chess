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

import { PATTERN_SAMPLING_SCALES, measureBoardPattern } from './boardEvidence';
import type { GrayscaleImage, PatternEvidence } from './boardEvidence';
import {
  MIN_DETECTION_CONFIDENCE,
  combineDetectionConfidence,
  insetCorners,
  isConvexQuadrilateral,
  normalizePoints,
  reorderCorners,
  scoreCandidate,
  validateCandidateGeometry,
} from './boardDetection';
import type { CandidateDiagnostic } from './types';
import type { BoardDetectionResult, VisionSize } from './types';

function emptyResult(size: VisionSize, message: string, candidateCount = 0, diagnostics: CandidateDiagnostic[] = []): BoardDetectionResult {
  return {
    success: false,
    imageSize: size,
    processingSize: size,
    corners: [],
    confidence: 0,
    candidateCount,
    message,
    diagnostics,
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

    OpenCV.findContours(dilated, contours, RetrievalModes.RETR_LIST, ContourApproximationModes.CHAIN_APPROX_SIMPLE);

    const processingBuffer = blurred.toBuffer('uint8');
    const grayscale: GrayscaleImage = {
      width: processingBuffer.cols,
      height: processingBuffer.rows,
      data: processingBuffer.buffer,
    };

    const candidates: {
      corners: { x: number; y: number }[];
      boundaryScore: number;
      pattern: PatternEvidence;
      confidence: number;
      samplingScale: number;
      samplingCorners: { x: number; y: number }[];
    }[] = [];
    const diagnostics: CandidateDiagnostic[] = [];

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

        const processingPoints = approx.getAll().map((point) => ({
          x: point.x,
          y: point.y,
        }));
        const orderedProcessing = reorderCorners(processingPoints);
        if (!isConvexQuadrilateral(orderedProcessing)) {
          continue;
        }

        const points = orderedProcessing.map((point) => ({
          x: point.x / scale,
          y: point.y / scale,
        }));
        const ordered = reorderCorners(points);

        const geometry = validateCandidateGeometry(ordered, originalSize);
        if (!geometry.valid) {
          continue;
        }

        const boundaryScore = scoreCandidate(ordered, originalSize);
        if (boundaryScore < MIN_DETECTION_CONFIDENCE * 0.5) {
          continue;
        }

        // A detected outer quadrilateral often includes a board border or a
        // surrounding table edge. Evaluate a few inward sampling regions and
        // keep the strongest board evidence, while retaining the outer corners
        // for calibration.
        let bestSamplingScale = 1;
        let bestSamplingCorners = orderedProcessing;
        let bestPattern = measureBoardPattern(grayscale, orderedProcessing);
        let bestConfidence = combineDetectionConfidence(boundaryScore, bestPattern);

        for (const samplingScale of PATTERN_SAMPLING_SCALES) {
          if (samplingScale === 1) {
            continue;
          }
          const samplingProcessing = insetCorners(orderedProcessing, samplingScale);
          const pattern = measureBoardPattern(grayscale, samplingProcessing);
          const confidence = combineDetectionConfidence(boundaryScore, pattern);
          if (
            confidence > bestConfidence ||
            (confidence === bestConfidence && pattern.score > bestPattern.score)
          ) {
            bestConfidence = confidence;
            bestPattern = pattern;
            bestSamplingScale = samplingScale;
            bestSamplingCorners = samplingProcessing;
          }
        }

        const confidence = bestConfidence;
        const samplingCorners = bestSamplingCorners.map((point) => ({
          x: point.x / scale,
          y: point.y / scale,
        }));

        candidates.push({
          corners: ordered,
          boundaryScore,
          pattern: bestPattern,
          confidence,
          samplingScale: bestSamplingScale,
          samplingCorners,
        });
        diagnostics.push({
          corners: ordered,
          samplingCorners,
          samplingScale: bestSamplingScale,
          boundaryScore,
          pattern: bestPattern,
          confidence,
        });
      } finally {
        approx.release();
      }
    }

    if (candidates.length === 0) {
      return {
        ...emptyResult(originalSize, 'Board not detected. Try a clearer, top-down view with the whole board visible.', 0, diagnostics),
        processingSize,
      };
    }

    candidates.sort(
      (a, b) => b.confidence - a.confidence || b.boundaryScore - a.boundaryScore,
    );
    diagnostics.sort(
      (a, b) => b.confidence - a.confidence || b.boundaryScore - a.boundaryScore,
    );
    const best = candidates[0];

    if (best.confidence < MIN_DETECTION_CONFIDENCE) {
      return {
        ...emptyResult(
          originalSize,
          best.pattern.valid
            ? 'Board not detected with enough evidence. Improve lighting and framing, then retry.'
            : 'A rectangular outline was found, but its interior does not look like an 8 × 8 chessboard. Improve lighting and keep the whole board visible, then retry.',
          candidates.length,
          diagnostics,
        ),
        processingSize,
        confidence: best.confidence,
      };
    }

    const normalized = normalizePoints(best.corners, originalSize);
    return {
      success: true,
      imageSize: originalSize,
      processingSize,
      corners: normalized,
      confidence: best.confidence,
      candidateCount: candidates.length,
      message: 'Board candidate detected. Review and adjust the corners before confirming.',
      diagnostics,
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