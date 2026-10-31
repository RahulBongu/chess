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
import { validateGrayscaleFrame } from './grayscaleFrame';

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

export async function detectBoardFromMat(src: Mat): Promise<BoardDetectionResult> {
  const originalSize: VisionSize = { width: src.cols, height: src.rows };

  let gray: Mat | null = null;
  let blurred: Mat | null = null;
  let edges: Mat | null = null;
  let edgesLow: Mat | null = null;
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
    edgesLow = Mat.create(targetHeight, targetWidth, DataTypes.CV_8UC1);
    dilated = Mat.create(targetHeight, targetWidth, DataTypes.CV_8UC1);
    resized = Mat.create(targetHeight, targetWidth, DataTypes.CV_8UC3);

    OpenCV.resize(src, resized, Size.create(targetWidth, targetHeight), 0, 0, InterpolationFlags.INTER_AREA);
    OpenCV.cvtColor(resized, gray, ColorConversionCodes.COLOR_BGR2GRAY);
    OpenCV.GaussianBlur(gray, blurred, Size.create(5, 5), 0);
    // Use a lower threshold pair as well as the original pair. A monitor or
    // table edge is usually strong, but a chessboard drawn on a screen can have
    // weaker boundaries after blur and JPEG compression.
    OpenCV.Canny(blurred, edgesLow, 30, 90);
    OpenCV.Canny(blurred, edges, 60, 180);
    OpenCV.bitwise_or(edgesLow, edges, edges);

    const kernel = OpenCV.getStructuringElement(MorphShapes.MORPH_RECT, Size.create(3, 3));
    OpenCV.dilate(edges, dilated, kernel, Point.create(-1, -1), 1, BorderTypes.BORDER_CONSTANT, Scalar.create(1));

    OpenCV.findContours(dilated, contours, RetrievalModes.RETR_LIST, ContourApproximationModes.CHAIN_APPROX_SIMPLE);

    const processingBuffer = blurred.toBuffer('uint8');
    const grayscale: GrayscaleImage = {
      width: processingBuffer.cols,
      height: processingBuffer.rows,
      data: processingBuffer.buffer,
    };

    const seenCandidates = new Set<string>();
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
      let processingPoints: { x: number; y: number }[] | null = null;
      try {
        OpenCV.approxPolyDP(contour, approx, epsilon, true);
        if (approx.length === 4) {
          processingPoints = approx.getAll().map((point) => ({
            x: point.x,
            y: point.y,
          }));
        } else if (contour.length >= 4) {
          // A board can be present as a contour whose outline is not simplified
          // to exactly four points. Its minimum-area rectangle is a plausible
          // quadrilateral candidate and is scored by the same pattern evidence.
          const pointCount = contour.length;
          const contourData: number[] = [];
          for (const point of contour.getAll()) {
            contourData.push(point.x, point.y);
          }
          const contourMat = Mat.create(pointCount, 1, DataTypes.CV_32FC2, contourData);
          try {
            const rect = OpenCV.minAreaRect(contourMat);
            if (rect.width > 1 && rect.height > 1) {
              const angle = (rect.angle * Math.PI) / 180;
              const cosine = Math.cos(angle);
              const sine = Math.sin(angle);
              const halfWidth = rect.width / 2;
              const halfHeight = rect.height / 2;
              processingPoints = [
                [-halfWidth, -halfHeight],
                [halfWidth, -halfHeight],
                [halfWidth, halfHeight],
                [-halfWidth, halfHeight],
              ].map(([localX, localY]) => ({
                x: rect.x + localX * cosine - localY * sine,
                y: rect.y + localX * sine + localY * cosine,
              }));
            }
          } finally {
            contourMat.release();
          }
        }

        if (!processingPoints) {
          continue;
        }

        const orderedProcessing = reorderCorners(processingPoints);
        if (!isConvexQuadrilateral(orderedProcessing)) {
          continue;
        }

        const candidateKey = orderedProcessing
          .map((point) => `${Math.round(point.x)},${Math.round(point.y)}`)
          .join('|');
        if (seenCandidates.has(candidateKey)) {
          continue;
        }
        seenCandidates.add(candidateKey);

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
    edgesLow?.release();
    dilated?.release();
    resized?.release();
    contours.release?.();
  }
}

export async function detectBoardFromBase64(base64: string): Promise<BoardDetectionResult> {
  // Still-photo path: base64 is an encoded JPEG/PNG image.
  const src = Mat.createFromBase64(base64);
  return detectBoardFromMat(src);
}

export async function detectBoardFromGrayscaleFrame(
  base64: string,
  width: number,
  height: number,
): Promise<BoardDetectionResult> {
  const validation = validateGrayscaleFrame(base64, width, height);
  if (!validation.ok) {
    return emptyResult({ width: Math.max(1, width || 1), height: Math.max(1, height || 1) }, validation.message);
  }

  // Raw single-channel pixels must use the buffer API, not createFromBase64.
  const gray = Mat.createFromBuffer('uint8', height, width, 1, validation.buffer);
  const color = Mat.create(height, width, DataTypes.CV_8UC3);
  OpenCV.cvtColor(gray, color, ColorConversionCodes.COLOR_GRAY2BGR);
  gray.release();
  return detectBoardFromMat(color);
}