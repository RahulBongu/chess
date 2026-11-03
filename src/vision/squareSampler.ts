/**
 * Square Observation Sampler for Handheld Chess Navigation.
 *
 * Samples 64 squares directly from an in-memory grayscale camera frame using the
 * projective board homography. Computes occupancy, mean luminance, contrast, and
 * piece-color estimates in sub-millisecond execution time on mobile.
 */

import {
  algebraicToColRow,
  colRowToAlgebraic,
  getBoardHomography,
  type BoardOrientation,
  type Point2D,
} from './perspectiveOverlay.ts';
import { projectSquarePoint, type GrayscaleImage, type SquareHomography } from './boardEvidence.ts';
import type { ChessSquare } from '../chess/ChessGame.ts';
import type { PieceColor, SquareObservation } from './moveRecognizer.ts';

export interface SamplerOptions {
  /** Number of center samples along each axis (e.g. 3 -> 3x3 = 9 samples). Default 3. */
  centerSamplesPerAxis?: number;
  /** Radius of center sampling grid in unit-square fractions. Default 0.22. */
  centerRadius?: number;
  /** Luminance delta threshold against square background to consider occupied. Default 18. */
  occupancyThreshold?: number;
}

/**
 * Returns how many of the given points fall strictly inside the decoded
 * image. Points on the boundary count as visible.
 */
function countInsideFrame(
  points: { x: number; y: number }[],
  width: number,
  height: number,
): number {
  let inside = 0;
  for (const p of points) {
    if (
      Number.isFinite(p.x) &&
      Number.isFinite(p.y) &&
      p.x >= 0 &&
      p.y >= 0 &&
      p.x <= width - 1 &&
      p.y <= height - 1
    ) {
      inside += 1;
    }
  }
  return inside;
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

/**
 * Samples all 64 squares from a grayscale camera frame.
 */
export function sampleSquareObservations(
  image: GrayscaleImage,
  corners: Point2D[],
  orientation: BoardOrientation = 'white_bottom',
  options: SamplerOptions = {},
): SquareObservation[] | null {
  if (!image || !image.data || image.width < 10 || image.height < 10) {
    return null;
  }

  const homography = getBoardHomography(corners);
  if (!homography) {
    return null;
  }

  const samplesPerAxis = options.centerSamplesPerAxis ?? 3;
  const centerRadius = options.centerRadius ?? 0.22;
  const occupancyThreshold = options.occupancyThreshold ?? 18;

  const observations: SquareObservation[] = [];

  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const squareName = colRowToAlgebraic(col, row, orientation) as ChessSquare;

      // 1. Sample central region (where a piece stands)
      const uCenter = (col + 0.5) / 8;
      const vCenter = (row + 0.5) / 8;
      const uStep = (centerRadius * 2) / (samplesPerAxis - 1 || 1) / 8;
      const vStep = (centerRadius * 2) / (samplesPerAxis - 1 || 1) / 8;
      const uStart = uCenter - centerRadius / 8;
      const vStart = vCenter - centerRadius / 8;

      let centerSum = 0;
      let centerCount = 0;
      const centerValues: number[] = [];
      const projectedCenter: { x: number; y: number }[] = [];

      for (let sy = 0; sy < samplesPerAxis; sy += 1) {
        for (let sx = 0; sx < samplesPerAxis; sx += 1) {
          const u = uStart + sx * uStep;
          const v = vStart + sy * vStep;
          const pt = projectSquarePoint(homography, u, v);
          projectedCenter.push(pt);
          const val = sampleBilinear(image, pt.x, pt.y);
          if (Number.isFinite(val)) {
            centerSum += val;
            centerCount += 1;
            centerValues.push(val);
          }
        }
      }

      const meanCenter = centerCount > 0 ? centerSum / centerCount : 128;

      // Center variance / standard deviation
      let varianceSum = 0;
      for (const val of centerValues) {
        varianceSum += (val - meanCenter) ** 2;
      }
      const stdDevCenter = centerCount > 1 ? Math.sqrt(varianceSum / centerCount) : 0;

      // 2. Sample square corner patches (bare square surface background)
      // Chess pieces rarely occupy the extreme corners of a square
      const cornerOffsets = [0.15, 0.85];
      let bgSum = 0;
      let bgCount = 0;
      const projectedBg: { x: number; y: number }[] = [];

      for (const ov of cornerOffsets) {
        for (const ou of cornerOffsets) {
          const u = (col + ou) / 8;
          const v = (row + ov) / 8;
          const pt = projectSquarePoint(homography, u, v);
          projectedBg.push(pt);
          const val = sampleBilinear(image, pt.x, pt.y);
          if (Number.isFinite(val)) {
            bgSum += val;
            bgCount += 1;
          }
        }
      }

      const meanBg = bgCount > 0 ? bgSum / bgCount : meanCenter;
      const deltaFromBg = Math.abs(meanCenter - meanBg);

      // 2.5 Visibility of this square inside the decoded frame (0..1).
      // Off-frame squares sample clamped edge pixels which proves nothing.
      const allProjected = projectedCenter.concat(projectedBg);
      const visibleCount = countInsideFrame(allProjected, image.width, image.height);
      const visibility = allProjected.length > 0 ? visibleCount / allProjected.length : 1;
      const frameCredible = visibility >= 0.5;

      // 3. Occupancy determination
      // High contrast between center and corner background or high center texture indicates a piece
      let isOccupied = deltaFromBg >= occupancyThreshold || stdDevCenter >= 22;
      if (!frameCredible) {
        // The square is substantially outside the camera frame: report honestly unknown.
        isOccupied = false;
      }

      // 4. Color estimation:
      // White pieces reflect high luminance (typically > 140) or are much brighter than the background.
      // Black pieces absorb light (typically < 100) or are much darker than the background.
      let color: PieceColor = null;
      if (isOccupied && frameCredible) {
        if (meanCenter >= 135 || meanCenter > meanBg + 25) {
          color = 'w';
        } else if (meanCenter <= 105 || meanCenter < meanBg - 25) {
          color = 'b';
        } else {
          // Moderate intensity: estimate based on side relative to background
          color = meanCenter >= meanBg ? 'w' : 'b';
        }
      }

      // Confidence is scaled by how much of the square is actually in frame.
      const baseConfidence = Math.min(1.0, Math.max(0.4, (deltaFromBg + stdDevCenter) / 50));
      const confidence = baseConfidence * (0.45 + 0.55 * visibility);

      observations.push({
        square: squareName,
        occupied: isOccupied,
        color,
        luminance: Math.round(meanCenter),
        confidence,
        visibility: Math.round(visibility * 1000) / 1000,
      });
    }
  }

  return observations;
}
