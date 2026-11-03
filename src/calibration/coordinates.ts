/**
 * Shared coordinate conversions for board calibration and rectification.
 *
 * Conventions
 * -----------
 * - A touch location is preview-local: (0, 0) is the top-left of the exact
 *   View that received the touch. It is not a screen coordinate and does not
 *   include safe-area, scroll, or preview offsets.
 * - Normalized calibration coordinates are fractions of that preview:
 *   x = 0 is the left edge, x = 1 is the right edge, y = 0 is the top edge and
 *   y = 1 is the bottom edge.
 * - Decoded image coordinates are continuous image-boundary coordinates:
 *   (0, 0) is the top-left boundary and (W, H) is the bottom-right boundary.
 *   x = 1 therefore maps to W, not W - 1. This matches the source points used
 *   by the perspective homography in `src/vision/rectify.ts`; individual pixel
 *   centers remain at integer coordinates 0..W-1.
 * - Captures returned by expo-camera are already rotated to the device
 *   orientation, so the returned `width` and `height` describe the decoded
 *   image. No additional EXIF rotation is applied in the app.
 * - The captured-image review container has the same aspect ratio as the
 *   decoded image, rounded to an integer height. `resizeMode="contain"` may
 *   therefore introduce less than one displayed pixel of letterboxing; the
 *   live camera preview is controlled separately by expo-camera.
 */

import type { NormalizedPoint } from './geometry.ts';

export interface DisplaySize {
  width: number;
  height: number;
}

export interface PixelPoint {
  x: number;
  y: number;
}

export function clampUnit(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

/**
 * Returns the integer display box used for an image at a fixed maximum width.
 * Height is rounded to the nearest pixel, so the displayed aspect ratio can
 * differ from the source ratio by at most 0.5 / height.
 */
export function computeAspectFitDisplaySize(
  sourceWidth: number,
  sourceHeight: number,
  maxWidth: number,
): DisplaySize {
  if (
    !Number.isFinite(sourceWidth) ||
    !Number.isFinite(sourceHeight) ||
    !Number.isFinite(maxWidth) ||
    sourceWidth <= 0 ||
    sourceHeight <= 0 ||
    maxWidth <= 0
  ) {
    throw new Error('Image and display dimensions must be finite and positive.');
  }

  return {
    width: maxWidth,
    height: Math.max(1, Math.round(maxWidth * (sourceHeight / sourceWidth))),
  };
}

/**
 * Converts a preview-local touch into clamped normalized coordinates.
 * Touches exactly on, or a fraction of a pixel outside, an edge map to 0 or 1.
 */
export function normalizePreviewTouch(
  locationX: number,
  locationY: number,
  displayWidth: number,
  displayHeight: number,
): NormalizedPoint {
  if (
    !Number.isFinite(displayWidth) ||
    !Number.isFinite(displayHeight) ||
    displayWidth <= 0 ||
    displayHeight <= 0
  ) {
    throw new Error('Preview dimensions must be finite and positive.');
  }

  return {
    x: clampUnit(locationX / displayWidth),
    y: clampUnit(locationY / displayHeight),
  };
}

/** Converts one normalized point to continuous decoded-image coordinates. */
export function normalizedToPixelPoint(
  point: NormalizedPoint,
  imageWidth: number,
  imageHeight: number,
): PixelPoint {
  if (
    !Number.isFinite(imageWidth) ||
    !Number.isFinite(imageHeight) ||
    imageWidth <= 0 ||
    imageHeight <= 0
  ) {
    throw new Error('Image dimensions must be finite and positive.');
  }

  return {
    x: point.x * imageWidth,
    y: point.y * imageHeight,
  };
}

export function normalizedToPixelCorners(
  points: NormalizedPoint[],
  imageWidth: number,
  imageHeight: number,
): PixelPoint[] {
  return points.map((point) => normalizedToPixelPoint(point, imageWidth, imageHeight));
}

/** Converts a continuous decoded-image coordinate back to [0, 1]. */
export function pixelToNormalizedPoint(
  point: PixelPoint,
  imageWidth: number,
  imageHeight: number,
): NormalizedPoint {
  if (
    !Number.isFinite(imageWidth) ||
    !Number.isFinite(imageHeight) ||
    imageWidth <= 0 ||
    imageHeight <= 0
  ) {
    throw new Error('Image dimensions must be finite and positive.');
  }

  return {
    x: clampUnit(point.x / imageWidth),
    y: clampUnit(point.y / imageHeight),
  };
}

/**
 * Recognizes the normalized corner payload produced by camera setup. The
 * board-review screen also accepts legacy pixel payloads, so this is a
 * heuristic rather than a validity check.
 */
export function isNormalizedCornerSet(points: NormalizedPoint[]): boolean {
  return (
    points.length === 4 &&
    points.every(
      (point) =>
        Number.isFinite(point.x) &&
        Number.isFinite(point.y) &&
        point.x >= 0 &&
        point.x <= 1.05 &&
        point.y >= 0 &&
        point.y <= 1.05,
    )
  );
}
