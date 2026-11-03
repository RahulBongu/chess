/**
 * Frame-to-preview coordinate projection (Milestone 5.4C).
 *
 * Traces the full path a board corner travels from the camera sensor to the
 * on-screen overlay:
 *
 *   sensor Y-plane (cropRect, downscaled by the analyzer)
 *     → frame pixel coordinate (0..width, 0..height, boundary convention)
 *     → normalised frame coordinate (0..1)
 *     → display rotation (imageProxy.imageInfo.rotationDegrees, 0/90/180/270)
 *     → PreviewView FILL_CENTER scale (uniform s = max(vw/Wd, vh/Hd), centered,
 *        overflow cropped)
 *     → optional horizontal mirror (front camera only)
 *     → view pixel coordinate (0..viewWidth, 0..viewHeight)
 *
 * Normalised frame coordinates are the currency exchanged between the
 * detector / tracker and the overlay. A naive `u * viewWidth` mapping is
 * wrong whenever the analysis crop aspect does not match the view aspect or
 * the sensor requires a 90/270 rotation; this module composes the transform
 * explicitly and is fully unit-tested in Node.
 *
 * Note: FILL_CENTER is the PreviewView ScaleType used by LiveCameraView.kt.
 * A rear camera is never mirrored; `mirror: 'horizontal'` exists for
 * correctness tests and future front-camera support.
 */

export type FrameMirror = 'none' | 'horizontal';
export type FrameScaleType = 'FILL_CENTER';

export interface FrameToViewOptions {
  /** Decoded analysis-frame width in pixels (post-crop, post-downscale). */
  frameWidth: number;
  /** Decoded analysis-frame height in pixels. */
  frameHeight: number;
  /** clockwise rotation in degrees needed to make the image upright. */
  rotationDegrees: number;
  /** Preview view width in pixels. */
  viewWidth: number;
  /** Preview view height in pixels. */
  viewHeight: number;
  /** Preview scaling mode. Default 'FILL_CENTER'. */
  scaleType?: FrameScaleType;
  /** Mirroring applied by the display layer. Default 'none'. */
  mirror?: FrameMirror;
}

export interface FrameToViewTransform {
  /** Maps a normalised frame point (u, v in 0..1) to view pixels. */
  mapNormalized(u: number, v: number): { x: number; y: number };
  /** Maps a frame pixel point (x, y) to view pixels. */
  mapPixels(x: number, y: number): { x: number; y: number };
  /**
   * Inverse of `mapNormalized`: view pixel -> normalised frame coordinate.
   * Exact algebraic inverse (no iteration), so round-trip error is
   * float-rounding only. Points outside the view map outside [0, 1].
   */
  unmapView(x: number, y: number): { u: number; v: number };
  /** Uniform scale applied to the rotated content (FILL_CENTER). */
  scale: number;
  /** Horizontal offset of the rotated content inside the view (may be negative). */
  offsetX: number;
  /** Vertical offset of the rotated content inside the view (may be negative). */
  offsetY: number;
  /** Displayed (rotated) content width in "frame pixels". */
  displayWidth: number;
  /** Displayed (rotated) content height in "frame pixels". */
  displayHeight: number;
}

/**
 * Builds the composed projection. Throws on invalid inputs because a wrong
 * mapping must never silently render a misregistered overlay.
 */
export function createFrameToViewTransform(options: FrameToViewOptions): FrameToViewTransform {
  const {
    frameWidth,
    frameHeight,
    rotationDegrees,
    viewWidth,
    viewHeight,
    scaleType = 'FILL_CENTER',
    mirror = 'none',
  } = options;

  if (!(frameWidth > 0) || !(frameHeight > 0) || !(viewWidth > 0) || !(viewHeight > 0)) {
    throw new Error('Frame and view dimensions must be positive.');
  }

  const rotation = ((Math.round(rotationDegrees) % 360) + 360) % 360;
  if (![0, 90, 180, 270].includes(rotation)) {
    throw new Error(`Unsupported rotationDegrees: ${rotationDegrees} (expected 0, 90, 180 or 270).`);
  }

  const swap = rotation === 90 || rotation === 270;
  const displayWidth = swap ? frameHeight : frameWidth;
  const displayHeight = swap ? frameWidth : frameHeight;

  if (scaleType !== 'FILL_CENTER') {
    throw new Error(`Unsupported scaleType: ${scaleType} (only FILL_CENTER is implemented and tested).`);
  }

  // Uniform scale so the rotated content fills the view, overflow cropped and centred.
  const scale = Math.max(viewWidth / displayWidth, viewHeight / displayHeight);
  const dispW = displayWidth * scale;
  const dispH = displayHeight * scale;
  const offsetX = (viewWidth - dispW) / 2;
  const offsetY = (viewHeight - dispH) / 2;

  const mapRotatedNormalized = (u: number, v: number): { u: number; v: number } => {
    switch (rotation) {
      case 0:
        return { u, v };
      case 90:
        // CW 90: new (u', v') = (1 - v, u)
        return { u: 1 - v, v: u };
      case 180:
        return { u: 1 - u, v: 1 - v };
      case 270:
        // CW 270: new (u', v') = (v, 1 - u)
        return { u: v, v: 1 - u };
      default:
        return { u, v };
    }
  };

  const mapNormalized = (u: number, v: number) => {
    let rotated = mapRotatedNormalized(u, v);
    if (mirror === 'horizontal') {
      rotated = { u: 1 - rotated.u, v: rotated.v };
    }
    return {
      x: offsetX + rotated.u * dispW,
      y: offsetY + rotated.v * dispH,
    };
  };

  const mapPixels = (x: number, y: number) => mapNormalized(x / frameWidth, y / frameHeight);

  // Exact inverse: view pixels -> normalised frame coordinates.
  const unmapView = (x: number, y: number): { u: number; v: number } => {
    let ru = (x - offsetX) / dispW;
    let rv = (y - offsetY) / dispH;
    if (mirror === 'horizontal') {
      ru = 1 - ru;
    }
    switch (rotation) {
      case 90:
        // forward: (u', v') = (1 - v, u)  =>  u = v', v = 1 - u'
        return { u: rv, v: 1 - ru };
      case 180:
        return { u: 1 - ru, v: 1 - rv };
      case 270:
        // forward: (u', v') = (v, 1 - u)  =>  u = 1 - v', v = u'
        return { u: 1 - rv, v: ru };
      default:
        return { u: ru, v: rv };
    }
  };

  return {
    mapNormalized,
    mapPixels,
    unmapView,
    scale,
    offsetX,
    offsetY,
    displayWidth,
    displayHeight,
  };
}

/** Convenience helper: maps an array of normalised corners to view pixels. */
export function mapNormalizedCornersToView(
  corners: { x: number; y: number }[],
  options: FrameToViewOptions,
): { x: number; y: number }[] {
  const transform = createFrameToViewTransform(options);
  return corners.map((corner) => transform.mapNormalized(corner.x, corner.y));
}
