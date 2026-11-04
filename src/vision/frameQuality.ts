/**
 * Lightweight frame focus/blur measurement for the live camera.
 *
 * The native analyzer emits raw 8-bit grayscale frames. Motion blur and
 * defocus lower high-frequency energy long before a square classifier starts
 * misreading individual pieces, so focus is measured once per detection frame
 * and used to reject (not average) unusable evidence.
 *
 * The metric is the normalized Laplacian variance:
 *
 *   sharpness = variance(Laplacian(luma)) / (variance(luma) + 1)
 *
 * On the repository's FENiT reference image the sharp score is ~0.33, a mild
 * Gaussian blur (sigma 0.8) scores ~0.048, and a strong hand-shake blur
 * (sigma >= 1.5) scores <= 0.007. The live pipeline rejects frames below
 * FUSION_MIN_SHARPNESS (0.010) before they enter temporal fusion.
 *
 * The frame is sampled on a stride grid (target ~240 px on the long edge) so
 * the measurement costs a few milliseconds even on a 1080p stream. The
 * discrete Laplacian is divided by stride^4 to keep the score roughly
 * resolution-independent.
 */

export interface FrameFocus {
  /** Normalized focus score; lower is blurrier. */
  sharpness: number;
  luminanceVariance: number;
  laplacianVariance: number;
  /** Number of sampled pixels used for the measurement. */
  sampleCount: number;
  /** Sampling stride, for diagnostics. */
  stride: number;
}

export interface FrameFocusOptions {
  /** Target maximum dimension of the sampled grid. Default 240. */
  maxDimension?: number;
}

const EMPTY_FOCUS: FrameFocus = {
  sharpness: 0,
  luminanceVariance: 0,
  laplacianVariance: 0,
  sampleCount: 0,
  stride: 1,
};

export function computeFrameSharpness(
  data: Uint8Array | ArrayLike<number>,
  width: number,
  height: number,
  options: FrameFocusOptions = {},
): FrameFocus {
  if (
    !data ||
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width <= 0 ||
    height <= 0 ||
    data.length !== width * height
  ) {
    return EMPTY_FOCUS;
  }

  const maxDimension = Math.max(32, Math.floor(options.maxDimension ?? 240));
  const stride = Math.max(1, Math.ceil(Math.max(width, height) / maxDimension));
  const startX = stride;
  const startY = stride;
  const endX = width - 1 - stride;
  const endY = height - 1 - stride;
  if (endX <= startX || endY <= startY) {
    return EMPTY_FOCUS;
  }

  let sampleCount = 0;
  let sum = 0;
  let sumSquares = 0;
  let lapSum = 0;
  let lapSumSquares = 0;
  let lapCount = 0;

  for (let y = startY; y <= endY; y += stride) {
    const row = y * width;
    for (let x = startX; x <= endX; x += stride) {
      const center = Number(data[row + x]);
      if (!Number.isFinite(center)) return EMPTY_FOCUS;
      sampleCount += 1;
      sum += center;
      sumSquares += center * center;

      const left = Number(data[row + x - stride]);
      const right = Number(data[row + x + stride]);
      const up = Number(data[row - stride * width + x]);
      const down = Number(data[row + stride * width + x]);
      if (
        !Number.isFinite(left) ||
        !Number.isFinite(right) ||
        !Number.isFinite(up) ||
        !Number.isFinite(down)
      ) {
        return EMPTY_FOCUS;
      }
      const laplacian = 4 * center - left - right - up - down;
      lapSum += laplacian;
      lapSumSquares += laplacian * laplacian;
      lapCount += 1;
    }
  }

  if (sampleCount === 0 || lapCount === 0) {
    return EMPTY_FOCUS;
  }

  const mean = sum / sampleCount;
  const luminanceVariance = Math.max(0, sumSquares / sampleCount - mean * mean);
  const lapMean = lapSum / lapCount;
  // Raw Laplacian variance scales with stride^4 for a smooth image; normalize
  // so the same threshold works at every sampled resolution.
  const lapVariance = Math.max(
    0,
    (lapSumSquares / lapCount - lapMean * lapMean) / (stride * stride * stride * stride),
  );
  const sharpness = lapVariance / (luminanceVariance + 1);

  return {
    sharpness,
    luminanceVariance,
    laplacianVariance: lapVariance,
    sampleCount,
    stride,
  };
}
