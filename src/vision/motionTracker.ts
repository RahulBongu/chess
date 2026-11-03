/**
 * On-device camera-motion board tracker (Milestone 5.4C).
 *
 * Implements actual geometric tracking instead of "hold the last detected
 * corners and smooth them":
 *
 * 1. Shi-Tomasi (min-eigenvalue) feature detection, grid-distributed,
 *    preferring the board quadrilateral.
 * 2. Pyramidal Lucas-Kanade tracking (3 levels, 9x9 window, up to 14
 *    iterations per level) on the raw Y-plane grayscale frames.
 * 3. Forward-backward round-trip rejection (each feature must track back to
 *    its origin within `maxRoundTripErrorPx`).
 * 4. RANSAC homography estimation (Hartley normalisation + Jacobi smallest
 *    eigenvector solver; deterministic seed), with refit over the inlier set.
 * 5. Sanity validation: finite + orientation-preserving homography, strictly
 *    convex remapped corners, bounded scale and translation.
 * 6. Tracking quality score from inlier ratio, round-trip error and spatial
 *    distribution of inliers.
 * 7. Elapsed-time HOLDING / LOST behaviour. A held quadrilateral is never
 *    presented as current tracking: this tracker returns `corners: null`
 *    unless a fresh, validated homography was computed for the frame.
 *
 * Coordinate convention: frames are single-channel grayscale images with
 * pixels indexed as `data[y * width + x]`; the pixel-grid spans [0, width]
 * x [0, height] boundary convention used elsewhere in the app.
 *
 * This module is pure TypeScript with no native dependencies, so the same
 * implementation runs in the app and in Node-based verification harnesses.
 */

import { isConvexQuadrilateral } from './boardDetection.ts';
import type { Point2D } from './perspectiveOverlay.ts';

export interface GrayBuffer {
  width: number;
  height: number;
  data: Uint8Array;
}

export type MotionTrackState = 'uninitialized' | 'tracking' | 'holding' | 'lost';

export interface MotionTrackOptions {
  /** Pyramid depth, ≥ 1. Default 3. */
  pyramidLevels?: number;
  /** Hard cap on detected features. Default 120. */
  maxFeatures?: number;
  /** Minimum Shi-Tomasi min-eigenvalue response. Default 60. */
  minCornerResponse?: number;
  /** Grid cell size in level-0 pixels for spatial distribution. Default 28. */
  gridCellSize?: number;
  /** Patch half-window radius in level pixels (window = (2r+1)²). Default 4. */
  warpWindowRadius?: number;
  /** Maximum LK iterations per level. Default 14. */
  maxIterations?: number;
  /** Forward-backward round-trip rejection threshold, level-0 px. Default 1.5. */
  maxRoundTripErrorPx?: number;
  /** RANSAC reprojection threshold, level-0 px. Default 1.4. */
  ransacThresholdPx?: number;
  /** RANSAC sampling iterations. Default 220. */
  ransacIterations?: number;
  /** Minimum inliers required for a valid homography. Default 10. */
  minInliers?: number;
  /** Minimum inlier ratio (inliers / considered features). Default 0.45. */
  minInlierRatio?: number;
  /** Frames update may fail while still holding the last good fix. Default 1500 ms. */
  maxHoldMs?: number;
  /** Deterministic RANSAC seed. Default 1234567. */
  ransacSeed?: number;
}

export interface TrackedCornerSet {
  /** Normalized [0,1] frame coordinates, ordered TL, TR, BR, BL (matches detector output). */
  normalized: [Point2D, Point2D, Point2D, Point2D];
  /** Pixel coordinates in frame space (boundary convention; 0..width/height). */
  pixels: [Point2D, Point2D, Point2D, Point2D];
}

export interface MotionTrackResult {
  state: MotionTrackState;
  /** Freshly validated corners for THIS frame, or null while holding/lost. */
  corners: TrackedCornerSet | null;
  /** 0..1 quality from inlier ratio, round-trip error and inlier spread. */
  confidence: number;
  inliers: number;
  inlierRatio: number;
  meanRoundTripErrorPx: number | null;
  featureCount: number;
  reason?: string;
  /** When holding/lost, ms since the last frame that produced a fresh fix. */
  sinceGoodMs?: number;
  /**
   * Freshly estimated frame-to-frame homography (row-major 3x3, h[8]=1),
   * or null when no validated fix exists for this frame. Exposed so tests
   * can measure reprojection error against synthetic ground truth.
   */
  homography: Float64Array | null;
}

interface Feature {
  x: number;
  y: number;
}

interface TrackedFeature extends Feature {
  tx: number;
  ty: number;
  fbError: number;
}

const DEFAULTS = {
  pyramidLevels: 3,
  maxFeatures: 120,
  minCornerResponse: 60,
  gridCellSize: 28,
  warpWindowRadius: 4,
  maxIterations: 14,
  maxRoundTripErrorPx: 1.5,
  ransacThresholdPx: 1.4,
  ransacIterations: 220,
  minInliers: 10,
  minInlierRatio: 0.45,
  maxHoldMs: 1500,
  ransacSeed: 1234567,
};

/** Deterministic PRNG (mulberry32) so RANSAC behaves identically on every device. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function sampleBilinearF(image: Float32Array, width: number, height: number, x: number, y: number): number {
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return Number.NaN;
  }
  const clampedX = Math.min(width - 1, Math.max(0, x));
  const clampedY = Math.min(height - 1, Math.max(0, y));
  const x0 = Math.floor(clampedX);
  const y0 = Math.floor(clampedY);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const fx = clampedX - x0;
  const fy = clampedY - y0;
  const i00 = y0 * width + x0;
  const i01 = y0 * width + x1;
  const i10 = y1 * width + x0;
  const i11 = y1 * width + x1;
  const top = image[i00] * (1 - fx) + image[i01] * fx;
  const bottom = image[i10] * (1 - fx) + image[i11] * fx;
  return top * (1 - fy) + bottom * fy;
}

function samplePatch(
  image: Float32Array,
  width: number,
  height: number,
  cx: number,
  cy: number,
  radius: number,
): Float32Array | null {
  const size = (radius * 2 + 1) ** 2;
  const patch = new Float32Array(size);
  let index = 0;
  for (let dy = -radius; dy <= radius; dy += 1) {
    for (let dx = -radius; dx <= radius; dx += 1) {
      const px = cx + dx;
      const py = cy + dy;
      if (px < -0.5 || py < -0.5 || px > width - 0.5 || py > height - 0.5) {
        return null;
      }
      patch[index] = sampleBilinearF(image, width, height, px, py);
      if (!Number.isFinite(patch[index])) {
        return null;
      }
      index += 1;
    }
  }
  return patch;
}

/** Downsamples one level: separable [1,4,6,4,1]/16 Gaussian, then 2x stride. */
function downsample(src: Float32Array, width: number, height: number): { data: Float32Array; width: number; height: number } {
  const temp = new Float32Array(src.length);
  const kernel = [1, 4, 6, 4, 1];
  const norm = 16;
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      let acc = 0;
      for (let k = -2; k <= 2; k += 1) {
        const xx = Math.min(width - 1, Math.max(0, x + k));
        acc += kernel[k + 2] * src[row + xx];
      }
      temp[row + x] = acc / norm;
    }
  }
  const outW = Math.max(1, Math.floor(width / 2));
  const outH = Math.max(1, Math.floor(height / 2));
  const out = new Float32Array(outW * outH);
  for (let oy = 0; oy < outH; oy += 1) {
    for (let ox = 0; ox < outW; ox += 1) {
      let acc = 0;
      for (let k = -2; k <= 2; k += 1) {
        const yy = Math.min(height - 1, Math.max(0, oy * 2 + k));
        acc += kernel[k + 2] * temp[yy * width + ox * 2];
      }
      out[oy * outW + ox] = acc / norm;
    }
  }
  return { data: out, width: outW, height: outH };
}

function buildPyramid(
  frame: GrayBuffer,
  levels: number,
): { data: Float32Array; width: number; height: number }[] {
  if (!frame || frame.width < 2 || frame.height < 2 || frame.data.length < frame.width * frame.height) {
    throw new Error('GrayBuffer is invalid.');
  }
  const base = new Float32Array(frame.width * frame.height);
  for (let i = 0; i < base.length; i += 1) {
    base[i] = frame.data[i];
  }
  const pyramid: { data: Float32Array; width: number; height: number }[] = [
    { data: base, width: frame.width, height: frame.height },
  ];
  for (let level = 1; level < levels; level += 1) {
    const prev = pyramid[level - 1];
    const next = downsample(prev.data, prev.width, prev.height);
    pyramid.push(next);
    if (next.width < 16 || next.height < 16) {
      break;
    }
  }
  return pyramid;
}

/** Central-difference gradients per level. */
function buildGradients(pyramid: { data: Float32Array; width: number; height: number }[]) {
  return pyramid.map((level) => {
    const { data, width, height } = level;
    const gx = new Float32Array(data.length);
    const gy = new Float32Array(data.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const xm = Math.max(0, x - 1);
        const xp = Math.min(width - 1, x + 1);
        const ym = Math.max(0, y - 1);
        const yp = Math.min(height - 1, y + 1);
        gx[y * width + x] = (data[y * width + xp] - data[y * width + xm]) * 0.5;
        gy[y * width + x] = (data[yp * width + x] - data[ym * width + x]) * 0.5;
      }
    }
    return { gx, gy };
  });
}

function trackPoint(
  oldPyramid: { data: Float32Array; width: number; height: number }[],
  oldGradients: { gx: Float32Array; gy: Float32Array }[],
  newPyramid: { data: Float32Array; width: number; height: number }[],
  newGradients: { gx: Float32Array; gy: Float32Array }[],
  startX: number,
  startY: number,
  options: Required<Pick<MotionTrackOptions, 'warpWindowRadius' | 'maxIterations'>>,
): { x: number; y: number } | null {
  const radius = options.warpWindowRadius;
  let shiftX = 0;
  let shiftY = 0;

  for (let level = oldPyramid.length - 1; level >= 0; level -= 1) {
    const oldLevel = oldPyramid[level];
    const newLevel = newPyramid[level];
    const scale = 1 / 2 ** level;
    const srcX = startX * scale;
    const srcY = startY * scale;
    const patch = samplePatch(oldLevel.data, oldLevel.width, oldLevel.height, srcX, srcY, radius);
    if (!patch) {
      return null;
    }
    const gx = newGradients[level].gx;
    const gy = newGradients[level].gy;

    let curX = shiftX * scale;
    let curY = shiftY * scale;
    for (let iter = 0; iter < options.maxIterations; iter += 1) {
      let sxx = 0;
      let sxy = 0;
      let syy = 0;
      let sx = 0;
      let sy = 0;
      let index = 0;
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = srcX + curX + dx;
          const ny = srcY + curY + dy;
          const gX = sampleBilinearF(gx, newLevel.width, newLevel.height, nx, ny);
          const gY = sampleBilinearF(gy, newLevel.width, newLevel.height, nx, ny);
          const residual = sampleBilinearF(newLevel.data, newLevel.width, newLevel.height, nx, ny) - patch[index];
          sxx += gX * gX;
          sxy += gX * gY;
          syy += gY * gY;
          sx += gX * residual;
          sy += gY * residual;
          index += 1;
        }
      }
      const det = sxx * syy - sxy * sxy + 1e-3;
      const deltaX = -(syy * sx - sxy * sy) / det;
      const deltaY = -(sxx * sy - sxy * sx) / det;
      curX += deltaX;
      curY += deltaY;
      if (Math.abs(deltaX) + Math.abs(deltaY) < 0.02) {
        break;
      }
    }
    shiftX = curX / scale;
    shiftY = curY / scale;
  }

  return { x: startX + shiftX, y: startY + shiftY };
}

/** Hartley normalisation similarity (translate centroid to origin, RMS distance = sqrt(2)). */
function similarityForNormalization(points: { x: number; y: number }[]) {
  const n = points.length;
  let meanX = 0;
  let meanY = 0;
  for (const p of points) {
    meanX += p.x;
    meanY += p.y;
  }
  meanX /= n;
  meanY /= n;
  let meanDist = 0;
  for (const p of points) {
    meanDist += Math.hypot(p.x - meanX, p.y - meanY);
  }
  meanDist /= n;
  const scale = meanDist > 1e-9 ? Math.SQRT2 / meanDist : 1;
  return { meanX, meanY, scale };
}

function applySimilarity(
  similarity: { meanX: number; meanY: number; scale: number },
  point: { x: number; y: number },
) {
  return {
    x: (point.x - similarity.meanX) * similarity.scale,
    y: (point.y - similarity.meanY) * similarity.scale,
  };
}

/** Jacobi eigenvalue decomposition for a symmetric 9x9 matrix. Returns (eigenvalues, eigenvectors). */
function jacobiEigen(matrix: Float64Array): { eigenvalues: Float64Array; eigenvectors: Float64Array } {
  const n = 9;
  const a = Float64Array.from(matrix);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    v[i * n + i] = 1;
  }
  for (let sweep = 0; sweep < 80; sweep += 1) {
    // Off-diagonal norm
    let off = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        off += a[i * n + j] * a[i * n + j];
      }
    }
    if (off < 1e-18) {
      break;
    }
    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-18) {
          continue;
        }
        const app = a[p * n + p];
        const aqq = a[q * n + q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        for (let k = 0; k < n; k += 1) {
          const akp = a[k * n + p];
          const akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k += 1) {
          const apk = a[p * n + k];
          const aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k += 1) {
          const vkp = v[k * n + p];
          const vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const eigenvalues = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    eigenvalues[i] = a[i * n + i];
  }
  return { eigenvalues, eigenvectors: v };
}

/** Solves the 8x9 DLT system Ah = 0 for (possibly >4) correspondences. H is row-major[9]. */
function solveDlt(
  srcPoints: { x: number; y: number }[],
  dstPoints: { x: number; y: number }[],
): Float64Array | null {
  const rows = srcPoints.length * 2;
  // A^T A (9x9 symmetric)
  const ata = new Float64Array(81);
  for (let i = 0; i < srcPoints.length; i += 1) {
    const { x, y } = srcPoints[i];
    const { x: xp, y: yp } = dstPoints[i];
    const rowA = [-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp];
    const rowB = [0, 0, 0, -x, -y, -1, yp * x, yp * y, yp];
    for (const row of [rowA, rowB]) {
      for (let r = 0; r < 9; r += 1) {
        const rv = row[r];
        if (rv === 0) {
          continue;
        }
        for (let c = r; c < 9; c += 1) {
          ata[r * 9 + c] += rv * row[c];
        }
      }
    }
  }
  // Mirror the symmetric part
  for (let r = 0; r < 9; r += 1) {
    for (let c = 0; c < r; c += 1) {
      ata[r * 9 + c] = ata[c * 9 + r];
    }
  }
  const { eigenvalues, eigenvectors } = jacobiEigen(ata);
  let minIndex = 0;
  for (let i = 1; i < 9; i += 1) {
    if (eigenvalues[i] < eigenvalues[minIndex]) {
      minIndex = i;
    }
  }
  const h = new Float64Array(9);
  for (let i = 0; i < 9; i += 1) {
    h[i] = eigenvectors[i * 9 + minIndex];
  }
  if (!Number.isFinite(h[8]) || Math.abs(h[8]) < 1e-12) {
    return null;
  }
  const inv = 1 / h[8];
  for (let i = 0; i < 9; i += 1) {
    h[i] *= inv;
  }
  if (!Number.isFinite(h[0])) {
    return null;
  }
  void rows;
  return h;
}

function multiplyHomographies(a: Float64Array, b: Float64Array): Float64Array {
  // row-major 3x3 multiply a * b
  const out = new Float64Array(9);
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[r * 3 + c] =
        a[r * 3 + 0] * b[0 * 3 + c] + a[r * 3 + 1] * b[1 * 3 + c] + a[r * 3 + 2] * b[2 * 3 + c];
    }
  }
  return out;
}

function similarityMatrix(sim: { meanX: number; meanY: number; scale: number }): Float64Array {
  // [s 0 -s*meanX; 0 s -s*meanY; 0 0 1]
  return Float64Array.from([
    sim.scale, 0, -sim.scale * sim.meanX,
    0, sim.scale, -sim.scale * sim.meanY,
    0, 0, 1,
  ]);
}

function similarityInverse(sim: { meanX: number; meanY: number; scale: number }): Float64Array {
  const inv = 1 / sim.scale;
  return Float64Array.from([
    inv, 0, sim.meanX,
    0, inv, sim.meanY,
    0, 0, 1,
  ]);
}

export function projectHomography(h: Float64Array, x: number, y: number): { x: number; y: number } {
  const denominator = h[6] * x + h[7] * y + h[8];
  if (Math.abs(denominator) < 1e-12) {
    return { x: Number.NaN, y: Number.NaN };
  }
  return {
    x: (h[0] * x + h[1] * y + h[2]) / denominator,
    y: (h[3] * x + h[4] * y + h[5]) / denominator,
  };
}

/**
 * RANSAC homography from point correspondences.
 * Returns the refitted homography and the inlier indices, or null when the
 * model cannot reach the requested inlier quality.
 */
export function estimateHomographyRansac(
  srcPoints: { x: number; y: number }[],
  dstPoints: { x: number; y: number }[],
  options: {
    thresholdPx?: number;
    iterations?: number;
    minInliers?: number;
    seed?: number;
  } = {},
): { homography: Float64Array; inliers: number[] } | null {
  if (srcPoints.length !== dstPoints.length || srcPoints.length < 4) {
    return null;
  }
  const threshold = options.thresholdPx ?? DEFAULTS.ransacThresholdPx;
  const iterations = options.iterations ?? DEFAULTS.ransacIterations;
  const minInliers = options.minInliers ?? DEFAULTS.minInliers;
  const seed = options.seed ?? DEFAULTS.ransacSeed;
  const rng = mulberry32(seed);

  const simSrc = similarityForNormalization(srcPoints);
  const simDst = similarityForNormalization(dstPoints);
  const srcNorm = srcPoints.map((p) => applySimilarity(simSrc, p));
  const dstNorm = dstPoints.map((p) => applySimilarity(simDst, p));

  const tSrc = similarityMatrix(simSrc);
  const tDstInv = similarityInverse(simDst);

  const reprojectError = (h: Float64Array) => {
    const errors = new Float64Array(srcPoints.length);
    for (let i = 0; i < srcPoints.length; i += 1) {
      const projected = projectHomography(h, srcNorm[i].x, srcNorm[i].y);
      if (!Number.isFinite(projected.x)) {
        errors[i] = Number.POSITIVE_INFINITY;
        continue;
      }
      errors[i] = Math.hypot(projected.x - dstNorm[i].x, projected.y - dstNorm[i].y);
    }
    return errors;
  };

  let bestInliers: number[] = [];
  for (let iter = 0; iter < iterations; iter += 1) {
    const picked: number[] = [];
    while (picked.length < 4) {
      const index = Math.floor(rng() * srcPoints.length);
      if (!picked.includes(index)) {
        picked.push(index);
      }
    }
    const s = picked.map((i) => srcNorm[i]);
    const d = picked.map((i) => dstNorm[i]);
    const hNorm = solveDlt(s, d);
    if (!hNorm) {
      continue;
    }
    const errors = reprojectError(hNorm);
    const inliers: number[] = [];
    for (let i = 0; i < errors.length; i += 1) {
      if (errors[i] <= threshold) {
        inliers.push(i);
      }
    }
    if (inliers.length > bestInliers.length) {
      bestInliers = inliers;
    }
    if (bestInliers.length >= srcPoints.length) {
      break;
    }
  }

  if (bestInliers.length < minInliers) {
    return null;
  }

  // Refit on the inlier set in normalized coordinates.
  const fitSrc = bestInliers.map((i) => srcNorm[i]);
  const fitDst = bestInliers.map((i) => dstNorm[i]);
  let finalNorm = solveDlt(fitSrc, fitDst);
  if (!finalNorm) {
    const s = bestInliers.slice(0, 4).map((i) => srcNorm[i]);
    const d = bestInliers.slice(0, 4).map((i) => dstNorm[i]);
    finalNorm = solveDlt(s, d);
  }
  if (!finalNorm) {
    return null;
  }

  const final = multiplyHomographies(tDstInv, multiplyHomographies(finalNorm, tSrc));
  if (Math.abs(final[8]) < 1e-12) {
    return null;
  }
  const inv = 1 / final[8];
  for (let i = 0; i < 9; i += 1) {
    final[i] *= inv;
  }

  // Re-verify in pixel space
  const pixelErrors = srcPoints.map((p, i) => {
    const projected = projectHomography(final, p.x, p.y);
    return Math.hypot(projected.x - dstPoints[i].x, projected.y - dstPoints[i].y);
  });
  const finalInliers = pixelErrors
    .map((err, i) => ({ err, i }))
    .filter(({ err }) => err <= threshold * 2)
    .map(({ i }) => i);

  if (finalInliers.length < minInliers) {
    return null;
  }

  return { homography: final, inliers: finalInliers };
}

/** Grid-distributed Shi-Tomasi corners, preferring pixels inside the board quad. */
export function detectShiTomasiFeatures(
  level: { data: Float32Array; width: number; height: number },
  restrictTo: { x: number; y: number }[] | null,
  options: { maxFeatures: number; minResponse: number; gridCellSize: number },
): Feature[] {
  const { data, width, height } = level;
  const cell = options.gridCellSize;
  const ixx = new Float32Array(data.length);
  const iyy = new Float32Array(data.length);
  const ixy = new Float32Array(data.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const xm = Math.max(0, x - 1);
      const xp = Math.min(width - 1, x + 1);
      const ym = Math.max(0, y - 1);
      const yp = Math.min(height - 1, y + 1);
      const gx = (data[y * width + xp] - data[y * width + xm]) * 0.5;
      const gy = (data[yp * width + x] - data[ym * width + x]) * 0.5;
      ixx[y * width + x] = gx * gx;
      iyy[y * width + x] = gy * gy;
      ixy[y * width + x] = gx * gy;
    }
  }

  // Restriction bounding box (board quad inflated by 15%)
  let minX = -Infinity;
  let minY = -Infinity;
  let maxX = Infinity;
  let maxY = Infinity;
  if (restrictTo && restrictTo.length === 4) {
    minX = Math.min(...restrictTo.map((p) => p.x));
    maxX = Math.max(...restrictTo.map((p) => p.x));
    minY = Math.min(...restrictTo.map((p) => p.y));
    maxY = Math.max(...restrictTo.map((p) => p.y));
    const padX = (maxX - minX) * 0.15;
    const padY = (maxY - minY) * 0.15;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;
  }

  const win = 2; // 5x5 window
  const cellW = Math.max(1, Math.ceil(width / cell));
  const cellH = Math.max(1, Math.ceil(height / cell));
  const picked: { x: number; y: number; response: number }[] = [];
  const perCell = 2;

  for (let cy = 0; cy < cellH; cy += 1) {
    for (let cx = 0; cx < cellW; cx += 1) {
      const x0 = cx * cell;
      const x1 = Math.min(width, x0 + cell);
      const y0 = cy * cell;
      const y1 = Math.min(height, y0 + cell);
      // Fast skip completely-outside cells
      if (x1 < minX || x0 > maxX || y1 < minY || y0 > maxY) {
        continue;
      }
      const cellCandidates: { x: number; y: number; response: number }[] = [];
      for (let y = Math.max(win, y0); y < Math.min(height - win, y1); y += 1) {
        for (let x = Math.max(win, x0); x < Math.min(width - win, x1); x += 1) {
          if (x < minX || x > maxX || y < minY || y > maxY) {
            continue;
          }
          let sxx = 0;
          let syy = 0;
          let sxy = 0;
          for (let wy = -win; wy <= win; wy += 1) {
            const row = (y + wy) * width;
            for (let wx = -win; wx <= win; wx += 1) {
              const idx = row + x + wx;
              sxx += ixx[idx];
              syy += iyy[idx];
              sxy += ixy[idx];
            }
          }
          const track = sxx + syy;
          const det = sxx * syy - sxy * sxy;
          const minEigen = (track - Math.sqrt(Math.max(0, track * track - 4 * det))) / 2;
          if (minEigen > options.minResponse) {
            cellCandidates.push({ x, y, response: minEigen });
          }
        }
      }
      cellCandidates.sort((a, b) => b.response - a.response);
      for (let i = 0; i < Math.min(perCell, cellCandidates.length); i += 1) {
        picked.push(cellCandidates[i]);
      }
    }
  }

  picked.sort((a, b) => b.response - a.response);
  return picked.slice(0, options.maxFeatures).map(({ x, y }) => ({ x, y }));
}

function quadArea(points: { x: number; y: number }[]): number {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    const q = points[(i + 1) % points.length];
    area += p.x * q.y - q.x * p.y;
  }
  return Math.abs(area) / 2;
}

/** Signed area (positive = counter-clockwise in screen coordinates). */
function signedQuadArea(points: { x: number; y: number }[]): number {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    const q = points[(i + 1) % points.length];
    area += p.x * q.y - q.x * p.y;
  }
  return area / 2;
}

export class BoardMotionTracker {
  private readonly options: Required<MotionTrackOptions>;
  private prevPyramid: { data: Float32Array; width: number; height: number }[] | null = null;
  private prevGradients: { gx: Float32Array; gy: Float32Array }[] | null = null;
  private prevCornersNorm: [Point2D, Point2D, Point2D, Point2D] | null = null;
  private prevCornersPx: [Point2D, Point2D, Point2D, Point2D] | null = null;
  private frameWidth = 0;
  private frameHeight = 0;
  private state: MotionTrackState = 'uninitialized';
  private lastGoodTimeMs: number | null = null;
  private sinceGoodMs: number | undefined = undefined;

  constructor(options: MotionTrackOptions = {}) {
    this.options = {
      pyramidLevels: Math.max(1, options.pyramidLevels ?? DEFAULTS.pyramidLevels),
      maxFeatures: options.maxFeatures ?? DEFAULTS.maxFeatures,
      minCornerResponse: options.minCornerResponse ?? DEFAULTS.minCornerResponse,
      gridCellSize: options.gridCellSize ?? DEFAULTS.gridCellSize,
      warpWindowRadius: options.warpWindowRadius ?? DEFAULTS.warpWindowRadius,
      maxIterations: options.maxIterations ?? DEFAULTS.maxIterations,
      maxRoundTripErrorPx: options.maxRoundTripErrorPx ?? DEFAULTS.maxRoundTripErrorPx,
      ransacThresholdPx: options.ransacThresholdPx ?? DEFAULTS.ransacThresholdPx,
      ransacIterations: options.ransacIterations ?? DEFAULTS.ransacIterations,
      minInliers: options.minInliers ?? DEFAULTS.minInliers,
      minInlierRatio: options.minInlierRatio ?? DEFAULTS.minInlierRatio,
      maxHoldMs: options.maxHoldMs ?? DEFAULTS.maxHoldMs,
      ransacSeed: options.ransacSeed ?? DEFAULTS.ransacSeed,
    };
  }

  getState(): MotionTrackState {
    return this.state;
  }

  needsReseed(): boolean {
    return this.state === 'uninitialized' || this.state === 'lost';
  }

  /**
   * Seeds the tracker with a trusted set of corners (assisted calibration or a
   * high-confidence detection) plus the frame they were derived from.
   */
  seed(frame: GrayBuffer, normalizedCorners: Point2D[], nowMs?: number): boolean {
    if (!normalizedCorners || normalizedCorners.length !== 4) {
      return false;
    }
    const cornersPx = normalizedCorners.map((p) => ({
      x: p.x * frame.width,
      y: p.y * frame.height,
    }));
    if (!isConvexQuadrilateral(cornersPx)) {
      return false;
    }
    this.prevPyramid = buildPyramid(frame, this.options.pyramidLevels);
    this.prevGradients = buildGradients(this.prevPyramid);
    this.prevCornersNorm = normalizedCorners.map((p) => ({ x: p.x, y: p.y })) as [Point2D, Point2D, Point2D, Point2D];
    this.prevCornersPx = cornersPx as [Point2D, Point2D, Point2D, Point2D];
    this.frameWidth = frame.width;
    this.frameHeight = frame.height;
    this.state = 'tracking';
    this.lastGoodTimeMs = nowMs ?? Date.now();
    this.sinceGoodMs = undefined;
    return true;
  }

  reset(): void {
    this.prevPyramid = null;
    this.prevGradients = null;
    this.prevCornersNorm = null;
    this.prevCornersPx = null;
    this.frameWidth = 0;
    this.frameHeight = 0;
    this.state = 'uninitialized';
    this.lastGoodTimeMs = null;
    this.sinceGoodMs = undefined;
  }

  /**
   * Tracks the board between the previously seeded/tracked frame and `frame`.
   * The returned corners are always freshly computed for this frame; when the
   * fix cannot be validated, `corners` is null and the caller must fall back
   * to hold/lost handling (and eventually re-seed from a detection).
   */
  update(frame: GrayBuffer, nowMs?: number): MotionTrackResult {
    const now = nowMs ?? Date.now();
    if (this.state === 'lost') {
      const sinceGood = this.lastGoodTimeMs !== null ? now - this.lastGoodTimeMs : undefined;
      return {
        state: 'lost',
        corners: null,
        confidence: 0,
        inliers: 0,
        inlierRatio: 0,
        meanRoundTripErrorPx: null,
        featureCount: 0,
        reason: 'Tracking lost; re-detection or re-calibration required.',
        sinceGoodMs: sinceGood,
        homography: null,
      };
    }
    if (
      !this.prevPyramid ||
      !this.prevGradients ||
      !this.prevCornersPx ||
      frame.width !== this.frameWidth ||
      frame.height !== this.frameHeight
    ) {
      this.state = 'uninitialized';
      return {
        state: 'uninitialized',
        corners: null,
        confidence: 0,
        inliers: 0,
        inlierRatio: 0,
        meanRoundTripErrorPx: null,
        featureCount: 0,
        reason: 'Motion tracker is not initialised; seed from a detection or calibration.',
        homography: null,
      };
    }

    let pyramid: { data: Float32Array; width: number; height: number }[];
    let gradients: { gx: Float32Array; gy: Float32Array }[];
    try {
      pyramid = buildPyramid(frame, this.options.pyramidLevels);
      gradients = buildGradients(pyramid);
    } catch (error) {
      return this.fail(now, error instanceof Error ? error.message : 'Frame pyramid build failed.');
    }

    const features = detectShiTomasiFeatures(
      this.prevPyramid[0],
      this.prevCornersPx,
      {
        maxFeatures: this.options.maxFeatures,
        minResponse: this.options.minCornerResponse,
        gridCellSize: this.options.gridCellSize,
      },
    );

    if (features.length < 12) {
      return this.fail(now, `Only ${features.length} trackable features in the board region.`);
    }

    // Forward tracking + round-trip verification
    const tracked: TrackedFeature[] = [];
    for (const feature of features) {
      const forward = trackPoint(
        this.prevPyramid,
        this.prevGradients,
        pyramid,
        gradients,
        feature.x,
        feature.y,
        {
          warpWindowRadius: this.options.warpWindowRadius,
          maxIterations: this.options.maxIterations,
        },
      );
      if (!forward) {
        continue;
      }
      const backward = trackPoint(
        pyramid,
        gradients,
        this.prevPyramid,
        this.prevGradients,
        forward.x,
        forward.y,
        {
          warpWindowRadius: this.options.warpWindowRadius,
          maxIterations: this.options.maxIterations,
        },
      );
      if (!backward) {
        continue;
      }
      const fbError = Math.hypot(backward.x - feature.x, backward.y - feature.y);
      tracked.push({ x: feature.x, y: feature.y, tx: forward.x, ty: forward.y, fbError });
    }

    const survivors = tracked.filter((t) => t.fbError <= this.options.maxRoundTripErrorPx);
    if (survivors.length < this.options.minInliers) {
      return this.fail(
        now,
        `Only ${survivors.length} features survived round-trip verification.`,
      );
    }

    const estimation = estimateHomographyRansac(
      survivors.map((t) => ({ x: t.x, y: t.y })),
      survivors.map((t) => ({ x: t.tx, y: t.ty })),
      {
        thresholdPx: this.options.ransacThresholdPx,
        iterations: this.options.ransacIterations,
        minInliers: this.options.minInliers,
        seed: this.options.ransacSeed,
      },
    );

    if (!estimation) {
      return this.fail(now, 'RANSAC could not fit a stable homography to the tracked features.');
    }

    const inlierSet = estimation.inliers.map((i) => survivors[i]);
    const mappedCorners = this.prevCornersPx.map((p) =>
      projectHomography(estimation.homography, p.x, p.y),
    );

    const sanity = this.validateMappedCorners(mappedCorners);
    if (!sanity.valid) {
      return this.fail(now, sanity.reason ?? 'Mapped corners failed sanity checks.');
    }

    const meanFb = inlierSet.reduce((sum, t) => sum + t.fbError, 0) / inlierSet.length;
    const inlierRatio = inlierSet.length / tracked.length;

    // Enforce the documented minimum inlier ratio: a homography supported by
    // only a small fraction of the tracked features (large jump, partial
    // occlusion, degenerate match) must not be reported as fresh tracking.
    if (inlierRatio < this.options.minInlierRatio) {
      return this.fail(
        now,
        `Inlier ratio ${inlierRatio.toFixed(2)} is below the required ${this.options.minInlierRatio.toFixed(2)} ` +
          `(${inlierSet.length}/${tracked.length} features).`,
      );
    }

    // Spatial distribution of inliers relative to the board area
    const xs = inlierSet.map((t) => t.tx);
    const ys = inlierSet.map((t) => t.ty);
    const featSpanX = Math.max(...xs) - Math.min(...xs);
    const featSpanY = Math.max(...ys) - Math.min(...ys);
    const quad = quadArea(mappedCorners);
    const distribution = quad > 0 ? clamp01((featSpanX * featSpanY) / quad) : 0;

    const fbQuality = clamp01(1 - meanFb / this.options.maxRoundTripErrorPx);
    const distributionQuality = clamp01(distribution / 0.45);
    const quality = clamp01(0.55 * inlierRatio + 0.25 * fbQuality + 0.2 * distributionQuality);

    const pixels = mappedCorners as [Point2D, Point2D, Point2D, Point2D];
    const normalized = pixels.map((p) => ({
      x: p.x / this.frameWidth,
      y: p.y / this.frameHeight,
    })) as [Point2D, Point2D, Point2D, Point2D];

    this.prevPyramid = pyramid;
    this.prevGradients = gradients;
    this.prevCornersPx = pixels;
    this.prevCornersNorm = normalized;
    this.state = 'tracking';
    this.lastGoodTimeMs = now;
    this.sinceGoodMs = undefined;

    return {
      state: 'tracking',
      corners: { normalized, pixels },
      confidence: quality,
      inliers: inlierSet.length,
      inlierRatio,
      meanRoundTripErrorPx: meanFb,
      featureCount: features.length,
      homography: estimation.homography,
    };
  }

  private fail(now: number, reason: string): MotionTrackResult {
    const sinceGood = this.lastGoodTimeMs !== null ? now - this.lastGoodTimeMs : null;
    const withinHold =
      sinceGood !== null && sinceGood <= this.options.maxHoldMs;
    this.state = withinHold ? 'holding' : 'lost';
    this.sinceGoodMs = sinceGood ?? undefined;
    return {
      state: this.state,
      corners: null,
      confidence: 0,
      inliers: 0,
      inlierRatio: 0,
      meanRoundTripErrorPx: null,
      featureCount: 0,
      reason,
      sinceGoodMs: this.sinceGoodMs,
      homography: null,
    };
  }

  private validateMappedCorners(
    corners: { x: number; y: number }[],
  ): { valid: boolean; reason?: string } {
    if (corners.some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y))) {
      return { valid: false, reason: 'Homography produced non-finite corners.' };
    }
    // Orientation preservation (affine part must not flip)
    // (checked implicitly via convexity below, kept explicit for clarity)
    if (!isConvexQuadrilateral(corners as Point2D[])) {
      return { valid: false, reason: 'Mapped corners are not a strictly convex quadrilateral.' };
    }
    // Orientation preservation: a valid physical camera motion can never flip
    // the winding of the board quad. Rejects mirrored/inside-out homographies
    // that are still geometrically convex.
    const prevWinding = signedQuadArea(this.prevCornersPx!);
    const newWinding = signedQuadArea(corners);
    if (
      Math.abs(prevWinding) > 1e-9 &&
      Math.abs(newWinding) > 1e-9 &&
      Math.sign(prevWinding) !== Math.sign(newWinding)
    ) {
      return { valid: false, reason: 'Homography flipped the board orientation.' };
    }
    // Bounded drift: spans may not change by more than 2x per update
    const prev = this.prevCornersPx!;
    const prevSpanX = Math.max(...prev.map((p) => p.x)) - Math.min(...prev.map((p) => p.x));
    const prevSpanY = Math.max(...prev.map((p) => p.y)) - Math.min(...prev.map((p) => p.y));
    const newSpanX = Math.max(...corners.map((p) => p.x)) - Math.min(...corners.map((p) => p.x));
    const newSpanY = Math.max(...corners.map((p) => p.y)) - Math.min(...corners.map((p) => p.y));
    if (prevSpanX <= 0 || prevSpanY <= 0) {
      return { valid: false, reason: 'Previous board span is degenerate.' };
    }
    const ratioX = newSpanX / prevSpanX;
    const ratioY = newSpanY / prevSpanY;
    if (ratioX < 0.5 || ratioX > 2.0 || ratioY < 0.5 || ratioY > 2.0) {
      return { valid: false, reason: `Scale change out of bounds (x=${ratioX.toFixed(2)}, y=${ratioY.toFixed(2)}).` };
    }
    // Fully outside the frame?
    const marginX = this.frameWidth * 0.6;
    const marginY = this.frameHeight * 0.6;
    if (
      corners.every((p) => p.x < -marginX || p.x > this.frameWidth + marginX) ||
      corners.every((p) => p.y < -marginY || p.y > this.frameHeight + marginY)
    ) {
      return { valid: false, reason: 'The board left the camera frame.' };
    }
    return { valid: true };
  }
}
