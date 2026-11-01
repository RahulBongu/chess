import { getPixelLuminanceAndRgb, type AnyImageData } from './classifySquare.ts';
import type {
  PieceClassifierModel,
  PieceClassificationResult,
  PiecePredictionCandidate,
  PieceType,
} from './types.ts';
import { PIECE_TYPES } from './types.ts';

export const HOG_CELL_SIZE = 16;
export const HOG_PATCH_SIZE = 64;
export const HOG_NUM_BINS = 8;
export const HOG_CELLS_PER_AXIS = 4; // 64 / 16 = 4 cells horizontally and vertically
export const HOG_BLOCKS_PER_AXIS = 3; // 4 - 2 + 1 = 3 overlapping blocks
export const HOG_BLOCK_SIZE = 2; // 2x2 cells per block
export const HOG_FEATURE_DIMENSION =
  HOG_BLOCKS_PER_AXIS * HOG_BLOCKS_PER_AXIS * (HOG_BLOCK_SIZE * HOG_BLOCK_SIZE) * HOG_NUM_BINS; // 3 * 3 * 4 * 8 = 288

export const DEFAULT_PIECE_CONFIDENCE_THRESHOLD = 0.6;
export const DEFAULT_PIECE_MARGIN_THRESHOLD = 0.2;

/**
 * Validates image integrity and throws an informative error if malformed.
 */
function validatePatchImage(image: AnyImageData | null | undefined): void {
  if (!image) {
    throw new Error('Image data is null or undefined.');
  }
  if (!Number.isFinite(image.width) || image.width <= 0) {
    throw new Error(`Invalid image width: ${image.width}. Must be a positive integer.`);
  }
  if (!Number.isFinite(image.height) || image.height <= 0) {
    throw new Error(`Invalid image height: ${image.height}. Must be a positive integer.`);
  }
  if (!image.data || image.data.length === 0) {
    throw new Error('Image byte buffer is empty or missing.');
  }
}

/**
 * Extracts a deterministic 288-bin Histogram of Oriented Gradients (HOG) descriptor
 * from a normalized 64x64 piece region within a square patch.
 *
 * Parameters:
 * - 64x64 canonical patch (sampled from square center if square is 100x100)
 * - 4x4 grid of 16x16 pixel cells
 * - 8 unsigned orientation bins (0° to 180°, 22.5° per bin)
 * - 3x3 overlapping blocks with 2x2 cell neighborhood and L2-norm normalization
 * - Total descriptor size: 3 * 3 * 4 * 8 = 288 features
 *
 * Memory: Writes into `scratchBuffer` if provided, preventing garbage-collection overhead.
 */
export function extractPieceHogFeatures(
  image: AnyImageData,
  options?: {
    roi?: { x: number; y: number; width: number; height: number };
    scratchBuffer?: Float32Array;
  },
): Float32Array {
  validatePatchImage(image);

  const out = options?.scratchBuffer ?? new Float32Array(HOG_FEATURE_DIMENSION);
  if (out.length !== HOG_FEATURE_DIMENSION) {
    throw new Error(
      `Scratch buffer length ${out.length} does not match expected HOG feature dimension ${HOG_FEATURE_DIMENSION}.`,
    );
  }
  out.fill(0);

  // Determine sampling bounding box (defaults to central 64x64 if 100x100 square)
  const roi = options?.roi ?? {
    x: image.width >= 100 ? Math.round((image.width - HOG_PATCH_SIZE) / 2) : 0,
    y: image.height >= 100 ? Math.round((image.height - HOG_PATCH_SIZE) / 2) : 0,
    width: Math.min(image.width, HOG_PATCH_SIZE),
    height: Math.min(image.height, HOG_PATCH_SIZE),
  };

  const xStep = roi.width / HOG_PATCH_SIZE;
  const yStep = roi.height / HOG_PATCH_SIZE;

  // 1. Compute pixel gradients and accumulate into 4x4 cell histograms (each has 8 bins)
  // cellHist[cellY][cellX][bin]
  const cellHist = new Float32Array(HOG_CELLS_PER_AXIS * HOG_CELLS_PER_AXIS * HOG_NUM_BINS);

  for (let py = 1; py < HOG_PATCH_SIZE - 1; py += 1) {
    const srcY = roi.y + py * yStep;
    const srcYPrev = roi.y + (py - 1) * yStep;
    const srcYNext = roi.y + (py + 1) * yStep;
    const cellY = Math.min(HOG_CELLS_PER_AXIS - 1, Math.floor(py / HOG_CELL_SIZE));

    for (let px = 1; px < HOG_PATCH_SIZE - 1; px += 1) {
      const srcX = roi.x + px * xStep;
      const srcXPrev = roi.x + (px - 1) * xStep;
      const srcXNext = roi.x + (px + 1) * xStep;
      const cellX = Math.min(HOG_CELLS_PER_AXIS - 1, Math.floor(px / HOG_CELL_SIZE));

      // Sobel / central difference gradients
      const left = getPixelLuminanceAndRgb(image, srcXPrev, srcY).lum;
      const right = getPixelLuminanceAndRgb(image, srcXNext, srcY).lum;
      const up = getPixelLuminanceAndRgb(image, srcX, srcYPrev).lum;
      const down = getPixelLuminanceAndRgb(image, srcX, srcYNext).lum;

      const gx = right - left;
      const gy = down - up;

      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag < 1.0) {
        continue;
      }

      // Unsigned angle in degrees [0, 180)
      let angle = (Math.atan2(gy, gx) * 180) / Math.PI;
      if (angle < 0) {
        angle += 180;
      }
      if (angle >= 180) {
        angle = 0;
      }

      // Linear interpolation between the two nearest bin centers
      const binWidth = 180 / HOG_NUM_BINS; // 22.5 degrees
      const continuousBin = angle / binWidth;
      const bin0 = Math.floor(continuousBin) % HOG_NUM_BINS;
      const bin1 = (bin0 + 1) % HOG_NUM_BINS;
      const weight1 = continuousBin - Math.floor(continuousBin);
      const weight0 = 1 - weight1;

      const cellBase = (cellY * HOG_CELLS_PER_AXIS + cellX) * HOG_NUM_BINS;
      cellHist[cellBase + bin0] += mag * weight0;
      cellHist[cellBase + bin1] += mag * weight1;
    }
  }

  // 2. Block normalization: 3x3 blocks of 2x2 cells, with L2-Hys normalization
  let outIdx = 0;
  const eps = 1e-4;

  for (let by = 0; by < HOG_BLOCKS_PER_AXIS; by += 1) {
    for (let bx = 0; bx < HOG_BLOCKS_PER_AXIS; bx += 1) {
      const blockStart = outIdx;
      let sumSq = 0;

      // Concatenate 2x2 cells
      for (let cy = 0; cy < HOG_BLOCK_SIZE; cy += 1) {
        const cellY = by + cy;
        for (let cx = 0; cx < HOG_BLOCK_SIZE; cx += 1) {
          const cellX = bx + cx;
          const cellBase = (cellY * HOG_CELLS_PER_AXIS + cellX) * HOG_NUM_BINS;
          for (let b = 0; b < HOG_NUM_BINS; b += 1) {
            const val = cellHist[cellBase + b];
            out[outIdx] = val;
            sumSq += val * val;
            outIdx += 1;
          }
        }
      }

      // L2 norm across block features
      const norm = Math.sqrt(sumSq + eps * eps);
      for (let i = blockStart; i < outIdx; i += 1) {
        out[i] /= norm;
      }
    }
  }

  return out;
}

/**
 * Validates that an explicitly supplied piece classifier model has valid dimensions.
 */
export function validatePieceClassifierModel(model: PieceClassifierModel): void {
  if (!model) {
    throw new Error('Model is null or undefined.');
  }
  if (!model.classes || model.classes.length !== 6) {
    throw new Error(
      `Invalid model class count: expected 6 classes, received ${model.classes?.length ?? 0}.`,
    );
  }
  if (model.featureDimension !== HOG_FEATURE_DIMENSION) {
    throw new Error(
      `Invalid model feature dimension: expected ${HOG_FEATURE_DIMENSION}, received ${model.featureDimension}.`,
    );
  }
  if (!model.weights || model.weights.length !== 6) {
    throw new Error(
      `Invalid model weight matrix: expected 6 class rows, received ${model.weights?.length ?? 0}.`,
    );
  }
  for (let c = 0; c < 6; c += 1) {
    const row = model.weights[c];
    if (!row || row.length !== HOG_FEATURE_DIMENSION) {
      throw new Error(
        `Model weight row for class index ${c} has invalid length: expected ${HOG_FEATURE_DIMENSION}, received ${row?.length ?? 0}.`,
      );
    }
  }
  if (!model.biases || model.biases.length !== 6) {
    throw new Error(
      `Invalid model bias vector: expected 6 values, received ${model.biases?.length ?? 0}.`,
    );
  }
}

/**
 * Checks whether a candidate piece classification causes an invariant rule conflict.
 * Zero-silent-error policy: Flags the conflict without silently modifying the prediction.
 */
export function checkPieceInvariantConflict(
  squareName: string,
  candidateType: PieceType,
  color?: 'white' | 'black' | 'uncertain',
): { hasConflict: boolean; reason?: string } {
  if (!squareName || squareName.length < 2) {
    return { hasConflict: false };
  }

  const rank = squareName.charAt(1);

  // FIDE Rule: Pawns cannot exist on the 1st or 8th rank
  if (candidateType === 'pawn' && (rank === '1' || rank === '8')) {
    const colorLabel = color && color !== 'uncertain' ? `${color} ` : '';
    return {
      hasConflict: true,
      reason: `Rule invariant violation: ${colorLabel}pawn cannot exist on rank ${rank}.`,
    };
  }

  return { hasConflict: false };
}

/**
 * Evaluates an extracted square patch against an explicitly supplied 6-class model.
 *
 * Requirements:
 * - If model is null or undefined, returns a safe unclassified result with metadata.
 * - Calibrates raw decision logits using Softmax with temperature scaling.
 * - Computes candidate ranking, top probability, and margin to runner-up.
 * - Marks prediction uncertain if confidence is low, margin is narrow, or color is uncertain.
 * - Does NOT silently change predicted piece types to force chess invariants.
 */
export function classifyPieceFromPatch(
  patch: AnyImageData,
  options: {
    model: PieceClassifierModel | null | undefined;
    color?: 'white' | 'black' | 'uncertain';
    squareName?: string;
    confidenceThreshold?: number;
    marginThreshold?: number;
    scratchBuffer?: Float32Array;
  },
): PieceClassificationResult {
  validatePatchImage(patch);

  const features = extractPieceHogFeatures(patch, {
    scratchBuffer: options.scratchBuffer,
  });

  const color = options.color ?? 'uncertain';
  const confidenceThreshold = options.confidenceThreshold ?? DEFAULT_PIECE_CONFIDENCE_THRESHOLD;
  const marginThreshold = options.marginThreshold ?? DEFAULT_PIECE_MARGIN_THRESHOLD;

  // 1. Model absence check
  if (!options.model) {
    return {
      predictedType: null,
      confidence: 0,
      margin: 0,
      isUncertain: true,
      candidates: [],
      features,
      color,
      reason: 'No piece classification model provided. Automatic piece-type recognition skipped.',
    };
  }

  validatePieceClassifierModel(options.model);

  const { classes, weights, biases } = options.model;
  const temperature = options.model.temperature && options.model.temperature > 0 ? options.model.temperature : 1.0;

  // 2. Compute logits: logit_c = bias_c + sum(weights_c,i * feat_i)
  const logits = new Float32Array(6);
  let maxLogit = -Infinity;

  for (let c = 0; c < 6; c += 1) {
    let dot = biases[c];
    const wRow = weights[c];
    for (let i = 0; i < HOG_FEATURE_DIMENSION; i += 1) {
      dot += wRow[i] * features[i];
    }
    logits[c] = dot;
    if (dot > maxLogit) {
      maxLogit = dot;
    }
  }

  // 3. Calibrated Softmax: P(c) = exp((logit_c - maxLogit) / T) / sum
  const expValues = new Float32Array(6);
  let expSum = 0;

  for (let c = 0; c < 6; c += 1) {
    const val = Math.exp((logits[c] - maxLogit) / temperature);
    expValues[c] = val;
    expSum += val;
  }

  const candidates: PiecePredictionCandidate[] = [];
  for (let c = 0; c < 6; c += 1) {
    candidates.push({
      type: classes[c],
      score: logits[c],
      probability: expSum > 0 ? expValues[c] / expSum : 1 / 6,
    });
  }

  // Sort candidates descending by probability
  candidates.sort((a, b) => b.probability - a.probability);

  const top1 = candidates[0];
  const top2 = candidates[1];

  const confidence = top1 ? top1.probability : 0;
  const margin = top1 && top2 ? top1.probability - top2.probability : confidence;

  // 4. Uncertainty & Invariant Evaluation
  let isUncertain = false;
  let reason: string | undefined;

  if (color === 'uncertain') {
    isUncertain = true;
    reason = 'Piece color estimate is uncertain.';
  } else if (confidence < confidenceThreshold) {
    isUncertain = true;
    reason = `Top prediction confidence ${(confidence * 100).toFixed(1)}% is below threshold ${(confidenceThreshold * 100).toFixed(1)}%.`;
  } else if (margin < marginThreshold) {
    isUncertain = true;
    reason = `Prediction margin ${(margin * 100).toFixed(1)}% between ${top1.type} and ${top2.type} is ambiguous (< ${(marginThreshold * 100).toFixed(1)}%).`;
  }

  // 5. Invariant Conflict Check (Zero-Silent-Error Policy)
  if (options.squareName && top1) {
    const invariantCheck = checkPieceInvariantConflict(options.squareName, top1.type, color);
    if (invariantCheck.hasConflict) {
      isUncertain = true;
      reason = invariantCheck.reason ?? 'Square location violates structural chess rules.';
    }
  }

  return {
    predictedType: top1 ? top1.type : null,
    confidence,
    margin,
    isUncertain,
    candidates,
    features,
    color,
    reason,
  };
}
