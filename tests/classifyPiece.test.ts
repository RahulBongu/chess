/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  checkPieceInvariantConflict,
  classifyPieceFromPatch,
  DEFAULT_PIECE_CONFIDENCE_THRESHOLD,
  DEFAULT_PIECE_MARGIN_THRESHOLD,
  extractPieceHogFeatures,
  HOG_FEATURE_DIMENSION,
  validatePieceClassifierModel,
} from '../src/vision/classifyPiece.ts';
import type { AnyImageData } from '../src/vision/classifySquare.ts';
import type { PieceClassifierModel, PieceType } from '../src/vision/types.ts';
import {
  defaultPieceClassifierModel,
  getDefaultPieceClassifierModel,
  PIECE_CLASSES,
  PIECE_MODEL_VERSION,
} from '../src/vision/pieceModelWeights.ts';
import {
  experimentalPieceClassifierModel,
  getExperimentalPieceClassifierModel,
} from '../src/vision/experimentalPieceModelWeights.ts';

/**
 * Creates a synthetic uniform or patterned grayscale image buffer for testing.
 */
function createSyntheticPatch(width: number, height: number, fillVal: number = 200): AnyImageData {
  const data = new Uint8Array(width * height);
  data.fill(fillVal);
  return { width, height, data };
}

/**
 * Creates a synthetic mock model with controlled weights and biases.
 */
function createMockModel(overrides?: Partial<PieceClassifierModel>): PieceClassifierModel {
  const classes: PieceType[] = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'];
  const weights: number[][] = [];
  const biases: number[] = [0, 0, 0, 0, 0, 0];

  for (let c = 0; c < 6; c += 1) {
    const row = new Array(HOG_FEATURE_DIMENSION).fill(0);
    // Give each class a distinct positive sensitivity to specific feature indices
    row[c * 40] = 2.5;
    row[c * 40 + 1] = 1.5;
    weights.push(row);
  }

  return {
    name: 'test-mock-piece-classifier',
    version: '1.0.0-test',
    featureDimension: HOG_FEATURE_DIMENSION,
    classes,
    weights,
    biases,
    temperature: 1.0,
    metadata: {
      trainedOn: 'synthetic-unit-test-fixtures',
      license: 'MIT',
    },
    ...overrides,
  };
}

describe('offline chess piece HOG feature extraction and classification', () => {
  it('1. HOG feature-vector dimensions and mathematical repeatability', () => {
    const patch = createSyntheticPatch(64, 64, 180);

    // Draw a dark vertical bar in the center to create gradients
    for (let y = 10; y < 54; y += 1) {
      for (let x = 28; x < 36; x += 1) {
        patch.data[y * 64 + x] = 30;
      }
    }

    const feat1 = extractPieceHogFeatures(patch);
    assert.equal(feat1.length, HOG_FEATURE_DIMENSION, 'Must produce exactly 288 HOG features');

    // Run again and verify exact bit-for-bit repeatability
    const feat2 = extractPieceHogFeatures(patch);
    assert.deepEqual(feat1, feat2, 'Feature extraction must be strictly deterministic');

    // Verify scratch buffer reuse
    const scratch = new Float32Array(HOG_FEATURE_DIMENSION);
    const feat3 = extractPieceHogFeatures(patch, { scratchBuffer: scratch });
    assert.equal(feat3, scratch, 'Must return and write into the supplied scratch buffer');
    assert.deepEqual(feat3, feat1);
  });

  it('2. Rejects malformed images with descriptive errors', () => {
    // Null/undefined image
    assert.throws(() => extractPieceHogFeatures(null as any), /Image data is null or undefined/);
    assert.throws(() => extractPieceHogFeatures(undefined as any), /Image data is null or undefined/);

    // Non-positive dimensions
    assert.throws(() => extractPieceHogFeatures({ width: 0, height: 64, data: new Uint8Array(64) }), /Invalid image width/);
    assert.throws(() => extractPieceHogFeatures({ width: 64, height: -5, data: new Uint8Array(64) }), /Invalid image height/);

    // Empty byte buffer
    assert.throws(() => extractPieceHogFeatures({ width: 64, height: 64, data: new Uint8Array(0) }), /Image byte buffer is empty/);

    // Invalid scratch buffer size
    const badScratch = new Float32Array(100);
    assert.throws(
      () => extractPieceHogFeatures(createSyntheticPatch(64, 64), { scratchBuffer: badScratch }),
      /Scratch buffer length 100 does not match expected HOG feature dimension 288/,
    );
  });

  it('3. Rejects invalid model dimensions safely', () => {
    const patch = createSyntheticPatch(64, 64);

    // Invalid class count
    const badClassesModel = createMockModel({ classes: ['pawn', 'rook'] as any });
    assert.throws(() => classifyPieceFromPatch(patch, { model: badClassesModel }), /Invalid model class count: expected 6/);

    // Invalid feature dimension
    const badDimModel = createMockModel({ featureDimension: 128 });
    assert.throws(() => classifyPieceFromPatch(patch, { model: badDimModel }), /Invalid model feature dimension: expected 288/);

    // Invalid weight matrix row count
    const badRowsModel = createMockModel({ weights: [[1, 2], [3, 4]] });
    assert.throws(() => classifyPieceFromPatch(patch, { model: badRowsModel }), /Invalid model weight matrix: expected 6 class rows/);

    // Invalid weight column count
    const badColsModel = createMockModel();
    badColsModel.weights[0] = new Array(200).fill(0); // 200 instead of 288
    assert.throws(() => classifyPieceFromPatch(patch, { model: badColsModel }), /Model weight row for class index 0 has invalid length/);

    // Invalid bias vector length
    const badBiasesModel = createMockModel({ biases: [0, 0, 0] });
    assert.throws(() => classifyPieceFromPatch(patch, { model: badBiasesModel }), /Invalid model bias vector: expected 6 values/);
  });

  it('4. Handles model absence cleanly without crashing', () => {
    const patch = createSyntheticPatch(64, 64);
    const result = classifyPieceFromPatch(patch, { model: null, color: 'white' });

    assert.equal(result.predictedType, null);
    assert.equal(result.confidence, 0);
    assert.equal(result.margin, 0);
    assert.equal(result.isUncertain, true);
    assert.equal(result.candidates.length, 0);
    assert.equal(result.features.length, 288);
    assert.match(result.reason ?? '', /No piece classification model provided/);
  });

  it('5. Ranked candidate output, Softmax normalization, and probability ordering', () => {
    const patch = createSyntheticPatch(64, 64);
    // Draw feature on class 3 (rook) index
    for (let y = 10; y < 54; y += 1) {
      for (let x = 20; x < 44; x += 1) {
        patch.data[y * 64 + x] = 30;
      }
    }

    const model = createMockModel();
    // Bias rook strongly
    model.biases[3] = 4.0; // class index 3 = rook

    const result = classifyPieceFromPatch(patch, { model, color: 'white' });

    assert.equal(result.predictedType, 'rook');
    assert.equal(result.candidates.length, 6);

    // Verify candidates are strictly sorted descending by probability
    for (let i = 0; i < result.candidates.length - 1; i += 1) {
      assert.ok(
        result.candidates[i].probability >= result.candidates[i + 1].probability,
        'Candidates must be sorted descending by probability',
      );
    }

    // Verify Softmax probability normalization
    const probSum = result.candidates.reduce((s, c) => s + c.probability, 0);
    assert.ok(Math.abs(probSum - 1.0) < 1e-4, 'Candidate probabilities must sum to 1.0');

    // Verify confidence and margin calculations
    assert.equal(result.confidence, result.candidates[0].probability);
    assert.equal(result.margin, result.candidates[0].probability - result.candidates[1].probability);
  });

  it('6. Low-confidence and tied predictions trigger uncertainty flags', () => {
    const patch = createSyntheticPatch(64, 64, 128); // Blank patch produces near-zero gradients
    const model = createMockModel();
    // Zero all biases so all 6 classes tie with equal logit 0.0
    model.biases = [0, 0, 0, 0, 0, 0];

    const result = classifyPieceFromPatch(patch, {
      model,
      color: 'white',
      confidenceThreshold: 0.60,
      marginThreshold: 0.20,
    });

    // In a 6-way tie, each class gets P = 1/6 = 16.7%
    assert.ok(result.confidence < 0.25);
    assert.ok(result.margin < 0.05);
    assert.equal(result.isUncertain, true);
    assert.match(result.reason ?? '', /below threshold/);
  });

  it('7. Color candidate mapping (white vs black vs uncertain)', () => {
    const patch = createSyntheticPatch(64, 64);
    const model = createMockModel();

    // White piece
    const whiteRes = classifyPieceFromPatch(patch, { model, color: 'white' });
    assert.equal(whiteRes.color, 'white');

    // Black piece
    const blackRes = classifyPieceFromPatch(patch, { model, color: 'black' });
    assert.equal(blackRes.color, 'black');

    // Uncertain color triggers uncertainty in the result
    const uncertainRes = classifyPieceFromPatch(patch, { model, color: 'uncertain' });
    assert.equal(uncertainRes.color, 'uncertain');
    assert.equal(uncertainRes.isUncertain, true);
    assert.match(uncertainRes.reason ?? '', /Piece color estimate is uncertain/);
  });

  it('8. Invariant conflicts are flagged for review, NEVER silently corrected (Zero-Silent-Error policy)', () => {
    const patch = createSyntheticPatch(64, 64);
    const model = createMockModel();
    // Bias pawn strongly
    model.biases[0] = 5.0; // index 0 = pawn

    // Case A: Pawn on rank 4 (legal)
    const legalRes = classifyPieceFromPatch(patch, {
      model,
      color: 'white',
      squareName: 'e4',
      confidenceThreshold: 0.50,
      marginThreshold: 0.10,
    });
    assert.equal(legalRes.predictedType, 'pawn');
    assert.equal(legalRes.isUncertain, false);

    // Case B: Pawn on rank 1 (ILLEGAL by chess rules!)
    const illegalRank1Res = classifyPieceFromPatch(patch, {
      model,
      color: 'white',
      squareName: 'e1',
      confidenceThreshold: 0.50,
      marginThreshold: 0.10,
    });
    // The classifier must NOT silently change 'pawn' to 'rook' or 'queen'!
    assert.equal(illegalRank1Res.predictedType, 'pawn', 'Must NOT silently alter predicted piece type');
    assert.equal(illegalRank1Res.isUncertain, true, 'Rule invariant violation must flag isUncertain: true');
    assert.match(illegalRank1Res.reason ?? '', /Rule invariant violation/);
    assert.match(illegalRank1Res.reason ?? '', /pawn cannot exist on rank 1/);

    // Case C: Pawn on rank 8 (ILLEGAL by chess rules!)
    const illegalRank8Res = classifyPieceFromPatch(patch, {
      model,
      color: 'black',
      squareName: 'd8',
      confidenceThreshold: 0.50,
      marginThreshold: 0.10,
    });
    assert.equal(illegalRank8Res.predictedType, 'pawn');
    assert.equal(illegalRank8Res.isUncertain, true);
    assert.match(illegalRank8Res.reason ?? '', /pawn cannot exist on rank 8/);
  });

  it('9. Direct invariant check utility function behavior', () => {
    assert.equal(checkPieceInvariantConflict('e4', 'pawn').hasConflict, false);
    assert.equal(checkPieceInvariantConflict('e1', 'rook').hasConflict, false);
    assert.equal(checkPieceInvariantConflict('e1', 'king').hasConflict, false);
    assert.equal(checkPieceInvariantConflict('e8', 'queen').hasConflict, false);

    assert.equal(checkPieceInvariantConflict('a1', 'pawn', 'white').hasConflict, true);
    assert.equal(checkPieceInvariantConflict('h8', 'pawn', 'black').hasConflict, true);
    assert.match(
      checkPieceInvariantConflict('a1', 'pawn', 'white').reason ?? '',
      /white pawn cannot exist on rank 1/,
    );
  });

  it('10. Python/TypeScript HOG feature numerical parity on fixed synthetic patch', () => {
    // Fixed deterministic synthetic patch: 64x64 with vertical bar and diagonal slash
    const patch = createSyntheticPatch(64, 64, 200);
    for (let y = 12; y < 52; y += 1) {
      for (let x = 28; x < 36; x += 1) {
        patch.data[y * 64 + x] = 40;
      }
    }
    for (let i = 0; i < 16; i += 1) {
      patch.data[(20 + i) * 64 + (36 + i)] = 40;
    }

    const tsHog = extractPieceHogFeatures(patch);
    assert.equal(tsHog.length, 288);

    // Reference values extracted from scripts/train_piece_classifier.py extract_hog_288
    const expectedIndicesAndValues: [number, number][] = [
      [0, 0.000000],
      [8, 0.208792],
      [16, 0.000000],
      [32, 0.127084],
      [64, 0.160086],
      [128, 0.457891],
      [200, 0.954480],
      [256, 0.923760],
    ];

    for (const [idx, expectedVal] of expectedIndicesAndValues) {
      const diff = Math.abs(tsHog[idx] - expectedVal);
      assert.ok(
        diff < 1e-5,
        `Feature index ${idx} value ${tsHog[idx].toFixed(6)} diverges from Python reference ${expectedVal.toFixed(6)} (diff ${diff})`,
      );
    }
  });

  it('11. Exported defaultPieceClassifierModel structural and mathematical integrity', () => {
    const model = getDefaultPieceClassifierModel();
    assert.ok(model, 'defaultPieceClassifierModel must be defined');
    assert.equal(model, defaultPieceClassifierModel);

    // Validate structural schema
    assert.doesNotThrow(() => validatePieceClassifierModel(model));

    assert.equal(model.name, 'hog-staunton-micro-v1');
    assert.equal(model.version, PIECE_MODEL_VERSION);
    assert.equal(model.featureDimension, 288);
    assert.deepEqual(model.classes, ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']);
    assert.equal(model.weights.length, 6);
    for (let c = 0; c < 6; c += 1) {
      assert.equal(model.weights[c].length, 288);
      // Ensure weights are real trained floating point values (not zeros or dummy numbers)
      const hasNonZero = model.weights[c].some((w) => Math.abs(w) > 1e-4);
      assert.ok(hasNonZero, `Class ${c} (${model.classes[c]}) must have trained non-zero weights`);
    }
    assert.equal(model.biases.length, 6);

    // Verify metadata provenance disclosures
    assert.ok(model.metadata);
    assert.equal(model.metadata?.license, 'MIT');
    assert.equal(model.metadata?.evaluationType, 'SYNTHETIC-ONLY');
    assert.equal(model.metadata?.trainDenominator, 1080);
    assert.equal(model.metadata?.testDenominator, 360);
    assert.ok((model.metadata?.top1Accuracy ?? 0) > 0.95);
    assert.ok((model.metadata?.top2Accuracy ?? 0) >= 0.99);
  });

  it('12. End-to-end inference with exported defaultPieceClassifierModel', () => {
    const model = getDefaultPieceClassifierModel();

    // Create a pawn-like synthetic patch (small spherical head + conical stem)
    const pawnPatch = createSyntheticPatch(64, 64, 215);
    // Draw base
    for (let y = 50; y <= 58; y += 1) {
      for (let x = 20; x <= 44; x += 1) {
        pawnPatch.data[y * 64 + x] = 40;
      }
    }
    // Draw stem
    for (let y = 28; y < 50; y += 1) {
      for (let x = 28; x <= 36; x += 1) {
        pawnPatch.data[y * 64 + x] = 40;
      }
    }
    // Draw round head
    for (let y = 14; y < 28; y += 1) {
      for (let x = 25; x <= 39; x += 1) {
        if ((x - 32) ** 2 + (y - 20) ** 2 <= 7 ** 2) {
          pawnPatch.data[y * 64 + x] = 40;
        }
      }
    }

    const res = classifyPieceFromPatch(pawnPatch, {
      model,
      color: 'white',
      squareName: 'e4',
    });

    assert.equal(res.color, 'white');
    assert.equal(res.candidates.length, 6);
    assert.ok(res.predictedType !== null);

    // Probabilities must sum to 1.0
    const probSum = res.candidates.reduce((s, c) => s + c.probability, 0);
    assert.ok(Math.abs(probSum - 1.0) < 1e-4);

    // Candidates must be ordered descending
    for (let i = 0; i < 5; i += 1) {
      assert.ok(res.candidates[i].probability >= res.candidates[i + 1].probability);
    }

    // Confidence and margin must match definitions
    assert.equal(res.confidence, res.candidates[0].probability);
    assert.equal(res.margin, res.candidates[0].probability - res.candidates[1].probability);
  });

  it('13. Integrity and structure of experimentalPieceClassifierModel (Phase 4.3E candidate)', () => {
    const expModel = getExperimentalPieceClassifierModel();
    validatePieceClassifierModel(expModel);

    assert.equal(expModel.featureDimension, 288);
    assert.equal(expModel.classes.length, 6);
    assert.equal(expModel.weights.length, 6);
    assert.equal(expModel.biases.length, 6);
    for (let c = 0; c < 6; c += 1) {
      assert.equal(expModel.weights[c].length, 288);
      const hasNonZero = expModel.weights[c].some((w) => Math.abs(w) > 1e-4);
      assert.ok(hasNonZero, `Candidate Class ${c} must have non-zero weights`);
    }

    assert.ok(expModel.metadata);
    assert.equal(expModel.metadata?.license, 'CC BY 4.0');
    assert.equal(expModel.metadata?.evaluationType, 'HELD-OUT-PHYSICAL-SET-4');
    assert.equal(expModel.metadata?.trainDenominator, 4608);
    assert.equal(expModel.metadata?.testDenominator, 768);

    // Verify production model isolation: defaultPieceClassifierModel remains synthetic
    const prodModel = getDefaultPieceClassifierModel();
    assert.equal(prodModel.metadata?.evaluationType, 'SYNTHETIC-ONLY');
    assert.equal(prodModel.metadata?.license, 'MIT');
    assert.notEqual(prodModel.version, expModel.version);
  });

  it('14. Python/TypeScript parity on experimentalPieceClassifierModel inference', () => {
    // Fixed deterministic test patch matching test 10
    const patch = createSyntheticPatch(64, 64, 200);
    for (let y = 12; y < 52; y += 1) {
      for (let x = 28; x < 36; x += 1) {
        patch.data[y * 64 + x] = 40;
      }
    }
    for (let i = 0; i < 16; i += 1) {
      patch.data[(20 + i) * 64 + (36 + i)] = 40;
    }

    const expModel = getExperimentalPieceClassifierModel();
    const res = classifyPieceFromPatch(patch, {
      model: expModel,
      color: 'white',
      squareName: 'e4',
    });

    assert.equal(res.predictedType, 'knight');

    // Expected values from Python reference (evaluate_real_benchmark.py / predict_piece)
    // Candidate Logits: [0.375428, 7.039955, -2.02584, -3.722581, -3.693206, 2.026261]
    // Candidate Probs:  [0.001265, 0.991985, 0.000115, 0.000021, 0.000022, 0.006593]
    const expectedProbs: Record<string, number> = {
      pawn: 0.001265,
      knight: 0.991985,
      bishop: 0.000115,
      rook: 0.000021,
      queen: 0.000022,
      king: 0.006593,
    };

    for (const cand of res.candidates) {
      const exp = expectedProbs[cand.type];
      const diff = Math.abs(cand.probability - exp);
      assert.ok(
        diff < 1e-4,
        `Probability for ${cand.type} (${cand.probability.toFixed(6)}) diverges from Python (${exp.toFixed(6)}) by ${diff}`,
      );
    }
  });

  it('15. Python/TypeScript parity on 100x100 BGR mobile sampling and HOG feature extraction', () => {
    // Deterministic 100x100 BGR test fixture with 4 colored quadrants and central circular piece
    const width = 100;
    const height = 100;
    const data = new Uint8Array(width * height * 3);

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = (y * width + x) * 3;
        if (y < 50 && x < 50) {
          // Top-left: Pure Red (BGR [0, 0, 255])
          data[idx] = 0; data[idx + 1] = 0; data[idx + 2] = 255;
        } else if (y < 50 && x >= 50) {
          // Top-right: Pure Green (BGR [0, 255, 0])
          data[idx] = 0; data[idx + 1] = 255; data[idx + 2] = 0;
        } else if (y >= 50 && x < 50) {
          // Bottom-left: Pure Blue (BGR [255, 0, 0])
          data[idx] = 255; data[idx + 1] = 0; data[idx + 2] = 0;
        } else {
          // Bottom-right: Wood tone (BGR [140, 180, 210])
          data[idx] = 140; data[idx + 1] = 180; data[idx + 2] = 210;
        }
      }
    }

    const cy = 50, cx = 50, rad = 20;
    for (let y = cy - rad; y <= cy + rad; y += 1) {
      for (let x = cx - rad; x <= cx + rad; x += 1) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2) {
          const idx = (y * width + x) * 3;
          data[idx] = 30; data[idx + 1] = 30; data[idx + 2] = 30;
        }
      }
    }

    const bgrImg = { width, height, data, channels: 3 as const, colorOrder: 'bgr' as const };
    const tsHog = extractPieceHogFeatures(bgrImg, { roi: { x: 0, y: 0, width: 100, height: 100 } });
    assert.equal(tsHog.length, 288);

    // Python reference values from verify_preprocessing_parity.py on identical 100x100 BGR fixture
    const expectedFeaturePoints: [number, number][] = [
      [0, 0.000000],
      [8, 0.592673],
      [32, 0.288792],
      [64, 0.320126],
      [128, 0.120523],
      [200, 0.130177],
      [256, 0.365449],
    ];

    for (const [idx, expectedVal] of expectedFeaturePoints) {
      const diff = Math.abs(tsHog[idx] - expectedVal);
      assert.ok(
        diff < 1e-4,
        `Feature index ${idx} value ${tsHog[idx].toFixed(6)} diverges from Python reference ${expectedVal.toFixed(6)} (diff ${diff})`,
      );
    }

    // End-to-end inference parity with production model
    const prodModel = getDefaultPieceClassifierModel();
    const resProd = classifyPieceFromPatch(bgrImg, {
      model: prodModel,
      roi: { x: 0, y: 0, width: 100, height: 100 },
    });
    assert.equal(resProd.predictedType, 'knight');
    assert.ok(Math.abs(resProd.confidence - 0.977434) < 1e-4);
    assert.ok(Math.abs(resProd.margin - 0.955119) < 1e-4);

    // End-to-end inference parity with experimental candidate model
    const expModel = getExperimentalPieceClassifierModel();
    const resExp = classifyPieceFromPatch(bgrImg, {
      model: expModel,
      roi: { x: 0, y: 0, width: 100, height: 100 },
    });
    assert.equal(resExp.predictedType, 'knight');
    assert.ok(Math.abs(resExp.confidence - 0.726998) < 1e-4);
    assert.ok(Math.abs(resExp.margin - 0.516422) < 1e-4);
  });
});
