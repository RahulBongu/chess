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
});
