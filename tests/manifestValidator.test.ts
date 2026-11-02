/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  getExpectedSplitForSet,
  validateFenPlacement,
  validateManifestCollection,
  validateManifestRecord,
} from '../src/vision/manifestValidator.ts';
import type { CaptureManifestRecord } from '../src/vision/manifestValidator.ts';

function createValidIsolatedRecord(overrides: Partial<CaptureManifestRecord> = {}): CaptureManifestRecord {
  return {
    dataset_version: '1.0.0',
    source_id: 'author_capture_2026',
    set_id: 'SET01',
    split: 'train',
    piece_instance_id: 'SET01_W_P01',
    piece_type: 'pawn',
    piece_color: 'white',
    scene_id: 'SET01_SCENE_01',
    board_fen: null,
    square_coord: 'e4',
    camera_angle: 'zenith',
    lighting_setup: 'diffuse_daylight',
    original_image_path: 'raw/SET01/SET01_SCENE_01_zenith.jpg',
    derived_crop_path: 'crops/train/SET01/SET01_SCENE_01_e4_W_P01.png',
    crop_type: 'isolated_piece',
    occlusion_level: 'none',
    photographer: 'Rahul Bongu',
    rights_holder: 'Rahul Bongu',
    release_record: 'author_property_release_20261102.pdf',
    data_license: 'CC0-1.0',
    code_license: 'MIT',
    permission_status: 'verified_author_consent',
    annotation_status: 'verified',
    quality_notes: 'isolated piece sweep',
    ...overrides,
  };
}

function createValidContextRecord(overrides: Partial<CaptureManifestRecord> = {}): CaptureManifestRecord {
  return {
    dataset_version: '1.0.0',
    source_id: 'author_capture_2026',
    set_id: 'SET01',
    split: 'train',
    piece_instance_id: 'SET01_W_K01',
    piece_type: 'king',
    piece_color: 'white',
    scene_id: 'SET01_SCENE_START_01',
    board_fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    square_coord: 'e1',
    camera_angle: 'white_perspective',
    lighting_setup: 'diffuse_daylight',
    original_image_path: 'raw/SET01/SET01_SCENE_START_01_white_angle.jpg',
    derived_crop_path: 'crops/train/SET01/SET01_SCENE_START_01_e1_W_K01.png',
    crop_type: 'full_board_context',
    occlusion_level: 'none',
    photographer: 'Rahul Bongu',
    rights_holder: 'Rahul Bongu',
    release_record: 'author_property_release_20261102.pdf',
    data_license: 'CC0-1.0',
    code_license: 'MIT',
    permission_status: 'verified_author_consent',
    annotation_status: 'verified',
    quality_notes: 'starting position king',
    ...overrides,
  };
}

function createValidEmptyRecord(overrides: Partial<CaptureManifestRecord> = {}): CaptureManifestRecord {
  return {
    dataset_version: '1.0.0',
    source_id: 'author_capture_2026',
    set_id: 'SET01',
    split: 'train',
    piece_instance_id: 'EMPTY',
    piece_type: 'empty',
    piece_color: 'none',
    scene_id: 'SET01_SCENE_EMPTY_01',
    board_fen: '8/8/8/8/8/8/8/8',
    square_coord: 'd4',
    camera_angle: 'zenith',
    lighting_setup: 'diffuse_daylight',
    original_image_path: 'raw/SET01/SET01_SCENE_EMPTY_01_zenith.jpg',
    derived_crop_path: 'crops/train/SET01/SET01_SCENE_EMPTY_01_d4_EMPTY.png',
    crop_type: 'empty_square',
    occlusion_level: 'none',
    photographer: 'Rahul Bongu',
    rights_holder: 'Rahul Bongu',
    release_record: 'author_property_release_20261102.pdf',
    data_license: 'CC0-1.0',
    code_license: 'MIT',
    permission_status: 'verified_author_consent',
    annotation_status: 'verified',
    quality_notes: 'empty board reference crop',
    ...overrides,
  };
}

describe('offline capture manifest validator [Milestone 5.2C]', () => {
  it('1. accepts valid isolated piece, empty square, and full-board context records', () => {
    const isolated = createValidIsolatedRecord();
    const context = createValidContextRecord();
    const empty = createValidEmptyRecord();

    assert.deepEqual(validateManifestRecord(isolated), []);
    assert.deepEqual(validateManifestRecord(context), []);
    assert.deepEqual(validateManifestRecord(empty), []);
  });

  it('2. correctly maps sets to expected splits per 70/15/15 protocol', () => {
    assert.equal(getExpectedSplitForSet('SET01'), 'train');
    assert.equal(getExpectedSplitForSet('SET14'), 'train');
    assert.equal(getExpectedSplitForSet('SET15'), 'val');
    assert.equal(getExpectedSplitForSet('SET17'), 'val');
    assert.equal(getExpectedSplitForSet('SET18'), 'holdout');
    assert.equal(getExpectedSplitForSet('SET20'), 'holdout');
    assert.equal(getExpectedSplitForSet('SET25'), null);
    assert.equal(getExpectedSplitForSet('INVALID'), null);
  });

  it('3. validates FEN placement syntax strictly', () => {
    assert.equal(validateFenPlacement('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'), true);
    assert.equal(validateFenPlacement('8/8/8/8/8/8/8/8'), true);
    assert.equal(validateFenPlacement('4k3/8/8/8/8/8/8/4K3'), true);

    // Invalid FENs
    assert.equal(validateFenPlacement('8/8/8/8/8/8/8'), false); // Only 7 ranks
    assert.equal(validateFenPlacement('9/8/8/8/8/8/8/8'), false); // Rank adds to 9
    assert.equal(validateFenPlacement('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX'), false); // Invalid char X
    assert.equal(validateFenPlacement(''), false);
  });

  it('4. detects missing required fields and malformed IDs', () => {
    const record = createValidIsolatedRecord({
      dataset_version: '',
      set_id: 'SET_ONE',
      piece_instance_id: 'W_PAWN_1',
      square_coord: 'z9',
    });
    const errors = validateManifestRecord(record);

    assert.ok(errors.some((e) => e.includes("Missing required 'dataset_version'")));
    assert.ok(errors.some((e) => e.includes("Invalid 'set_id'")));
    assert.ok(errors.some((e) => e.includes("Invalid 'piece_instance_id'")));
    assert.ok(errors.some((e) => e.includes("Invalid or missing 'square_coord'")));
  });

  it('5. detects piece type, color, and instance ID code mismatches', () => {
    // Type is knight, but instance says P (pawn)
    const typeMismatch = createValidIsolatedRecord({
      piece_type: 'knight',
      piece_instance_id: 'SET01_W_P01',
    });
    const errors1 = validateManifestRecord(typeMismatch);
    assert.ok(errors1.some((e) => e.includes("does not match piece_type 'knight'")));

    // Color is black, but instance says W (white)
    const colorMismatch = createValidIsolatedRecord({
      piece_color: 'black',
      piece_instance_id: 'SET01_W_P01',
    });
    const errors2 = validateManifestRecord(colorMismatch);
    assert.ok(errors2.some((e) => e.includes("does not match piece_color 'black'")));

    // Empty piece with non-none color
    const emptyInconsistent = createValidIsolatedRecord({
      piece_type: 'empty',
      piece_color: 'white',
      piece_instance_id: 'EMPTY',
    });
    const errors3 = validateManifestRecord(emptyInconsistent);
    assert.ok(errors3.some((e) => e.includes("piece_color must be 'none' when piece_type is 'empty'")));
  });

  it('6. detects missing FEN on full-board context crops', () => {
    const contextNoFen = createValidContextRecord({
      board_fen: null,
    });
    const errors = validateManifestRecord(contextNoFen);
    assert.ok(errors.some((e) => e.includes("Missing or invalid 'board_fen'")));
  });

  it('7. detects non-compliant licensing and missing property release', () => {
    const nonCompliant = createValidIsolatedRecord({
      data_license: 'CC-BY-NC-4.0',
      code_license: 'GPL-3.0',
      release_record: '',
    });
    const errors = validateManifestRecord(nonCompliant);

    assert.ok(errors.some((e) => e.includes("data must be 'CC0-1.0'")));
    assert.ok(errors.some((e) => e.includes("code must be 'MIT'")));
    assert.ok(errors.some((e) => e.includes("Missing required 'release_record'")));
  });

  it('8. detects split mismatch against 70/15/15 protocol in single records', () => {
    // SET01 belongs to train, but is marked holdout
    const splitMismatch = createValidIsolatedRecord({
      set_id: 'SET01',
      split: 'holdout',
    });
    const errors = validateManifestRecord(splitMismatch);
    assert.ok(errors.some((e) => e.includes("Split mismatch for set 'SET01'")));
  });

  it('9. detects duplicate derived crop paths across a collection', () => {
    const rec1 = createValidIsolatedRecord({
      derived_crop_path: 'crops/train/SET01/crop_01.png',
    });
    const rec2 = createValidIsolatedRecord({
      square_coord: 'f3',
      derived_crop_path: 'crops/train/SET01/crop_01.png', // duplicate!
    });

    const result = validateManifestCollection([rec1, rec2]);
    assert.equal(result.valid, false);
    assert.ok(result.errors.some((e) => e.includes('[Duplicate Path]')));
  });

  it('10. detects cross-split set leakage across a collection', () => {
    const rec1 = createValidIsolatedRecord({
      set_id: 'SET01',
      split: 'train',
      derived_crop_path: 'crops/train/SET01/crop_01.png',
    });
    const rec2 = createValidIsolatedRecord({
      set_id: 'SET01',
      split: 'val', // leaked into val!
      derived_crop_path: 'crops/val/SET01/crop_02.png',
    });

    const result = validateManifestCollection([rec1, rec2]);
    assert.equal(result.valid, false);
    assert.ok(result.errors.some((e) => e.includes('[Set Leakage Violation]')));
  });

  it('11. detects cross-split piece instance leakage across a collection', () => {
    const rec1 = createValidIsolatedRecord({
      set_id: 'SET01',
      split: 'train',
      piece_instance_id: 'SET01_W_P01',
      derived_crop_path: 'crops/train/SET01/crop_01.png',
    });
    const rec2 = createValidIsolatedRecord({
      set_id: 'SET15', // SET15 is in val
      split: 'val',
      piece_instance_id: 'SET01_W_P01', // SET01 pawn erroneously tagged in SET15 val!
      derived_crop_path: 'crops/val/SET15/crop_02.png',
    });

    const result = validateManifestCollection([rec1, rec2]);
    assert.equal(result.valid, false);
    assert.ok(result.errors.some((e) => e.includes('[Piece Leakage Violation]')));
  });

  it('12. validates a multi-set collection with clean summary metrics', () => {
    const recTrain1 = createValidIsolatedRecord({
      set_id: 'SET01',
      split: 'train',
      piece_instance_id: 'SET01_W_P01',
      derived_crop_path: 'crops/train/SET01/p01.png',
    });
    const recTrain2 = createValidEmptyRecord({
      set_id: 'SET01',
      split: 'train',
      derived_crop_path: 'crops/train/SET01/empty_d4.png',
    });
    const recVal = createValidContextRecord({
      set_id: 'SET15',
      split: 'val',
      piece_instance_id: 'SET15_W_K01',
      derived_crop_path: 'crops/val/SET15/k01.png',
      original_image_path: 'raw/SET15/scene_start.jpg',
    });
    const recHoldout = createValidIsolatedRecord({
      set_id: 'SET18',
      split: 'holdout',
      piece_instance_id: 'SET18_W_P01',
      derived_crop_path: 'crops/holdout/SET18/p01.png',
      original_image_path: 'raw/SET18/scene_iso.jpg',
    });

    const result = validateManifestCollection([recTrain1, recTrain2, recVal, recHoldout]);
    assert.equal(result.valid, true);
    assert.equal(result.errors.length, 0);

    assert.equal(result.summary.totalRecords, 4);
    assert.deepEqual(result.summary.distinctSets, ['SET01', 'SET15', 'SET18']);
    assert.deepEqual(result.summary.distinctPieceInstances, ['SET01_W_P01', 'SET15_W_K01', 'SET18_W_P01']);
    assert.equal(result.summary.cropsByType.isolated_piece, 2);
    assert.equal(result.summary.cropsByType.empty_square, 1);
    assert.equal(result.summary.cropsByType.full_board_context, 1);
    assert.equal(result.summary.cropsBySplit.train, 2);
    assert.equal(result.summary.cropsBySplit.val, 1);
    assert.equal(result.summary.cropsBySplit.holdout, 1);
  });
});
