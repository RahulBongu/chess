/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  countPiecesInPlacement,
  getExpectedSplitForSet,
  validateCompleteFen,
  validateFenPlacement,
  validateManifestCollection,
  validateManifestRecord,
  validatePlacementOnly,
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
    piece_placement: null,
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
    piece_placement: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR',
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
    board_fen: '8/8/8/8/8/8/8/8 w - - 0 1',
    piece_placement: '8/8/8/8/8/8/8/8',
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

describe('offline capture manifest validator [Milestone 5.2D]', () => {
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

  it('4. distinguishes placement-only strings from complete six-field FENs', () => {
    const validPlacement = 'r1bq1rk1/ppp1bppp/2np1n2/1B2p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1';
    const completeFen = 'r1bq1rk1/ppp1bppp/2np1n2/1B2p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8';

    // Placement validation
    assert.equal(validatePlacementOnly(validPlacement).valid, true);
    assert.equal(validatePlacementOnly(completeFen).valid, false); // Contains spaces!

    // Complete FEN validation
    assert.equal(validateCompleteFen(completeFen).valid, true);
    assert.equal(validateCompleteFen(validPlacement).valid, false); // Has only 1 token!

    // Incomplete tokens
    assert.equal(validateCompleteFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq').valid, false);
    // Invalid active color
    assert.equal(validateCompleteFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1').valid, false);
    // Invalid castling
    assert.equal(validateCompleteFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KKqq - 0 1').valid, false);
    // Invalid fullmove
    assert.equal(validateCompleteFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0').valid, false);
  });

  it('5. independently verifies piece counts across all four Plan A midgame positions [Gate 1]', () => {
    // Position 1: Opening Transition (Ruy Lopez / Berlin Defense)
    const p1 = 'r1bq1rk1/ppp1bppp/2np1n2/1B2p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1';
    const c1 = countPiecesInPlacement(p1);
    assert.equal(c1.totalWhite, 16);
    assert.equal(c1.totalBlack, 16);
    assert.equal(c1.totalPieces, 32); // NOT 30!
    assert.equal(c1.emptySquares, 32);
    assert.equal(c1.white['K'], 1);
    assert.equal(c1.black['k'], 1);
    assert.equal(c1.white['P'], 8);
    assert.equal(c1.black['p'], 8);

    // Position 2: Tactical Complex Middlegame
    const p2 = 'r3r1k1/ppqbbppp/2p2n2/8/3N4/2PB4/PP1Q1PPP/R3R1K1';
    const c2 = countPiecesInPlacement(p2);
    assert.equal(c2.totalWhite, 12);
    assert.equal(c2.totalBlack, 13);
    assert.equal(c2.totalPieces, 25); // NOT 24!
    assert.equal(c2.emptySquares, 39);
    assert.equal(c2.white['K'], 1);
    assert.equal(c2.black['k'], 1);
    assert.equal(c2.white['P'], 6);
    assert.equal(c2.black['p'], 6);
    assert.equal(c2.white['Q'], 1);
    assert.equal(c2.black['q'], 1);

    // Position 3: Asymmetric Pawn Structure & Opposite Castling
    const p3 = '3r1rk1/pb1n1p1p/1p4p1/2p5/2P2PP1/P1N1B2P/1P6/2KR3R';
    const c3 = countPiecesInPlacement(p3);
    assert.equal(c3.totalWhite, 11);
    assert.equal(c3.totalBlack, 11);
    assert.equal(c3.totalPieces, 22); // NOT 20!
    assert.equal(c3.emptySquares, 42);
    assert.equal(c3.white['K'], 1);
    assert.equal(c3.black['k'], 1);
    assert.equal(c3.white['P'], 6);
    assert.equal(c3.black['p'], 6);
    assert.equal(c3.white['Q'] || 0, 0); // Queens traded
    assert.equal(c3.black['q'] || 0, 0);

    // Position 4: Simplified Technical Endgame
    const p4 = '4r3/3R1p2/5kp1/p6p/P6P/5KP1/5P2/8';
    const c4 = countPiecesInPlacement(p4);
    assert.equal(c4.totalWhite, 6);
    assert.equal(c4.totalBlack, 6);
    assert.equal(c4.totalPieces, 12); // NOT 10!
    assert.equal(c4.emptySquares, 52);
    assert.equal(c4.white['K'], 1);
    assert.equal(c4.black['k'], 1);
    assert.equal(c4.white['R'], 1);
    assert.equal(c4.black['r'], 1);
    assert.equal(c4.white['P'], 4);
    assert.equal(c4.black['p'], 4);
  });

  it('6. verifies complete six-field FEN ground truth for all four positions [Gate 2]', () => {
    const fens = [
      'r1bq1rk1/ppp1bppp/2np1n2/1B2p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7',
      'r3r1k1/ppqbbppp/2p2n2/8/3N4/2PB4/PP1Q1PPP/R3R1K1 w - - 4 16',
      '3r1rk1/pb1n1p1p/1p4p1/2p5/2P2PP1/P1N1B2P/1P6/2KR3R w - - 0 20',
      '4r3/3R1p2/5kp1/p6p/P6P/5KP1/5P2/8 w - - 2 34',
    ];

    for (const fen of fens) {
      const res = validateCompleteFen(fen);
      assert.equal(res.valid, true, `FEN failed validation: ${fen} - ${res.error}`);
    }
  });

  it('7. detects missing required fields and malformed IDs', () => {
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

  it('8. detects piece type, color, and instance ID code mismatches', () => {
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

  it('9. detects missing FEN or placement on full-board context crops', () => {
    const contextNoFen = createValidContextRecord({
      board_fen: null,
      piece_placement: null,
    });
    const errors = validateManifestRecord(contextNoFen);
    assert.ok(errors.some((e) => e.includes("Missing required 'board_fen' or 'piece_placement'")));
  });

  it('10. detects non-compliant licensing and missing property release', () => {
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

  it('11. detects duplicate derived crop paths across a collection', () => {
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

  it('12. detects cross-split set leakage across a collection', () => {
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

  it('13. detects cross-split piece instance leakage across a collection', () => {
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

  it('14. validates a multi-category pilot collection spanning all required capture categories [Gate 4]', () => {
    const pilotRecords: CaptureManifestRecord[] = [
      // Category 1: Full-board starting position (Zenith, Diffuse)
      createValidContextRecord({
        scene_id: 'SET01_SCENE_01',
        camera_angle: 'zenith',
        lighting_setup: 'diffuse_daylight',
        derived_crop_path: 'crops/train/SET01/cat1_c01_e1.png',
      }),
      // Category 2: Full-board starting position oblique (White perspective, Diffuse)
      createValidContextRecord({
        scene_id: 'SET01_SCENE_04',
        camera_angle: 'white_perspective',
        lighting_setup: 'diffuse_daylight',
        derived_crop_path: 'crops/train/SET01/cat2_c04_e1.png',
      }),
      // Category 3: Full-board Position 1 (Opening Transition, 32 pieces)
      createValidContextRecord({
        scene_id: 'SET01_SCENE_13',
        board_fen: 'r1bq1rk1/ppp1bppp/2np1n2/1B2p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7',
        piece_placement: 'r1bq1rk1/ppp1bppp/2np1n2/1B2p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1',
        derived_crop_path: 'crops/train/SET01/cat3_pos1_e1.png',
      }),
      // Category 4: Full-board Position 2 (Tactical Middlegame, 25 pieces)
      createValidContextRecord({
        scene_id: 'SET01_SCENE_17',
        board_fen: 'r3r1k1/ppqbbppp/2p2n2/8/3N4/2PB4/PP1Q1PPP/R3R1K1 w - - 4 16',
        piece_placement: 'r3r1k1/ppqbbppp/2p2n2/8/3N4/2PB4/PP1Q1PPP/R3R1K1',
        derived_crop_path: 'crops/train/SET01/cat4_pos2_d2.png',
        piece_instance_id: 'SET01_W_Q01',
        piece_type: 'queen',
        square_coord: 'd2',
      }),
      // Category 5: Full-board Position 3 (Asymmetric Castling, 22 pieces)
      createValidContextRecord({
        scene_id: 'SET01_SCENE_21',
        board_fen: '3r1rk1/pb1n1p1p/1p4p1/2p5/2P2PP1/P1N1B2P/1P6/2KR3R w - - 0 20',
        piece_placement: '3r1rk1/pb1n1p1p/1p4p1/2p5/2P2PP1/P1N1B2P/1P6/2KR3R',
        derived_crop_path: 'crops/train/SET01/cat5_pos3_c1.png',
        piece_instance_id: 'SET01_W_K01',
        piece_type: 'king',
        square_coord: 'c1',
      }),
      // Category 6: Full-board Position 4 (Simplified Endgame, 12 pieces)
      createValidContextRecord({
        scene_id: 'SET01_SCENE_25',
        board_fen: '4r3/3R1p2/5kp1/p6p/P6P/5KP1/5P2/8 w - - 2 34',
        piece_placement: '4r3/3R1p2/5kp1/p6p/P6P/5KP1/5P2/8',
        derived_crop_path: 'crops/train/SET01/cat6_pos4_d7.png',
        piece_instance_id: 'SET01_W_R01',
        piece_type: 'rook',
        square_coord: 'd7',
      }),
      // Category 7: Sub-phase A1 Piece Registration (Queen on d4)
      createValidIsolatedRecord({
        scene_id: 'SET01_ISO_REF_W_Q01',
        piece_instance_id: 'SET01_W_Q01',
        piece_type: 'queen',
        square_coord: 'd4',
        derived_crop_path: 'crops/train/SET01/cat7_ref_w_q01.png',
      }),
      // Category 8: Sub-phase A1 Piece Registration (King on e4)
      createValidIsolatedRecord({
        scene_id: 'SET01_ISO_REF_W_K01',
        piece_instance_id: 'SET01_W_K01',
        piece_type: 'king',
        square_coord: 'e4',
        derived_crop_path: 'crops/train/SET01/cat8_ref_w_k01.png',
      }),
      // Category 9: Sub-phase A2 Sweep Instance (Pawn on e4)
      createValidIsolatedRecord({
        scene_id: 'SET01_SWEEP_W_P01_E4',
        piece_instance_id: 'SET01_W_P01',
        piece_type: 'pawn',
        square_coord: 'e4',
        derived_crop_path: 'crops/train/SET01/cat9_sweep_w_p01_e4.png',
      }),
      // Category 10: Sub-phase A3 Empty Square Reference (Empty on d4)
      createValidEmptyRecord({
        scene_id: 'SET01_EMPTY_D4',
        derived_crop_path: 'crops/train/SET01/cat10_empty_d4.png',
      }),
    ];

    const result = validateManifestCollection(pilotRecords);
    assert.equal(result.valid, true, `Pilot manifest validation failed: ${result.errors.join('; ')}`);
    assert.equal(result.errors.length, 0);
    assert.equal(result.summary.totalRecords, 10);
    assert.equal(result.summary.distinctSets.length, 1);
    assert.equal(result.summary.distinctSets[0], 'SET01');
    assert.equal(result.summary.cropsByType.full_board_context, 6);
    assert.equal(result.summary.cropsByType.isolated_piece, 3);
    assert.equal(result.summary.cropsByType.empty_square, 1);
  });

  it('15. verifies that move sequence 1.e4 e5 2.Nf3 Nc6 3.Bb5 Nf6 4.d3 d6 5.Nc3 Be7 6.O-O O-O yields exact FEN w - - 4 7 [Gate 1]', async () => {
    const { Chess } = await import('chess.js');
    const chess = new Chess();
    const moves = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'Nf6', 'd3', 'd6', 'Nc3', 'Be7', 'O-O', 'O-O'];
    for (const m of moves) {
      chess.move(m);
    }
    const fen = chess.fen();
    assert.equal(fen, 'r1bq1rk1/ppp1bppp/2np1n2/1B2p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7');
    assert.equal(chess.turn(), 'w');
    assert.equal(chess.inCheck(), false);
  });

  it('16. reconciles full dataset crop accounting across all categories and splits [Gate 2 & 3]', () => {
    // 20 physical sets
    const numSets = 20;

    // Full-Board Plan A Context Scenes
    const fullBoardScenesPerSet = 28;
    const totalFullBoardScenes = numSets * fullBoardScenesPerSet; // 560
    const contextCrops = totalFullBoardScenes * 64; // 35,840

    // Isolated Sweeps
    const sweepPhotosPerSet = 12 * 64; // 768
    const totalSweepPhotos = numSets * sweepPhotosPerSet; // 15,360
    const sweepCrops = totalSweepPhotos; // 15,360

    // Physical Piece Reference Photos (Object Registration)
    const pieceRefPhotosPerSet = 32;
    const totalPieceRefPhotos = numSets * pieceRefPhotosPerSet; // 640
    const pieceRefCrops = totalPieceRefPhotos; // 640

    // Empty-Board Reference Photos
    const emptyBoardPhotosPerSet = 3;
    const totalEmptyBoardPhotos = numSets * emptyBoardPhotosPerSet; // 60
    const canonicalEmptyCrops = numSets * 64; // 1,280
    const totalMultiLightingEmptyCrops = numSets * 3 * 64; // 3,840
    const reservedStressEmptyCrops = totalMultiLightingEmptyCrops - canonicalEmptyCrops; // 2,560

    // Source Photos
    const totalSourcePhotos = totalFullBoardScenes + totalSweepPhotos + totalPieceRefPhotos + totalEmptyBoardPhotos;
    assert.equal(totalSourcePhotos, 16620); // 560 + 15,360 + 640 + 60 = 16,620

    // Initial Canonical Training Pool (Context + Sweep + Canonical Empty)
    const canonicalDatasetPool = contextCrops + sweepCrops + canonicalEmptyCrops;
    assert.equal(canonicalDatasetPool, 52480); // 35,840 + 15,360 + 1,280 = 52,480

    // Split Distribution of Canonical Pool (70% Train, 15% Val, 15% Holdout)
    const trainCrops = 14 * (28 * 64 + 768 + 64);
    const valCrops = 3 * (28 * 64 + 768 + 64);
    const holdoutCrops = 3 * (28 * 64 + 768 + 64);

    assert.equal(trainCrops, 36736); // 14 sets × 2,624 crops/set
    assert.equal(valCrops, 7872); // 3 sets × 2,624 crops/set
    assert.equal(holdoutCrops, 7872); // 3 sets × 2,624 crops/set
    assert.equal(trainCrops + valCrops + holdoutCrops, 52480);

    // Reserved Non-Initial-Training Crops (QC and Stress Test)
    assert.equal(pieceRefCrops, 640);
    assert.equal(reservedStressEmptyCrops, 2560);
    assert.equal(canonicalDatasetPool + pieceRefCrops + reservedStressEmptyCrops, 55680);
  });
});
