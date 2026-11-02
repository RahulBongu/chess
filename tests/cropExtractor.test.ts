/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  buildCropFilename,
  extractCropsForMode,
  extractSquareCropsFromBoard,
  parsePlacementToSquareMap,
} from '../src/vision/cropExtractor.ts';
import type { CropExtractionRequest } from '../src/vision/cropExtractor.ts';
import type { GrayscaleImage } from '../src/vision/boardEvidence.ts';
import type { RgbImage } from '../src/vision/types.ts';

const START_PLACEMENT = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
const START_FEN = `${START_PLACEMENT} w KQkq - 0 1`;

const START_PIECE_MAP: Record<string, string> = {
  a8: 'SET01_B_R01',
  b8: 'SET01_B_N01',
  c8: 'SET01_B_B01',
  d8: 'SET01_B_Q01',
  e8: 'SET01_B_K01',
  f8: 'SET01_B_B02',
  g8: 'SET01_B_N02',
  h8: 'SET01_B_R02',
  a7: 'SET01_B_P01',
  b7: 'SET01_B_P02',
  c7: 'SET01_B_P03',
  d7: 'SET01_B_P04',
  e7: 'SET01_B_P05',
  f7: 'SET01_B_P06',
  g7: 'SET01_B_P07',
  h7: 'SET01_B_P08',
  a2: 'SET01_W_P01',
  b2: 'SET01_W_P02',
  c2: 'SET01_W_P03',
  d2: 'SET01_W_P04',
  e2: 'SET01_W_P05',
  f2: 'SET01_W_P06',
  g2: 'SET01_W_P07',
  h2: 'SET01_W_P08',
  a1: 'SET01_W_R01',
  b1: 'SET01_W_N01',
  c1: 'SET01_W_B01',
  d1: 'SET01_W_Q01',
  e1: 'SET01_W_K01',
  f1: 'SET01_W_B02',
  g1: 'SET01_W_N02',
  h1: 'SET01_W_R02',
};

function makeGrayscaleImage(fill = 0): GrayscaleImage {
  return {
    width: 800,
    height: 800,
    data: new Uint8Array(800 * 800).fill(fill),
  };
}

function run(request: CropExtractionRequest) {
  return extractCropsForMode(makeGrayscaleImage(120), request);
}

function singlePiecePlacement(pieceChar: string, square: string): string {
  const file = square.charCodeAt(0) - 97;
  const rank = parseInt(square[1], 10);
  const ranks: string[] = [];
  for (let currentRank = 8; currentRank >= 1; currentRank--) {
    if (currentRank !== rank) {
      ranks.push('8');
      continue;
    }
    const before = file === 0 ? '' : String(file);
    const after = file === 7 ? '' : String(7 - file);
    ranks.push(`${before}${pieceChar}${after}`);
  }
  return ranks.join('/');
}

function pieceCharFromCanonicalId(pieceInstanceId: string): string {
  const match = pieceInstanceId.match(/^SET01_([WB])_([PNBRQK])[0-9]{2}$/);
  assert.ok(match, `Unexpected piece identity ${pieceInstanceId}`);
  const char = match![2];
  return match![1] === 'W' ? char : char.toLowerCase();
}

describe('offline square crop extraction semantics [Milestone 5.2H, synthetic fixtures]', () => {
  it('1. parses valid placement strictly and rejects malformed placement instead of treating it as empty', () => {
    const map = parsePlacementToSquareMap(START_PLACEMENT);
    assert.equal(map['a1'].pieceChar, 'R');
    assert.equal(map['e1'].pieceType, 'king');
    assert.equal(map['e8'].pieceColor, 'black');
    assert.equal(map['e4'], undefined);

    assert.throws(() => parsePlacementToSquareMap('8/8/8/8/8/8/8'), /Invalid placement/);
    assert.throws(() => parsePlacementToSquareMap('9/8/8/8/8/8/8/8'), /Invalid placement/);
    assert.throws(() => parsePlacementToSquareMap('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX'), /Invalid placement/);
  });

  it('2. emits exactly 64 full-board context crops with mapped identities and deterministic filenames', () => {
    const result = run({
      mode: 'full_board_context',
      boardFen: START_FEN,
      pieceInstanceMap: START_PIECE_MAP,
      setId: 'SET01',
    });

    assert.equal(result.sourcePhotoCount, 1);
    assert.equal(result.derivedCropCount, 64);
    assert.equal(result.crops.length, 64);
    assert.equal(result.cropType, 'full_board_context');

    const a8 = result.crops.find((crop) => crop.square === 'a8')!;
    assert.equal(a8.pieceChar, 'r');
    assert.equal(a8.pieceType, 'rook');
    assert.equal(a8.pieceColor, 'black');
    assert.equal(a8.pieceInstanceId, 'SET01_B_R01');
    assert.equal(a8.physicalTagId, 'S01-B-R01');

    const e1 = result.crops.find((crop) => crop.square === 'e1')!;
    assert.equal(e1.pieceInstanceId, 'SET01_W_K01');
    assert.equal(e1.physicalTagId, 'S01-W-K01');

    const e4 = result.crops.find((crop) => crop.square === 'e4')!;
    assert.equal(e4.pieceType, 'empty');
    assert.equal(e4.pieceColor, 'none');
    assert.equal(e4.pieceInstanceId, 'EMPTY');
    assert.equal(e4.physicalTagId, 'NONE');

    const filenames = result.crops.map((crop) =>
      buildCropFilename('SET01_SCENE_START_ZENITH', crop.square, crop.pieceInstanceId)
    );
    assert.equal(new Set(filenames).size, 64);
    assert.equal(new Set(result.crops.map((crop) => crop.pieceInstanceId)).size, 33); // 32 pieces + EMPTY
  });

  it('3. emits exactly one occupied-target crop for an isolated sweep/reference photo', () => {
    const placement = singlePiecePlacement('P', 'e4');
    const result = run({
      mode: 'occupied_target_only',
      placement,
      targetSquare: 'e4',
      targetPieceInstanceId: 'SET01_W_P01',
      setId: 'SET01',
    });

    assert.equal(result.sourcePhotoCount, 1);
    assert.equal(result.derivedCropCount, 1);
    assert.equal(result.crops.length, 1);
    assert.equal(result.cropType, 'isolated_piece');
    assert.equal(result.crops[0].square, 'e4');
    assert.equal(result.crops[0].pieceInstanceId, 'SET01_W_P01');
    assert.equal(result.crops[0].physicalTagId, 'S01-W-P01');
  });

  it('4. maps 32 reference photographs to exactly 32 target crops, one per source photograph', () => {
    const ids = [
      ...Array.from({ length: 8 }, (_, i) => `SET01_W_P${String(i + 1).padStart(2, '0')}`),
      ...Array.from({ length: 8 }, (_, i) => `SET01_B_P${String(i + 1).padStart(2, '0')}`),
      'SET01_W_N01',
      'SET01_W_N02',
      'SET01_B_N01',
      'SET01_B_N02',
      'SET01_W_B01',
      'SET01_W_B02',
      'SET01_B_B01',
      'SET01_B_B02',
      'SET01_W_R01',
      'SET01_W_R02',
      'SET01_B_R01',
      'SET01_B_R02',
      'SET01_W_Q01',
      'SET01_W_K01',
      'SET01_B_Q01',
      'SET01_B_K01',
    ];

    const filenames: string[] = [];
    let derivedTotal = 0;
    for (const id of ids) {
      const char = pieceCharFromCanonicalId(id);
      const placement = singlePiecePlacement(char, 'd4');
      const result = run({
        mode: 'occupied_target_only',
        placement,
        targetSquare: 'd4',
        targetPieceInstanceId: id,
        setId: 'SET01',
      });
      assert.equal(result.sourcePhotoCount, 1);
      assert.equal(result.derivedCropCount, 1);
      assert.equal(result.crops.length, 1);
      derivedTotal += result.derivedCropCount;
      filenames.push(buildCropFilename(`SET01_REF_${id}`, 'd4', id));
    }

    assert.equal(ids.length, 32);
    assert.equal(derivedTotal, 32);
    assert.equal(new Set(filenames).size, 32);
  });

  it('5. emits exactly 64 empty-square crops for an empty-board lighting reference', () => {
    const result = run({
      mode: 'empty_board_grid',
      setId: 'SET01',
    });

    assert.equal(result.sourcePhotoCount, 1);
    assert.equal(result.derivedCropCount, 64);
    assert.equal(result.cropType, 'empty_square');
    assert.equal(result.crops.length, 64);
    assert.ok(
      result.crops.every(
        (crop) =>
          crop.pieceType === 'empty' &&
          crop.pieceColor === 'none' &&
          crop.pieceInstanceId === 'EMPTY' &&
          crop.physicalTagId === 'NONE'
      )
    );
  });

  it('6. rejects wrong target coordinates, mismatched identities, and ambiguous occupancy', () => {
    const onePawn = singlePiecePlacement('P', 'e4');
    assert.throws(
      () =>
        run({
          mode: 'occupied_target_only',
          placement: onePawn,
          targetSquare: 'd4',
          targetPieceInstanceId: 'SET01_W_P01',
          setId: 'SET01',
        }),
      /does not match the only occupied square/
    );

    assert.throws(
      () =>
        run({
          mode: 'occupied_target_only',
          placement: onePawn,
          targetSquare: 'e4',
          targetPieceInstanceId: 'SET01_W_Q01',
          setId: 'SET01',
        }),
      /is a queen, expected pawn/
    );

    assert.throws(
      () =>
        run({
          mode: 'occupied_target_only',
          placement: onePawn,
          targetSquare: 'e4',
          targetPieceInstanceId: 'SET02_W_P01',
          setId: 'SET01',
        }),
      /belongs to SET02/
    );

    assert.throws(
      () =>
        run({
          mode: 'occupied_target_only',
          placement: START_PLACEMENT,
          targetSquare: 'e4',
          targetPieceInstanceId: 'SET01_W_P01',
          setId: 'SET01',
        }),
      /exactly one occupied square/
    );
  });

  it('7. rejects full-board extraction without a physical identity map, duplicate instances, and map/FEN mismatches', () => {
    assert.throws(
      () => run({ mode: 'full_board_context', placement: START_PLACEMENT, setId: 'SET01' }),
      /requires pieceInstanceMap/
    );

    const duplicateMap = { ...START_PIECE_MAP, a2: 'SET01_W_P01', b2: 'SET01_W_P01' };
    assert.throws(
      () =>
        run({
          mode: 'full_board_context',
          placement: START_PLACEMENT,
          pieceInstanceMap: duplicateMap,
          setId: 'SET01',
        }),
      /appears on multiple squares/
    );

    const wrongTypeMap = { ...START_PIECE_MAP, e1: 'SET01_W_Q01' };
    assert.throws(
      () =>
        run({
          mode: 'full_board_context',
          placement: START_PLACEMENT,
          pieceInstanceMap: wrongTypeMap,
          setId: 'SET01',
        }),
      /is a queen, expected king/
    );
  });

  it('8. keeps white-bottom and black-bottom coordinate associations correct', () => {
    const white = run({ mode: 'empty_board_grid', orientation: 'white-bottom' });
    assert.equal(white.crops[0].square, 'a8');
    assert.equal(white.crops[63].square, 'h1');

    const black = run({ mode: 'empty_board_grid', orientation: 'black-bottom' });
    assert.equal(black.crops[0].square, 'h1');
    assert.equal(black.crops[63].square, 'a8');
  });

  it('9. rejects non-canonical image dimensions instead of silently resizing', () => {
    const badImage: GrayscaleImage = {
      width: 600,
      height: 600,
      data: new Uint8Array(600 * 600),
    };

    assert.throws(
      () =>
        extractCropsForMode(badImage, {
          mode: 'empty_board_grid',
        }),
      /non-canonical image size/
    );
  });

  it('10. preserves 3-channel crop buffer stride at the low-level slicing boundary', () => {
    const data = new Uint8Array(800 * 800 * 3);
    const rgbImage: RgbImage = {
      width: 800,
      height: 800,
      data,
      channels: 3,
      colorOrder: 'bgr',
    };

    const crops = extractSquareCropsFromBoard(rgbImage);
    assert.equal(crops.length, 64);
    assert.equal(crops[0].channels, 3);
    assert.equal(crops[0].data.length, 30000);
  });
});
