/**
 * Offline Square Crop Extraction Engine
 * Milestone 5.2G: Extracts 64 canonical 100x100 square crops from 800x800 rectified board images.
 */

import { CANONICAL_BOARD_SIZE, CANONICAL_SQUARE_SIZE, GRID_DIMENSION } from './rectify.ts';
import type { AnyImageData } from './classifySquare.ts';
import type { BoardViewOrientation } from './types.ts';
import type { PieceType, PieceColor } from './manifestValidator.ts';

export interface ExtractedSquareCrop {
  square: string; // 'a1' through 'h8'
  fileIndex: number; // 0..7 (0 = a, 7 = h)
  rankIndex: number; // 0..7 (0 = 1, 7 = 8)
  col: number; // 0..7 (x-grid column on rectified plane)
  row: number; // 0..7 (y-grid row on rectified plane)
  bounds: { x: number; y: number; width: number; height: number };
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
  pieceType: PieceType;
  pieceColor: PieceColor;
  pieceChar: string | null;
}

const CHAR_TO_PIECE_TYPE: Record<string, PieceType> = {
  p: 'pawn',
  P: 'pawn',
  n: 'knight',
  N: 'knight',
  b: 'bishop',
  B: 'bishop',
  r: 'rook',
  R: 'rook',
  q: 'queen',
  Q: 'queen',
  k: 'king',
  K: 'king',
};

/**
 * Maps an 8-rank FEN placement string to a 64-square map of pieces.
 */
export function parsePlacementToSquareMap(
  placement: string
): Record<string, { pieceChar: string; pieceType: PieceType; pieceColor: PieceColor }> {
  const map: Record<string, { pieceChar: string; pieceType: PieceType; pieceColor: PieceColor }> = {};
  if (!placement || typeof placement !== 'string') return map;

  const firstToken = placement.trim().split(/\s+/)[0];
  const ranks = firstToken.split('/');
  if (ranks.length !== 8) return map;

  for (let rIdx = 0; rIdx < 8; rIdx++) {
    const rankNum = 8 - rIdx; // Rank 8 is first
    const rankStr = ranks[rIdx];
    let fileIdx = 0;

    for (const ch of rankStr) {
      if (ch >= '1' && ch <= '8') {
        fileIdx += parseInt(ch, 10);
      } else if ('pnbrqkPNBRQK'.includes(ch)) {
        if (fileIdx < 8) {
          const fileChar = String.fromCharCode(97 + fileIdx);
          const sq = `${fileChar}${rankNum}`;
          const isWhite = 'PNBRQK'.includes(ch);
          map[sq] = {
            pieceChar: ch,
            pieceType: CHAR_TO_PIECE_TYPE[ch] || 'empty',
            pieceColor: isWhite ? 'white' : 'black',
          };
          fileIdx++;
        }
      }
    }
  }
  return map;
}

/**
 * Slices 64 100x100 square crops from an 800x800 canonical rectified board image.
 */
export function extractSquareCropsFromBoard(
  image: AnyImageData,
  placement?: string | null,
  orientation: BoardViewOrientation = 'white-bottom'
): ExtractedSquareCrop[] {
  if (image.width !== CANONICAL_BOARD_SIZE || image.height !== CANONICAL_BOARD_SIZE) {
    throw new Error(
      `Cannot extract crops from non-canonical image size ${image.width}x${image.height}. Must be ${CANONICAL_BOARD_SIZE}x${CANONICAL_BOARD_SIZE}.`
    );
  }

  const channels = 'channels' in image && image.channels ? image.channels : image.data.length / (CANONICAL_BOARD_SIZE * CANONICAL_BOARD_SIZE);
  if (channels !== 1 && channels !== 3) {
    throw new Error(`Unsupported image channel depth: ${channels}. Must be 1 (grayscale) or 3 (RGB/BGR).`);
  }

  const pieceMap = placement ? parsePlacementToSquareMap(placement) : {};
  const crops: ExtractedSquareCrop[] = [];

  for (let row = 0; row < GRID_DIMENSION; row++) {
    for (let col = 0; col < GRID_DIMENSION; col++) {
      // Algebraic square mapping depending on viewpoint orientation
      let fileIdx: number;
      let rankIdx: number;

      if (orientation === 'white-bottom') {
        fileIdx = col;
        rankIdx = 7 - row;
      } else {
        fileIdx = 7 - col;
        rankIdx = row;
      }

      const fileChar = String.fromCharCode(97 + fileIdx);
      const rankNum = rankIdx + 1;
      const square = `${fileChar}${rankNum}`;

      // Bounding box on the 800x800 plane
      const startX = col * CANONICAL_SQUARE_SIZE;
      const startY = row * CANONICAL_SQUARE_SIZE;

      // Extract 100x100 pixel buffer
      const cropSize = CANONICAL_SQUARE_SIZE;
      const cropData = new Uint8Array(cropSize * cropSize * channels);

      for (let y = 0; y < cropSize; y++) {
        const srcY = startY + y;
        const srcRowOffset = srcY * CANONICAL_BOARD_SIZE * channels;
        const dstRowOffset = y * cropSize * channels;

        const srcStart = srcRowOffset + startX * channels;
        const srcEnd = srcStart + cropSize * channels;

        cropData.set(image.data.subarray(srcStart, srcEnd), dstRowOffset);
      }

      const pieceInfo = pieceMap[square];

      crops.push({
        square,
        fileIndex: fileIdx,
        rankIndex: rankIdx,
        col,
        row,
        bounds: {
          x: startX,
          y: startY,
          width: cropSize,
          height: cropSize,
        },
        data: cropData,
        width: cropSize,
        height: cropSize,
        channels,
        pieceType: pieceInfo ? pieceInfo.pieceType : 'empty',
        pieceColor: pieceInfo ? pieceInfo.pieceColor : 'none',
        pieceChar: pieceInfo ? pieceInfo.pieceChar : null,
      });
    }
  }

  return crops;
}
