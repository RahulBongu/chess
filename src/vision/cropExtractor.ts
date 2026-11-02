/**
 * Offline Square Crop Extraction Engine
 * Milestone 5.2H: Explicit full-board, isolated-target, and empty-board extraction modes.
 *
 * The engine is deliberately offline and side-effect free. It validates the supplied
 * ground truth and returns exactly the crops required by the selected capture mode.
 * Filesystem writing and manifest serialization live in scripts/extract_square_crops.py.
 */

import { CANONICAL_BOARD_SIZE, CANONICAL_SQUARE_SIZE, GRID_DIMENSION } from './rectify.ts';
import type { AnyImageData } from './classifySquare.ts';
import type { BoardViewOrientation } from './types.ts';
import { canonicalToPhysicalTag, validatePlacementOnly } from './manifestValidator.ts';
import type { CropType, PieceType, PieceColor } from './manifestValidator.ts';

export type CropExtractionMode =
  | 'full_board_context'
  | 'occupied_target_only'
  | 'empty_board_grid';

export type ExtractedCropType = Extract<CropType, 'full_board_context' | 'isolated_piece' | 'empty_square'>;

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

export interface CaptureCrop extends ExtractedSquareCrop {
  pieceInstanceId: string;
  physicalTagId: string;
  cropType: ExtractedCropType;
  extractionMode: CropExtractionMode;
}

export interface CropExtractionRequest {
  mode: CropExtractionMode;
  /** Placement-only ground truth. Mutually exclusive with boardFen. */
  placement?: string | null;
  /** Complete six-field FEN. Only the placement field is used for crop labels. */
  boardFen?: string | null;
  orientation?: BoardViewOrientation;
  setId?: string;
  /** Required for occupied_target_only. */
  targetSquare?: string;
  /** Required for occupied_target_only. Canonical SETxx_W_Pxx identifier. */
  targetPieceInstanceId?: string;
  /**
   * Required for full_board_context. Maps algebraic squares to canonical piece
   * instance IDs. Missing occupied squares or duplicate physical pieces are rejected.
   */
  pieceInstanceMap?: Record<string, string>;
}

export interface CropExtractionResult {
  mode: CropExtractionMode;
  cropType: ExtractedCropType;
  placement: string;
  sourcePhotoCount: 1;
  derivedCropCount: number;
  crops: CaptureCrop[];
}

export const CROP_TYPE_BY_MODE: Record<CropExtractionMode, ExtractedCropType> = {
  full_board_context: 'full_board_context',
  occupied_target_only: 'isolated_piece',
  empty_board_grid: 'empty_square',
};

export const EMPTY_PLACEMENT = '8/8/8/8/8/8/8/8';

const SQUARE_REGEX = /^[a-h][1-8]$/;
const CANONICAL_PIECE_REGEX = /^SET([0-9]{2})_([WB])_([PNBRQK])([0-9]{2})$/;
const SET_ID_REGEX = /^SET[0-9]{2}$/;
const SCENE_ID_REGEX = /^[A-Za-z0-9_-]+$/;

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

const TYPE_CHAR_TO_PIECE_TYPE: Record<string, PieceType> = {
  P: 'pawn',
  N: 'knight',
  B: 'bishop',
  R: 'rook',
  Q: 'queen',
  K: 'king',
};

function createEmptySquareMap(): Record<string, string> {
  const map: Record<string, string> = {};
  for (let rank = 1; rank <= 8; rank++) {
    for (let file = 0; file < 8; file++) {
      map[`${String.fromCharCode(97 + file)}${rank}`] = 'EMPTY';
    }
  }
  return map;
}

function getPlacementSource(request: CropExtractionRequest): string | null {
  if (request.placement && request.boardFen) {
    throw new Error('Provide either placement or boardFen, not both.');
  }
  if (request.boardFen) {
    const tokens = request.boardFen.trim().split(/\s+/);
    if (tokens.length !== 6) {
      throw new Error(
        `boardFen must contain exactly 6 fields, got ${tokens.length}. Use placement for placement-only ground truth.`
      );
    }
    return tokens[0];
  }
  if (request.placement) {
    return request.placement.trim();
  }
  return null;
}

/**
 * Maps an 8-rank FEN placement string to a square-by-square piece map.
 * Malformed placements are rejected instead of being silently treated as empty.
 */
export function parsePlacementToSquareMap(
  placement: string
): Record<string, { pieceChar: string; pieceType: PieceType; pieceColor: PieceColor }> {
  const check = validatePlacementOnly(placement);
  if (!check.valid) {
    throw new Error(`Invalid placement: ${check.error}`);
  }

  const map: Record<string, { pieceChar: string; pieceType: PieceType; pieceColor: PieceColor }> = {};
  const ranks = placement.trim().split('/');

  for (let rIdx = 0; rIdx < 8; rIdx++) {
    const rankNum = 8 - rIdx; // Rank 8 is first
    const rankStr = ranks[rIdx];
    let fileIdx = 0;

    for (const ch of rankStr) {
      if (ch >= '1' && ch <= '8') {
        fileIdx += parseInt(ch, 10);
      } else if ('pnbrqkPNBRQK'.includes(ch)) {
        const fileChar = String.fromCharCode(97 + fileIdx);
        const sq = `${fileChar}${rankNum}`;
        const isWhite = 'PNBRQK'.includes(ch);
        map[sq] = {
          pieceChar: ch,
          pieceType: CHAR_TO_PIECE_TYPE[ch],
          pieceColor: isWhite ? 'white' : 'black',
        };
        fileIdx++;
      }
    }
  }
  return map;
}

function parseCanonicalPieceId(pieceInstanceId: string): {
  setId: string;
  colorChar: string;
  typeChar: string;
  pieceType: PieceType;
} {
  const match = pieceInstanceId.match(CANONICAL_PIECE_REGEX);
  if (!match) {
    throw new Error(
      `Invalid piece_instance_id '${pieceInstanceId}'. Expected canonical format SETxx_W_P01.`
    );
  }
  return {
    setId: `SET${match[1]}`,
    colorChar: match[2],
    typeChar: match[3],
    pieceType: TYPE_CHAR_TO_PIECE_TYPE[match[3]],
  };
}

function assertPieceIdentity(
  pieceInstanceId: string,
  expected: { pieceType: PieceType; pieceColor: PieceColor; pieceChar: string },
  setId: string | undefined,
  square: string
): void {
  const parsed = parseCanonicalPieceId(pieceInstanceId);
  const expectedColorChar = expected.pieceColor === 'white' ? 'W' : 'B';
  if (parsed.colorChar !== expectedColorChar) {
    throw new Error(
      `Piece identity '${pieceInstanceId}' at ${square} has color '${parsed.colorChar}', expected '${expectedColorChar}'.`
    );
  }
  if (parsed.pieceType !== expected.pieceType) {
    throw new Error(
      `Piece identity '${pieceInstanceId}' at ${square} is a ${parsed.pieceType}, expected ${expected.pieceType}.`
    );
  }
  if (setId && parsed.setId !== setId) {
    throw new Error(
      `Piece identity '${pieceInstanceId}' belongs to ${parsed.setId}, expected set ${setId}.`
    );
  }
}

function resolveSetId(request: CropExtractionRequest, map: Record<string, string>): string {
  if (request.setId !== undefined) {
    if (!SET_ID_REGEX.test(request.setId)) {
      throw new Error(`Invalid set_id '${request.setId}'. Expected SET01-SET20.`);
    }
    return request.setId;
  }
  const firstId = Object.values(map).find((id) => id && id !== 'EMPTY');
  if (!firstId) {
    throw new Error('Cannot infer set_id from an empty piece instance map.');
  }
  return parseCanonicalPieceId(firstId).setId;
}

/**
 * Validates a full-board square-to-piece-instance map against the FEN occupancy map.
 */
export function validateFullBoardPieceMap(
  pieceMap: Record<string, string>,
  placementMap: Record<string, { pieceType: PieceType; pieceColor: PieceColor; pieceChar: string }>,
  setId?: string
): Record<string, string> {
  if (!pieceMap || typeof pieceMap !== 'object' || Array.isArray(pieceMap)) {
    throw new Error('full_board_context requires a pieceInstanceMap object.');
  }

  const resolvedSetId = resolveSetId({ mode: 'full_board_context', setId }, pieceMap);
  const seenInstances = new Set<string>();
  const validated: Record<string, string> = {};

  for (const [square, rawId] of Object.entries(pieceMap)) {
    if (!SQUARE_REGEX.test(square)) {
      throw new Error(`Invalid square '${square}' in pieceInstanceMap.`);
    }
    if (typeof rawId !== 'string' || rawId.trim() === '') {
      throw new Error(`Empty piece instance ID for square ${square}.`);
    }

    const id = rawId.trim();
    const expected = placementMap[square];
    if (!expected) {
      if (id !== 'EMPTY') {
        throw new Error(
          `pieceInstanceMap assigns '${id}' to empty square ${square}; FEN says the square is empty.`
        );
      }
      validated[square] = 'EMPTY';
      continue;
    }

    if (id === 'EMPTY') {
      throw new Error(`pieceInstanceMap leaves occupied square ${square} as EMPTY.`);
    }
    assertPieceIdentity(id, expected, resolvedSetId, square);
    if (seenInstances.has(id)) {
      throw new Error(
        `piece_instance_id '${id}' appears on multiple squares in one scene; a physical piece can occupy only one square.`
      );
    }
    seenInstances.add(id);
    validated[square] = id;
  }

  for (const square of Object.keys(placementMap)) {
    if (!validated[square]) {
      throw new Error(`pieceInstanceMap is missing occupied square ${square}.`);
    }
  }

  // Empty squares default to EMPTY; only explicitly supplied contradictions are rejected above.
  for (let rank = 1; rank <= 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const square = `${String.fromCharCode(97 + file)}${rank}`;
      if (!validated[square]) {
        validated[square] = 'EMPTY';
      }
    }
  }

  return validated;
}

function createCaptureCrops(
  image: AnyImageData,
  placement: string,
  orientation: BoardViewOrientation,
  pieceInstanceBySquare: Record<string, string>,
  mode: CropExtractionMode
): CaptureCrop[] {
  const cropType = CROP_TYPE_BY_MODE[mode];
  const slices = extractSquareCropsFromBoard(image, placement, orientation);
  return slices.map((crop) => {
    const pieceInstanceId = pieceInstanceBySquare[crop.square];
    if (!pieceInstanceId) {
      throw new Error(`No piece instance mapping for square ${crop.square}.`);
    }
    const physicalTagId = canonicalToPhysicalTag(pieceInstanceId);
    if (physicalTagId === 'INVALID') {
      throw new Error(`Invalid canonical piece identity '${pieceInstanceId}' for square ${crop.square}.`);
    }
    return {
      ...crop,
      pieceInstanceId,
      physicalTagId,
      cropType,
      extractionMode: mode,
    };
  });
}

/**
 * Extracts exactly the crops required by a capture mode.
 *
 * - full_board_context: 64 crops, explicit physical identity map required.
 * - occupied_target_only: 1 target crop, placement must contain exactly one piece.
 * - empty_board_grid: 64 empty crops, placement must contain no pieces.
 */
export function extractCropsForMode(
  image: AnyImageData,
  request: CropExtractionRequest
): CropExtractionResult {
  if (!(request.mode in CROP_TYPE_BY_MODE)) {
    throw new Error(`Unknown extraction mode '${request.mode}'.`);
  }

  const orientation = request.orientation ?? 'white-bottom';
  const placementSource = getPlacementSource(request);

  if (request.mode === 'empty_board_grid') {
    const placement = placementSource ?? EMPTY_PLACEMENT;
    const placementMap = parsePlacementToSquareMap(placement);
    if (Object.keys(placementMap).length !== 0) {
      throw new Error('empty_board_grid requires an empty placement with zero occupied squares.');
    }
    const emptyMap = createEmptySquareMap();
    const crops = createCaptureCrops(image, placement, orientation, emptyMap, request.mode);
    return {
      mode: request.mode,
      cropType: CROP_TYPE_BY_MODE[request.mode],
      placement,
      sourcePhotoCount: 1,
      derivedCropCount: crops.length,
      crops,
    };
  }

  if (!placementSource) {
    throw new Error(`${request.mode} requires placement or boardFen ground truth.`);
  }
  const placementMap = parsePlacementToSquareMap(placementSource);

  if (request.mode === 'occupied_target_only') {
    if (!request.targetSquare || !SQUARE_REGEX.test(request.targetSquare)) {
      throw new Error('occupied_target_only requires a valid targetSquare (a1-h8).');
    }
    if (!request.targetPieceInstanceId) {
      throw new Error('occupied_target_only requires targetPieceInstanceId.');
    }
    const occupiedSquares = Object.keys(placementMap);
    if (occupiedSquares.length !== 1) {
      throw new Error(
        `occupied_target_only requires exactly one occupied square, got ${occupiedSquares.length}.`
      );
    }
    const occupiedSquare = occupiedSquares[0];
    if (occupiedSquare !== request.targetSquare) {
      throw new Error(
        `Target square '${request.targetSquare}' does not match the only occupied square '${occupiedSquare}'.`
      );
    }
    const expected = placementMap[occupiedSquare];
    assertPieceIdentity(
      request.targetPieceInstanceId,
      expected,
      request.setId,
      occupiedSquare
    );

    const map = createEmptySquareMap();
    map[occupiedSquare] = request.targetPieceInstanceId;
    const crops = createCaptureCrops(image, placementSource, orientation, map, request.mode).filter(
      (crop) => crop.square === request.targetSquare
    );
    if (crops.length !== 1) {
      throw new Error(`Expected exactly one target crop for ${request.targetSquare}, got ${crops.length}.`);
    }
    return {
      mode: request.mode,
      cropType: CROP_TYPE_BY_MODE[request.mode],
      placement: placementSource,
      sourcePhotoCount: 1,
      derivedCropCount: crops.length,
      crops,
    };
  }

  // full_board_context
  if (Object.keys(placementMap).length === 0) {
    throw new Error('full_board_context requires at least one occupied square.');
  }
  if (!request.pieceInstanceMap) {
    throw new Error(
      'full_board_context requires pieceInstanceMap; refusing to invent physical piece identities from FEN types.'
    );
  }
  const validatedMap = validateFullBoardPieceMap(
    request.pieceInstanceMap,
    placementMap,
    request.setId
  );
  const crops = createCaptureCrops(image, placementSource, orientation, validatedMap, request.mode);
  if (crops.length !== 64) {
    throw new Error(`full_board_context must emit 64 crops, got ${crops.length}.`);
  }
  return {
    mode: request.mode,
    cropType: CROP_TYPE_BY_MODE[request.mode],
    placement: placementSource,
    sourcePhotoCount: 1,
    derivedCropCount: crops.length,
    crops,
  };
}

/**
 * Deterministic filename shared with scripts/extract_square_crops.py.
 */
export function buildCropFilename(
  sceneId: string,
  square: string,
  pieceInstanceId: string
): string {
  if (!SCENE_ID_REGEX.test(sceneId)) {
    throw new Error(
      `Invalid scene_id '${sceneId}'. Use letters, digits, underscore, or hyphen only.`
    );
  }
  if (!SQUARE_REGEX.test(square)) {
    throw new Error(`Invalid square '${square}'.`);
  }
  if (pieceInstanceId !== 'EMPTY' && !CANONICAL_PIECE_REGEX.test(pieceInstanceId)) {
    throw new Error(`Invalid piece_instance_id '${pieceInstanceId}'.`);
  }
  return `${sceneId}_${square}_${pieceInstanceId}.png`;
}

/**
 * Slices 64 100x100 square crops from an 800x800 canonical rectified board image.
 * This low-level helper assumes the image is already rectified and does not resize.
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

  const channels =
    'channels' in image && image.channels
      ? image.channels
      : image.data.length / (CANONICAL_BOARD_SIZE * CANONICAL_BOARD_SIZE);
  if (channels !== 1 && channels !== 3) {
    throw new Error(
      `Unsupported image channel depth: ${channels}. Must be 1 (grayscale) or 3 (RGB/BGR).`
    );
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
