/**
 * Physical Capture Manifest Schema and Offline Validator
 * Milestone 5.2D: Data-capture preparation, piece-instance tracking, FEN validation, and license gating.
 */

export type PieceType = 'pawn' | 'knight' | 'bishop' | 'rook' | 'queen' | 'king' | 'empty';
export type PieceColor = 'white' | 'black' | 'none';
export type CameraAngle = 'zenith' | 'white_perspective' | 'black_perspective' | 'lateral_flank';
export type LightingSetup = 'diffuse_daylight' | 'overhead_specular' | 'directional_oblique';
export type CropType = 'isolated_piece' | 'empty_square' | 'full_board_context' | 'full_board_raw';
export type OcclusionLevel = 'none' | 'partial' | 'severe';
export type SplitPartition = 'train' | 'val' | 'holdout';
export type PermissionStatus = 'verified_author_consent' | 'verified_owner_release';
export type AnnotationStatus = 'verified' | 'provisional' | 'unlabeled';

export interface CaptureManifestRecord {
  dataset_version: string;
  source_id: string;
  set_id: string;
  split: SplitPartition;
  piece_instance_id: string;
  physical_tag_id?: string | null;
  piece_type: PieceType;
  piece_color: PieceColor;
  scene_id: string;
  board_fen: string | null;
  piece_placement?: string | null;
  square_coord: string | null;
  camera_angle: CameraAngle;
  lighting_setup: LightingSetup;
  original_image_path: string;
  derived_crop_path: string;
  crop_type: CropType;
  occlusion_level: OcclusionLevel;
  photographer: string;
  rights_holder: string;
  release_record: string;
  data_license: string;
  code_license: string;
  permission_status: PermissionStatus;
  annotation_status: AnnotationStatus;
  quality_notes?: string;
}

export interface PiecePlacementCountSummary {
  white: Record<string, number>;
  black: Record<string, number>;
  totalWhite: number;
  totalBlack: number;
  totalPieces: number;
  emptySquares: number;
}

export interface ManifestSummary {
  totalRecords: number;
  distinctSets: string[];
  distinctPieceInstances: string[];
  distinctPhysicalTags: string[];
  distinctScenes: string[];
  cropsByType: Record<CropType, number>;
  cropsBySplit: Record<SplitPartition, number>;
  setSplitMap: Record<string, SplitPartition>;
}

export interface ManifestValidationResult {
  valid: boolean;
  errors: string[];
  summary: ManifestSummary;
}

const VALID_PIECE_TYPES = new Set<PieceType>([
  'pawn',
  'knight',
  'bishop',
  'rook',
  'queen',
  'king',
  'empty',
]);

const VALID_PIECE_COLORS = new Set<PieceColor>(['white', 'black', 'none']);

const VALID_CAMERA_ANGLES = new Set<CameraAngle>([
  'zenith',
  'white_perspective',
  'black_perspective',
  'lateral_flank',
]);

const VALID_LIGHTING_SETUPS = new Set<LightingSetup>([
  'diffuse_daylight',
  'overhead_specular',
  'directional_oblique',
]);

const VALID_CROP_TYPES = new Set<CropType>([
  'isolated_piece',
  'empty_square',
  'full_board_context',
  'full_board_raw',
]);

const VALID_OCCLUSION_LEVELS = new Set<OcclusionLevel>(['none', 'partial', 'severe']);

const VALID_SPLITS = new Set<SplitPartition>(['train', 'val', 'holdout']);

const VALID_PERMISSION_STATUSES = new Set<PermissionStatus>([
  'verified_author_consent',
  'verified_owner_release',
]);

const VALID_ANNOTATION_STATUSES = new Set<AnnotationStatus>([
  'verified',
  'provisional',
  'unlabeled',
]);

const SQUARE_REGEX = /^[a-h][1-8]$/;
const SET_ID_REGEX = /^SET[0-9]{2}$/;
const PIECE_INSTANCE_REGEX = /^SET[0-9]{2}_[WB]_[PNBRQK][0-9]{2}$/;
export const PHYSICAL_TAG_REGEX = /^S[0-9]{2}-[WB]-[PNBRQK][0-9]{2}$/;

/**
 * Maps a canonical manifest piece_instance_id (e.g. SET01_W_P01) to its physical underside tag ID (e.g. S01-W-P01).
 * Empty square identifier 'EMPTY' maps to 'NONE'.
 */
export function canonicalToPhysicalTag(canonicalId: string): string {
  if (!canonicalId || canonicalId === 'EMPTY') return 'NONE';
  const match = canonicalId.match(/^SET([0-9]{2})_([WB])_([PNBRQK][0-9]{2})$/);
  if (!match) return 'INVALID';
  return `S${match[1]}-${match[2]}-${match[3]}`;
}

/**
 * Maps a physical underside tag ID (e.g. S01-W-P01) to its canonical manifest piece_instance_id (e.g. SET01_W_P01).
 * Tag 'NONE' or 'EMPTY' maps to 'EMPTY'.
 */
export function physicalTagToCanonical(physicalTag: string): string {
  if (!physicalTag || physicalTag === 'NONE' || physicalTag === 'EMPTY') return 'EMPTY';
  const match = physicalTag.match(/^S([0-9]{2})-([WB])-([PNBRQK][0-9]{2})$/);
  if (!match) return 'INVALID';
  return `SET${match[1]}_${match[2]}_${match[3]}`;
}

/**
 * Expected split assignment based on the leak-free protocol:
 * Sets 01-14 -> train (70%)
 * Sets 15-17 -> val (15%)
 * Sets 18-20 -> holdout (15%)
 */
export function getExpectedSplitForSet(setId: string): SplitPartition | null {
  const match = setId.match(/^SET([0-9]{2})$/);
  if (!match) return null;
  const num = parseInt(match[1], 10);
  if (num >= 1 && num <= 14) return 'train';
  if (num >= 15 && num <= 17) return 'val';
  if (num >= 18 && num <= 20) return 'holdout';
  return null;
}

/**
 * Counts pieces by class and color from an 8-rank FEN placement string.
 */
export function countPiecesInPlacement(placement: string): PiecePlacementCountSummary {
  const white: Record<string, number> = {};
  const black: Record<string, number> = {};
  let totalWhite = 0;
  let totalBlack = 0;
  let emptySquares = 0;

  if (!placement || typeof placement !== 'string') {
    return { white, black, totalWhite: 0, totalBlack: 0, totalPieces: 0, emptySquares: 64 };
  }

  const firstToken = placement.trim().split(/\s+/)[0];
  const ranks = firstToken.split('/');

  for (const rank of ranks) {
    for (const ch of rank) {
      if (ch >= '1' && ch <= '8') {
        emptySquares += parseInt(ch, 10);
      } else if ('PNBRQK'.includes(ch)) {
        white[ch] = (white[ch] || 0) + 1;
        totalWhite++;
      } else if ('pnbrqk'.includes(ch)) {
        black[ch] = (black[ch] || 0) + 1;
        totalBlack++;
      }
    }
  }

  return {
    white,
    black,
    totalWhite,
    totalBlack,
    totalPieces: totalWhite + totalBlack,
    emptySquares,
  };
}

/**
 * Validates a placement-only string (strictly 8 ranks, no spaces, exactly 8 squares per rank).
 */
export function validatePlacementOnly(placement: string): { valid: boolean; error?: string } {
  if (!placement || typeof placement !== 'string') {
    return { valid: false, error: 'Placement string is empty or invalid.' };
  }
  const trimmed = placement.trim();
  if (trimmed.includes(' ')) {
    return {
      valid: false,
      error: "Placement string must not contain spaces (use complete FEN for 6-field strings).",
    };
  }
  const ranks = trimmed.split('/');
  if (ranks.length !== 8) {
    return { valid: false, error: `Placement must contain exactly 8 ranks separated by '/', got ${ranks.length}.` };
  }
  for (let r = 0; r < 8; r++) {
    const rank = ranks[r];
    let count = 0;
    for (const ch of rank) {
      if (ch >= '1' && ch <= '8') {
        count += parseInt(ch, 10);
      } else if ('pnbrqkPNBRQK'.includes(ch)) {
        count += 1;
      } else {
        return { valid: false, error: `Invalid character '${ch}' in rank ${8 - r}.` };
      }
    }
    if (count !== 8) {
      return { valid: false, error: `Rank ${8 - r} has ${count} squares instead of 8.` };
    }
  }
  return { valid: true };
}

/**
 * Validates a complete six-field FEN string.
 */
export function validateCompleteFen(fen: string): { valid: boolean; error?: string } {
  if (!fen || typeof fen !== 'string') {
    return { valid: false, error: 'FEN string is empty or invalid.' };
  }
  const tokens = fen.trim().split(/\s+/);
  if (tokens.length !== 6) {
    return {
      valid: false,
      error: `Complete FEN must contain exactly 6 space-separated fields, got ${tokens.length}. Use 'piece_placement' if only specifying placement.`,
    };
  }

  const [placement, activeColor, castling, enPassant, halfmove, fullmove] = tokens;

  // Field 1: Placement
  const placementCheck = validatePlacementOnly(placement);
  if (!placementCheck.valid) {
    return placementCheck;
  }

  // Field 2: Active Color
  if (activeColor !== 'w' && activeColor !== 'b') {
    return { valid: false, error: `Active color must be 'w' or 'b', got '${activeColor}'.` };
  }

  // Field 3: Castling Rights
  if (castling !== '-') {
    const validChars = new Set(['K', 'Q', 'k', 'q']);
    const seen = new Set<string>();
    for (const ch of castling) {
      if (!validChars.has(ch) || seen.has(ch)) {
        return { valid: false, error: `Invalid castling rights string: '${castling}'.` };
      }
      seen.add(ch);
    }
  }

  // Field 4: En Passant
  if (enPassant !== '-') {
    if (!/^[a-h][36]$/.test(enPassant)) {
      return {
        valid: false,
        error: `Invalid en-passant square: '${enPassant}'. Must be '-' or a valid square on rank 3 or 6.`,
      };
    }
  }

  // Field 5: Halfmove Clock
  const half = parseInt(halfmove, 10);
  if (isNaN(half) || half < 0 || String(half) !== halfmove) {
    return { valid: false, error: `Invalid halfmove clock: '${halfmove}'. Must be a non-negative integer.` };
  }

  // Field 6: Fullmove Number
  const full = parseInt(fullmove, 10);
  if (isNaN(full) || full < 1 || String(full) !== fullmove) {
    return { valid: false, error: `Invalid fullmove number: '${fullmove}'. Must be a positive integer >= 1.` };
  }

  return { valid: true };
}

/**
 * Validates FEN placement string syntax (8 ranks, each totaling 8 squares).
 * Retained for backwards compatibility; validates first token if multi-token.
 */
export function validateFenPlacement(fen: string): boolean {
  if (!fen || typeof fen !== 'string') return false;
  const placement = fen.trim().split(/\s+/)[0];
  return validatePlacementOnly(placement).valid;
}

/**
 * Validates a single manifest record.
 */
export function validateManifestRecord(
  record: CaptureManifestRecord,
  index = 0
): string[] {
  const errors: string[] = [];
  const prefix = `[Record #${index} Set:${record?.set_id || 'UNKNOWN'} Piece:${record?.piece_instance_id || 'UNKNOWN'}]`;

  if (!record || typeof record !== 'object') {
    return [`${prefix} Record is not a valid object`];
  }

  // Required string fields
  if (!record.dataset_version?.trim()) {
    errors.push(`${prefix} Missing required 'dataset_version'`);
  }
  if (!record.source_id?.trim()) {
    errors.push(`${prefix} Missing required 'source_id'`);
  }

  // Set ID validation
  if (!record.set_id || !SET_ID_REGEX.test(record.set_id)) {
    errors.push(`${prefix} Invalid 'set_id': '${record.set_id}'. Must match SET01-SET20 format.`);
  }

  // Split validation
  if (!VALID_SPLITS.has(record.split)) {
    errors.push(`${prefix} Invalid 'split': '${record.split}'. Expected train, val, or holdout.`);
  } else if (record.set_id && SET_ID_REGEX.test(record.set_id)) {
    const expected = getExpectedSplitForSet(record.set_id);
    if (expected && record.split !== expected) {
      errors.push(
        `${prefix} Split mismatch for set '${record.set_id}': assigned '${record.split}', expected '${expected}' per leak-free protocol.`
      );
    }
  }

  // Piece type & color validation
  if (!VALID_PIECE_TYPES.has(record.piece_type)) {
    errors.push(`${prefix} Unknown 'piece_type': '${record.piece_type}'.`);
  }
  if (!VALID_PIECE_COLORS.has(record.piece_color)) {
    errors.push(`${prefix} Unknown 'piece_color': '${record.piece_color}'.`);
  }

  // Piece-color consistency
  if (record.piece_type === 'empty') {
    if (record.piece_color !== 'none') {
      errors.push(`${prefix} Inconsistent empty piece: piece_color must be 'none' when piece_type is 'empty'.`);
    }
    if (record.piece_instance_id !== 'EMPTY') {
      errors.push(`${prefix} Empty square must have piece_instance_id 'EMPTY', got '${record.piece_instance_id}'.`);
    }
  } else {
    if (record.piece_color !== 'white' && record.piece_color !== 'black') {
      errors.push(`${prefix} Non-empty piece must have piece_color 'white' or 'black', got '${record.piece_color}'.`);
    }
    // Piece instance ID format
    if (!record.piece_instance_id || !PIECE_INSTANCE_REGEX.test(record.piece_instance_id)) {
      errors.push(
        `${prefix} Invalid 'piece_instance_id': '${record.piece_instance_id}'. Must match format SETxx_W_P01 (e.g. SET01_W_P01).`
      );
    } else {
      // Instance prefix must match set_id
      const instanceSet = record.piece_instance_id.slice(0, 5);
      if (instanceSet !== record.set_id) {
        errors.push(`${prefix} Piece instance prefix '${instanceSet}' does not match set_id '${record.set_id}'.`);
      }
      // Color letter consistency (W vs B)
      const instanceColorChar = record.piece_instance_id.charAt(6);
      const expectedColorChar = record.piece_color === 'white' ? 'W' : 'B';
      if (instanceColorChar !== expectedColorChar) {
        errors.push(
          `${prefix} Piece instance color code '${instanceColorChar}' does not match piece_color '${record.piece_color}'.`
        );
      }
      // Piece type letter consistency (P, N, B, R, Q, K)
      const instanceTypeChar = record.piece_instance_id.charAt(8);
      const typeCharMap: Record<PieceType, string> = {
        pawn: 'P',
        knight: 'N',
        bishop: 'B',
        rook: 'R',
        queen: 'Q',
        king: 'K',
        empty: 'E',
      };
      if (typeCharMap[record.piece_type] !== instanceTypeChar) {
        errors.push(
          `${prefix} Piece instance type code '${instanceTypeChar}' does not match piece_type '${record.piece_type}'.`
        );
      }
    }
  }

  // Physical tag ID validation & deterministic mapping (Gate 1)
  if (record.physical_tag_id !== undefined && record.physical_tag_id !== null && record.physical_tag_id.trim() !== '') {
    const tag = record.physical_tag_id.trim();
    if (record.piece_type === 'empty') {
      if (tag !== 'NONE' && tag !== 'EMPTY') {
        errors.push(`${prefix} Empty square 'physical_tag_id' must be 'NONE' or 'EMPTY', got '${tag}'.`);
      }
    } else {
      if (!PHYSICAL_TAG_REGEX.test(tag)) {
        errors.push(`${prefix} Invalid 'physical_tag_id': '${tag}'. Expected format Sxx-W-P01 (e.g. S01-W-P01).`);
      } else {
        const expectedTag = canonicalToPhysicalTag(record.piece_instance_id);
        if (tag !== expectedTag) {
          errors.push(
            `${prefix} Physical tag ID '${tag}' does not deterministically map to piece_instance_id '${record.piece_instance_id}' (expected '${expectedTag}').`
          );
        }
      }
    }
  }

  // Scene ID
  if (!record.scene_id?.trim()) {
    errors.push(`${prefix} Missing required 'scene_id'.`);
  }

  // Square coordinates
  if (record.crop_type !== 'full_board_raw') {
    if (!record.square_coord || !SQUARE_REGEX.test(record.square_coord)) {
      errors.push(
        `${prefix} Invalid or missing 'square_coord': '${record.square_coord}'. Must be a valid algebraic square (a1-h8).`
      );
    }
  }

  // FEN / Placement validation (required for full_board_context and full_board_raw)
  if (record.crop_type === 'full_board_context' || record.crop_type === 'full_board_raw') {
    if (record.board_fen) {
      const fenTokens = record.board_fen.trim().split(/\s+/);
      if (fenTokens.length === 6) {
        const fenCheck = validateCompleteFen(record.board_fen);
        if (!fenCheck.valid) {
          errors.push(`${prefix} Invalid complete 6-field 'board_fen': ${fenCheck.error}`);
        }
      } else {
        // Placement-only provided in board_fen field
        if (!validateFenPlacement(record.board_fen)) {
          errors.push(`${prefix} Invalid placement syntax in 'board_fen': '${record.board_fen}'.`);
        }
      }
    } else if (record.piece_placement) {
      const placeCheck = validatePlacementOnly(record.piece_placement);
      if (!placeCheck.valid) {
        errors.push(`${prefix} Invalid 'piece_placement': ${placeCheck.error}`);
      }
    } else {
      errors.push(`${prefix} Missing required 'board_fen' or 'piece_placement' for full-board crop.`);
    }
  }

  // Validate standalone piece_placement if present
  if (record.piece_placement) {
    const placeCheck = validatePlacementOnly(record.piece_placement);
    if (!placeCheck.valid) {
      errors.push(`${prefix} Invalid 'piece_placement': ${placeCheck.error}`);
    }
  }

  // Camera angle & lighting
  if (!VALID_CAMERA_ANGLES.has(record.camera_angle)) {
    errors.push(`${prefix} Unknown 'camera_angle': '${record.camera_angle}'.`);
  }
  if (!VALID_LIGHTING_SETUPS.has(record.lighting_setup)) {
    errors.push(`${prefix} Unknown 'lighting_setup': '${record.lighting_setup}'.`);
  }

  // Crop type & occlusion
  if (!VALID_CROP_TYPES.has(record.crop_type)) {
    errors.push(`${prefix} Unknown 'crop_type': '${record.crop_type}'.`);
  }
  if (!VALID_OCCLUSION_LEVELS.has(record.occlusion_level)) {
    errors.push(`${prefix} Unknown 'occlusion_level': '${record.occlusion_level}'.`);
  }

  // Paths
  if (!record.original_image_path?.trim()) {
    errors.push(`${prefix} Missing required 'original_image_path'.`);
  }
  if (!record.derived_crop_path?.trim()) {
    errors.push(`${prefix} Missing required 'derived_crop_path'.`);
  }

  // Provenance & Licensing (Gate 5)
  if (!record.photographer?.trim()) {
    errors.push(`${prefix} Missing required 'photographer' attribution.`);
  }
  if (!record.rights_holder?.trim()) {
    errors.push(`${prefix} Missing required 'rights_holder'.`);
  }
  if (!record.release_record?.trim()) {
    errors.push(`${prefix} Missing required 'release_record' document reference.`);
  }
  if (record.data_license !== 'CC0-1.0') {
    errors.push(`${prefix} Non-compliant 'data_license': '${record.data_license}'. Dataset data must be 'CC0-1.0'.`);
  }
  if (record.code_license !== 'MIT') {
    errors.push(`${prefix} Non-compliant 'code_license': '${record.code_license}'. Pipeline code must be 'MIT'.`);
  }
  if (!VALID_PERMISSION_STATUSES.has(record.permission_status)) {
    errors.push(`${prefix} Invalid or unverified 'permission_status': '${record.permission_status}'.`);
  }
  if (!VALID_ANNOTATION_STATUSES.has(record.annotation_status)) {
    errors.push(`${prefix} Unknown 'annotation_status': '${record.annotation_status}'.`);
  }

  return errors;
}

/**
 * Validates a complete collection of manifest records, checking intra-record
 * consistency, duplicate filenames, and inter-split leakage.
 */
export function validateManifestCollection(
  records: CaptureManifestRecord[]
): ManifestValidationResult {
  const errors: string[] = [];

  const setsSeen = new Set<string>();
  const pieceInstancesSeen = new Set<string>();
  const physicalTagsSeen = new Set<string>();
  const scenesSeen = new Set<string>();
  const derivedPathsSeen = new Map<string, number>();

  const setSplitMap: Record<string, SplitPartition> = {};
  const pieceSplitMap: Record<string, SplitPartition> = {};
  const tagPieceMap: Record<string, string> = {};
  const tagSplitMap: Record<string, SplitPartition> = {};

  const cropsByType: Record<CropType, number> = {
    isolated_piece: 0,
    empty_square: 0,
    full_board_context: 0,
    full_board_raw: 0,
  };

  const cropsBySplit: Record<SplitPartition, number> = {
    train: 0,
    val: 0,
    holdout: 0,
  };

  if (!Array.isArray(records) || records.length === 0) {
    return {
      valid: false,
      errors: ['Manifest is empty or not an array.'],
      summary: {
        totalRecords: 0,
        distinctSets: [],
        distinctPieceInstances: [],
        distinctPhysicalTags: [],
        distinctScenes: [],
        cropsByType,
        cropsBySplit,
        setSplitMap,
      },
    };
  }

  for (let i = 0; i < records.length; i++) {
    const record = records[i];
    const recordErrors = validateManifestRecord(record, i);
    errors.push(...recordErrors);

    if (record) {
      if (record.set_id) setsSeen.add(record.set_id);
      if (record.scene_id) scenesSeen.add(record.scene_id);
      if (record.piece_instance_id && record.piece_instance_id !== 'EMPTY') {
        pieceInstancesSeen.add(record.piece_instance_id);
      }
      if (record.physical_tag_id && record.physical_tag_id !== 'NONE' && record.physical_tag_id !== 'EMPTY') {
        physicalTagsSeen.add(record.physical_tag_id);

        // Check physical tag mapping consistency
        if (record.piece_instance_id) {
          const existingPiece = tagPieceMap[record.physical_tag_id];
          if (existingPiece && existingPiece !== record.piece_instance_id) {
            errors.push(
              `[Tag Identity Conflict] Physical tag '${record.physical_tag_id}' is mapped to multiple piece instances: '${existingPiece}' and '${record.piece_instance_id}'.`
            );
          } else {
            tagPieceMap[record.physical_tag_id] = record.piece_instance_id;
          }
        }

        // Check cross-split leakage for physical tag
        if (record.split) {
          const existingTagSplit = tagSplitMap[record.physical_tag_id];
          if (existingTagSplit && existingTagSplit !== record.split) {
            errors.push(
              `[Tag Leakage Violation] Physical tag '${record.physical_tag_id}' appears in multiple splits: '${existingTagSplit}' and '${record.split}'.`
            );
          } else {
            tagSplitMap[record.physical_tag_id] = record.split;
          }
        }
      }

      // Check duplicate derived crop paths
      if (record.derived_crop_path) {
        const prevIndex = derivedPathsSeen.get(record.derived_crop_path);
        if (prevIndex !== undefined) {
          errors.push(
            `[Duplicate Path] Derived crop path '${record.derived_crop_path}' occurs in record #${prevIndex} and record #${i}.`
          );
        } else {
          derivedPathsSeen.set(record.derived_crop_path, i);
        }
      }

      // Check cross-split leakage for physical set
      if (record.set_id && record.split) {
        const existingSplit = setSplitMap[record.set_id];
        if (existingSplit && existingSplit !== record.split) {
          errors.push(
            `[Set Leakage Violation] Set '${record.set_id}' appears in multiple splits: '${existingSplit}' and '${record.split}'.`
          );
        } else {
          setSplitMap[record.set_id] = record.split;
        }
      }

      // Check cross-split leakage for physical piece instance
      if (record.piece_instance_id && record.piece_instance_id !== 'EMPTY' && record.split) {
        const existingPieceSplit = pieceSplitMap[record.piece_instance_id];
        if (existingPieceSplit && existingPieceSplit !== record.split) {
          errors.push(
            `[Piece Leakage Violation] Physical piece instance '${record.piece_instance_id}' appears in multiple splits: '${existingPieceSplit}' and '${record.split}'.`
          );
        } else {
          pieceSplitMap[record.piece_instance_id] = record.split;
        }
      }

      // Counts
      if (VALID_CROP_TYPES.has(record.crop_type)) {
        cropsByType[record.crop_type] = (cropsByType[record.crop_type] || 0) + 1;
      }
      if (VALID_SPLITS.has(record.split)) {
        cropsBySplit[record.split] = (cropsBySplit[record.split] || 0) + 1;
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    summary: {
      totalRecords: records.length,
      distinctSets: Array.from(setsSeen).sort(),
      distinctPieceInstances: Array.from(pieceInstancesSeen).sort(),
      distinctPhysicalTags: Array.from(physicalTagsSeen).sort(),
      distinctScenes: Array.from(scenesSeen).sort(),
      cropsByType,
      cropsBySplit,
      setSplitMap,
    },
  };
}
