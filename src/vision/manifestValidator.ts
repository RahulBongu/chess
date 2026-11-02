/**
 * Physical Capture Manifest Schema and Offline Validator
 * Milestone 5.2C: Data-capture preparation, piece-instance tracking, and license gating.
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
  piece_type: PieceType;
  piece_color: PieceColor;
  scene_id: string;
  board_fen: string | null;
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

export interface ManifestSummary {
  totalRecords: number;
  distinctSets: string[];
  distinctPieceInstances: string[];
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
 * Validates FEN placement string syntax (8 ranks, each totaling 8 squares).
 */
export function validateFenPlacement(fen: string): boolean {
  if (!fen || typeof fen !== 'string') return false;
  const placement = fen.trim().split(/\s+/)[0];
  const ranks = placement.split('/');
  if (ranks.length !== 8) return false;

  for (const rank of ranks) {
    let count = 0;
    for (const char of rank) {
      if (char >= '1' && char <= '8') {
        count += parseInt(char, 10);
      } else if ('pnbrqkPNBRQK'.includes(char)) {
        count += 1;
      } else {
        return false;
      }
    }
    if (count !== 8) return false;
  }
  return true;
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

  // FEN validation (required for full_board_context and full_board_raw)
  if (record.crop_type === 'full_board_context' || record.crop_type === 'full_board_raw') {
    if (!record.board_fen || !validateFenPlacement(record.board_fen)) {
      errors.push(`${prefix} Missing or invalid 'board_fen' for full-board crop: '${record.board_fen}'.`);
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
  const scenesSeen = new Set<string>();
  const derivedPathsSeen = new Map<string, number>();

  const setSplitMap: Record<string, SplitPartition> = {};
  const pieceSplitMap: Record<string, SplitPartition> = {};

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
      distinctScenes: Array.from(scenesSeen).sort(),
      cropsByType,
      cropsBySplit,
      setSplitMap,
    },
  };
}
