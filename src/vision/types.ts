import type { PatternEvidence } from './boardEvidence';

export interface VisionSize {
  width: number;
  height: number;
}

export interface DetectedCorner {
  x: number;
  y: number;
}

export interface DetectionCandidate {
  corners: DetectedCorner[];
  area: number;
  rectangularity: number;
  confidence: number;
}

export interface CandidateDiagnostic {
  corners: DetectedCorner[];
  samplingCorners: DetectedCorner[];
  samplingScale: number;
  boundaryScore: number;
  pattern: PatternEvidence;
  confidence: number;
}

export interface BoardDetectionResult {
  success: boolean;
  imageSize: VisionSize;
  processingSize: VisionSize;
  corners: DetectedCorner[];
  confidence: number;
  candidateCount: number;
  message: string;
  diagnostics?: CandidateDiagnostic[];
}

export type BoardViewOrientation = 'white-bottom' | 'black-bottom';

export type ParityConsistency = 'consistent' | 'inconsistent' | 'unknown';

export interface Homography3x3 {
  m00: number;
  m01: number;
  m02: number;
  m10: number;
  m11: number;
  m12: number;
  m20: number;
  m21: number;
  m22: number;
}

export interface SquareRegion {
  /** Algebraic notation for the square ('a1'..'h8') under the active orientation. */
  name: string;
  /** File index: 0 = file a, 7 = file h. */
  file: number;
  /** Rank index: 0 = rank 1, 7 = rank 8. */
  rank: number;
  /** Canonical grid row index: 0 = top row (y: 0..100), 7 = bottom row (y: 700..800). */
  gridRow: number;
  /** Canonical grid column index: 0 = left col (x: 0..100), 7 = right col (x: 700..800). */
  gridCol: number;
  /** Bounding box in canonical 800x800 coordinate space. */
  bounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  /** Center point in canonical 800x800 coordinate space. */
  center: DetectedCorner;
  /** Four corners in canonical 800x800 coordinate space [TL, TR, BR, BL]. */
  canonicalCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner];
  /** Four corners projected back to source image coordinate space [TL, TR, BR, BL]. */
  sourceCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner];
  /** True if the square is light according to standard chess rules. */
  isLight: boolean;
}

export interface OrientationAssessment {
  view: BoardViewOrientation;
  parityConsistency: ParityConsistency;
  ambiguityWarning: string;
}

export interface RectifiedBoardGeometry {
  canonicalSize: VisionSize;
  sourceCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner];
  canonicalCorners: [DetectedCorner, DetectedCorner, DetectedCorner, DetectedCorner];
  forwardHomography: Homography3x3;
  inverseHomography: Homography3x3;
  reprojectionErrorPx: number;
  orientation: OrientationAssessment;
  squares: SquareRegion[];
  squareByName: Record<string, SquareRegion>;
}

export interface RgbImage {
  width: number;
  height: number;
  /** Row-major interleaved RGB byte data (length = width * height * 3). */
  data: Uint8Array | Uint8ClampedArray;
  channels: 3;
}

export type SquareOccupancy = 'empty' | 'occupied' | 'uncertain';

export type PieceColorEstimate = 'white' | 'black' | 'uncertain';

export interface SquareDiagnosticFeatures {
  surfaceLuminance: number;
  surfaceStdDev: number;
  centerLuminance: number;
  centerStdDev: number;
  contrastWithSurface: number;
  edgeIntensity: number;
  minLuminance: number;
  maxLuminance: number;
  luminanceRange: number;
  meanR?: number;
  meanG?: number;
  meanB?: number;
}

export interface ClassifiedSquare {
  name: string;
  file: number;
  rank: number;
  gridRow: number;
  gridCol: number;
  isLight: boolean;
  occupancy: SquareOccupancy;
  occupancyConfidence: number;
  pieceColor: PieceColorEstimate | null;
  colorConfidence: number;
  overallConfidence: number;
  features: SquareDiagnosticFeatures;
  reason?: string;
}

export interface BoardClassificationResult {
  view: BoardViewOrientation;
  squares: ClassifiedSquare[];
  squareByName: Record<string, ClassifiedSquare>;
  summary: {
    emptyCount: number;
    occupiedCount: number;
    uncertainCount: number;
    whitePieceCount: number;
    blackPieceCount: number;
    meanConfidence: number;
  };
}

