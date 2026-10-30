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