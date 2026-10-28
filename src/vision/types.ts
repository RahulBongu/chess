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

export interface BoardDetectionResult {
  success: boolean;
  imageSize: VisionSize;
  processingSize: VisionSize;
  corners: DetectedCorner[];
  confidence: number;
  candidateCount: number;
  message: string;
}