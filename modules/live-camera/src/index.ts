const { requireNativeViewManager } = require('expo-modules-core');

export interface LiveCameraFrameEvent {
  base64: string;
  width: number;
  height: number;
  /** clockwise rotation in degrees required to make the frame upright (0/90/180/270). */
  rotationDegrees?: number;
  /** which stream produced this frame: 'detection' (full resolution) or 'tracking' (low-res high-rate). */
  stream?: 'detection' | 'tracking';
  /** Cumulative frames analysed by the native analyzer for this stream. */
  analyzedFrames?: number;
  /** Cumulative frames skipped by the native analyzer (throttle or busy). */
  droppedFrames?: number;
  /**
   * 64 x 13 raw model scores (logits), board-view row-major. Present only
   * when native square classification ran for this frame. Crop pixels never
   * cross the bridge.
   */
  squareScores?: number[];
  /** Normalized corners used for this frame's square crops. */
  classificationCorners?: number[];
  /** Native ONNX inference duration for this frame. */
  inferenceMs?: number;
  /** Correlates this frame with the corner update that produced it. */
  classificationRequestId?: number;
}

export interface LiveCameraViewHandle {
  setClassificationCorners(corners: number[], requestId: number): Promise<void>;
  clearClassificationCorners(): Promise<void>;
  setClassificationEnabled(enabled: boolean): Promise<void>;
  releaseFrame(): Promise<void>;
  releaseTrackingFrame(): Promise<void>;
}

export interface LiveCameraViewProps {
  active?: boolean;
  onFrame?: (event: { nativeEvent: LiveCameraFrameEvent }) => void;
  onTrackingFrame?: (event: { nativeEvent: LiveCameraFrameEvent }) => void;
  onPreviewReady?: (event: { nativeEvent: { width: number; height: number } }) => void;
  onAnalysisReady?: (event: { nativeEvent: { width: number; height: number; stream?: string } }) => void;
  onCameraBound?: (event: { nativeEvent: { analysisEnabled: boolean; trackingEnabled: boolean } }) => void;
  onCameraError?: (event: { nativeEvent: { message: string } }) => void;
  /** Enable high-rate low-res tracking stream (default true). */
  trackingEnabled?: boolean;
  /** Target width for tracking stream (default 160). */
  trackingTargetWidth?: number;
  /** Target height for tracking stream (default 120). */
  trackingTargetHeight?: number;
  /** Minimum interval between tracking frames in ms (default 80 ~ 12.5 fps). */
  trackingIntervalMs?: number;
  style?: any;
}

export const LiveCameraView = requireNativeViewManager('LiveCamera');