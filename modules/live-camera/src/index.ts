const { requireNativeViewManager } = require('expo-modules-core');

export interface LiveCameraFrameEvent {
  base64: string;
  width: number;
  height: number;
}

export interface LiveCameraViewProps {
  active?: boolean;
  onFrame?: (event: { nativeEvent: LiveCameraFrameEvent }) => void;
  onPreviewReady?: (event: { nativeEvent: { width: number; height: number } }) => void;
  onAnalysisReady?: (event: { nativeEvent: { width: number; height: number } }) => void;
  onCameraBound?: (event: { nativeEvent: { analysisEnabled: boolean } }) => void;
  onCameraError?: (event: { nativeEvent: { message: string } }) => void;
  style?: any;
}

export const LiveCameraView = requireNativeViewManager('LiveCamera');