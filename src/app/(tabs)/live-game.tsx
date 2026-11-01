import { useCameraPermissions } from 'expo-camera';
import { useWindowDimensions } from 'react-native';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { detectBoardFromGrayscaleFrame } from '@/vision/detectBoard';
import { LiveCameraView } from '../../../modules/live-camera/src';
import type { DetectedCorner } from '@/vision/types';

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function cornerPixels(corner: DetectedCorner, width: number, height: number) {
  return { x: corner.x * width, y: corner.y * height };
}

function Segment({
  from,
  to,
  color,
  width = 2,
}: {
  from: { x: number; y: number };
  to: { x: number; y: number };
  color: string;
  width?: number;
}) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
  const midX = (from.x + to.x) / 2;
  const midY = (from.y + to.y) / 2;
  return (
    <View
      pointerEvents="none"
      style={{
        position: 'absolute',
        left: midX - length / 2,
        top: midY - width / 2,
        width: length,
        height: width,
        backgroundColor: color,
        transform: [{ rotate: `${angle}deg` }],
      }}
    />
  );
}

export default function LiveGameScreen() {
  const { width: windowWidth } = useWindowDimensions();
  const [permission, requestPermission] = useCameraPermissions();
  const [active, setActive] = useState(false);
  const [previewReady, setPreviewReady] = useState(false);
  const [analysisReady, setAnalysisReady] = useState(false);
  const [corners, setCorners] = useState<DetectedCorner[]>([]);
  const [status, setStatus] = useState('Start a live game to preview the board.');
  const [analysisStatus, setAnalysisStatus] = useState<string | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [diagnostics, setDiagnostics] = useState<string | null>(null);
  // Temporary diagnostics for live grayscale board detection.
  const [detectionMessage, setDetectionMessage] = useState<string | null>(null);
  const [detectionConfidence, setDetectionConfidence] = useState<number | null>(null);
  const busyRef = useRef(false);
  const cameraRef = useRef<any>(null);
  const startupTokenRef = useRef(0);
  const startupTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (startupTimeoutRef.current) {
        clearTimeout(startupTimeoutRef.current);
      }
    };
  }, []);

  const previewWidth = Math.min(windowWidth - 24, 420);
  const previewHeight = Math.round(previewWidth * (4 / 3));

  const handleFrame = useCallback(
    async (event: { nativeEvent: { base64: string; width: number; height: number } }) => {
      if (!active || busyRef.current) {
        return;
      }
      busyRef.current = true;
      try {
        const result = await detectBoardFromGrayscaleFrame(
          event.nativeEvent.base64,
          event.nativeEvent.width,
          event.nativeEvent.height,
        );
        setFrameCount((value) => value + 1);
        setDetectionMessage(result.message);
        setDetectionConfidence(result.confidence);
        if (result.success) {
          setCorners(result.corners);
          setAnalysisStatus('Board detected. Live analysis is running.');
        } else {
          setCorners([]);
          setAnalysisStatus(result.message);
        }
        const best = result.diagnostics?.[0];
        setDiagnostics(
          best
            ? `candidates=${result.candidateCount} boundary=${best.boundaryScore.toFixed(2)} pattern=${best.pattern.score.toFixed(2)} fit=${best.pattern.fit.toFixed(2)} alt=${best.pattern.alternation.toFixed(2)} contrast=${best.pattern.contrast.toFixed(1)} noise=${best.pattern.noise.toFixed(1)} scale=${best.samplingScale.toFixed(2)} reason=${best.pattern.reason ?? 'accepted'}`
            : `candidates=${result.candidateCount}`,
        );
      } catch (error) {
        setStatus(error instanceof Error ? error.message : 'Live frame analysis failed.');
      } finally {
        busyRef.current = false;
        cameraRef.current?.releaseFrame?.();
      }
    },
    [active],
  );

  const clearStartupTimeout = useCallback(() => {
    if (startupTimeoutRef.current) {
      clearTimeout(startupTimeoutRef.current);
      startupTimeoutRef.current = null;
    }
  }, []);

  const handlePreviewReady = useCallback(() => {
    clearStartupTimeout();
    setPreviewReady(true);
    setStatus('Live camera preview running. Analyzing frames...');
  }, [clearStartupTimeout]);

  const handleAnalysisReady = useCallback(() => {
    setAnalysisReady(true);
    setAnalysisStatus('Analysis frames received.');
  }, []);

  const handleCameraError = useCallback((event: { nativeEvent: { message: string } }) => {
    clearStartupTimeout();
    setActive(false);
    setPreviewReady(false);
    setAnalysisReady(false);
    setCorners([]);
    setStatus(`Live camera failed to start: ${event.nativeEvent.message}`);
  }, [clearStartupTimeout]);

  const handleStart = async () => {
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        setStatus('Camera permission is required for live game mode.');
        return;
      }
    }
    setCorners([]);
    setPreviewReady(false);
    setAnalysisReady(false);
    setAnalysisStatus(null);
    setDiagnostics(null);
    setFrameCount(0);
    setDetectionMessage(null);
    setDetectionConfidence(null);
    startupTokenRef.current += 1;
    clearStartupTimeout();
    setActive(true);
    setStatus('Starting live camera...');
    startupTimeoutRef.current = setTimeout(() => {
      if (previewReady) {
        return;
      }
      setActive(false);
      setPreviewReady(false);
      setAnalysisReady(false);
      setCorners([]);
      setStatus('Live camera preview did not start within 5 seconds. Analysis frames may arrive, but PreviewView is not streaming.');
    }, 5000);
  };

  const handleStop = () => {
    clearStartupTimeout();
    setActive(false);
    setPreviewReady(false);
    setAnalysisReady(false);
    setAnalysisStatus(null);
    setCorners([]);
    setStatus('Live game stopped. Camera released.');
  };

  if (!permission) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <Text style={styles.status}>Loading camera permission status...</Text>
      </SafeAreaView>
    );
  }

  if (!permission.granted && !active) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.centered}>
          <Text style={styles.title}>Live Game</Text>
          <Text style={styles.body}>
            Camera access is requested only when you start a live game. Frames are analyzed locally and are not saved.
          </Text>
          <Pressable style={styles.primaryButton} onPress={handleStart}>
            <Text style={styles.buttonText}>Start Live Game</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => {}} disabled>
            <Text style={styles.buttonText}>Stop</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Live Game</Text>
        <Text style={styles.body}>
          Stage 1 foundation: live preview, board overlay and local frame analysis. Piece recognition and engine moves are not implemented yet.
        </Text>

        <View style={[styles.previewContainer, { width: previewWidth, height: previewHeight }]}>
          <LiveCameraView
            ref={cameraRef}
            style={StyleSheet.absoluteFill}
            active={active}
            onFrame={handleFrame}
            onPreviewReady={handlePreviewReady}
            onAnalysisReady={handleAnalysisReady}
            onCameraError={handleCameraError}
          />
          {corners.length === 4 ? (
            <View style={StyleSheet.absoluteFill} pointerEvents="none">
              {(() => {
                const points = corners.map((corner) => cornerPixels(corner, previewWidth, previewHeight));
                const segments: React.ReactElement[] = [];
                for (let index = 0; index < 4; index += 1) {
                  segments.push(
                    <Segment
                      key={`outer-${index}`}
                      from={points[index]}
                      to={points[(index + 1) % 4]}
                      color="#00E5FF"
                      width={3}
                    />,
                  );
                }
                for (let i = 1; i < 8; i += 1) {
                  const t = i / 8;
                  segments.push(
                    <Segment
                      key={`vertical-${i}`}
                      from={{
                        x: lerp(points[0].x, points[1].x, t),
                        y: lerp(points[0].y, points[1].y, t),
                      }}
                      to={{
                        x: lerp(points[3].x, points[2].x, t),
                        y: lerp(points[3].y, points[2].y, t),
                      }}
                      color="rgba(0,229,255,0.55)"
                      width={1}
                    />,
                  );
                  segments.push(
                    <Segment
                      key={`horizontal-${i}`}
                      from={{
                        x: lerp(points[0].x, points[3].x, t),
                        y: lerp(points[0].y, points[3].y, t),
                      }}
                      to={{
                        x: lerp(points[1].x, points[2].x, t),
                        y: lerp(points[1].y, points[2].y, t),
                      }}
                      color="rgba(0,229,255,0.55)"
                      width={1}
                    />,
                  );
                }
                return segments;
              })()}
            </View>
          ) : null}
        </View>

        <Text style={styles.status}>{status}</Text>
        <Text style={styles.body}>
          Preview ready: {previewReady ? 'yes' : 'no'}; analysis ready: {analysisReady ? 'yes' : 'no'}; analyzed frames: {frameCount}
        </Text>
        {analysisStatus ? <Text style={styles.body}>{analysisStatus}</Text> : null}
        {diagnostics ? <Text style={styles.body}>{diagnostics}</Text> : null}
        <Text style={styles.body}>
          Detection: {corners.length === 4 ? 'board found' : 'no board'}; confidence:{' '}
          {detectionConfidence !== null ? detectionConfidence.toFixed(2) : 'n/a'}
        </Text>
        <Text style={styles.body}>Detection message: {detectionMessage ?? 'none'}</Text>
        {corners.length === 4 ? (
          <Text style={styles.body}>
            Corners:{' '}
            {corners.map((corner) => `(${corner.x.toFixed(3)}, ${corner.y.toFixed(3)})`).join(' ')}
          </Text>
        ) : null}

        <View style={styles.controls}>
          <Pressable
            style={active ? styles.disabledButton : styles.primaryButton}
            onPress={handleStart}
            disabled={active}>
            <Text style={styles.buttonText}>Start Live Game</Text>
          </Pressable>
          <Pressable
            style={!active ? styles.disabledButton : styles.secondaryButton}
            onPress={handleStop}
            disabled={!active}>
            <Text style={styles.buttonText}>Stop</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1 },
  content: { padding: 16, gap: 12, alignItems: 'center' },
  centered: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, gap: 12 },
  title: { fontSize: 22, fontWeight: '700', textAlign: 'center' },
  body: { fontSize: 13, textAlign: 'center', opacity: 0.8 },
  status: { fontSize: 14, textAlign: 'center', fontWeight: '600' },
  previewContainer: {
    position: 'relative',
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#000000',
  },
  controls: { width: '100%', flexDirection: 'row', gap: 8, justifyContent: 'center', flexWrap: 'wrap' },
  primaryButton: { paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, backgroundColor: '#208AEF', minWidth: 150, alignItems: 'center' },
  secondaryButton: { paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, backgroundColor: '#6C757D', minWidth: 120, alignItems: 'center' },
  disabledButton: { paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, backgroundColor: '#B0BEC5', minWidth: 120, alignItems: 'center' },
  buttonText: { color: '#FFFFFF', fontWeight: '600', fontSize: 14 },
});