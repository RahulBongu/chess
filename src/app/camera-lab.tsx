import { CameraView, useCameraPermissions } from 'expo-camera';
import { useWindowDimensions } from 'react-native';
import React, { useCallback, useRef, useState } from 'react';
import {
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { LiveCameraView } from '../../modules/live-camera/src';

type LabMode = 'idle' | 'A' | 'B' | 'C';

export default function CameraLabScreen() {
  const { width: windowWidth } = useWindowDimensions();
  const [permission, requestPermission] = useCameraPermissions();
  const [mode, setMode] = useState<LabMode>('idle');
  const [nativeBinding, setNativeBinding] = useState('not started');
  const [previewReady, setPreviewReady] = useState(false);
  const [analysisReady, setAnalysisReady] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [lastEvent, setLastEvent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cameraRef = useRef<any>(null);
  const busyRef = useRef(false);

  const previewWidth = Math.min(windowWidth - 24, 420);
  const previewHeight = Math.round(previewWidth * (4 / 3));

  const timestamp = () => new Date().toISOString();

  const resetDiagnostics = useCallback(() => {
    setNativeBinding('not started');
    setPreviewReady(false);
    setAnalysisReady(false);
    setFrameCount(0);
    setLastEvent(`${timestamp()}: reset`);
    setError(null);
  }, []);

  const stop = useCallback(() => {
    setMode('idle');
    setLastEvent(`${timestamp()}: stop requested`);
  }, []);

  const startModeA = useCallback(async () => {
    if (mode !== 'idle') {
      return;
    }
    resetDiagnostics();
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        setError('Camera permission denied.');
        setLastEvent(`${timestamp()}: permission denied`);
        return;
      }
    }
    setMode('A');
    setLastEvent(`${timestamp()}: Mode A started`);
  }, [mode, permission, requestPermission, resetDiagnostics]);

  const startModeB = useCallback(async () => {
    if (mode !== 'idle') {
      return;
    }
    resetDiagnostics();
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        setError('Camera permission denied.');
        setLastEvent(`${timestamp()}: permission denied`);
        return;
      }
    }
    setMode('B');
    setLastEvent(`${timestamp()}: Mode B started`);
  }, [mode, permission, requestPermission, resetDiagnostics]);

  const startModeC = useCallback(async () => {
    if (mode !== 'idle') {
      return;
    }
    resetDiagnostics();
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        setError('Camera permission denied.');
        setLastEvent(`${timestamp()}: permission denied`);
        return;
      }
    }
    setMode('C');
    setLastEvent(`${timestamp()}: Mode C started`);
  }, [mode, permission, requestPermission, resetDiagnostics]);

  const handleFrame = useCallback(async (event: { nativeEvent: { base64: string; width: number; height: number } }) => {
    if (busyRef.current) {
      return;
    }
    busyRef.current = true;
    try {
      setFrameCount((value) => value + 1);
      setLastEvent(`${timestamp()}: analysis frame ${event.nativeEvent.width}x${event.nativeEvent.height}`);
    } finally {
      busyRef.current = false;
      cameraRef.current?.releaseFrame?.();
    }
  }, []);

  const handlePreviewReady = useCallback((event: { nativeEvent: { width: number; height: number } }) => {
    setPreviewReady(true);
    setLastEvent(`${timestamp()}: preview streaming ${event.nativeEvent.width}x${event.nativeEvent.height}`);
  }, []);

  const handleAnalysisReady = useCallback((event: { nativeEvent: { width: number; height: number } }) => {
    setAnalysisReady(true);
    setLastEvent(`${timestamp()}: analysis ready ${event.nativeEvent.width}x${event.nativeEvent.height}`);
  }, []);

  const handleCameraBound = useCallback((event: { nativeEvent: { analysisEnabled: boolean } }) => {
    setNativeBinding(event.nativeEvent.analysisEnabled ? 'bound: preview + analysis' : 'bound: preview only');
    setLastEvent(`${timestamp()}: camera bound`);
  }, []);

  const handleCameraError = useCallback((event: { nativeEvent: { message: string } }) => {
    setError(event.nativeEvent.message);
    setLastEvent(`${timestamp()}: camera error: ${event.nativeEvent.message}`);
  }, []);

  const handleExpoCameraReady = useCallback(() => {
    setPreviewReady(true);
    setLastEvent(`${timestamp()}: Expo Camera onCameraReady`);
  }, []);

  const handleExpoCameraError = useCallback((event: { message?: string }) => {
    setError(event.message ?? 'Expo Camera mount error');
    setLastEvent(`${timestamp()}: Expo Camera mount error: ${event.message ?? 'unknown'}`);
  }, []);

  if (!permission) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <Text style={styles.status}>Loading camera permission status...</Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Camera Lab</Text>
        <Text style={styles.body}>
          Temporary diagnostic modes. Only one camera mode may be active. No frames are saved.
        </Text>

        <View style={[styles.previewContainer, { width: previewWidth, height: previewHeight }]}>
          {mode === 'A' && permission?.granted ? (
            <CameraView
              style={StyleSheet.absoluteFill}
              facing="back"
              onCameraReady={handleExpoCameraReady}
              onMountError={handleExpoCameraError}
            />
          ) : null}
          {mode === 'B' || mode === 'C' ? (
            <LiveCameraView
              ref={cameraRef}
              style={StyleSheet.absoluteFill}
              active={true}
              analysisEnabled={mode === 'C'}
              onFrame={handleFrame}
              onPreviewReady={handlePreviewReady}
              onAnalysisReady={handleAnalysisReady}
              onCameraBound={handleCameraBound}
              onCameraError={handleCameraError}
            />
          ) : null}
        </View>

        <Text style={styles.status}>Mode: {mode}</Text>
        <Text style={styles.body}>Permission: {permission.granted ? 'granted' : 'not granted'}</Text>
        <Text style={styles.body}>Binding: {nativeBinding}</Text>
        <Text style={styles.body}>Preview ready: {previewReady ? 'yes' : 'no'}</Text>
        <Text style={styles.body}>Analysis ready: {analysisReady ? 'yes' : 'no'}</Text>
        <Text style={styles.body}>Analyzed frames: {frameCount}</Text>
        {lastEvent ? <Text style={styles.body}>{lastEvent}</Text> : null}
        {error ? <Text style={styles.error}>{error}</Text> : null}

        <View style={styles.controls}>
          <Pressable style={mode === 'idle' ? styles.primaryButton : styles.disabledButton} onPress={startModeA} disabled={mode !== 'idle'}>
            <Text style={styles.buttonText}>Mode A: Expo preview</Text>
          </Pressable>
          <Pressable style={mode === 'idle' ? styles.primaryButton : styles.disabledButton} onPress={startModeB} disabled={mode !== 'idle'}>
            <Text style={styles.buttonText}>Mode B: CameraX preview</Text>
          </Pressable>
          <Pressable style={mode === 'idle' ? styles.primaryButton : styles.disabledButton} onPress={startModeC} disabled={mode !== 'idle'}>
            <Text style={styles.buttonText}>Mode C: CameraX + analysis</Text>
          </Pressable>
          <Pressable style={mode === 'idle' ? styles.disabledButton : styles.secondaryButton} onPress={stop} disabled={mode === 'idle'}>
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
  title: { fontSize: 22, fontWeight: '700', textAlign: 'center' },
  body: { fontSize: 13, textAlign: 'center', opacity: 0.8 },
  status: { fontSize: 14, textAlign: 'center', fontWeight: '600' },
  error: { fontSize: 13, textAlign: 'center', color: '#B00020', fontWeight: '600' },
  previewContainer: {
    position: 'relative',
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#000000',
  },
  controls: { width: '100%', flexDirection: 'row', gap: 8, justifyContent: 'center', flexWrap: 'wrap' },
  primaryButton: { paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, backgroundColor: '#208AEF', minWidth: 170, alignItems: 'center' },
  secondaryButton: { paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, backgroundColor: '#6C757D', minWidth: 120, alignItems: 'center' },
  disabledButton: { paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, backgroundColor: '#B0BEC5', minWidth: 120, alignItems: 'center' },
  buttonText: { color: '#FFFFFF', fontWeight: '600', fontSize: 14 },
});