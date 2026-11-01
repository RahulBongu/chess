import { CameraView, useCameraPermissions, type CameraMountError } from 'expo-camera';
import { Linking } from 'react-native';
import { router, useFocusEffect } from 'expo-router';
import React, { useCallback, useRef, useEffect, useState } from 'react';
import {
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from 'react-native';

import { deleteCachedCapture } from '@/vision/imageLifecycle';
import { resolveCameraAvailability } from '@/camera/availability';
import { detectBoardFromBase64 } from '@/vision/detectBoard';
import type { CandidateDiagnostic } from '@/vision/types';import { SafeAreaView } from 'react-native-safe-area-context';

import {
  NormalizedPoint,
  addCalibrationPoint,
  canAddCalibrationPoint,
  isValidCalibration,
  resetCalibrationPoints,
  undoCalibrationPoint,
} from '@/calibration';

const CORNER_LABELS = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left'] as const;

export default function CameraSetupScreen() {
  const { width } = useWindowDimensions();
  const cameraRef = useRef<CameraView>(null);
  const capturedImageUriRef = useRef<string | null>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraAvailable, setCameraAvailable] = useState<boolean | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraKey, setCameraKey] = useState(0);
  const [isScreenFocused, setIsScreenFocused] = useState(false);
  const [previewSize, setPreviewSize] = useState({ width: 0, height: 0 });
  const [points, setPoints] = useState<NormalizedPoint[]>([]);
  const [statusMessage, setStatusMessage] = useState(
    'Position the whole board in view, then tap Auto-detect board or its four outer corners.',
  );
  const [confirmedPoints, setConfirmedPoints] = useState<NormalizedPoint[] | null>(null);

  const [isDetecting, setIsDetecting] = useState(false);
  const [capturedImage, setCapturedImage] = useState<{ uri: string; width: number; height: number } | null>(null);
  const [detectedCorners, setDetectedCorners] = useState<NormalizedPoint[]>([]);
  const [selectedCorner, setSelectedCorner] = useState<number | null>(null);
  const [detectionMessage, setDetectionMessage] = useState<string | null>(null);
  const [detectionConfidence, setDetectionConfidence] = useState<number | null>(null);
  const [detectionDiagnostics, setDetectionDiagnostics] = useState<CandidateDiagnostic[]>([]);

  const previewWidth = Math.min(width - 24, 480);
  const previewHeight = Math.round(previewWidth * (4 / 3));

  const isHandedOffRef = useRef(false);

  const clearCapturedImage = useCallback((): boolean => {
    const uri = capturedImageUriRef.current;
    capturedImageUriRef.current = null;
    setCapturedImage(null);
    return deleteCachedCapture(uri);
  }, []);

  useFocusEffect(
    useCallback(() => {
      setIsScreenFocused(true);
      isHandedOffRef.current = false;

      return () => {
        // Expo Camera recommends unmounting the preview whenever its screen is
        // not focused. On Android, mounting a preview off-screen leaves CameraX
        // without a surface, so the preview never becomes ready when the tab is
        // opened later. Unmounting also releases the camera while on other tabs.
        setIsScreenFocused(false);
        setCameraReady(false);
        setCameraError(null);

        // If ownership was explicitly transferred to Board Review, do not delete the file!
        if (isHandedOffRef.current) {
          isHandedOffRef.current = false;
          return;
        }

        if (!capturedImageUriRef.current) {
          return;
        }

        const cleared = clearCapturedImage();
        setDetectedCorners([]);
        setSelectedCorner(null);
        setDetectionMessage(null);
        setDetectionConfidence(null);
        setDetectionDiagnostics([]);
        setStatusMessage(
          cleared
            ? 'Temporary capture cleared. Position the whole board in view, then tap Auto-detect board.'
            : 'The temporary capture may remain in the app cache. Position the whole board in view, then tap Auto-detect board.',
        );
      };
    }, [clearCapturedImage]),
  );

  useEffect(() => {
    if (!permission?.granted) {
      return;
    }

    let active = true;
    resolveCameraAvailability(Platform.OS, () => CameraView.isAvailableAsync()).then((available) => {
      // `null` means the platform is native, where availability is reported by
      // the CameraView mount callbacks rather than the web-only probe.
      if (active && available !== null) {
        setCameraAvailable(available);
      }
    });

    return () => {
      active = false;
    };
  }, [permission?.granted]);

  const handleEnableCamera = async () => {
    setStatusMessage('Requesting camera permission...');
    try {
      const result = await requestPermission();
      if (result.granted) {
        setStatusMessage('Position the whole board in view, then tap its four outer corners.');
      } else {
        setStatusMessage('Camera permission is required for board calibration.');
      }
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Unable to request camera permission.');
    }
  };

  const handleOpenSettings = async () => {
    try {
      await Linking.openSettings();
    } catch {
      setStatusMessage('Open your device settings to enable camera access.');
    }
  };

  const handlePreviewLayout = (event: { nativeEvent: { layout: { width: number; height: number } } }) => {
    const { width: layoutWidth, height: layoutHeight } = event.nativeEvent.layout;
    setPreviewSize({ width: layoutWidth, height: layoutHeight });
  };

  const handlePreviewPress = (event: { nativeEvent: { locationX: number; locationY: number } }) => {
    if (previewSize.width <= 0 || previewSize.height <= 0) {
      return;
    }

    const point = {
      x: event.nativeEvent.locationX / previewSize.width,
      y: event.nativeEvent.locationY / previewSize.height,
    };

    const validation = canAddCalibrationPoint(points, point);
    if (!validation.valid) {
      setStatusMessage(validation.reason ?? 'That point cannot be used.');
      return;
    }

    const nextPoints = addCalibrationPoint(points, point);
    if (!nextPoints) {
      setStatusMessage('That point cannot be used.');
      return;
    }

    setPoints(nextPoints);
    setConfirmedPoints(null);
    const nextCorner = CORNER_LABELS[nextPoints.length] ?? 'all corners';
    setStatusMessage(
      nextPoints.length < 4
        ? `${nextPoints.length} corner${nextPoints.length === 1 ? '' : 's'} selected. Next: ${nextCorner}.`
        : 'All four corners selected. Review and confirm calibration.',
    );
  };

  const handleUndo = () => {
    const nextPoints = undoCalibrationPoint(points);
    setPoints(nextPoints);
    setConfirmedPoints(null);
    setStatusMessage(
      nextPoints.length === 0
        ? 'Calibration reset. Tap the top-left corner to begin.'
        : `${nextPoints.length} corner${nextPoints.length === 1 ? '' : 's'} selected. Next: ${CORNER_LABELS[nextPoints.length]}.`,
    );
  };

  const handleReset = () => {
    setPoints(resetCalibrationPoints());
    setConfirmedPoints(null);
    setStatusMessage('Calibration reset. Tap the top-left corner to begin.');
  };

  const handleConfirm = () => {
    const validation = isValidCalibration(points);
    if (!validation.valid) {
      setStatusMessage(validation.reason ?? 'Calibration is invalid.');
      return;
    }

    setConfirmedPoints(points);
    setStatusMessage('Calibration accepted. The four corners are stored in memory for this session.');
  };

  const handleCameraReady = () => {
    setCameraReady(true);
    setCameraError(null);
    setStatusMessage((current) =>
      current === 'Camera preview could not start. Check camera access, then retry.'
        ? 'Camera preview is ready. Position the whole board in view.'
        : current,
    );
  };

  const handleCameraMountError = (event: CameraMountError) => {
    const detail = event.message?.trim();
    setCameraReady(false);
    setCameraError(detail ? `Camera preview failed: ${detail}` : 'Camera preview failed to start.');
    setStatusMessage('Camera preview could not start. Check camera access, then retry.');
  };

  const handleRetryCamera = () => {
    setCameraError(null);
    setCameraReady(false);
    setCameraKey((value) => value + 1);
  };

  const handleAutoDetect = async () => {
    if (isDetecting || !cameraReady || !cameraRef.current) {
      return;
    }

    setIsDetecting(true);
    setStatusMessage('Capturing image for board detection...');
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.8,
      });

      if (!photo?.uri) {
        throw new Error('The camera did not return image data.');
      }

      clearCapturedImage();
      capturedImageUriRef.current = photo.uri;
      setCapturedImage({ uri: photo.uri, width: photo.width, height: photo.height });
      setStatusMessage('Processing image locally...');

      if (!photo.base64) {
        clearCapturedImage();
        throw new Error('The camera did not return image data.');
      }

      const result = await detectBoardFromBase64(photo.base64);
      setDetectionMessage(result.message);
      setDetectionConfidence(result.confidence);
      setDetectionDiagnostics(result.diagnostics ?? []);

      if (result.success) {
        setDetectedCorners(result.corners);
        setSelectedCorner(null);
        setStatusMessage(result.message);
      } else {
        setDetectedCorners([]);
        setStatusMessage(result.message);
      }
    } catch (error) {
      setDetectionMessage(error instanceof Error ? error.message : 'Board detection failed.');
      setStatusMessage(error instanceof Error ? error.message : 'Board detection failed.');
    } finally {
      setIsDetecting(false);
    }
  };

  const handleRetake = () => {
    const cleared = clearCapturedImage();
    setDetectedCorners([]);
    setSelectedCorner(null);
    setDetectionMessage(null);
    setDetectionConfidence(null);
    setDetectionDiagnostics([]);
    setCameraReady(false);
    setStatusMessage(
      cleared
        ? 'Temporary capture cleared. Position the whole board in view, then tap Auto-detect board.'
        : 'Position the whole board in view, then tap Auto-detect board. The previous capture may remain in the app cache.',
    );
  };

  const handleImagePress = (event: { nativeEvent: { locationX: number; locationY: number } }) => {
    if (!capturedImage || selectedCorner === null) {
      return;
    }

    const displayWidth = Math.min(width - 24, 480);
    const displayHeight = Math.round(displayWidth * (capturedImage.height / capturedImage.width));
    const point = {
      x: Math.min(1, Math.max(0, event.nativeEvent.locationX / displayWidth)),
      y: Math.min(1, Math.max(0, event.nativeEvent.locationY / displayHeight)),
    };

    const next = [...detectedCorners];
    next[selectedCorner] = point;
    setDetectedCorners(next);
    setSelectedCorner(null);
  };

  const handleConfirmDetection = () => {
    const validation = isValidCalibration(detectedCorners);
    if (!validation.valid) {
      setStatusMessage(validation.reason ?? 'Calibration is invalid.');
      return;
    }

    setConfirmedPoints(detectedCorners);
    setCameraReady(false);
    setStatusMessage(
      'Calibration accepted from detected board corners. Tap Review & Reconstruct Position to proceed.',
    );
  };

  const handleReturnToBoard = () => {
    const cleared = clearCapturedImage();
    if (!cleared) {
      setStatusMessage('The temporary capture may remain in the app cache.');
    }
    router.back();
  };

  const renderOverlay = () => {
    if (previewSize.width <= 0 || previewSize.height <= 0) {
      return null;
    }

    const pixelPoints = points.map((point) => ({
      x: point.x * previewSize.width,
      y: point.y * previewSize.height,
    }));

    const segments: React.ReactElement[] = [];
    for (let index = 0; index < pixelPoints.length; index += 1) {
      const nextIndex = index + 1;
      if (nextIndex >= pixelPoints.length) {
        break;
      }
      const from = pixelPoints[index];
      const to = pixelPoints[nextIndex];
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const length = Math.hypot(dx, dy);
      const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
      const midX = (from.x + to.x) / 2;
      const midY = (from.y + to.y) / 2;
      segments.push(
        <View
          key={`segment-${index}`}
          style={[
            styles.segment,
            {
              left: midX - length / 2,
              top: midY - 1,
              width: length,
              transform: [{ rotate: `${angle}deg` }],
            },
          ]}
        />,
      );
    }

    if (pixelPoints.length === 4) {
      const from = pixelPoints[3];
      const to = pixelPoints[0];
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const length = Math.hypot(dx, dy);
      const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
      const midX = (from.x + to.x) / 2;
      const midY = (from.y + to.y) / 2;
      segments.push(
        <View
          key="segment-close"
          style={[
            styles.segment,
            {
              left: midX - length / 2,
              top: midY - 1,
              width: length,
              transform: [{ rotate: `${angle}deg` }],
            },
          ]}
        />,
      );
    }

    return (
      <>
        {segments}
        {pixelPoints.map((point, index) => (
          <View
            key={`point-${index}`}
            style={[
              styles.marker,
              {
                left: point.x - 14,
                top: point.y - 14,
              },
            ]}>
            <Text style={styles.markerText}>{index + 1}</Text>
          </View>
        ))}
      </>
    );
  };

  if (!permission) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.centered}>
          <Text style={styles.title}>Camera / Board Setup</Text>
          <Text style={styles.body}>Loading camera permission status...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!permission.granted) {
    const permanentlyDenied = !permission.canAskAgain;
    return (
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.content}>
          <Text style={styles.title}>Camera / Board Setup</Text>
          <Text style={styles.body}>
            Camera access is requested only when you tap Enable camera. A still image is captured only when you tap Auto-detect board. The image is processed locally on this device by OpenCV; it is not uploaded or saved to your gallery. The temporary JPEG may remain in the app&apos;s private cache while you review it, and the app deletes it after retake, confirmation, or leaving this screen.
          </Text>
          {permanentlyDenied ? (
            <>
              <Text style={styles.warning}>Camera permission was denied. Enable it in system settings, then return to this screen.</Text>
              <Pressable style={styles.primaryButton} onPress={handleOpenSettings} accessibilityRole="button">
                <Text style={styles.buttonText}>Open settings</Text>
              </Pressable>
            </>
          ) : (
            <Pressable style={styles.primaryButton} onPress={handleEnableCamera} accessibilityRole="button">
              <Text style={styles.buttonText}>Enable camera</Text>
            </Pressable>
          )}
          <Pressable style={styles.secondaryButton} onPress={handleReturnToBoard} accessibilityRole="button">
            <Text style={styles.secondaryButtonText}>Return to chessboard</Text>
          </Pressable>
        </ScrollView>
      </SafeAreaView>
    );
  }

  if (cameraAvailable === false) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.content}>
          <Text style={styles.title}>Camera / Board Setup</Text>
          <Text style={styles.warning}>No camera is available in this browser.</Text>
          <Pressable style={styles.secondaryButton} onPress={handleReturnToBoard} accessibilityRole="button">
            <Text style={styles.secondaryButtonText}>Return to chessboard</Text>
          </Pressable>
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <Text style={styles.title}>Camera / Board Setup</Text>
        <Text style={styles.body}>
          Position the phone so the entire 8 × 8 board is visible. Auto-detect captures one still image and processes it locally on this device. It checks the rectangular boundary and the alternating-square pattern inside it. You can also tap the outer corners in this order: top-left, top-right, bottom-right, bottom-left.
        </Text>

        {capturedImage ? (
          <View style={{ gap: 8, alignItems: 'center' }}>
            <View
              style={[
                styles.previewContainer,
                {
                  width: Math.min(width - 24, 480),
                  height: Math.round(Math.min(width - 24, 480) * (capturedImage.height / capturedImage.width)),
                },
              ]}>
              <Image
                source={{ uri: capturedImage.uri }}
                style={StyleSheet.absoluteFill}
                resizeMode="contain"
              />
              <Pressable style={StyleSheet.absoluteFill} onPress={handleImagePress} accessibilityLabel="Detected board image" />
              {(() => {
                const displayWidth = Math.min(width - 24, 480);
                const displayHeight = Math.round(displayWidth * (capturedImage.height / capturedImage.width));
                const pixelPoints = detectedCorners.map((point) => ({
                  x: point.x * displayWidth,
                  y: point.y * displayHeight,
                }));
                const segments: React.ReactElement[] = [];
                for (let index = 0; index < pixelPoints.length; index += 1) {
                  const nextIndex = (index + 1) % pixelPoints.length;
                  if (pixelPoints.length < 2 || (pixelPoints.length !== 4 && nextIndex === 0)) {
                    continue;
                  }
                  if (index === pixelPoints.length - 1 && pixelPoints.length !== 4) {
                    continue;
                  }
                  const from = pixelPoints[index];
                  const to = pixelPoints[nextIndex];
                  const dx = to.x - from.x;
                  const dy = to.y - from.y;
                  const length = Math.hypot(dx, dy);
                  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
                  const midX = (from.x + to.x) / 2;
                  const midY = (from.y + to.y) / 2;
                  segments.push(
                    <View
                      key={`detected-segment-${index}`}
                      style={[
                        styles.segment,
                        {
                          left: midX - length / 2,
                          top: midY - 1,
                          width: length,
                          transform: [{ rotate: `${angle}deg` }],
                        },
                      ]}
                    />,
                  );
                }
                return (
                  <>
                    {segments}
                    {pixelPoints.map((point, index) => (
                      <Pressable
                        key={`detected-point-${index}`}
                        onPress={() => setSelectedCorner(index)}
                        style={[
                          styles.marker,
                          selectedCorner === index && styles.selectedMarker,
                          {
                            left: point.x - 14,
                            top: point.y - 14,
                          },
                        ]}>
                        <Text style={styles.markerText}>{index + 1}</Text>
                      </Pressable>
                    ))}
                  </>
                );
              })()}
            </View>
            {detectionMessage ? <Text style={styles.status}>{detectionMessage}</Text> : null}
            {detectionConfidence !== null ? (
              <Text style={styles.status}>
                Detection score: {detectionConfidence.toFixed(2)} (heuristic quality, not a probability)
              </Text>
            ) : null}
            {detectionDiagnostics.length > 0 ? (
              <Text style={styles.body}>
                {detectionDiagnostics.length} candidate{detectionDiagnostics.length === 1 ? '' : 's'} evaluated. Best: boundary{' '}
                {detectionDiagnostics[0].boundaryScore.toFixed(2)}, pattern {detectionDiagnostics[0].pattern.score.toFixed(2)}, fit{' '}
                {detectionDiagnostics[0].pattern.fit.toFixed(2)}, alternation{' '}
                {detectionDiagnostics[0].pattern.alternation.toFixed(2)}, contrast{' '}
                {detectionDiagnostics[0].pattern.contrast.toFixed(1)}, noise{' '}
                {detectionDiagnostics[0].pattern.noise.toFixed(1)}, sampling scale{' '}
                {detectionDiagnostics[0].samplingScale.toFixed(2)}, orientation{' '}
                {detectionDiagnostics[0].pattern.orientation ?? 'n/a'}, reason:{' '}
                {detectionDiagnostics[0].pattern.reason ?? 'accepted'}.
              </Text>
            ) : null}
            {detectionDiagnostics.slice(0, 5).map((diagnostic, index) => (
              <Text key={`diagnostic-${index}`} style={styles.body}>
                #{index + 1}: corners{' '}
                {diagnostic.corners
                  .map((corner) => `(${corner.x.toFixed(0)}, ${corner.y.toFixed(0)})`)
                  .join(' ')}; sampling scale {diagnostic.samplingScale.toFixed(2)}; boundary{' '}
                {diagnostic.boundaryScore.toFixed(2)}; pattern {diagnostic.pattern.score.toFixed(2)}; reason:{' '}
                {diagnostic.pattern.reason ?? 'accepted'}.
              </Text>
            ))}
            <Text style={styles.body}>
              Tap a numbered corner to select it, then tap the image to move it. Confirm when the quadrilateral is valid.
            </Text>
            <View style={styles.controls}>
              <Pressable style={styles.secondaryButton} onPress={handleRetake} accessibilityRole="button">
                <Text style={styles.secondaryButtonText}>Retake image</Text>
              </Pressable>
              <Pressable
                style={isValidCalibration(detectedCorners).valid ? styles.primaryButton : styles.disabledButton}
                onPress={handleConfirmDetection}
                disabled={!isValidCalibration(detectedCorners).valid}
                accessibilityRole="button">
                <Text style={styles.buttonText}>Confirm calibration</Text>
              </Pressable>
            </View>
          </View>
        ) : (
          <View
            style={[styles.previewContainer, { width: previewWidth, height: previewHeight }]}
            onLayout={handlePreviewLayout}>
            {isScreenFocused ? (
              <CameraView
                ref={cameraRef}
                key={cameraKey}
                style={StyleSheet.absoluteFill}
                facing="back"
                onCameraReady={handleCameraReady}
                onMountError={handleCameraMountError}
              />
            ) : null}
            <Pressable style={StyleSheet.absoluteFill} onPress={handlePreviewPress} accessibilityLabel="Calibration preview" />
            {renderOverlay()}
            {isScreenFocused && !cameraReady && !cameraError ? (
              <View style={styles.previewStatus}>
                <Text style={styles.previewStatusText}>Starting camera...</Text>
              </View>
            ) : null}
            {cameraError ? (
              <View style={styles.previewStatus}>
                <Text style={styles.previewStatusText} numberOfLines={6}>
                  {cameraError}
                </Text>
              </View>
            ) : null}
          </View>
        )}

        {!capturedImage ? (
          <Pressable
            style={isDetecting || !cameraReady ? styles.disabledButton : styles.primaryButton}
            onPress={handleAutoDetect}
            disabled={isDetecting || !cameraReady}
            accessibilityRole="button">
            <Text style={styles.buttonText}>{isDetecting ? 'Detecting...' : 'Auto-detect board'}</Text>
          </Pressable>
        ) : null}

        <Text style={styles.status}>{statusMessage}</Text>

        <View style={styles.cornerProgress}>
          {CORNER_LABELS.map((label, index) => (
            <View key={label} style={styles.cornerProgressItem}>
              <Text style={index < points.length ? styles.cornerDone : styles.cornerPending}>{index + 1}</Text>
              <Text style={styles.cornerLabel}>{label}</Text>
            </View>
          ))}
        </View>

        <View style={styles.controls}>
          <Pressable style={styles.secondaryButton} onPress={handleUndo} disabled={points.length === 0} accessibilityRole="button">
            <Text style={points.length === 0 ? styles.disabledText : styles.secondaryButtonText}>Undo last corner</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={handleReset} disabled={points.length === 0} accessibilityRole="button">
            <Text style={points.length === 0 ? styles.disabledText : styles.secondaryButtonText}>Reset calibration</Text>
          </Pressable>
        </View>

        <Pressable
          style={isValidCalibration(points).valid ? styles.primaryButton : styles.disabledButton}
          onPress={handleConfirm}
          disabled={!isValidCalibration(points).valid}
          accessibilityRole="button">
          <Text style={styles.buttonText}>Confirm calibration</Text>
        </Pressable>

        {cameraError ? (
          <Pressable style={styles.secondaryButton} onPress={handleRetryCamera} accessibilityRole="button">
            <Text style={styles.secondaryButtonText}>Retry camera</Text>
          </Pressable>
        ) : null}

        {confirmedPoints ? (
          <View style={styles.successCard}>
            <Text style={styles.successTitle}>Calibration accepted</Text>
            <Text style={styles.successBody}>
              Four corners stored in memory for this session. They reset if the screen or app restarts.
            </Text>
            {confirmedPoints.map((point, index) => (
              <Text key={`${point.x}-${point.y}-${index}`} style={styles.cornerCoordinate}>
                {index + 1}. ({point.x.toFixed(3)}, {point.y.toFixed(3)})
              </Text>
            ))}
            <Pressable
              style={[styles.primaryButton, { marginTop: 8 }]}
              onPress={() => {
                const uriToTransfer = capturedImageUriRef.current ?? capturedImage?.uri;
                if (uriToTransfer) {
                  isHandedOffRef.current = true;
                  capturedImageUriRef.current = null;
                  setCapturedImage(null);
                  router.push({
                    pathname: '/board-review',
                    params: {
                      imageUri: uriToTransfer,
                      corners: confirmedPoints ? JSON.stringify(confirmedPoints) : undefined,
                      imageWidth: capturedImage?.width ? String(capturedImage.width) : undefined,
                      imageHeight: capturedImage?.height ? String(capturedImage.height) : undefined,
                    },
                  });
                } else {
                  router.push('/board-review');
                }
              }}
              accessibilityRole="button">
              <Text style={styles.buttonText}>Review & Reconstruct Position</Text>
            </Pressable>
          </View>
        ) : null}

        <Pressable
          style={styles.secondaryButton}
          onPress={() => router.push('/board-review')}
          accessibilityRole="button">
          <Text style={styles.secondaryButtonText}>Open Manual Position Review</Text>
        </Pressable>

        <Pressable style={styles.secondaryButton} onPress={handleReturnToBoard} accessibilityRole="button">
          <Text style={styles.secondaryButtonText}>Return to chessboard</Text>
        </Pressable>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  content: {
    padding: 16,
    gap: 12,
    alignItems: 'center',
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    gap: 12,
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    textAlign: 'center',
  },
  body: {
    fontSize: 14,
    textAlign: 'center',
    opacity: 0.8,
  },
  warning: {
    fontSize: 14,
    textAlign: 'center',
    color: '#B00020',
    fontWeight: '600',
  },
  status: {
    fontSize: 14,
    textAlign: 'center',
    fontWeight: '600',
  },
  previewContainer: {
    position: 'relative',
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#000000',
  },
  previewStatus: {
    ...StyleSheet.absoluteFill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  previewStatusText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
    paddingHorizontal: 16,
  },
  segment: {
    position: 'absolute',
    height: 3,
    backgroundColor: '#00E5FF',
    borderRadius: 2,
  },
  marker: {
    position: 'absolute',
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#FFD600',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#000000',
  },
  selectedMarker: {
    backgroundColor: '#FF6D00',
    borderColor: '#FFFFFF',
  },
  markerText: {
    color: '#000000',
    fontWeight: '700',
    fontSize: 13,
  },
  cornerProgress: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 8,
  },
  cornerProgressItem: {
    width: '48%',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 8,
    borderRadius: 8,
    backgroundColor: '#F4F4F6',
  },
  cornerDone: {
    width: 24,
    textAlign: 'center',
    fontWeight: '700',
    color: '#1B5E20',
  },
  cornerPending: {
    width: 24,
    textAlign: 'center',
    fontWeight: '700',
    color: '#757575',
  },
  cornerLabel: {
    fontSize: 13,
  },
  controls: {
    width: '100%',
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  primaryButton: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: '#208AEF',
    minWidth: 180,
    alignItems: 'center',
  },
  disabledButton: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: '#B0BEC5',
    minWidth: 180,
    alignItems: 'center',
  },
  secondaryButton: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 10,
    backgroundColor: '#6C757D',
    minWidth: 150,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFFFFF',
    fontWeight: '600',
    fontSize: 14,
  },
  secondaryButtonText: {
    color: '#FFFFFF',
    fontWeight: '600',
    fontSize: 14,
  },
  disabledText: {
    color: '#E0E0E0',
    fontWeight: '600',
    fontSize: 14,
  },
  successCard: {
    width: '100%',
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#E8F5E9',
    gap: 4,
  },
  successTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1B5E20',
  },
  successBody: {
    fontSize: 13,
    opacity: 0.8,
  },
  cornerCoordinate: {
    fontSize: 12,
    fontFamily: 'monospace',
  },
});