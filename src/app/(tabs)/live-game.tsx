import { useCameraPermissions } from 'expo-camera';
import { router } from 'expo-router';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from 'react-native';

import { detectBoardFromGrayscaleBuffer } from '@/vision/detectBoard';
import { validateGrayscaleFrame } from '@/vision/grayscaleFrame';
import { sampleSquareObservations } from '@/vision/squareSampler';
import { BoardMotionTracker } from '@/vision/motionTracker';
import { createFrameToViewTransform } from '@/vision/cameraProjection';
import { LiveCameraView } from '../../../modules/live-camera/src';
import {
  LiveGuidanceCoordinator,
  type LiveGuidanceModel,
} from '@/vision/liveGuidance';
import type { BoardOrientation, Point2D } from '@/vision/perspectiveOverlay';
import type { SquareObservation } from '@/vision/moveRecognizer';

function Segment({
  from,
  to,
  color,
  width = 2,
}: {
  from: Point2D;
  to: Point2D;
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
  const [orientation, setOrientation] = useState<BoardOrientation>('white_bottom');
  const [guidance, setGuidance] = useState<LiveGuidanceModel | null>(null);
  const [status, setStatus] = useState('Start live camera to track the physical board.');
  const [frameCount, setFrameCount] = useState(0);
  const [trackingFrameCount, setTrackingFrameCount] = useState(0);
  const [detectionFrameCount, setDetectionFrameCount] = useState(0);

  const coordinatorRef = useRef<LiveGuidanceCoordinator | null>(null);
  if (!coordinatorRef.current) {
    coordinatorRef.current = new LiveGuidanceCoordinator({ orientation });
  }

  const busyRef = useRef(false);
  const cameraRef = useRef<any>(null);
  const startupTokenRef = useRef(0);
  const startupTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const motionTrackerRef = useRef(new BoardMotionTracker());
  const lastDetectionMsRef = useRef(0);
  const lastFrameInfoRef = useRef<{ width: number; height: number; rotationDegrees: number } | null>(
    null,
  );
  const latestTrackingProposalRef = useRef<{
    corners: { normalized: Point2D[] } | null;
    confidence: number;
    inliers: number | null;
    timestamp: number;
  } | null>(null);

  useEffect(() => {
    coordinatorRef.current?.setOrientation(orientation);
  }, [orientation]);

  useEffect(() => {
    return () => {
      if (startupTimeoutRef.current) {
        clearTimeout(startupTimeoutRef.current);
      }
      motionTrackerRef.current.reset();
    };
  }, []);

  const previewWidth = Math.min(windowWidth - 24, 420);
  const previewHeight = Math.round(previewWidth * (4 / 3));

  // High-rate tracking frame handler (~12.5 fps, low-res)
  const handleTrackingFrame = useCallback(
    async (event: {
      nativeEvent: { base64: string; width: number; height: number; rotationDegrees?: number };
    }) => {
      if (!active) return;
      const frameWidth = event.nativeEvent.width;
      const frameHeight = event.nativeEvent.height;
      const validation = validateGrayscaleFrame(event.nativeEvent.base64, frameWidth, frameHeight);
      if (!validation.ok) return;
      const gray = validation.buffer;
      const grayFrame = { width: frameWidth, height: frameHeight, data: gray };
      const now = Date.now();

      // Update motion tracker on every tracking frame
      const proposal = motionTrackerRef.current.update(grayFrame, now);

      // Cache the latest tracking proposal for the detection pipeline
      latestTrackingProposalRef.current = {
        corners: proposal.corners ? { normalized: proposal.corners.normalized } : null,
        confidence: proposal.confidence,
        inliers: proposal.inliers,
        timestamp: now,
      };

      setTrackingFrameCount((v) => v + 1);
      cameraRef.current?.releaseTrackingFrame?.();
    },
    [active],
  );

  // Lower-rate detection frame handler (~4 fps, higher-res)
  const handleFrame = useCallback(
    async (event: {
      nativeEvent: { base64: string; width: number; height: number; rotationDegrees?: number };
    }) => {
      if (!active || busyRef.current) {
        return;
      }
      busyRef.current = true;
      try {
        const frameWidth = event.nativeEvent.width;
        const frameHeight = event.nativeEvent.height;
        const validation = validateGrayscaleFrame(event.nativeEvent.base64, frameWidth, frameHeight);
        if (!validation.ok) {
          setStatus(validation.message);
          return;
        }
        const gray = validation.buffer;
        const grayFrame = { width: frameWidth, height: frameHeight, data: gray };
        const now = Date.now();
        lastFrameInfoRef.current = {
          width: frameWidth,
          height: frameHeight,
          rotationDegrees: event.nativeEvent.rotationDegrees ?? 0,
        };

        // 1. Full detection when needed
        let detectedCorners: Point2D[] | null = null;
        let confidence = 0;
        const detectionDue =
          motionTrackerRef.current.needsReseed() || now - lastDetectionMsRef.current >= 1500;
        if (detectionDue) {
          const det = await detectBoardFromGrayscaleBuffer(gray, frameWidth, frameHeight);
          lastDetectionMsRef.current = now;
          if (det.success) {
            detectedCorners = det.corners;
            confidence = det.confidence;
            motionTrackerRef.current.seed(grayFrame, det.corners, now);
          }
        }

        setFrameCount((val) => val + 1);
        setDetectionFrameCount((v) => v + 1);

        const coordinator = coordinatorRef.current;

        // 2. Sample 64-square observations from the freshest fix.
        let observations: SquareObservation[] | null = null;
        if (coordinator) {
          const sampleCorners =
            coordinator.getTracker().getCorners() ??
            detectedCorners ??
            latestTrackingProposalRef.current?.corners?.normalized ??
            null;
          if (sampleCorners) {
            const pixelCorners: Point2D[] = sampleCorners.map((pt) => ({
              x: pt.x * frameWidth,
              y: pt.y * frameHeight,
            }));
            observations = sampleSquareObservations(grayFrame, pixelCorners, orientation);
          }
        }

        // 3. Guidance coordinator
        if (coordinator) {
          const trackingProposal = latestTrackingProposalRef.current
            ? {
                corners: latestTrackingProposalRef.current.corners!.normalized,
                confidence: latestTrackingProposalRef.current.confidence,
                inliers: latestTrackingProposalRef.current.inliers,
              }
            : null;

          const model = await coordinator.processFrame(
            detectedCorners,
            observations,
            false,
            confidence,
            now,
            trackingProposal,
          );
          setGuidance(model);
          setStatus(model.instruction);
        }
      } catch (error) {
        setStatus(error instanceof Error ? error.message : 'Live frame processing failed.');
      } finally {
        busyRef.current = false;
        cameraRef.current?.releaseFrame?.();
      }
    },
    [active, orientation],
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
    setStatus('Camera streaming. Aim rear camera at the chessboard.');
  }, [clearStartupTimeout]);

  const handleAnalysisReady = useCallback(() => {
    setAnalysisReady(true);
  }, []);

  const handleCameraError = useCallback(
    (event: { nativeEvent: { message: string } }) => {
      clearStartupTimeout();
      setActive(false);
      setPreviewReady(false);
      setAnalysisReady(false);
      setStatus(`Camera error: ${event.nativeEvent.message}`);
    },
    [clearStartupTimeout],
  );

  const handleStart = async () => {
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        setStatus('Camera permission is required for live chess navigator.');
        return;
      }
    }
    setPreviewReady(false);
    setAnalysisReady(false);
    setFrameCount(0);
    setTrackingFrameCount(0);
    setDetectionFrameCount(0);
    startupTokenRef.current += 1;
    clearStartupTimeout();
    setActive(true);
    setStatus('Starting rear camera...');
    motionTrackerRef.current.reset();
    lastDetectionMsRef.current = 0;
    lastFrameInfoRef.current = null;
    latestTrackingProposalRef.current = null;
    startupTimeoutRef.current = setTimeout(() => {
      if (previewReady) return;
      setActive(false);
      setPreviewReady(false);
      setAnalysisReady(false);
      setStatus('Live camera preview did not start within 5 seconds.');
    }, 5000);
  };

  const handleStop = () => {
    clearStartupTimeout();
    setActive(false);
    setPreviewReady(false);
    setAnalysisReady(false);
    motionTrackerRef.current.reset();
    lastDetectionMsRef.current = 0;
    lastFrameInfoRef.current = null;
    latestTrackingProposalRef.current = null;
    setStatus('Live camera stopped.');
  };

  const toggleOrientation = () => {
    setOrientation((prev) => (prev === 'white_bottom' ? 'black_bottom' : 'white_bottom'));
  };

  const handleGoToReview = () => {
    handleStop();
    router.push({
      pathname: '/board-review',
      params: {
        orientation,
        fen: guidance?.fen,
      },
    });
  };

  if (!permission) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <Text style={styles.status}>Checking camera permission...</Text>
      </SafeAreaView>
    );
  }

  if (!permission.granted && !active) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.centered}>
          <Text style={styles.title}>Live Chess Navigator</Text>
          <Text style={styles.body}>
            Hold phone and aim at your chessboard. Live guidance overlays optimal moves onto physical squares.
          </Text>
          <Pressable style={styles.primaryButton} onPress={handleStart}>
            <Text style={styles.buttonText}>Enable Camera & Start</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  const geometry = guidance?.perspectiveGeometry;
  const trackingDegraded = guidance?.trackingState === 'holding';
  const boundaryColor = trackingDegraded ? '#F59E0B' : '#00E5FF';
  const gridColor = trackingDegraded ? 'rgba(245, 158, 11, 0.4)' : 'rgba(0, 229, 255, 0.45)';

  const frameInfo = lastFrameInfoRef.current;
  const viewTransform = frameInfo
    ? createFrameToViewTransform({
        frameWidth: frameInfo.width,
        frameHeight: frameInfo.height,
        rotationDegrees: frameInfo.rotationDegrees,
        viewWidth: previewWidth,
        viewHeight: previewHeight,
        scaleType: 'FILL_CENTER',
      })
    : null;
  const viewPt = (pt: Point2D): Point2D =>
    viewTransform ? viewTransform.mapNormalized(pt.x, pt.y) : pt;

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content}>
        {/* Top HUD Header */}
        <View style={styles.hudHeader}>
          <View style={styles.badgeRow}>
            <View
              style={[
                styles.statusBadge,
                guidance?.trackingState === 'tracking'
                  ? styles.badgeSuccess
                  : guidance?.trackingState === 'holding'
                    ? styles.badgeWarning
                    : styles.badgeDefault,
              ]}>
              <Text style={styles.badgeText}>
                {guidance?.trackingState ? guidance.trackingState.toUpperCase() : 'SEARCHING'}
              </Text>
            </View>
            <View style={styles.turnBadge}>
              <Text style={styles.turnText}>
                {guidance?.turn === 'w' ? 'WHITE TO MOVE' : 'BLACK TO MOVE'}
              </Text>
            </View>
            {guidance?.evaluationScore !== null && guidance?.evaluationScore !== undefined ? (
              <View style={styles.evalBadge}>
                <Text style={styles.evalText}>
                  {guidance.evaluationScore > 0 ? `+${guidance.evaluationScore}` : `${guidance.evaluationScore}`} cp
                </Text>
              </View>
            ) : null}
          </View>
          <Text style={styles.instructionBanner}>
            {guidance?.instruction ?? status}
          </Text>
        </View>

        {/* Live Camera View with Perspective Overlays */}
        <View style={[styles.previewContainer, { width: previewWidth, height: previewHeight }]}>
          <LiveCameraView
            ref={cameraRef}
            style={StyleSheet.absoluteFill}
            active={active}
            onFrame={handleFrame}
            onTrackingFrame={handleTrackingFrame}
            onPreviewReady={handlePreviewReady}
            onAnalysisReady={handleAnalysisReady}
            onCameraError={handleCameraError}
            trackingEnabled={true}
            trackingTargetWidth={160}
            trackingTargetHeight={120}
            trackingIntervalMs={80}
          />

          {/* Perspective-Correct Overlays */}
          {geometry?.corners ? (
            <View style={StyleSheet.absoluteFill} pointerEvents="none">
              {/* 1. Internal Perspective Grid Lines */}
              {geometry.gridLines?.map((line) => (
                <Segment
                  key={line.key}
                  from={viewPt(line.from)}
                  to={viewPt(line.to)}
                  color={gridColor}
                  width={1}
                />
              ))}

              {/* 2. Outer Board Boundary */}
              {geometry.corners.map((corner, i) => {
                const nextCorner = geometry.corners![(i + 1) % 4];
                return (
                  <Segment
                    key={`outer-${i}`}
                    from={viewPt(corner)}
                    to={viewPt(nextCorner)}
                    color={boundaryColor}
                    width={2.5}
                  />
                );
              })}

              {/* 3. Source Square Perspective Quad Highlight (Glowing Cyan) */}
              {geometry.sourceQuad ? (
                <>
                  {geometry.sourceQuad.map((pt, i) => {
                    const nextPt = geometry.sourceQuad![(i + 1) % 4];
                    return (
                      <Segment
                        key={`src-quad-${i}`}
                        from={viewPt(pt)}
                        to={viewPt(nextPt)}
                        color="#00E5FF"
                        width={3.5}
                      />
                    );
                  })}
                </>
              ) : null}

              {/* 4. Destination Square Perspective Quad Highlight (Glowing Green) */}
              {geometry.destinationQuad ? (
                <>
                  {geometry.destinationQuad.map((pt, i) => {
                    const nextPt = geometry.destinationQuad![(i + 1) % 4];
                    return (
                      <Segment
                        key={`dst-quad-${i}`}
                        from={viewPt(pt)}
                        to={viewPt(nextPt)}
                        color="#00E676"
                        width={3.5}
                      />
                    );
                  })}
                </>
              ) : null}

              {/* 5. Perspective-Projected Move Arrow */}
              {geometry.arrow ? (
                <>
                  {/* Arrow Shaft */}
                  <Segment
                    from={viewPt(geometry.arrow.from)}
                    to={viewPt(geometry.arrow.to)}
                    color="#00E676"
                    width={4}
                  />
                  {/* Arrowhead Left Wing */}
                  <Segment
                    from={viewPt(geometry.arrow.to)}
                    to={viewPt(geometry.arrow.headLeft)}
                    color="#00E676"
                    width={4}
                  />
                  {/* Arrowhead Right Wing */}
                  <Segment
                    from={viewPt(geometry.arrow.to)}
                    to={viewPt(geometry.arrow.headRight)}
                    color="#00E676"
                    width={4}
                  />
                </>
              ) : null}
            </View>
          ) : null}
        </View>

        {/* Action Controls & Navigation */}
        <View style={styles.controlsRow}>
          <Pressable
            style={active ? styles.secondaryButton : styles.primaryButton}
            onPress={active ? handleStop : handleStart}>
            <Text style={styles.buttonText}>{active ? 'Stop Camera' : 'Start Camera'}</Text>
          </Pressable>

          <Pressable style={styles.utilityButton} onPress={toggleOrientation}>
            <Text style={styles.utilityButtonText}>
              {orientation === 'white_bottom' ? 'White Bottom' : 'Black Bottom'}
            </Text>
          </Pressable>

          <Pressable style={styles.utilityButton} onPress={handleGoToReview}>
            <Text style={styles.utilityButtonText}>Review / Fix</Text>
          </Pressable>
        </View>

        {/* Promotion Confirmation (explicit, never automatic) */}
        {guidance?.pendingPromotion ? (
          <View style={styles.promotionCard}>
            <Text style={styles.promotionTitle}>
              Pawn promoted on {guidance.pendingPromotion.to}. Choose the piece:
            </Text>
            <View style={styles.promotionButtons}>
              {(
                [
                  ['q', 'Queen'],
                  ['r', 'Rook'],
                  ['b', 'Bishop'],
                  ['n', 'Knight'],
                ] as const
              ).map(([piece, label]) => (
                <Pressable
                  key={piece}
                  style={styles.promotionButton}
                  onPress={() => {
                    const ok = coordinatorRef.current?.getRecognizer().confirmPromotion(piece);
                    setStatus(
                      ok
                        ? `Promotion confirmed: ${label}.`
                        : 'Promotion confirmation rejected; tap Review / Fix.',
                    );
                  }}>
                  <Text style={styles.buttonText}>{label}</Text>
                </Pressable>
              ))}
            </View>
          </View>
        ) : null}

        {/* Telemetry info */}
        <View style={styles.telemetryCard}>
          <Text style={styles.telemetryText}>
            Engine: {guidance?.engineName ?? 'Chal v2.0.0'} (Local NNUE)
          </Text>
          <Text style={styles.telemetryText}>
            Track: {guidance?.trackingState?.toUpperCase() ?? '—'} · src {guidance?.trackingSource ?? '—'} · inliers{' '}
            {guidance?.trackInliers ?? '—'} · conf {Math.round((guidance?.trackingConfidence ?? 0) * 100)}%
          </Text>
          <Text style={styles.telemetryText}>
            Frames: det {detectionFrameCount} · trk {trackingFrameCount} · Status: {guidance?.recognizerStatus ?? 'ready'}
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0A0E17',
  },
  content: {
    padding: 12,
    alignItems: 'center',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#F0F4F8',
    marginBottom: 8,
  },
  body: {
    fontSize: 14,
    color: '#94A3B8',
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 20,
  },
  hudHeader: {
    width: '100%',
    backgroundColor: '#1E293B',
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
  },
  badgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
    gap: 8,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
  },
  badgeSuccess: {
    backgroundColor: '#059669',
  },
  badgeWarning: {
    backgroundColor: '#D97706',
  },
  badgeDefault: {
    backgroundColor: '#475569',
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  turnBadge: {
    backgroundColor: '#334155',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
  },
  turnText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#E2E8F0',
  },
  evalBadge: {
    backgroundColor: '#1E1B4B',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
  },
  evalText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#A5B4FC',
  },
  instructionBanner: {
    fontSize: 15,
    fontWeight: '700',
    color: '#38BDF8',
    lineHeight: 20,
  },
  previewContainer: {
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#000000',
    marginBottom: 12,
  },
  controlsRow: {
    flexDirection: 'row',
    gap: 8,
    width: '100%',
    justifyContent: 'center',
    marginBottom: 12,
  },
  primaryButton: {
    backgroundColor: '#2563EB',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  secondaryButton: {
    backgroundColor: '#DC2626',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  utilityButton: {
    backgroundColor: '#334155',
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 13,
  },
  utilityButtonText: {
    color: '#E2E8F0',
    fontWeight: '600',
    fontSize: 12,
  },
  status: {
    color: '#CBD5E1',
    fontSize: 13,
    textAlign: 'center',
    marginVertical: 4,
  },
  promotionCard: {
    width: '100%',
    backgroundColor: '#1E293B',
    borderColor: '#38BDF8',
    borderWidth: 1,
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
  },
  promotionTitle: {
    color: '#F0F4F8',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  promotionButtons: {
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
  },
  promotionButton: {
    backgroundColor: '#0E7490',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 8,
  },
  telemetryCard: {
    width: '100%',
    backgroundColor: '#111827',
    padding: 10,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#1F2937',
  },
  telemetryText: {
    color: '#64748B',
    fontSize: 11,
    fontFamily: 'monospace',
  },
});