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
import {
  LiveDiagnostics,
  createEmptySnapshot,
  describePipelineStage,
  type LiveDiagSnapshot,
  type StageLatencySummary,
} from '@/vision/liveDiagnostics';
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
  // 5.4F diagnostics: counters update at frame rate in a ref (no React
  // render), and a 2 Hz snapshot drives the HUD.
  const diagnosticsRef = useRef<LiveDiagnostics | null>(null);
  if (!diagnosticsRef.current) {
    diagnosticsRef.current = new LiveDiagnostics();
  }
  const [diag, setDiag] = useState<LiveDiagSnapshot>(() => createEmptySnapshot());

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
  // Detection runs at a different resolution than motion tracking (640x480
  // vs 352x288 on the target phone). The motion tracker requires the seed and
  // update frames to share dimensions, so seed it from the latest tracking
  // frame rather than the detection frame.
  const latestTrackingFrameRef = useRef<{
    frame: { width: number; height: number; data: Uint8Array };
    timestamp: number;
  } | null>(null);
  const previousRecognizerStatusRef = useRef<string | null>(null);
  const lastEngineResultCountRef = useRef(0);

  useEffect(() => {
    coordinatorRef.current?.setOrientation(orientation);
  }, [orientation]);

  // HUD refresh at 2 Hz; frame handlers only touch the diagnostics ref.
  useEffect(() => {
    if (!active) {
      return;
    }
    const interval = setInterval(() => {
      setDiag(diagnosticsRef.current!.snapshot(Date.now()));
    }, 500);
    return () => clearInterval(interval);
  }, [active]);

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
      nativeEvent: {
        base64: string;
        width: number;
        height: number;
        rotationDegrees?: number;
        analyzedFrames?: number;
        droppedFrames?: number;
      };
    }) => {
      const diagnostics = diagnosticsRef.current!;
      // Native only emits the next frame after release, so always release.
      if (!active) {
        cameraRef.current?.releaseTrackingFrame?.();
        return;
      }

      const native = event.nativeEvent;
      // Reception counters increment BEFORE decoding or tracking.
      diagnostics.noteNativeTracking(native.analyzedFrames, native.droppedFrames);
      diagnostics.counters.trackingEventsReceived += 1;
      const received = diagnostics.counters.trackingEventsReceived;
      if (received <= 3 || received % 50 === 0) {
        console.log(
          `[LiveCameraDiag] tracking rx=${received} native=${native.analyzedFrames ?? '—'} skipped=${native.droppedFrames ?? '—'} size=${native.width}x${native.height}`,
        );
      }

      try {
        const decodeStart = performance.now();
        const validation = validateGrayscaleFrame(native.base64, native.width, native.height);
        diagnostics.counters.frameDecodes += 1;
        diagnostics.record('decode', performance.now() - decodeStart);
        if (!validation.ok) {
          diagnostics.counters.trackingDecodeFailures += 1;
          return;
        }

        const grayFrame = { width: native.width, height: native.height, data: validation.buffer };
        const now = Date.now();
        latestTrackingFrameRef.current = { frame: grayFrame, timestamp: now };
        diagnostics.counters.motionTrackerAttempts += 1;
        const trackerStart = performance.now();
        const proposal = motionTrackerRef.current.update(grayFrame, now);
        diagnostics.record('tracker', performance.now() - trackerStart);

        const valid = proposal.state === 'tracking' && proposal.corners !== null;
        if (valid) {
          diagnostics.counters.motionTrackerValid += 1;
        }
        // Cache the freshest fix for the lower-rate detection pipeline. A
        // null entry lets the coordinator fall back to its own tracker.
        latestTrackingProposalRef.current = {
          corners: valid && proposal.corners ? { normalized: proposal.corners.normalized } : null,
          confidence: proposal.confidence,
          inliers: proposal.inliers,
          timestamp: now,
        };
      } catch {
        diagnostics.counters.trackingDecodeFailures += 1;
      } finally {
        cameraRef.current?.releaseTrackingFrame?.();
      }
    },
    [active],
  );

  // Lower-rate detection frame handler (~4 fps, higher-res)
  const handleFrame = useCallback(
    async (event: {
      nativeEvent: {
        base64: string;
        width: number;
        height: number;
        rotationDegrees?: number;
        analyzedFrames?: number;
        droppedFrames?: number;
      };
    }) => {
      const diagnostics = diagnosticsRef.current!;
      if (!active) {
        cameraRef.current?.releaseFrame?.();
        return;
      }

      const native = event.nativeEvent;
      // Reception counters increment BEFORE decoding or board detection.
      diagnostics.noteNativeDetection(native.analyzedFrames, native.droppedFrames);
      diagnostics.counters.detectionEventsReceived += 1;
      const received = diagnostics.counters.detectionEventsReceived;
      if (received <= 3 || received % 25 === 0) {
        console.log(
          `[LiveCameraDiag] detection rx=${received} native=${native.analyzedFrames ?? '—'} skipped=${native.droppedFrames ?? '—'} size=${native.width}x${native.height}`,
        );
      }

      if (busyRef.current) {
        diagnostics.counters.detectionSkippedBusy += 1;
        cameraRef.current?.releaseFrame?.();
        return;
      }

      busyRef.current = true;
      const handlerStart = performance.now();
      try {
        const decodeStart = performance.now();
        const validation = validateGrayscaleFrame(native.base64, native.width, native.height);
        diagnostics.counters.frameDecodes += 1;
        diagnostics.record('decode', performance.now() - decodeStart);
        if (!validation.ok) {
          diagnostics.counters.detectionDecodeFailures += 1;
          setStatus(validation.message);
          return;
        }

        const frameWidth = native.width;
        const frameHeight = native.height;
        const grayFrame = { width: frameWidth, height: frameHeight, data: validation.buffer };
        lastFrameInfoRef.current = {
          width: frameWidth,
          height: frameHeight,
          rotationDegrees: native.rotationDegrees ?? 0,
        };

        // 1. Full detection when needed
        let detectedCorners: Point2D[] | null = null;
        let confidence = 0;
        const detectionDue =
          motionTrackerRef.current.needsReseed() ||
          Date.now() - lastDetectionMsRef.current >= 1500;
        if (detectionDue) {
          diagnostics.counters.fullDetectionAttempts += 1;
          const detStart = performance.now();
          const det = await detectBoardFromGrayscaleBuffer(validation.buffer, frameWidth, frameHeight);
          diagnostics.record('detection', performance.now() - detStart);
          lastDetectionMsRef.current = Date.now();
          diagnostics.counters.candidatesGenerated += det.candidateCount ?? 0;
          if (det.success) {
            diagnostics.counters.candidatesAccepted += 1;
            detectedCorners = det.corners;
            confidence = det.confidence;
            const seedFrame = latestTrackingFrameRef.current;
            if (!seedFrame || Date.now() - seedFrame.timestamp > 1000) {
              diagnostics.counters.motionTrackerSeedSkippedNoFrame += 1;
            } else {
              const seeded = motionTrackerRef.current.seed(
                seedFrame.frame,
                det.corners,
                Date.now(),
              );
              if (seeded) {
                diagnostics.counters.motionTrackerSeeds += 1;
              } else {
                diagnostics.counters.motionTrackerSeedFailures += 1;
              }
            }
          }
        }

        const coordinator = coordinatorRef.current;
        const now = Date.now();

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
            const sampleStart = performance.now();
            observations = sampleSquareObservations(grayFrame, pixelCorners, orientation);
            diagnostics.counters.squareSamplingOps += 1;
            diagnostics.record('sampling', performance.now() - sampleStart);
          } else {
            diagnostics.counters.squareSamplingSkipped += 1;
          }
        }

        // 3. Guidance coordinator. Only pass a fresh motion-tracked fix;
        // a stale proposal must never keep an overlay alive.
        if (coordinator) {
          const cached = latestTrackingProposalRef.current;
          const fresh = cached && cached.corners && now - cached.timestamp <= 500;
          const trackingProposal =
            fresh && cached.corners
              ? {
                  corners: cached.corners.normalized,
                  confidence: cached.confidence,
                  inliers: cached.inliers,
                }
              : null;

          const guidanceStart = performance.now();
          const model = await coordinator.processFrame(
            detectedCorners,
            observations,
            false,
            confidence,
            now,
            trackingProposal,
          );
          diagnostics.record('guidance', performance.now() - guidanceStart);
          diagnostics.record('moveRecognition', model.stageTimings.recognizerMs);

          if (model.engineResultCount !== lastEngineResultCountRef.current) {
            lastEngineResultCountRef.current = model.engineResultCount;
            diagnostics.counters.engineResults = model.engineResultCount;
            if (model.engineLatencyMs !== null) {
              diagnostics.record('engine', model.engineLatencyMs);
            }
          }

          if (
            model.recognizerStatus === 'move_detected' &&
            previousRecognizerStatusRef.current !== 'move_detected'
          ) {
            diagnostics.counters.movesRecognized += 1;
          }
          if (model.recognizerStatus === 'uncertain') {
            diagnostics.counters.uncertainObservations += 1;
          }
          if (observations && model.trackingState !== 'tracking') {
            diagnostics.counters.observationsSkippedUntrusted += 1;
          }
          previousRecognizerStatusRef.current = model.recognizerStatus;

          const overlayTrusted =
            model.trackingState === 'tracking' && model.perspectiveGeometry.corners !== null;
          if (overlayTrusted) {
            diagnostics.counters.trustedOverlayFrames += 1;
            diagnostics.record('overlayModel', performance.now() - handlerStart);
          }

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
    diagnosticsRef.current!.reset(Date.now());
    setDiag(createEmptySnapshot());
    startupTokenRef.current += 1;
    clearStartupTimeout();
    setActive(true);
    setStatus('Starting rear camera...');
    motionTrackerRef.current.reset();
    lastDetectionMsRef.current = 0;
    lastFrameInfoRef.current = null;
    latestTrackingProposalRef.current = null;
    latestTrackingFrameRef.current = null;
    previousRecognizerStatusRef.current = null;
    lastEngineResultCountRef.current = 0;
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
    latestTrackingFrameRef.current = null;
    previousRecognizerStatusRef.current = null;
    lastEngineResultCountRef.current = 0;
    // Keep the final interval's numbers visible after stopping.
    setDiag(diagnosticsRef.current!.snapshot(Date.now()));
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
  const pipeline = describePipelineStage(
    diag,
    guidance?.trackingState ?? null,
    Boolean(guidance?.perspectiveGeometry.corners),
  );
  const fmtRate = (value: number | null) => (value === null ? '—' : `${value.toFixed(1)}/s`);
  const sessionSeconds =
    diag.sessionElapsedMs === null ? null : Math.round(diag.sessionElapsedMs / 1000);
  const fmtLatency = (summary: StageLatencySummary) =>
    summary.count === 0 || summary.medianMs === null
      ? '—'
      : `${summary.medianMs}/${summary.p95Ms} (n=${summary.count})`;
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
          <Text style={styles.telemetryText}>Pipeline: {pipeline.label}</Text>
          <Text style={styles.telemetryText}>
            Session {sessionSeconds ?? '—'}s avg: trk {fmtRate(diag.sessionRates.trackingReceivedPerSec)}{' '}
            (native analysed {diag.counters.nativeTrackingAnalyzed ?? '—'} · skipped{' '}
            {diag.counters.nativeTrackingDropped ?? '—'},{' '}
            {fmtRate(diag.sessionRates.trackingDroppedPerSec)}) · det{' '}
            {fmtRate(diag.sessionRates.detectionReceivedPerSec)} (native analysed{' '}
            {diag.counters.nativeDetectionAnalyzed ?? '—'} · skipped{' '}
            {diag.counters.nativeDetectionDropped ?? '—'},{' '}
            {fmtRate(diag.sessionRates.detectionDroppedPerSec)})
          </Text>
          <Text style={styles.telemetryText}>
            Tracker {diag.counters.motionTrackerAttempts} updates · {diag.counters.motionTrackerValid} valid ·
            seeds {diag.counters.motionTrackerSeeds} (fail {diag.counters.motionTrackerSeedFailures}, no frame{' '}
            {diag.counters.motionTrackerSeedSkippedNoFrame}) · detection{' '}
            {diag.counters.fullDetectionAttempts} runs · {diag.counters.candidatesGenerated} candidates ·{' '}
            {diag.counters.candidatesAccepted} accepted
          </Text>
          <Text style={styles.telemetryText}>
            Samples {diag.counters.squareSamplingOps} (skipped {diag.counters.squareSamplingSkipped}) · moves{' '}
            {diag.counters.movesRecognized} · uncertain {diag.counters.uncertainObservations} · untrusted{' '}
            {diag.counters.observationsSkippedUntrusted} · overlay {diag.counters.trustedOverlayFrames}
          </Text>
          <Text style={styles.telemetryText}>
            Latency median/p95 ms: decode {fmtLatency(diag.latency.decode)} · trk{' '}
            {fmtLatency(diag.latency.tracker)} · det {fmtLatency(diag.latency.detection)} · samp{' '}
            {fmtLatency(diag.latency.sampling)} · guide {fmtLatency(diag.latency.guidance)} · move{' '}
            {fmtLatency(diag.latency.moveRecognition)} · engine {fmtLatency(diag.latency.engine)} · overlay{' '}
            {fmtLatency(diag.latency.overlayModel)}
          </Text>
          <Text style={styles.telemetryText}>
            Status: {guidance?.recognizerStatus ?? 'ready'} · decode failures T
            {diag.counters.trackingDecodeFailures}/D{diag.counters.detectionDecodeFailures} · busy skips{' '}
            {diag.counters.detectionSkippedBusy} · engine results {diag.counters.engineResults}
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