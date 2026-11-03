/**
 * On-device diagnostics for the live-camera pipeline (Milestone 5.4F).
 *
 * These counters deliberately separate frame *reception* from frame
 * *processing*: a camera can deliver frames while board detection rejects
 * every one of them. Counters are plain mutable numbers incremented from the
 * frame handlers, and a throttled snapshot drives the HUD so that camera
 * frames never cause a React render each.
 *
 * No image data, timestamps or identifiers are persisted or transmitted.
 */

export interface StageLatencySummary {
  /** Number of samples backing the summary (0 means the stage never ran). */
  count: number;
  medianMs: number | null;
  p95Ms: number | null;
  maxMs: number | null;
}

export interface LiveDiagCounters {
  /** Cumulative frames analysed by the native tracking analyzer. */
  nativeTrackingAnalyzed: number | null;
  /** Cumulative frames skipped by the native tracking analyzer (throttle or busy). */
  nativeTrackingDropped: number | null;
  /** Tracking events actually received by JavaScript. */
  trackingEventsReceived: number;
  /** Tracking events whose grayscale payload failed decoding/validation. */
  trackingDecodeFailures: number;
  /** Motion-tracker update attempts. */
  motionTrackerAttempts: number;
  /** Motion-tracker updates that returned a fresh validated fix. */
  motionTrackerValid: number;
  /** Successful motion-tracker seeds from an accepted board detection. */
  motionTrackerSeeds: number;
  /** Seeds rejected by the tracker (invalid geometry / feature failure). */
  motionTrackerSeedFailures: number;
  /** Accepted detections that could not seed because no fresh tracking frame existed. */
  motionTrackerSeedSkippedNoFrame: number;
  /** Cumulative frames analysed by the native detection analyzer. */
  nativeDetectionAnalyzed: number | null;
  /** Cumulative frames skipped by the native detection analyzer. */
  nativeDetectionDropped: number | null;
  /** Detection events actually received by JavaScript. */
  detectionEventsReceived: number;
  /** Detection events ignored because a previous frame was still processing. */
  detectionSkippedBusy: number;
  /** Detection events whose grayscale payload failed decoding/validation. */
  detectionDecodeFailures: number;
  /** Full board-detection attempts. */
  fullDetectionAttempts: number;
  /** Candidate quadrilaterals produced by full detection. */
  candidatesGenerated: number;
  /** Candidate quadrilaterals accepted by full detection. */
  candidatesAccepted: number;
  /** Completed 64-square sampling operations. */
  squareSamplingOps: number;
  /** Detection frames where sampling was skipped because no corners existed. */
  squareSamplingSkipped: number;
  /** Frames carrying native 64 x 13 square-model evidence. */
  classificationFrames: number;
  /** Frames where native classification was requested but returned no scores. */
  classificationFailures: number;
  /** Boards adopted after passing the legal-position evidence gate. */
  positionAdoptions: number;
  /** Legal moves recognized and applied to the tracked game. */
  movesRecognized: number;
  /** Frames the recognizer reported as uncertain/ambiguous. */
  uncertainObservations: number;
  /** Observation sets discarded because tracking was not trusted. */
  observationsSkippedUntrusted: number;
  /** Detection frames whose overlay geometry was trusted for rendering. */
  trustedOverlayFrames: number;
  /** Grayscale decodes performed (tracking + detection). */
  frameDecodes: number;
  /** Engine searches that produced an accepted result. */
  engineResults: number;
}

export interface LiveDiagRates {
  trackingReceivedPerSec: number | null;
  trackingNativeAnalyzedPerSec: number | null;
  trackingDroppedPerSec: number | null;
  detectionReceivedPerSec: number | null;
  detectionNativeAnalyzedPerSec: number | null;
  detectionDroppedPerSec: number | null;
  motionTrackerUpdatesPerSec: number | null;
}

/** Session-average rates, immune to short-window spikes under congestion. */
export type LiveDiagSessionRates = LiveDiagRates;

export interface LiveDiagLatency {
  decode: StageLatencySummary;
  tracker: StageLatencySummary;
  detection: StageLatencySummary;
  sampling: StageLatencySummary;
  cropPrep: StageLatencySummary;
  classifier: StageLatencySummary;
  guidance: StageLatencySummary;
  moveRecognition: StageLatencySummary;
  engine: StageLatencySummary;
  overlayModel: StageLatencySummary;
}

export interface LiveDiagSnapshot {
  /** Monotonic-ish wall clock timestamp of the snapshot (Date.now). */
  atMs: number;
  /** Interval since the previous snapshot, or null for the first one. */
  intervalMs: number | null;
  /** Time since the diagnostics were reset (session start). */
  sessionElapsedMs: number | null;
  counters: LiveDiagCounters;
  rates: LiveDiagRates;
  /** Totals divided by sessionElapsedMs since reset. */
  sessionRates: LiveDiagSessionRates;
  latency: LiveDiagLatency;
}

export type LivePipelineStageCode =
  | 'no_frames'
  | 'frames_no_board'
  | 'board_not_tracked'
  | 'tracking_lost'
  | 'tracking_holding'
  | 'tracking_no_samples'
  | 'tracking_uncertain'
  | 'tracking_active'
  | 'guidance_active';

export interface LivePipelineStage {
  code: LivePipelineStageCode;
  label: string;
}

export function createEmptyCounters(): LiveDiagCounters {
  return {
    nativeTrackingAnalyzed: null,
    nativeTrackingDropped: null,
    trackingEventsReceived: 0,
    trackingDecodeFailures: 0,
    motionTrackerAttempts: 0,
    motionTrackerValid: 0,
    motionTrackerSeeds: 0,
    motionTrackerSeedFailures: 0,
    motionTrackerSeedSkippedNoFrame: 0,
    nativeDetectionAnalyzed: null,
    nativeDetectionDropped: null,
    detectionEventsReceived: 0,
    detectionSkippedBusy: 0,
    detectionDecodeFailures: 0,
    fullDetectionAttempts: 0,
    candidatesGenerated: 0,
    candidatesAccepted: 0,
    squareSamplingOps: 0,
    squareSamplingSkipped: 0,
    classificationFrames: 0,
    classificationFailures: 0,
    positionAdoptions: 0,
    movesRecognized: 0,
    uncertainObservations: 0,
    observationsSkippedUntrusted: 0,
    trustedOverlayFrames: 0,
    frameDecodes: 0,
    engineResults: 0,
  };
}

/**
 * Median and nearest-rank p95 for a list of non-negative millisecond values.
 * Returns null statistics when no samples exist so callers can never mistake
 * "never ran" for "took 0 ms".
 */
export function summarizeLatency(samples: readonly number[]): StageLatencySummary {
  const values = samples.filter((value) => Number.isFinite(value) && value >= 0);
  if (values.length === 0) {
    return { count: 0, medianMs: null, p95Ms: null, maxMs: null };
  }

  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  const p95Index = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1);

  return {
    count: sorted.length,
    medianMs: round(median),
    p95Ms: round(sorted[p95Index]),
    maxMs: round(sorted[sorted.length - 1]),
  };
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

type StageKey = keyof LiveDiagLatency;

const STAGE_KEYS: readonly StageKey[] = [
  'decode',
  'tracker',
  'detection',
  'sampling',
  'cropPrep',
  'classifier',
  'guidance',
  'moveRecognition',
  'engine',
  'overlayModel',
];

const DEFAULT_WINDOW = 240;

/**
 * Bounded latency windows plus the counters above. Safe to update at frame
 * rate; `snapshot()` is intended to run at HUD rate (2-4 Hz).
 */
export class LiveDiagnostics {
  readonly counters: LiveDiagCounters = createEmptyCounters();
  private readonly windows = new Map<StageKey, number[]>();
  private readonly windowSize: number;
  private lastSnapshotAtMs: number | null = null;
  private startedAtMs: number | null = null;
  private lastCounters: LiveDiagCounters = { ...this.counters };

  constructor(windowSize = DEFAULT_WINDOW) {
    this.windowSize = Math.max(1, Math.floor(windowSize));
    for (const key of STAGE_KEYS) {
      this.windows.set(key, []);
    }
  }

  record(stage: StageKey, ms: number): void {
    if (!Number.isFinite(ms) || ms < 0) {
      return;
    }
    const window = this.windows.get(stage);
    if (!window) {
      return;
    }
    window.push(ms);
    if (window.length > this.windowSize) {
      window.shift();
    }
  }

  /** Latest cumulative native counters, reported with every emitted frame. */
  noteNativeTracking(analyzed: number | undefined, dropped: number | undefined): void {
    if (typeof analyzed === 'number' && Number.isFinite(analyzed)) {
      this.counters.nativeTrackingAnalyzed = analyzed;
    }
    if (typeof dropped === 'number' && Number.isFinite(dropped)) {
      this.counters.nativeTrackingDropped = dropped;
    }
  }

  noteNativeDetection(analyzed: number | undefined, dropped: number | undefined): void {
    if (typeof analyzed === 'number' && Number.isFinite(analyzed)) {
      this.counters.nativeDetectionAnalyzed = analyzed;
    }
    if (typeof dropped === 'number' && Number.isFinite(dropped)) {
      this.counters.nativeDetectionDropped = dropped;
    }
  }

  reset(nowMs?: number): void {
    Object.assign(this.counters, createEmptyCounters());
    for (const key of STAGE_KEYS) {
      this.windows.set(key, []);
    }
    this.startedAtMs = nowMs ?? Date.now();
    this.lastSnapshotAtMs = nowMs ?? null;
    this.lastCounters = { ...this.counters };
  }

  snapshot(nowMs: number = Date.now()): LiveDiagSnapshot {
    const previousAt = this.lastSnapshotAtMs;
    const intervalMs = previousAt === null ? null : Math.max(0, nowMs - previousAt);
    const seconds = intervalMs !== null && intervalMs > 0 ? intervalMs / 1000 : null;

    const rate = (current: number, previous: number): number | null =>
      seconds === null ? null : round((current - previous) / seconds);
    // Native counters are reset when the camera session stops, so the first
    // observation after a session start can legitimately be measured from 0.
    const nativeRate = (current: number | null, previous: number | null): number | null =>
      seconds === null || current === null ? null : round((current - (previous ?? 0)) / seconds);

    const previous = this.lastCounters;
    const rates: LiveDiagRates = {
      trackingReceivedPerSec: rate(
        this.counters.trackingEventsReceived,
        previous.trackingEventsReceived,
      ),
      trackingNativeAnalyzedPerSec: nativeRate(
        this.counters.nativeTrackingAnalyzed,
        previous.nativeTrackingAnalyzed,
      ),
      trackingDroppedPerSec: nativeRate(
        this.counters.nativeTrackingDropped,
        previous.nativeTrackingDropped,
      ),
      detectionReceivedPerSec: rate(
        this.counters.detectionEventsReceived,
        previous.detectionEventsReceived,
      ),
      detectionNativeAnalyzedPerSec: nativeRate(
        this.counters.nativeDetectionAnalyzed,
        previous.nativeDetectionAnalyzed,
      ),
      detectionDroppedPerSec: nativeRate(
        this.counters.nativeDetectionDropped,
        previous.nativeDetectionDropped,
      ),
      motionTrackerUpdatesPerSec: rate(
        this.counters.motionTrackerAttempts,
        previous.motionTrackerAttempts,
      ),
    };

    const latency = {} as LiveDiagLatency;
    for (const key of STAGE_KEYS) {
      latency[key] = summarizeLatency(this.windows.get(key) ?? []);
    }

    const sessionElapsedMs =
      this.startedAtMs === null ? null : Math.max(0, nowMs - this.startedAtMs);
    const sessionSeconds =
      sessionElapsedMs !== null && sessionElapsedMs > 0 ? sessionElapsedMs / 1000 : null;
    const sessionRate = (current: number, available: boolean): number | null =>
      sessionSeconds === null || !available ? null : round(current / sessionSeconds);
    const sessionRates: LiveDiagSessionRates = {
      trackingReceivedPerSec: sessionRate(
        this.counters.trackingEventsReceived,
        sessionSeconds !== null,
      ),
      trackingNativeAnalyzedPerSec: sessionRate(
        this.counters.nativeTrackingAnalyzed ?? 0,
        this.counters.nativeTrackingAnalyzed !== null,
      ),
      trackingDroppedPerSec: sessionRate(
        this.counters.nativeTrackingDropped ?? 0,
        this.counters.nativeTrackingDropped !== null,
      ),
      detectionReceivedPerSec: sessionRate(
        this.counters.detectionEventsReceived,
        sessionSeconds !== null,
      ),
      detectionNativeAnalyzedPerSec: sessionRate(
        this.counters.nativeDetectionAnalyzed ?? 0,
        this.counters.nativeDetectionAnalyzed !== null,
      ),
      detectionDroppedPerSec: sessionRate(
        this.counters.nativeDetectionDropped ?? 0,
        this.counters.nativeDetectionDropped !== null,
      ),
      motionTrackerUpdatesPerSec: sessionRate(
        this.counters.motionTrackerAttempts,
        sessionSeconds !== null,
      ),
    };

    this.lastSnapshotAtMs = nowMs;
    this.lastCounters = { ...this.counters };

    return {
      atMs: nowMs,
      intervalMs,
      sessionElapsedMs,
      counters: { ...this.counters },
      rates,
      sessionRates,
      latency,
    };
  }
}

export function createEmptySnapshot(): LiveDiagSnapshot {
  return {
    atMs: 0,
    intervalMs: null,
    sessionElapsedMs: null,
    counters: createEmptyCounters(),
    rates: {
      trackingReceivedPerSec: null,
      trackingNativeAnalyzedPerSec: null,
      trackingDroppedPerSec: null,
      detectionReceivedPerSec: null,
      detectionNativeAnalyzedPerSec: null,
      detectionDroppedPerSec: null,
      motionTrackerUpdatesPerSec: null,
    },
    sessionRates: {
      trackingReceivedPerSec: null,
      trackingNativeAnalyzedPerSec: null,
      trackingDroppedPerSec: null,
      detectionReceivedPerSec: null,
      detectionNativeAnalyzedPerSec: null,
      detectionDroppedPerSec: null,
      motionTrackerUpdatesPerSec: null,
    },
    latency: {
      decode: summarizeLatency([]),
      tracker: summarizeLatency([]),
      detection: summarizeLatency([]),
      sampling: summarizeLatency([]),
      cropPrep: summarizeLatency([]),
      classifier: summarizeLatency([]),
      guidance: summarizeLatency([]),
      moveRecognition: summarizeLatency([]),
      engine: summarizeLatency([]),
      overlayModel: summarizeLatency([]),
    },
  };
}

/**
 * Current tracker state as seen this frame. Covers both the coordinator
 * (`uncalibrated`) and the motion tracker (`uninitialized`).
 */
export type PipelineTrackingState =
  | 'uncalibrated'
  | 'uninitialized'
  | 'tracking'
  | 'holding'
  | 'lost'
  | null;

/**
 * Classifies what the pipeline is actually doing so the HUD never shows a
 * generic "searching" state that hides a frame-delivery failure.
 *
 * Cumulative counters decide whether a stage was *ever* reached, but the
 * current tracker state takes precedence once tracking has started: a board
 * that just left the frame must not keep showing "guidance active" because an
 * earlier frame was trusted.
 */
export function describePipelineStage(
  snapshot: LiveDiagSnapshot,
  trackingState: PipelineTrackingState = null,
  hasGuidance = false,
): LivePipelineStage {
  const c = snapshot.counters;
  const framesReceived = c.trackingEventsReceived + c.detectionEventsReceived;
  const everTracked =
    c.motionTrackerValid > 0 ||
    c.trustedOverlayFrames > 0 ||
    trackingState === 'tracking' ||
    trackingState === 'holding';

  if (framesReceived === 0) {
    return { code: 'no_frames', label: '1/5 NO FRAMES REACHED JS' };
  }
  if (c.candidatesAccepted === 0) {
    return { code: 'frames_no_board', label: '2/5 FRAMES RECEIVED · NO BOARD CANDIDATE' };
  }
  if (!everTracked) {
    return { code: 'board_not_tracked', label: '3/5 BOARD FOUND · TRACKING NOT INITIALISED' };
  }
  if (
    everTracked &&
    (trackingState === 'lost' ||
      trackingState === 'uncalibrated' ||
      trackingState === 'uninitialized')
  ) {
    return { code: 'tracking_lost', label: '4/5 BOARD LOST · OVERLAY SUPPRESSED' };
  }
  if (trackingState === 'holding') {
    return { code: 'tracking_holding', label: '4/5 TRACKING HOLDING · OVERLAY DEGRADED' };
  }
  if (c.squareSamplingOps === 0) {
    return { code: 'tracking_no_samples', label: '4/5 TRACKING · NO SQUARE SAMPLES' };
  }
  if (hasGuidance && c.trustedOverlayFrames > 0) {
    return { code: 'guidance_active', label: '5/5 GUIDANCE ACTIVE · OVERLAY RENDERED' };
  }
  if (c.uncertainObservations > 0 && c.trustedOverlayFrames === 0) {
    return { code: 'tracking_uncertain', label: '4/5 TRACKING · OBSERVATIONS UNCERTAIN' };
  }
  return { code: 'tracking_active', label: '4/5 TRACKING · AWAITING GUIDANCE' };
}
