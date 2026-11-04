/**
 * Temporal Board Corner Tracker and Stabilizer for Handheld Chess Navigation.
 *
 * Stabilizes board corner estimates across consecutive video frames using adaptive
 * exponential moving average (EMA) smoothing, outlier rejection, convexity enforcement,
 * and an elapsed-time-based occlusion grace period to prevent UI flicker when hands cross the board.
 */

import { isConvexQuadrilateral } from './boardDetection.ts';
import type { Point2D } from './perspectiveOverlay.ts';

export type TrackingState = 'uncalibrated' | 'tracking' | 'holding' | 'lost';

/** Frame-quality hints supplied by the live camera pipeline. */
export interface TrackerFrameQuality {
  /** True when the frame is too blurred for reliable corner evidence. */
  motionBlur?: boolean;
  /** Confidence that the square grid matches the board for this frame (0..1). */
  alignmentConfidence?: number;
}

export interface BoardTrackerOptions {
  /** Smoothing factor alpha for EMA at the nominal 250 ms frame interval. Default 0.35. */
  smoothingFactor?: number;
  /** Maximum normalized corner distance allowed before treating as an outlier. Default 0.12. */
  maxOutlierDistance?: number;
  /** Number of frames to hold last known good corners during detection dropout. Default 8. */
  gracePeriodFrames?: number;
  /** Maximum elapsed time in ms to hold last known good corners during dropout. Default 2000 ms. */
  gracePeriodMs?: number;
  /** Minimum elapsed time in ms an outlier must persist before adapting. Default 750 ms. */
  outlierSustainMs?: number;
  /** Minimum confidence required from the motion tracker before its corners are accepted. Default 0.45. */
  minTrackedConfidence?: number;
  /** Consistent reacquisition frames required after a discontinuity. Default 2. */
  reacquireFrames?: number;
  /** Maximum distance between consecutive reacquisition proposals. Default 0.05. */
  outlierConsistencyDistance?: number;
  /** Minimum grid alignment confidence required to accept corners. Default 0.35. */
  minAlignmentConfidence?: number;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function cornerDistance(
  a: [Point2D, Point2D, Point2D, Point2D],
  b: [Point2D, Point2D, Point2D, Point2D],
): number {
  let maximum = 0;
  for (let index = 0; index < 4; index += 1) {
    maximum = Math.max(
      maximum,
      Math.hypot(a[index].x - b[index].x, a[index].y - b[index].y),
    );
  }
  return maximum;
}

export class BoardTracker {
  private corners: [Point2D, Point2D, Point2D, Point2D] | null = null;
  private state: TrackingState = 'uncalibrated';
  private confidence = 0;
  private consecutiveLostFrames = 0;
  private consecutiveDetectedFrames = 0;
  private outlierSpikeCount = 0;
  private firstOutlierTimeMs: number | null = null;
  private lastObservedTimeMs: number | null = null;
  private pendingOutlierCorners: [Point2D, Point2D, Point2D, Point2D] | null = null;
  private outlierConfirmationCount = 0;
  private holdReason: string | null = null;
  private isLocked = false;

  private readonly smoothingFactor: number;
  private readonly maxOutlierDistance: number;
  private readonly gracePeriodFrames: number;
  private readonly gracePeriodMs: number;
  private readonly outlierSustainMs: number;
  private readonly minTrackedConfidence: number;
  private readonly reacquireFrames: number;
  private readonly outlierConsistencyDistance: number;
  private readonly minAlignmentConfidence: number;

  constructor(options: BoardTrackerOptions = {}) {
    this.smoothingFactor = options.smoothingFactor ?? 0.35;
    this.maxOutlierDistance = options.maxOutlierDistance ?? 0.12;
    this.gracePeriodFrames = options.gracePeriodFrames ?? 8;
    this.gracePeriodMs = options.gracePeriodMs ?? 2000;
    this.outlierSustainMs = options.outlierSustainMs ?? 750;
    this.minTrackedConfidence = options.minTrackedConfidence ?? 0.45;
    this.reacquireFrames = Math.max(2, Math.floor(options.reacquireFrames ?? 2));
    this.outlierConsistencyDistance = options.outlierConsistencyDistance ?? 0.05;
    this.minAlignmentConfidence = options.minAlignmentConfidence ?? 0.35;
  }

  /**
   * Seeds the tracker with known corners (e.g. from assisted calibration or confirmed detection).
   */
  seed(corners: Point2D[], nowMs?: number): boolean {
    if (!corners || corners.length !== 4) {
      return false;
    }

    if (!isConvexQuadrilateral(corners)) {
      return false;
    }

    const now = nowMs ?? Date.now();
    this.corners = [
      { x: corners[0].x, y: corners[0].y },
      { x: corners[1].x, y: corners[1].y },
      { x: corners[2].x, y: corners[2].y },
      { x: corners[3].x, y: corners[3].y },
    ];
    this.state = 'tracking';
    this.confidence = 1.0;
    this.consecutiveLostFrames = 0;
    this.consecutiveDetectedFrames = 1;
    this.outlierSpikeCount = 0;
    this.firstOutlierTimeMs = null;
    this.lastObservedTimeMs = now;
    this.pendingOutlierCorners = null;
    this.outlierConfirmationCount = 0;
    this.holdReason = null;
    return true;
  }

  /**
   * Locks current corners to freeze tracking adjustments during sensitive play.
   */
  setLocked(locked: boolean): void {
    this.isLocked = locked;
  }

  getIsLocked(): boolean {
    return this.isLocked;
  }

  /**
   * Updates tracking state.
   *
   * @param detectedCorners Corners from the full detector this frame, or null.
   * @param rawConfidence Detector confidence in 0..1.
   * @param nowMs Injected monotonic time (tests / deterministic playback).
   * @param trackedCorners Corners proposed by the on-device motion tracker
   *   (fresh for THIS frame, already validated by it), or null. Used only
   *   when no usable detection is present.
   * @param trackedConfidence Motion tracker quality in 0..1; proposals below
   *   `minTrackedConfidence` are ignored.
   * @param quality Frame-quality hints. Blurred or misaligned frames are never
   *   allowed to move the corners; the tracker holds the last good fix.
   *
   * A held quadrilateral is never presented as fresh tracking: when neither
   * input yields a usable proposal the tracker degrades to HOLDING (within
   * the grace window) or LOST, and the caller must suppress confident
   * overlays and recognizer updates.
   */
  update(
    detectedCorners: Point2D[] | null,
    rawConfidence = 0,
    nowMs?: number,
    trackedCorners: Point2D[] | null = null,
    trackedConfidence = 0,
    quality: TrackerFrameQuality = {},
  ): TrackingState {
    if (this.isLocked && this.corners) {
      return this.state;
    }

    const hasExplicitTime = typeof nowMs === 'number';
    const now = nowMs ?? Date.now();

    const degradedReason =
      quality.motionBlur === true
        ? 'motion blur'
        : quality.alignmentConfidence !== undefined &&
            Number.isFinite(quality.alignmentConfidence) &&
            quality.alignmentConfidence < this.minAlignmentConfidence
          ? 'grid misaligned'
          : null;

    if (degradedReason) {
      return this.applyQualityHold(degradedReason, now, hasExplicitTime);
    }

    const isValidDetected =
      detectedCorners &&
      detectedCorners.length === 4 &&
      isConvexQuadrilateral(detectedCorners);

    const isValidTracked =
      !isValidDetected &&
      trackedCorners &&
      trackedCorners.length === 4 &&
      isConvexQuadrilateral(trackedCorners) &&
      trackedConfidence >= this.minTrackedConfidence;

    // Effective fresh observation for this frame: detection preferred,
    // motion-tracked proposal as fallback (quality-capped, same gates).
    const observedCorners: Point2D[] | null = isValidDetected
      ? detectedCorners
      : isValidTracked
        ? trackedCorners
        : null;
    const observedConfidence = isValidDetected
      ? rawConfidence
      : Math.min(trackedConfidence, 0.75);

    const isValidCandidate = observedCorners !== null;

    if (isValidCandidate && observedCorners) {
      const newCorners: [Point2D, Point2D, Point2D, Point2D] = [
        { x: observedCorners[0].x, y: observedCorners[0].y },
        { x: observedCorners[1].x, y: observedCorners[1].y },
        { x: observedCorners[2].x, y: observedCorners[2].y },
        { x: observedCorners[3].x, y: observedCorners[3].y },
      ];

      if (!this.corners || this.state === 'lost' || this.state === 'uncalibrated') {
        // Direct initialization from uncalibrated / lost state
        this.corners = newCorners;
        this.confidence = Math.max(0.4, observedConfidence);
        this.state = 'tracking';
        this.consecutiveLostFrames = 0;
        this.consecutiveDetectedFrames = 1;
        this.outlierSpikeCount = 0;
        this.firstOutlierTimeMs = null;
        this.lastObservedTimeMs = now;
        this.pendingOutlierCorners = null;
        this.outlierConfirmationCount = 0;
        this.holdReason = null;
        return this.state;
      }

      // Compute max displacement across all 4 vertices
      let maxDist = 0;
      for (let i = 0; i < 4; i += 1) {
        const dx = newCorners[i].x - this.corners[i].x;
        const dy = newCorners[i].y - this.corners[i].y;
        const dist = Math.hypot(dx, dy);
        if (dist > maxDist) {
          maxDist = dist;
        }
      }

      if (maxDist <= this.maxOutlierDistance) {
        // Ordinary handheld motion: smooth with a time-aware EMA. A fixed
        // alpha would over-smooth a fast pan and lag a slow drift, so the
        // per-frame factor is derived from elapsed time when available.
        const alpha = hasExplicitTime
          ? clamp(
              1 - Math.pow(1 - this.smoothingFactor, this.elapsedSinceObserved(now) / 250),
              0.08,
              0.9,
            )
          : this.smoothingFactor;
        const smoothed: [Point2D, Point2D, Point2D, Point2D] = [
          {
            x: (1 - alpha) * this.corners[0].x + alpha * newCorners[0].x,
            y: (1 - alpha) * this.corners[0].y + alpha * newCorners[0].y,
          },
          {
            x: (1 - alpha) * this.corners[1].x + alpha * newCorners[1].x,
            y: (1 - alpha) * this.corners[1].y + alpha * newCorners[1].y,
          },
          {
            x: (1 - alpha) * this.corners[2].x + alpha * newCorners[2].x,
            y: (1 - alpha) * this.corners[2].y + alpha * newCorners[2].y,
          },
          {
            x: (1 - alpha) * this.corners[3].x + alpha * newCorners[3].x,
            y: (1 - alpha) * this.corners[3].y + alpha * newCorners[3].y,
          },
        ];

        if (isConvexQuadrilateral(smoothed)) {
          this.corners = smoothed;
        } else {
          this.corners = newCorners;
        }

        this.confidence = (1 - alpha) * this.confidence + alpha * Math.max(0.5, observedConfidence);
        this.state = 'tracking';
        this.consecutiveLostFrames = 0;
        this.consecutiveDetectedFrames += 1;
        this.outlierSpikeCount = 0;
        this.firstOutlierTimeMs = null;
        this.lastObservedTimeMs = now;
        this.pendingOutlierCorners = null;
        this.outlierConfirmationCount = 0;
        this.holdReason = null;
        return this.state;
      }

      // Large displacement detected: could be a sudden hand crossing or an
      // intentional pan. Adapt only after two consecutive proposals agree on
      // the new location (or after the sustained-time fallback), never on a
      // single blown-up frame.
      const consistentWithPending =
        this.pendingOutlierCorners !== null &&
        cornerDistance(newCorners, this.pendingOutlierCorners) <=
          this.outlierConsistencyDistance;

      if (consistentWithPending) {
        this.outlierConfirmationCount += 1;
        this.outlierSpikeCount += 1;
      } else {
        this.pendingOutlierCorners = newCorners;
        this.outlierConfirmationCount = 1;
        this.outlierSpikeCount = 1;
        this.firstOutlierTimeMs = now;
      }

      const outlierDuration = now - (this.firstOutlierTimeMs ?? now);
      if (
        this.outlierConfirmationCount >= this.reacquireFrames ||
        outlierDuration >= this.outlierSustainMs
      ) {
        // Sustained, consistent shift: adopt the new board position.
        this.corners = newCorners;
        this.outlierSpikeCount = 0;
        this.firstOutlierTimeMs = null;
        this.pendingOutlierCorners = null;
        this.outlierConfirmationCount = 0;
        this.state = 'tracking';
        this.consecutiveLostFrames = 0;
        this.consecutiveDetectedFrames += 1;
        this.lastObservedTimeMs = now;
        this.confidence = Math.max(0.5, observedConfidence);
        this.holdReason = null;
      } else {
        // Single spike: hold current position instead of warping to it.
        this.state = 'holding';
        this.holdReason = `board discontinuity (${this.outlierConfirmationCount}/${this.reacquireFrames})`;
      }

      return this.state;
    }

    // Neither detection nor a usable motion-tracked proposal was available
    // this frame (or it failed validation).
    this.consecutiveDetectedFrames = 0;
    this.outlierSpikeCount = 0;
    this.firstOutlierTimeMs = null;
    this.pendingOutlierCorners = null;
    this.outlierConfirmationCount = 0;

    if (!this.corners) {
      // Stay LOST once the grace period has actually expired. Reporting
      // "uncalibrated" again makes a temporary dropout look like the board was
      // never found and loses the reacquisition signal the UI depends on.
      this.state = this.state === 'lost' ? 'lost' : 'uncalibrated';
      this.confidence = 0;
      this.holdReason = 'no board detection';
      return this.state;
    }

    this.consecutiveLostFrames += 1;
    const elapsedLostMs = this.lastObservedTimeMs !== null ? now - this.lastObservedTimeMs : 0;

    // Grace period evaluation: hold if within both time bound and frame bound
    const withinTimeBound = elapsedLostMs <= this.gracePeriodMs;
    const withinFrameBound = this.consecutiveLostFrames <= this.gracePeriodFrames;

    if (withinTimeBound && withinFrameBound) {
      this.state = 'holding';
      this.confidence = Math.max(0, this.confidence * 0.9);
      this.holdReason = 'board not detected';
      return this.state;
    }

    // Beyond grace period: tracking lost
    this.state = 'lost';
    this.confidence = 0;
    this.corners = null;
    this.holdReason = 'tracking lost';
    return this.state;
  }

  private elapsedSinceObserved(now: number): number {
    if (this.lastObservedTimeMs === null) return 250;
    return Math.max(1, now - this.lastObservedTimeMs);
  }

  private applyQualityHold(
    reason: string,
    now: number,
    hasExplicitTime: boolean,
  ): TrackingState {
    this.consecutiveDetectedFrames = 0;
    this.outlierSpikeCount = 0;
    this.firstOutlierTimeMs = null;
    this.pendingOutlierCorners = null;
    this.outlierConfirmationCount = 0;

    if (!this.corners) {
      this.state = 'uncalibrated';
      this.confidence = 0;
      this.holdReason = reason;
      return this.state;
    }

    this.consecutiveLostFrames += 1;
    const elapsedLostMs = this.lastObservedTimeMs !== null ? now - this.lastObservedTimeMs : 0;
    const withinTimeBound = elapsedLostMs <= this.gracePeriodMs;
    const withinFrameBound = this.consecutiveLostFrames <= this.gracePeriodFrames;

    if (withinTimeBound && withinFrameBound) {
      this.state = 'holding';
      // Decay the fix slightly, but never present it as fresh tracking.
      const decay = hasExplicitTime ? Math.max(0.9, 1 - this.elapsedSinceObserved(now) / 4000) : 0.9;
      this.confidence = Math.max(0, this.confidence * decay);
      this.holdReason = reason;
      return this.state;
    }

    this.state = 'lost';
    this.confidence = 0;
    this.corners = null;
    this.holdReason = `${reason}; tracking lost`;
    return this.state;
  }

  /**
   * Returns the current stabilized corners, or null if uncalibrated/lost.
   */
  getCorners(): [Point2D, Point2D, Point2D, Point2D] | null {
    if (!this.corners) {
      return null;
    }
    return [
      { x: this.corners[0].x, y: this.corners[0].y },
      { x: this.corners[1].x, y: this.corners[1].y },
      { x: this.corners[2].x, y: this.corners[2].y },
      { x: this.corners[3].x, y: this.corners[3].y },
    ];
  }

  getState(): TrackingState {
    return this.state;
  }

  getConfidence(): number {
    return this.confidence;
  }

  getConsecutiveDetectedFrames(): number {
    return this.consecutiveDetectedFrames;
  }

  getConsecutiveLostFrames(): number {
    return this.consecutiveLostFrames;
  }

  getLastObservedTimeMs(): number | null {
    return this.lastObservedTimeMs;
  }

  /** Why the tracker is holding, or null while tracking normally. */
  getHoldReason(): string | null {
    return this.holdReason;
  }

  /** Discontinuity reacquisition progress, for telemetry and tests. */
  getReacquireProgress(): { confirmations: number; required: number } | null {
    if (this.pendingOutlierCorners === null) return null;
    return {
      confirmations: this.outlierConfirmationCount,
      required: this.reacquireFrames,
    };
  }

  reset(): void {
    this.corners = null;
    this.state = 'uncalibrated';
    this.confidence = 0;
    this.consecutiveLostFrames = 0;
    this.consecutiveDetectedFrames = 0;
    this.outlierSpikeCount = 0;
    this.firstOutlierTimeMs = null;
    this.lastObservedTimeMs = null;
    this.pendingOutlierCorners = null;
    this.outlierConfirmationCount = 0;
    this.holdReason = null;
    this.isLocked = false;
  }
}
