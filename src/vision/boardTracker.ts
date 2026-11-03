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

export interface BoardTrackerOptions {
  /** Smoothing factor alpha for EMA. Higher = more responsive, lower = smoother. Default 0.35. */
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
  private isLocked = false;

  private readonly smoothingFactor: number;
  private readonly maxOutlierDistance: number;
  private readonly gracePeriodFrames: number;
  private readonly gracePeriodMs: number;
  private readonly outlierSustainMs: number;
  private readonly minTrackedConfidence: number;

  constructor(options: BoardTrackerOptions = {}) {
    this.smoothingFactor = options.smoothingFactor ?? 0.35;
    this.maxOutlierDistance = options.maxOutlierDistance ?? 0.12;
    this.gracePeriodFrames = options.gracePeriodFrames ?? 8;
    this.gracePeriodMs = options.gracePeriodMs ?? 2000;
    this.outlierSustainMs = options.outlierSustainMs ?? 750;
    this.minTrackedConfidence = options.minTrackedConfidence ?? 0.45;
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
  ): TrackingState {
    if (this.isLocked && this.corners) {
      return this.state;
    }

    const now = nowMs ?? Date.now();

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
        // Normal handheld motion: apply EMA smoothing
        const alpha = this.smoothingFactor;
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
        return this.state;
      }

      // Large displacement detected: could be a sudden hand crossing or intentional pan
      this.outlierSpikeCount += 1;
      if (this.firstOutlierTimeMs === null) {
        this.firstOutlierTimeMs = now;
      }
      this.pendingOutlierCorners = newCorners;

      const outlierDuration = now - this.firstOutlierTimeMs;
      if (outlierDuration >= this.outlierSustainMs || this.outlierSpikeCount >= 3) {
        // Sustained shift across time threshold: adapt to the new board position
        this.corners = newCorners;
        this.outlierSpikeCount = 0;
        this.firstOutlierTimeMs = null;
        this.pendingOutlierCorners = null;
        this.state = 'tracking';
        this.consecutiveLostFrames = 0;
        this.lastObservedTimeMs = now;
      } else {
        // Single spike: hold current position and maintain tracking
        this.state = 'holding';
      }

      return this.state;
    }

    // Neither detection nor a usable motion-tracked proposal was available
    // this frame (or it failed validation).
    this.consecutiveDetectedFrames = 0;
    this.outlierSpikeCount = 0;
    this.firstOutlierTimeMs = null;
    this.pendingOutlierCorners = null;

    if (!this.corners) {
      this.state = 'uncalibrated';
      this.confidence = 0;
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
      return this.state;
    }

    // Beyond grace period: tracking lost
    this.state = 'lost';
    this.confidence = 0;
    this.corners = null;
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
    this.isLocked = false;
  }
}
