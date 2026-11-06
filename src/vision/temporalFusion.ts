/**
 * Bounded temporal fusion of per-square 13-class classifier evidence.
 *
 * A single camera frame is not a board observation. Hand shake, mild blur,
 * rolling-shutter distortion and square-crop misalignment all move probability
 * mass between classes for a few frames. This module keeps a short, bounded
 * history of complete classified frames and fuses them per square, weighting
 * frames by measured quality and expiring old evidence so a previous board
 * position cannot outvote a move that has already happened.
 *
 * Design constraints:
 * - The window is measured in time, not just frame count. The live detection
 *   stream is throttled to ~250 ms/frame (see LiveCameraView.frameIntervalMs),
 *   so 1.5 s keeps roughly six frames: enough for the three stable frames the
 *   move recognizer already requires, while a piece move ages out quickly.
 * - Quality is weighted, not guessed: focus, alignment, visibility and
 *   classifier confidence reduce a frame's vote. A badly blurred frame is
 *   rejected rather than averaged into the position.
 * - Occupancy (empty vs occupied) and identity/colour are retained separately
 *   so an ambiguous pawn colour does not masquerade as an empty square.
 * - Frames are never queued without bound. At most `maxFrames` complete
 *   64-square frames are retained.
 */

import type { ChessSquare } from '../chess/ChessGame.ts';
import type { SquareObservation } from './moveRecognizer.ts';
import {
  PIECE_CLASSES,
  classProbabilitiesToTop,
  type PieceClass,
} from './pieceRecognition.ts';

export const FUSION_WINDOW_MS = 2500;
export const FUSION_MAX_FRAMES = 8;
/**
 * Recency half-life for frame weighting. Old frames still contribute (so a
 * noisy frame is averaged out), but their influence decays quickly enough
 * that a move's post-move frames can outweigh pre-move evidence inside the
 * bounded window. At 250 ms/frame this is roughly two frames.
 */
export const FUSION_RECENCY_HALF_LIFE_MS = 500;

/**
 * Normalized focus floor. The score is `variance(Laplacian) / (variance(luma)
 * + 1)`, measured on the detector's grayscale frame. On the repository's
 * FENiT reference image the sharp score is ~0.33, a mild Gaussian blur
 * (sigma 0.8) scores ~0.048, and a strong hand-shake blur (sigma >= 1.5)
 * scores <= 0.007. 0.010 therefore rejects strong blur while keeping mildly
 * soft frames.
 */
export const FUSION_MIN_SHARPNESS = 0.01;
export const FUSION_MIN_ALIGNMENT = 0.35;
export const FUSION_MIN_FRAME_WEIGHT = 0.05;
/**
 * Minimum per-square weight for a square to enter the fused observation set.
 * This must stay below FUSION_MIN_FRAME_WEIGHT: an accepted weak frame still
 * covers every square (as low-quality evidence), otherwise it would silently
 * create missing squares and halt recognition.
 */
export const FUSION_MIN_SQUARE_WEIGHT = 0.04;
export const FUSION_UNCERTAIN_PROBABILITY = 0.55;
export const FUSION_UNCERTAIN_MARGIN = 0.08;

/**
 * Consecutive complete frames that must show the same reliable square delta
 * before the current epoch is discarded. A single frame with two confident
 * classifier flips is far more likely to be noise than a legal move; a real
 * move persists for every subsequent frame until the board changes again.
 */
export const FUSION_TRANSITION_CORROBORATION_FRAMES = 2;
/**
 * Maximum gap between corroborating transition frames. A rejected/blurred
 * frame may sit between them, but a long pause means the candidate is stale
 * and must be re-proposed from fresh evidence.
 */
export const FUSION_TRANSITION_MAX_GAP_MS = 1000;
/** A single legal move changes at most four squares (castling). */
export const FUSION_MAX_TRANSITION_SQUARES = 4;

/** Why a retained evidence epoch was discarded. */
export type FusionEpochResetKind =
  | 'occupancy'
  | 'colour'
  | 'mixed'
  | 'discontinuity';

export interface FusionEpochResetDetail {
  atMs: number;
  kind: FusionEpochResetKind;
  squares: ChessSquare[];
  /** Human-readable deltas such as `e2:white_pawn->empty`. */
  changes: string[];
  corroboratingFrames: number;
}

export interface FusionPendingTransition {
  kind: FusionEpochResetKind;
  squares: ChessSquare[];
  changes: string[];
  /** Accepted frames that carried this exact delta. */
  frames: number;
  ageMs: number;
}

export interface FusionDiscardedTransition {
  atMs: number;
  reason: 'signature_changed' | 'not_corroborated' | 'expired' | 'evidence_expired';
  kind: FusionEpochResetKind;
  squares: ChessSquare[];
  changes: string[];
}

/** AddFrame rejection reasons, kept separate from epoch transitions. */
export interface FusionRejectionCounts {
  incomplete: number;
  occlusion: number;
  stale: number;
  blur: number;
  focus: number;
  alignment: number;
  visibility: number;
  lowWeight: number;
}

/** Evidence quality supplied by the live camera / tracker for one frame. */
export interface FrameQuality {
  /** Normalized focus score (0..1-ish); lower is blurrier. */
  sharpness?: number;
  /** Mean frame luma in 0..255, used for physical-visibility diagnostics. */
  meanLuminance?: number;
  /** Explicit motion-blur flag from a caller that has a better estimator. */
  motionBlur?: boolean;
  /** Confidence that the square grid was aligned to the board for this frame (0..1). */
  alignmentConfidence?: number;
  /** True when a hand or other object occludes a meaningful part of the board. */
  isHandOccluded?: boolean;
  /** Capture timestamp for this frame, if it differs from the processing time. */
  capturedAtMs?: number;
}

export interface FusionFrameResult {
  accepted: boolean;
  reason: string | null;
  weight: number;
  /** True when this frame confirmed a persistent board transition. */
  transitionReset?: boolean;
  transitionKind?: FusionEpochResetKind | null;
  transitionSquares?: ChessSquare[];
}

export interface FusedSquareEvidence {
  square: ChessSquare;
  probabilities: number[];
  /** Probability that the square is occupied (1 - P(empty)). */
  occupancyProbability: number;
  /** Probability mass assigned to White pieces (conditional on occupancy). */
  whiteProbability: number;
  /** Probability mass assigned to Black pieces (conditional on occupancy). */
  blackProbability: number;
  topClass: PieceClass;
  topProbability: number;
  secondProbability: number;
  /** Total quality-weighted evidence behind this square. */
  weight: number;
  /** Number of retained frames that contributed to this square. */
  frames: number;
  lastObservedMs: number;
  uncertain: boolean;
  provenance: 'observed';
}

export interface FusionSnapshot {
  observations: SquareObservation[];
  squares: FusedSquareEvidence[];
  acceptedFrames: number;
  rejectedFrames: number;
  lastRejection: string | null;
  uncertainSquares: ChessSquare[];
  missingSquares: ChessSquare[];
  averageFrameWeight: number;
  /** False while a square has no live evidence in the window. */
  hasFullCoverage: boolean;
  /** Frames discarded because a corroborated board transition was detected. */
  transitionResets: number;
  /** Breakdown of the transitions above by evidence kind. */
  transitionResetsByKind: Record<FusionEpochResetKind, number>;
  /** Most recent corroborated transition, or null. */
  lastEpochReset: FusionEpochResetDetail | null;
  /** Candidate transition that has not yet been corroborated. */
  pendingTransition: FusionPendingTransition | null;
  /** Candidate transitions discarded because they were isolated noise. */
  discardedTransitions: number;
  /** Most recent discarded candidate, or null. */
  lastDiscardedTransition: FusionDiscardedTransition | null;
  /** Accepted frames carrying low-confidence squares whose top class flickered. */
  lowConfidenceFluctuations: number;
  /** Bounded sample of the most recent low-confidence flicker squares. */
  lastLowConfidenceSquares: ChessSquare[];
  /** AddFrame rejection counts by reason since the last reset. */
  rejectionCounts: FusionRejectionCounts;
}

interface FusionFrame {
  timestampMs: number;
  weight: number;
  observations: SquareObservation[];
}

interface TransitionChange {
  fromClass: PieceClass;
  toClass: PieceClass;
  kind: 'occupancy' | 'colour';
}

interface PendingTransition {
  signature: string;
  changes: Map<ChessSquare, TransitionChange>;
  kind: FusionEpochResetKind;
  frames: FusionFrame[];
  startedAtMs: number;
  lastSeenMs: number;
}

export interface TemporalFusionOptions {
  windowMs?: number;
  maxFrames?: number;
  minFrameWeight?: number;
  minSharpness?: number;
  minAlignment?: number;
  minSquareWeight?: number;
  uncertainProbability?: number;
  uncertainMargin?: number;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function isCompleteFrame(observations: SquareObservation[] | null | undefined): boolean {
  if (!observations || observations.length !== 64) return false;
  const seen = new Set<string>();
  for (const observation of observations) {
    const probabilities = observation.classProbabilities;
    if (!probabilities || probabilities.length !== PIECE_CLASSES.length) return false;
    if (!probabilities.every((value) => Number.isFinite(value))) return false;
    if (seen.has(observation.square)) return false;
    seen.add(observation.square);
  }
  return seen.size === 64;
}

function topClassOf(observation: SquareObservation | undefined): PieceClass | null {
  if (!observation?.classProbabilities) return null;
  const top = classProbabilitiesToTop(observation.classProbabilities);
  return top ? PIECE_CLASSES[top.index] : null;
}

function isReliableObservation(observation: SquareObservation): boolean {
  if (observation.uncertain) return false;
  if ((observation.visibility ?? 1) < 0.6) return false;
  const top = observation.classProbabilities
    ? classProbabilitiesToTop(observation.classProbabilities)
    : null;
  if (!top) return false;
  return top.probability >= 0.6 && top.probability - top.secondProbability >= 0.08;
}

/**
 * Keeps a bounded rolling window of complete frame evidence and produces one
 * fused observation per square.
 */
export class TemporalSquareFusion {
  private frames: FusionFrame[] = [];
  private rejectedFrames = 0;
  private lastRejection: string | null = null;
  private transitionResets = 0;
  private transitionResetsByKind: Record<FusionEpochResetKind, number> = {
    occupancy: 0,
    colour: 0,
    mixed: 0,
    discontinuity: 0,
  };
  private lastEpochReset: FusionEpochResetDetail | null = null;
  private pendingTransition: PendingTransition | null = null;
  private discardedTransitions = 0;
  private lastDiscardedTransition: FusionDiscardedTransition | null = null;
  private lowConfidenceFluctuations = 0;
  private lastLowConfidenceSquares: ChessSquare[] = [];
  private rejectionCounts: FusionRejectionCounts = {
    incomplete: 0,
    occlusion: 0,
    stale: 0,
    blur: 0,
    focus: 0,
    alignment: 0,
    visibility: 0,
    lowWeight: 0,
  };
  /**
   * Stable reference position for the current epoch. It is deliberately not
   * `frames[0]`: the anchor survives eviction so one-square-per-frame drift
   * can still be recognised, while pending transitions are corroborated
   * before they can replace it.
   */
  private anchor: Map<ChessSquare, SquareObservation> | null = null;

  private readonly windowMs: number;
  private readonly maxFrames: number;
  private readonly minFrameWeight: number;
  private readonly minSharpness: number;
  private readonly minAlignment: number;
  private readonly minSquareWeight: number;
  private readonly uncertainProbability: number;
  private readonly uncertainMargin: number;

  constructor(options: TemporalFusionOptions = {}) {
    this.windowMs = options.windowMs ?? FUSION_WINDOW_MS;
    this.maxFrames = Math.max(1, Math.floor(options.maxFrames ?? FUSION_MAX_FRAMES));
    this.minFrameWeight = options.minFrameWeight ?? FUSION_MIN_FRAME_WEIGHT;
    this.minSharpness = options.minSharpness ?? FUSION_MIN_SHARPNESS;
    this.minAlignment = options.minAlignment ?? FUSION_MIN_ALIGNMENT;
    this.minSquareWeight = options.minSquareWeight ?? FUSION_MIN_SQUARE_WEIGHT;
    this.uncertainProbability = options.uncertainProbability ?? FUSION_UNCERTAIN_PROBABILITY;
    this.uncertainMargin = options.uncertainMargin ?? FUSION_UNCERTAIN_MARGIN;
  }

  reset(): void {
    this.frames = [];
    this.rejectedFrames = 0;
    this.lastRejection = null;
    this.transitionResets = 0;
    this.transitionResetsByKind = {
      occupancy: 0,
      colour: 0,
      mixed: 0,
      discontinuity: 0,
    };
    this.lastEpochReset = null;
    this.pendingTransition = null;
    this.discardedTransitions = 0;
    this.lastDiscardedTransition = null;
    this.lowConfidenceFluctuations = 0;
    this.lastLowConfidenceSquares = [];
    this.rejectionCounts = {
      incomplete: 0,
      occlusion: 0,
      stale: 0,
      blur: 0,
      focus: 0,
      alignment: 0,
      visibility: 0,
      lowWeight: 0,
    };
    this.anchor = null;
  }

  getFrames(): readonly FusionFrame[] {
    return this.frames;
  }

  /**
   * Adds one complete classified frame. Returns whether the frame was used.
   * Rejected frames never contribute to the fused position.
   */
  addFrame(
    observations: SquareObservation[] | null | undefined,
    quality: FrameQuality = {},
    nowMs: number = Date.now(),
  ): FusionFrameResult {
    this.evict(nowMs);
    this.refreshEpoch(nowMs);

    if (!isCompleteFrame(observations) || !observations) {
      this.reject('incomplete square evidence', 'incomplete');
      return { accepted: false, reason: this.lastRejection, weight: 0 };
    }
    if (quality.isHandOccluded) {
      this.reject('hand occlusion', 'occlusion');
      return { accepted: false, reason: this.lastRejection, weight: 0 };
    }

    const capturedAtMs = quality.capturedAtMs ?? nowMs;
    if (capturedAtMs > nowMs + 50 || nowMs - capturedAtMs > this.windowMs) {
      this.reject('stale frame', 'stale');
      return { accepted: false, reason: this.lastRejection, weight: 0 };
    }
    if (quality.motionBlur) {
      this.reject('motion blur', 'blur');
      return { accepted: false, reason: this.lastRejection, weight: 0 };
    }

    const sharpness = quality.sharpness ?? 1;
    if (!Number.isFinite(sharpness) || sharpness < this.minSharpness) {
      this.reject(
        `low focus (${Number.isFinite(sharpness) ? sharpness.toFixed(3) : 'invalid'})`,
        'focus',
      );
      return { accepted: false, reason: this.lastRejection, weight: 0 };
    }

    const alignment =
      quality.alignmentConfidence === undefined ? 1 : clamp01(quality.alignmentConfidence);
    if (alignment < this.minAlignment) {
      this.reject(`grid alignment too low (${alignment.toFixed(2)})`, 'alignment');
      return { accepted: false, reason: this.lastRejection, weight: 0 };
    }

    const visibilityCount = observations.filter(
      (observation) => (observation.visibility ?? 1) >= 0.6,
    ).length;
    const coverage = visibilityCount / 64;
    if (coverage < 0.5) {
      this.reject(`incomplete visibility (${visibilityCount}/64 squares)`, 'visibility');
      return { accepted: false, reason: this.lastRejection, weight: 0 };
    }

    let topProbabilitySum = 0;
    for (const observation of observations) {
      const top = observation.classProbabilities
        ? classProbabilitiesToTop(observation.classProbabilities)
        : null;
      topProbabilitySum += top?.probability ?? 0;
    }
    const meanTopProbability = topProbabilitySum / 64;
    const confidenceFactor = Math.max(
      0.25,
      Math.min(1, (meanTopProbability - 0.2) / 0.5),
    );
    const sharpnessFactor = Math.max(
      0.1,
      Math.min(1, sharpness / Math.max(this.minSharpness * 3, 1e-6)),
    );
    const weight = sharpnessFactor * (0.35 + 0.65 * alignment) * coverage * confidenceFactor;

    if (weight < this.minFrameWeight) {
      this.reject(`frame quality too low (weight ${weight.toFixed(2)})`, 'lowWeight');
      return { accepted: false, reason: this.lastRejection, weight: 0 };
    }

    const frame: FusionFrame = { timestampMs: capturedAtMs, weight, observations };

    // Seed the epoch anchor on the first accepted frame. The anchor survives
    // frame eviction so one-square-per-frame drift is still detected.
    if (!this.anchor) {
      this.anchor = this.toObservationMap(observations);
    }

    const diagnosis = this.diagnoseFrame(observations);
    this.recordLowConfidenceFluctuation(diagnosis.lowConfidenceSquares);

    // A chess move changes at least two squares. Isolated classifier noise can
    // also produce two confident flips, so a candidate delta is *held* until
    // two consecutive accepted frames show the same reliable delta. Until then
    // the old epoch remains authoritative and its evidence is not discarded.
    if (diagnosis.changes.size >= 2) {
      const signature = this.transitionSignature(diagnosis.changes);
      const kind = this.transitionKind(diagnosis.changes);
      const previous = this.pendingTransition;
      const sameCandidate =
        previous !== null &&
        previous.signature === signature &&
        capturedAtMs >= previous.lastSeenMs &&
        capturedAtMs - previous.lastSeenMs <= FUSION_TRANSITION_MAX_GAP_MS;

      if (sameCandidate) {
        previous.frames.push(frame);
        previous.lastSeenMs = capturedAtMs;
        if (previous.frames.length >= FUSION_TRANSITION_CORROBORATION_FRAMES) {
          const corroboratingFrames = previous.frames.slice(-this.maxFrames);
          const squares = [...previous.changes.keys()].sort((left, right) =>
            left.localeCompare(right),
          );
          this.frames = corroboratingFrames;
          this.anchor = this.toObservationMap(observations);
          this.transitionResets += 1;
          this.transitionResetsByKind[previous.kind] += 1;
          this.lastEpochReset = {
            atMs: capturedAtMs,
            kind: previous.kind,
            squares,
            changes: this.formatChanges(previous.changes),
            corroboratingFrames: previous.frames.length,
          };
          this.pendingTransition = null;
          this.lastRejection = null;
          return {
            accepted: true,
            reason: null,
            weight,
            transitionReset: true,
            transitionKind: previous.kind,
            transitionSquares: squares,
          };
        }
        this.lastRejection = null;
        return { accepted: true, reason: null, weight };
      }

      if (previous) {
        this.discardPendingTransition('signature_changed', capturedAtMs);
      }
      this.pendingTransition = {
        signature,
        changes: diagnosis.changes,
        kind,
        frames: [frame],
        startedAtMs: capturedAtMs,
        lastSeenMs: capturedAtMs,
      };
      this.lastRejection = null;
      return { accepted: true, reason: null, weight };
    }

    // The frame agrees with the anchor on every reliable square. Any pending
    // candidate was an isolated fluctuation; keep the existing evidence.
    if (this.pendingTransition) {
      this.discardPendingTransition('not_corroborated', capturedAtMs);
    }

    this.pushFrame(frame);
    this.lastRejection = null;
    return { accepted: true, reason: null, weight };
  }

  /** Fuses all retained, unexpired frames at `nowMs`. */
  snapshot(nowMs: number = Date.now()): FusionSnapshot {
    this.evict(nowMs);
    this.refreshEpoch(nowMs);

    const probabilitiesBySquare = new Map<
      ChessSquare,
      { sums: number[]; weight: number; frames: number; lastObservedMs: number }
    >();

    for (const frame of this.frames) {
      const ageMs = Math.max(0, nowMs - frame.timestampMs);
      const recencyWeight = Math.exp(-ageMs / FUSION_RECENCY_HALF_LIFE_MS);
      for (const observation of frame.observations) {
        const probabilities = observation.classProbabilities;
        if (!probabilities || probabilities.length !== PIECE_CLASSES.length) continue;
        const visibilityWeight = Math.max(0, Math.min(1, observation.visibility ?? 1));
        const squareWeight = frame.weight * visibilityWeight * recencyWeight;
        if (squareWeight <= 0) continue;
        const existing = probabilitiesBySquare.get(observation.square);
        if (!existing) {
          probabilitiesBySquare.set(observation.square, {
            sums: probabilities.map((value) => value * squareWeight),
            weight: squareWeight,
            frames: 1,
            lastObservedMs: frame.timestampMs,
          });
        } else {
          for (let index = 0; index < probabilities.length; index += 1) {
            existing.sums[index] += probabilities[index] * squareWeight;
          }
          existing.weight += squareWeight;
          existing.frames += 1;
          existing.lastObservedMs = Math.max(existing.lastObservedMs, frame.timestampMs);
        }
      }
    }

    const observations: SquareObservation[] = [];
    const squares: FusedSquareEvidence[] = [];
    const uncertainSquares: ChessSquare[] = [];
    const missingSquares: ChessSquare[] = [];

    for (let rank = 8; rank >= 1; rank -= 1) {
      for (let file = 0; file < 8; file += 1) {
        const square = `${String.fromCharCode(97 + file)}${rank}` as ChessSquare;
        const evidence = probabilitiesBySquare.get(square);
        if (!evidence || evidence.weight < this.minSquareWeight) {
          missingSquares.push(square);
          continue;
        }

        const probabilities = evidence.sums.map((value) => value / evidence.weight);
        const top = classProbabilitiesToTop(probabilities);
        if (!top) {
          missingSquares.push(square);
          continue;
        }

        const emptyProbability = probabilities[0];
        const occupancyProbability = clamp01(1 - emptyProbability);
        let whiteMass = 0;
        let blackMass = 0;
        for (let index = 1; index < probabilities.length; index += 1) {
          if (PIECE_CLASSES[index].startsWith('white_')) whiteMass += probabilities[index];
          else blackMass += probabilities[index];
        }
        const nonEmptyMass = Math.max(whiteMass + blackMass, 1e-6);
        const whiteProbability = whiteMass / nonEmptyMass;
        const blackProbability = blackMass / nonEmptyMass;
        const topClass = PIECE_CLASSES[top.index];
        const margin = top.probability - top.secondProbability;

        const lowEvidence = evidence.weight < this.minSquareWeight * 1.5;
        const classAmbiguous =
          top.probability < this.uncertainProbability || margin < this.uncertainMargin;
        const occupancyAmbiguous =
          topClass === 'empty'
            ? occupancyProbability > 1 - this.uncertainProbability
            : occupancyProbability < this.uncertainProbability;
        const colourAmbiguous =
          topClass !== 'empty' && Math.max(whiteProbability, blackProbability) < 0.6;
        const uncertain =
          lowEvidence || classAmbiguous || occupancyAmbiguous || colourAmbiguous;

        const occupied = !uncertain && occupancyProbability >= 0.5;
        const color: 'w' | 'b' | null = occupied
          ? whiteProbability >= blackProbability
            ? 'w'
            : 'b'
          : null;

        const fused: FusedSquareEvidence = {
          square,
          probabilities,
          occupancyProbability,
          whiteProbability,
          blackProbability,
          topClass,
          topProbability: top.probability,
          secondProbability: top.secondProbability,
          weight: evidence.weight,
          frames: evidence.frames,
          lastObservedMs: evidence.lastObservedMs,
          uncertain,
          provenance: 'observed',
        };
        squares.push(fused);
        if (uncertain) uncertainSquares.push(square);

        observations.push({
          square,
          occupied,
          color,
          confidence: top.probability,
          visibility: 1,
          classProbabilities: probabilities,
          occupiedProbability: occupancyProbability,
          whiteProbability,
          blackProbability,
          uncertain,
          provenance: 'observed',
          evidenceFrames: evidence.frames,
          lastObservedMs: evidence.lastObservedMs,
        });
      }
    }

    const averageFrameWeight =
      this.frames.length > 0
        ? this.frames.reduce((sum, frame) => sum + frame.weight, 0) / this.frames.length
        : 0;

    return {
      observations,
      squares,
      acceptedFrames: this.frames.length,
      rejectedFrames: this.rejectedFrames,
      lastRejection: this.lastRejection,
      uncertainSquares,
      missingSquares,
      averageFrameWeight,
      hasFullCoverage: missingSquares.length === 0 && observations.length === 64,
      transitionResets: this.transitionResets,
      transitionResetsByKind: { ...this.transitionResetsByKind },
      lastEpochReset: this.lastEpochReset,
      pendingTransition: this.pendingTransition
        ? {
            kind: this.pendingTransition.kind,
            squares: [...this.pendingTransition.changes.keys()].sort((left, right) =>
              left.localeCompare(right),
            ),
            changes: this.formatChanges(this.pendingTransition.changes),
            frames: this.pendingTransition.frames.length,
            ageMs: Math.max(0, nowMs - this.pendingTransition.startedAtMs),
          }
        : null,
      discardedTransitions: this.discardedTransitions,
      lastDiscardedTransition: this.lastDiscardedTransition,
      lowConfidenceFluctuations: this.lowConfidenceFluctuations,
      lastLowConfidenceSquares: [...this.lastLowConfidenceSquares],
      rejectionCounts: { ...this.rejectionCounts },
    };
  }

  /**
   * Computes reliable square deltas against the epoch anchor. Differences that
   * fail the confidence/margin/visibility gate are reported separately as
   * low-confidence fluctuations; they can never trigger an epoch transition.
   */
  private diagnoseFrame(observations: SquareObservation[]): {
    changes: Map<ChessSquare, TransitionChange>;
    lowConfidenceSquares: ChessSquare[];
  } {
    const changes = new Map<ChessSquare, TransitionChange>();
    const lowConfidenceSquares: ChessSquare[] = [];
    if (!this.anchor) return { changes, lowConfidenceSquares };

    for (const observation of observations) {
      const before = this.anchor.get(observation.square);
      if (!before) continue;
      const beforeClass = topClassOf(before);
      const afterClass = topClassOf(observation);
      if (!beforeClass || !afterClass || beforeClass === afterClass) continue;

      if (!isReliableObservation(before) || !isReliableObservation(observation)) {
        lowConfidenceSquares.push(observation.square);
        continue;
      }

      // A genuine chess move changes occupancy (empty <-> occupied) or piece
      // colour (capture). Knight-vs-bishop flicker within one colour is
      // classifier ambiguity and must be fused, never treated as a transition.
      const isOccupancyChange = (beforeClass === 'empty') !== (afterClass === 'empty');
      const beforeColor = beforeClass.startsWith('white_')
        ? 'w'
        : beforeClass.startsWith('black_')
          ? 'b'
          : null;
      const afterColor = afterClass.startsWith('white_')
        ? 'w'
        : afterClass.startsWith('black_')
          ? 'b'
          : null;
      const isColourChange =
        beforeColor !== null && afterColor !== null && beforeColor !== afterColor;
      if (!isOccupancyChange && !isColourChange) continue;

      changes.set(observation.square, {
        fromClass: beforeClass,
        toClass: afterClass,
        kind: isOccupancyChange ? 'occupancy' : 'colour',
      });
    }
    return { changes, lowConfidenceSquares };
  }

  private transitionKind(
    changes: Map<ChessSquare, TransitionChange>,
  ): FusionEpochResetKind {
    if (changes.size > FUSION_MAX_TRANSITION_SQUARES) return 'discontinuity';
    let hasOccupancy = false;
    let hasColour = false;
    for (const change of changes.values()) {
      if (change.kind === 'occupancy') hasOccupancy = true;
      else hasColour = true;
    }
    if (hasOccupancy && hasColour) return 'mixed';
    return hasOccupancy ? 'occupancy' : 'colour';
  }

  private transitionSignature(changes: Map<ChessSquare, TransitionChange>): string {
    return [...changes.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([square, change]) => `${square}:${change.toClass}`)
      .join('|');
  }

  private formatChanges(changes: Map<ChessSquare, TransitionChange>): string[] {
    return [...changes.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([square, change]) => `${square}:${change.fromClass}->${change.toClass}`);
  }

  private toObservationMap(
    observations: SquareObservation[],
  ): Map<ChessSquare, SquareObservation> {
    const map = new Map<ChessSquare, SquareObservation>();
    for (const observation of observations) {
      map.set(observation.square, observation);
    }
    return map;
  }

  private recordLowConfidenceFluctuation(squares: ChessSquare[]): void {
    if (squares.length === 0) return;
    this.lowConfidenceFluctuations += 1;
    this.lastLowConfidenceSquares = squares.slice(0, 8);
  }

  private pushFrame(frame: FusionFrame): void {
    this.frames.push(frame);
    while (this.frames.length > this.maxFrames) {
      this.frames.shift();
    }
  }

  private discardPendingTransition(
    reason: FusionDiscardedTransition['reason'],
    atMs: number,
  ): void {
    const pending = this.pendingTransition;
    if (!pending) return;
    this.discardedTransitions += 1;
    this.lastDiscardedTransition = {
      atMs,
      reason,
      kind: pending.kind,
      squares: [...pending.changes.keys()].sort((left, right) => left.localeCompare(right)),
      changes: this.formatChanges(pending.changes),
    };
    this.pendingTransition = null;
  }

  /**
   * Drops an uncorroborated candidate after a long gap, and releases a stale
   * anchor once no retained frame can support it. Expiry is not a board
   * transition: no retained evidence remains to be corrupted.
   */
  private refreshEpoch(nowMs: number): void {
    if (this.pendingTransition) {
      const gapMs = nowMs - this.pendingTransition.lastSeenMs;
      if (gapMs < 0 || gapMs > FUSION_TRANSITION_MAX_GAP_MS) {
        this.discardPendingTransition('expired', nowMs);
      }
    }
    if (this.frames.length === 0) {
      if (this.pendingTransition) {
        this.discardPendingTransition('evidence_expired', nowMs);
      }
      this.anchor = null;
    }
  }

  private evict(nowMs: number): void {
    const cutoff = nowMs - this.windowMs;
    this.frames = this.frames.filter((frame) => frame.timestampMs >= cutoff);
    if (this.frames.length > this.maxFrames) {
      this.frames = this.frames.slice(this.frames.length - this.maxFrames);
    }
  }

  private reject(reason: string, kind: keyof FusionRejectionCounts): void {
    this.rejectedFrames += 1;
    this.lastRejection = reason;
    this.rejectionCounts[kind] += 1;
  }
}
