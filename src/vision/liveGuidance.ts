/**
 * Live Guidance Coordinator for Handheld Chess Navigation.
 *
 * Integrates:
 * - BoardTracker (corner tracking & stabilization across handheld camera movement)
 * - MoveRecognizer (temporal differential legal move detection)
 * - Local ChessEngine (Chal v2 NNUE evaluation with stale result suppression)
 * - PerspectiveOverlay (projective mapping of highlights and move arrows onto camera preview)
 */

import { BoardTracker, type TrackingState } from './boardTracker.ts';
import {
  MoveRecognizer,
  type MoveRecognitionResult,
  type PendingPromotion,
  type RecognizedMove,
  type SquareObservation,
} from './moveRecognizer.ts';
import {
  computeMoveArrowGeometry,
  computePerspectiveGridLines,
  computeSquarePerspectiveQuad,
  type ArrowGeometry,
  type BoardOrientation,
  type GridLine,
  type PerspectiveQuad,
  type Point2D,
} from './perspectiveOverlay.ts';
import { createDefaultEngine, type EngineSelection } from '../chess/engineSelection.ts';
import type { ChessEngine, EngineAnalysisResult } from '../chess/engine.ts';
import { STARTING_FEN, type ChessSquare } from '../chess/ChessGame.ts';
import {
  observationsFromFen,
  recognizePositionFromObservations,
  suggestPositionFromObservations,
  summarizeSquareEvidence,
  verifyStartingPosition,
  type DiagnosticSquareDetail,
} from './pieceRecognition.ts';
import {
  TemporalSquareFusion,
  type FrameQuality,
  type FusionSnapshot,
  type TemporalFusionOptions,
} from './temporalFusion.ts';

export type TrackingSource = 'detector' | 'motion' | 'hold' | 'none';

/**
 * User-visible position lifecycle. These replace raw "missing square" errors:
 * a transient blur or hand occlusion is a stabilization/reacquisition state,
 * never a failed recognition of the last confirmed position.
 */
export type LivePositionState =
  | 'REACQUIRING_BOARD'
  | 'STABILIZING'
  | 'COLLECTING_EVIDENCE'
  | 'POSITION_CANDIDATE'
  | 'POSITION_CONFIRMED';

export interface MotionTrackProposal {
  /** Normalized corners fresh for this frame (as produced by BoardMotionTracker). */
  corners: Point2D[];
  /** Motion tracker quality, 0..1. */
  confidence: number;
  /** Number of RANSAC inliers, for telemetry. */
  inliers?: number | null;
}

/**
 * Per-stage wall-clock timings for a single processFrame call, in
 * milliseconds. These measure the JavaScript/vision pipeline only; camera
 * capture, native event delivery and React rendering are measured separately.
 */
export interface LiveStageTimings {
  trackerMs: number;
  recognizerMs: number;
  geometryMs: number;
  totalMs: number;
}

export interface LiveRecognitionState {
  /** True only after a legal, stable 13-class position has been adopted. */
  positionConfirmed: boolean;
  /** Consecutive frames that produced the same candidate position. */
  adoptionStreak: number;
  occupiedSquares: number;
  averageConfidence: number;
  uncertainSquares: string[];
  /** Why the last candidate position was rejected, if it was. */
  reason: string | null;
  /** Best current model proposal, even when it has not been adopted yet. */
  candidateFen: string | null;
  /** Number of frames that carried 64 x 13 classifier evidence. */
  classifiedFrames: number;
  /** Classified frames currently buffered for initial adoption averaging. */
  adoptionFrames: number;
  /** Configured averaging window for initial adoption. */
  adoptionWindow: number;
  /** Configured consecutive-candidate requirement. */
  adoptionConfirmations: number;
  /** Native ONNX inference time for the most recent classified frame. */
  inferenceMs: number | null;
  /** Native RGB square crop preparation duration for the most recent classified frame. */
  cropPrepMs: number | null;
  /** Probabilities and top classes for key diagnostic squares (kings, center). */
  diagnosticSquares: DiagnosticSquareDetail[];
  /** True when the standard-start check resolved genuinely uncertain squares from the prior. */
  startUsedPrior: boolean;
  /** Squares whose starting-layout value came from the bounded prior. */
  startPriorResolvedSquares: string[];
}

/** Temporal fusion telemetry for the most recent frame. */
export interface LiveFusionState {
  acceptedFrames: number;
  rejectedFrames: number;
  lastRejection: string | null;
  uncertainSquares: string[];
  missingSquares: string[];
  hasFullCoverage: boolean;
  averageFrameWeight: number;
  transitionResets: number;
  /** Measured frame focus (normalized Laplacian variance). */
  sharpness: number | null;
  /** Grid-alignment confidence supplied by the camera pipeline. */
  alignmentConfidence: number | null;
  /** Whether the most recent complete frame was accepted into the window. */
  frameAccepted: boolean;
}

export interface LiveGuidanceModel {
  trackingState: TrackingState;
  /** Why the corner tracker is holding, or null while tracking normally. */
  trackingHoldReason: string | null;
  /** Which input produced the current fix, or 'hold'/'none' when degraded. */
  trackingSource: TrackingSource;
  /** BoardTracker confidence (0..1). */
  trackingConfidence: number;
  /** RANSAC inliers from the motion tracker for this frame, if any. */
  trackInliers: number | null;
  /**
   * User-visible position lifecycle. A transient blur, hand crossing or
   * missing square is reported as stabilization/reacquisition, never as a
   * failed recognition of the last confirmed position.
   */
  positionState: LivePositionState;
  recognizerStatus: string;
  turn: 'w' | 'b';
  fen: string;
  instruction: string;
  bestMove: RecognizedMove | null;
  evaluationScore: number | null;
  engineName: string;
  isEngineAnalyzing: boolean;
  /** Set while a pawn promotion awaits explicit piece confirmation. */
  pendingPromotion: PendingPromotion | null;
  /**
   * Legal move supported by uncertain changed squares. The confirmed FEN is
   * preserved until enough unanimous candidate frames arrive.
   */
  provisionalMove: RecognizedMove | null;
  provisionalStreak: number;
  /** Piece-aware board adoption and classifier telemetry. */
  positionConfirmed: boolean;
  /** True when camera observations conclusively verify the standard 32-piece starting setup. */
  startReady: boolean;
  initializationMode: 'standard_start' | 'arbitrary_position';
  recognition: LiveRecognitionState;
  /** Bounded temporal fusion diagnostics for this frame. */
  fusion: LiveFusionState;
  /** Stage timings for this frame (JS pipeline only). */
  stageTimings: LiveStageTimings;
  /** Wall-clock duration of the last accepted engine search, if any. */
  engineLatencyMs: number | null;
  /** Monotonic count of accepted engine results, for diagnostics. */
  engineResultCount: number;
  perspectiveGeometry: {
    corners: [Point2D, Point2D, Point2D, Point2D] | null;
    sourceQuad: PerspectiveQuad | null;
    destinationQuad: PerspectiveQuad | null;
    arrow: ArrowGeometry | null;
    gridLines: GridLine[] | null;
  };
}

export interface LiveGuidanceOptions {
  orientation?: BoardOrientation;
  engine?: ChessEngine;
  initialFen?: string;
  initialTurn?: 'w' | 'b';
  /**
   * Initialisation strategy:
   * - 'standard_start' (default): verifies 32-piece initial setup with empty center;
   *   adopts the verified starting FEN safely without per-square piece errors.
   * - 'arbitrary_position': full 13-class per-square recognition for midgames/endgames.
   */
  initializationMode?: 'standard_start' | 'arbitrary_position';
  /**
   * Production live camera sets this true: engine analysis and move
   * recognition stay disabled until the 13-class model produces a legal,
   * stable position. Legacy/offline callers that already know the position
   * can leave it false.
   */
  requirePositionEvidence?: boolean;
  /**
   * Number of classified frames averaged before the initial position gate
   * runs. A static board's per-frame noise is largely independent, so
   * averaging reduces single-frame square errors. Ignored after adoption.
   * Defaults to 4 when `requirePositionEvidence` is set, 1 otherwise.
   */
  adoptionWindow?: number;
  /** Consecutive averaged candidates required before a position is adopted. */
  adoptionConfirmations?: number;
  /** Tuning for the bounded temporal square-evidence fusion window. */
  fusionOptions?: TemporalFusionOptions;
}

export class LiveGuidanceCoordinator {
  private tracker: BoardTracker;
  private recognizer: MoveRecognizer;
  private engineSelection: EngineSelection;
  private engine: ChessEngine;
  private orientation: BoardOrientation;

  private currentAnalysis: EngineAnalysisResult | null = null;
  private analyzedFen: string | null = null;
  private isAnalyzing = false;
  private searchGeneration = 0;
  private lastEngineLatencyMs: number | null = null;
  private engineResultCount = 0;

  private readonly requirePositionEvidence: boolean;
  private readonly adoptionWindow: number;
  private readonly adoptionConfirmations: number;
  private initializationMode: 'standard_start' | 'arbitrary_position';
  private adoptionFrames: SquareObservation[][] = [];
  private turnHint: 'w' | 'b';
  private positionConfirmed: boolean;
  /** Visual evidence matches the untouched 32-piece start; awaiting explicit user confirmation. */
  private startVerified = false;
  private adoptionStreak = 0;
  private adoptionCandidateFen: string | null = null;
  /** Best-effort placement offered to the assisted editor; never auto-adopted. */
  private suggestedFen: string | null = null;
  private recognitionReason: string | null = null;
  private recognitionSummary = {
    occupiedSquares: 0,
    averageConfidence: 0,
    uncertainSquares: [] as string[],
    diagnosticSquares: [] as DiagnosticSquareDetail[],
  };
  private classifiedFrames = 0;
  private lastInferenceMs: number | null = null;
  private lastCropPrepMs: number | null = null;

  private fusion: TemporalSquareFusion;
  private fusionSnapshot: FusionSnapshot | null = null;
  private lastFusionFrameAccepted = false;
  private lastFrameSharpness: number | null = null;
  private lastAlignmentConfidence: number | null = null;
  private startUsedPrior = false;
  private startPriorResolvedSquares: string[] = [];
  private positionState: LivePositionState = 'REACQUIRING_BOARD';

  constructor(options: LiveGuidanceOptions = {}) {
    this.orientation = options.orientation ?? 'white_bottom';
    this.tracker = new BoardTracker();
    this.recognizer = new MoveRecognizer({ initialFen: options.initialFen });
    this.requirePositionEvidence = options.requirePositionEvidence ?? false;
    this.initializationMode = options.initializationMode ?? 'standard_start';
    this.adoptionWindow = Math.max(
      1,
      Math.floor(options.adoptionWindow ?? (this.requirePositionEvidence ? 4 : 1)),
    );
    this.adoptionConfirmations = Math.max(1, Math.floor(options.adoptionConfirmations ?? 2));
    this.turnHint = options.initialTurn ?? 'w';
    this.fusion = new TemporalSquareFusion(options.fusionOptions);
    // A caller-supplied FEN is already a confirmed position. An internal
    // starting-position default is not: the camera must prove the board.
    this.positionConfirmed = Boolean(options.initialFen);

    if (options.engine) {
      this.engine = options.engine;
      this.engineSelection = {
        engine: options.engine,
        engineName: 'chal-2.0.0',
        fallbackReason: null,
      };
    } else {
      this.engineSelection = createDefaultEngine();
      this.engine = this.engineSelection.engine;
    }
  }

  setOrientation(orientation: BoardOrientation): void {
    if (orientation !== this.orientation) {
      // Evidence gathered in the old orientation maps to different algebraic
      // squares. Never average across an orientation change.
      this.clearAdoptionEvidence();
    }
    this.orientation = orientation;
  }

  getOrientation(): BoardOrientation {
    return this.orientation;
  }

  isPositionConfirmed(): boolean {
    return this.positionConfirmed;
  }

  isStartVerified(): boolean {
    return this.startVerified && !this.positionConfirmed;
  }

  confirmStartingPosition(): boolean {
    if (this.positionConfirmed) return true;
    // Explicit confirmation is only valid after the current averaged frame
    // actually verified the untouched starting layout.
    if (!this.startVerified) return false;
    if (this.recognizer.adoptPosition(STARTING_FEN)) {
      this.positionConfirmed = true;
      this.clearAdoptionEvidence();
      this.recognitionReason = null;
      this.currentAnalysis = null;
      this.analyzedFen = null;
      this.startVerified = false;
      return true;
    }
    return false;
  }

  setInitializationMode(mode: 'standard_start' | 'arbitrary_position'): void {
    if (this.initializationMode !== mode) {
      this.initializationMode = mode;
      if (!this.positionConfirmed) {
        this.clearAdoptionEvidence();
      }
    }
  }

  getInitializationMode(): 'standard_start' | 'arbitrary_position' {
    return this.initializationMode;
  }

  /** Records native inference and crop preparation latency without changing board state. */
  noteClassification(inferenceMs: number, cropPrepMs?: number): void {
    if (Number.isFinite(inferenceMs) && inferenceMs >= 0) {
      this.lastInferenceMs = inferenceMs;
    }
    if (typeof cropPrepMs === 'number' && Number.isFinite(cropPrepMs) && cropPrepMs >= 0) {
      this.lastCropPrepMs = cropPrepMs;
    }
  }

  /**
   * Sets the side to move. Before adoption this is the hint used to build the
   * candidate FEN. After adoption the placement is preserved and only the
   * side-to-move field changes, which triggers a fresh engine search.
   */
  setTurn(turn: 'w' | 'b'): boolean {
    this.turnHint = turn;
    if (!this.positionConfirmed) {
      this.clearAdoptionEvidence();
      return true;
    }
    const ok = this.recognizer.setTurn(turn);
    if (ok) {
      this.currentAnalysis = null;
      this.analyzedFen = null;
      this.isAnalyzing = false;
      this.searchGeneration += 1;
      this.engine.stop();
    }
    return ok;
  }

  getTurnHint(): 'w' | 'b' {
    return this.turnHint;
  }

  /**
   * Seeds the board tracker with known corners from assisted calibration.
   */
  seedCorners(corners: Point2D[], nowMs?: number): boolean {
    return this.tracker.seed(corners, nowMs);
  }

  /**
   * Adopts a position the user has explicitly confirmed or corrected.
   *
   * Trust here is anchored to the user, not to the classifier. The baseline is
   * synthesised from the FEN so a noisy frame cannot immediately masquerade as
   * a move, and any in-flight search from the previous position is dropped.
   */
  adoptUserPosition(fen: string): boolean {
    const observations = observationsFromFen(fen);
    if (!observations) {
      return false;
    }
    // The editor can express an empty board; the live tracker cannot use one.
    // Reuse the same structural gate as automatic adoption so a user-confirmed
    // position still has to be a playable chess position (both kings, legal
    // counts, no pawns on the back rank).
    const validated = recognizePositionFromObservations(observations, { turn: this.turnHint });
    if (!validated.ok) {
      return false;
    }
    if (!this.recognizer.adoptPosition(fen)) {
      return false;
    }
    this.positionConfirmed = true;
    this.clearAdoptionEvidence();
    this.recognitionReason = null;
    this.currentAnalysis = null;
    this.analyzedFen = null;
    this.isAnalyzing = false;
    this.searchGeneration += 1;
    this.engine.stop();
    return true;
  }

  /**
   * Ingests a new video frame's corners and square observations.
   *
   * @param trackedProposal Fresh, validated corners from the on-device
   *   motion tracker (already quality-checked by it), or null. When no
   *   detection is present this proposal drives the tracker.
   */
  async processFrame(
    detectedCorners: Point2D[] | null,
    observations: SquareObservation[] | null = null,
    isHandOccluded = false,
    rawConfidence = 0,
    nowMs?: number,
    trackedProposal: MotionTrackProposal | null = null,
    quality: FrameQuality = {},
  ): Promise<LiveGuidanceModel> {
    const frameStart = nowMsForTiming();
    const frameNow =
      typeof nowMs === 'number' && Number.isFinite(nowMs) ? nowMs : Date.now();

    this.lastFrameSharpness =
      typeof quality.sharpness === 'number' && Number.isFinite(quality.sharpness)
        ? quality.sharpness
        : null;
    this.lastAlignmentConfidence =
      typeof quality.alignmentConfidence === 'number' &&
      Number.isFinite(quality.alignmentConfidence)
        ? quality.alignmentConfidence
        : null;

    const frameQuality: FrameQuality = {
      ...quality,
      isHandOccluded: quality.isHandOccluded ?? isHandOccluded,
    };

    // 1. Update temporal corner tracker (detection preferred, motion-tracked
    // fallback). Blurred or misaligned frames hold the last good fix instead
    // of moving the grid to a wrong pose.
    const trackingState = this.tracker.update(
      detectedCorners,
      rawConfidence,
      nowMs,
      trackedProposal?.corners ?? null,
      trackedProposal?.confidence ?? 0,
      {
        motionBlur: frameQuality.motionBlur,
        alignmentConfidence: frameQuality.alignmentConfidence,
      },
    );
    const stabilizedCorners = this.tracker.getCorners();
    const trackingConfidence = this.tracker.getConfidence();
    const trackerMs = nowMsForTiming() - frameStart;

    const trackingSource: TrackingSource =
      detectedCorners && detectedCorners.length === 4 && trackingState === 'tracking'
        ? 'detector'
        : trackedProposal && trackedProposal.corners.length === 4
          ? 'motion'
          : trackingState === 'holding'
            ? 'hold'
            : 'none';

    // 2. Fuse classifier evidence BEFORE the move recognizer sees it. Distinct
    // failure modes map to distinct states:
    //   - Geometry unreliable (trackingState != 'tracking' / alignment low):
    //     recognizerTrusted is false, fusion not fed, recognizer paused,
    //     positionState = REACQUIRING_BOARD. Evidence cannot advance.
    //   - Blurred / physically occluded (sharpness < FUSION_MIN_SHARPNESS,
    //     motionBlur, or explicit isHandOccluded=true from the caller):
    //     fusion.addFrame rejects the frame, lastFusionFrameAccepted=false,
    //     positionState = STABILIZING (blur) or REACQUIRING_BOARD (loss).
    //     Explicit hand occlusion also resets the window so corrupted squares
    //     cannot vote into adoption history.
    //   - Visible board with uncertain classifications (low margins, >6
    //     uncertain squares): still fed to fusion; fusion marks squares
    //     uncertain per-square but does not reset. Start/move gates must
    //     still refuse to confirm/commit without sufficient confident evidence.
    //   - Sufficient fused + verified evidence: positionState becomes
    //     POSITION_CONFIRMED and only then engine/best-move/arrows are enabled.
    // When the board is lost or a hand covers it, old evidence must not survive
    // into the next board pose.
    const recognizerTrusted = trackingState === 'tracking';
    const hasClassEvidence =
      Boolean(observations) &&
      observations!.length === 64 &&
      observations!.every(
        (observation) =>
          Array.isArray(observation.classProbabilities) &&
          observation.classProbabilities!.length === 13,
      );

    if (trackingState === 'lost' || trackingState === 'uncalibrated') {
      this.resetFusionEvidence();
      this.clearAdoptionEvidence();
      if (!this.positionConfirmed) {
        this.recognitionReason = 'Locating board...';
      }
    } else if (isHandOccluded) {
      // Occluded squares are systematically darker and sampling is misaligned.
      // Never let them vote, and do not average them into adoption evidence.
      this.resetFusionEvidence();
      this.clearAdoptionEvidence();
      if (!this.positionConfirmed) {
        this.recognitionReason = 'Hand over board. Clear the view to recognize pieces.';
      }
    }

    let fusionFrameAccepted = false;
    if (
      observations &&
      observations.length === 64 &&
      hasClassEvidence &&
      recognizerTrusted &&
      !isHandOccluded
    ) {
      const added = this.fusion.addFrame(observations, frameQuality, frameNow);
      fusionFrameAccepted = added.accepted;
    }
    this.lastFusionFrameAccepted = fusionFrameAccepted;
    const snapshot = this.fusion.snapshot(frameNow);
    this.fusionSnapshot = snapshot;
    const fusedReady = Boolean(
      snapshot && snapshot.hasFullCoverage && snapshot.observations.length === 64,
    );
    const fusedUsable = recognizerTrusted && !isHandOccluded && fusedReady;

    // 3. Update move recognizer ONLY on trustworthy, fused evidence. When the
    // board is holding/lost or the frame is blurred, sampled squares are
    // misaligned and must never trigger a move. The confirmed FEN is kept.
    let recognitionResult: MoveRecognitionResult | null = null;
    const recognizerStart = nowMsForTiming();

    if (observations && observations.length === 64 && recognizerTrusted) {
      if (hasClassEvidence) {
        this.classifiedFrames += 1;

        if (isHandOccluded) {
          // Adoption evidence was already cleared above; keep the confirmed
          // position untouched and pause the recognizer so a pre-hand stable
          // count cannot commit the move on the first frame after the hand.
          if (this.positionConfirmed) {
            recognitionResult = this.recognizer.processFrame(observations, true);
          }
        } else if (!this.positionConfirmed) {
          if (fusionFrameAccepted) {
            this.adoptionFrames.push(observations);
            if (this.adoptionFrames.length > this.adoptionWindow) {
              this.adoptionFrames.shift();
            }
          }
          const minAcceptedFrames = Math.min(this.adoptionWindow, 3);
          const averaged =
            this.adoptionFrames.length >= this.adoptionWindow &&
            fusedUsable &&
            snapshot !== null &&
            snapshot.acceptedFrames >= minAcceptedFrames
              ? snapshot.observations
              : null;

          if (averaged) {
            this.updateRecognitionSummary(averaged);

            if (this.initializationMode === 'standard_start') {
              // A known starting position is only ever adopted after the
              // layout is verified AND the user explicitly confirms it. The
              // expected layout may resolve at most three genuinely uncertain
              // squares, and only after repeated fused evidence. Confident
              // contradictions still block.
              const priorEligible = snapshot !== null && snapshot.acceptedFrames >= 3;
              const startCheck = verifyStartingPosition(averaged, this.orientation, {
                maxUncertainSquares: priorEligible ? 3 : 0,
                priorFen: STARTING_FEN,
              });
              this.startVerified = startCheck.ok;
              this.startUsedPrior = startCheck.usedPrior;
              this.startPriorResolvedSquares = startCheck.priorResolvedSquares;
              const suggestion = suggestPositionFromObservations(averaged, {
                turn: this.turnHint,
              });
              this.suggestedFen = suggestion.ok ? suggestion.fen : null;
              this.adoptionStreak = 0;
              this.adoptionCandidateFen = null;
              if (startCheck.ok) {
                this.recognitionReason = startCheck.usedPrior
                  ? `Standard start verified with ${startCheck.priorResolvedSquares.length} uncertain square${startCheck.priorResolvedSquares.length === 1 ? '' : 's'} resolved from the expected layout. Tap Confirm Start.`
                  : 'Standard starting position verified. Tap Confirm Start to enable guidance.';
              } else {
                this.recognitionReason = `${startCheck.reason ?? 'Not a standard starting position.'} Switch to Custom Board or tap Set Position.`;
              }
            } else {
              this.startVerified = false;
              this.startUsedPrior = false;
              this.startPriorResolvedSquares = [];
              const candidate = recognizePositionFromObservations(averaged, {
                turn: this.turnHint,
                maxUncertainSquares: 0,
              });
              const suggestion = candidate.ok
                ? candidate
                : suggestPositionFromObservations(averaged, { turn: this.turnHint });
              this.suggestedFen = suggestion.ok ? suggestion.fen : null;

              if (candidate.ok) {
                if (this.adoptionCandidateFen === candidate.fen) {
                  this.adoptionStreak += 1;
                } else {
                  this.adoptionCandidateFen = candidate.fen;
                  this.adoptionStreak = 1;
                }
                this.recognitionReason =
                  `Verifying position (${this.adoptionStreak}/${this.adoptionConfirmations})...`;
                if (this.adoptionStreak >= this.adoptionConfirmations) {
                  // Seed the baseline from the accepted FEN, not from the noisy
                  // frame that produced it. Delta tracking must compare future
                  // frames against the believed position.
                  if (this.recognizer.adoptPosition(candidate.fen)) {
                    this.positionConfirmed = true;
                    this.clearAdoptionEvidence();
                    // Seed the new epoch with the frame that completed
                    // adoption, so engine/arrow gating has immediate fresh
                    // evidence. Pre-adoption frames are already discarded.
                    this.seedFusionEvidence(observations, frameQuality, frameNow);
                    this.recognitionReason = null;
                    this.currentAnalysis = null;
                    this.analyzedFen = null;
                  } else {
                    this.recognitionReason = 'The candidate position could not be adopted.';
                  }
                }
              } else {
                this.adoptionStreak = 0;
                this.adoptionCandidateFen = null;
                this.recognitionReason = candidate.reason;
              }
            }
          } else {
            this.updateRecognitionSummary(fusedReady ? snapshot!.observations : observations);
            if (!fusionFrameAccepted && snapshot?.lastRejection) {
              this.recognitionReason = `Waiting for a clearer frame (${snapshot.lastRejection})...`;
            } else {
              this.recognitionReason = `Collecting board evidence (${snapshot?.acceptedFrames ?? this.adoptionFrames.length}/${this.adoptionWindow})...`;
            }
          }
        } else if (fusedUsable) {
          this.updateRecognitionSummary(snapshot!.observations);
          recognitionResult = this.recognizer.processFrame(
            snapshot!.observations,
            false,
          );
        } else {
          // Confirmed position, but this frame is blurred/misaligned or fusion
          // has not accumulated full coverage. Keep the last confirmed FEN and
          // prevent any partly-built stable/provisional streak from advancing.
          this.updateRecognitionSummary(observations);
          this.recognizer.pause();
        }
      } else if (!this.requirePositionEvidence) {
        // Legacy/offline path: occupancy observations with a known position.
        recognitionResult = this.recognizer.processFrame(observations, isHandOccluded);
        this.positionConfirmed = true;
      }
    } else if (observations && observations.length === 64 && this.positionConfirmed) {
      // Geometry is not trustworthy (holding/lost): pause the recognizer so
      // stability and provisional candidates cannot advance from misaligned
      // square crops. The confirmed FEN is preserved.
      this.recognizer.pause();
    }

    // A committed move starts a new evidence epoch: the frame that triggered
    // the commit is post-move, while the fusion window may still contain the
    // old position. Re-seed from the confirmed FEN's frame so the next
    // comparison cannot drift back.
    if (recognitionResult && recognitionResult.move) {
      this.resetFusionEvidence();
      if (observations && hasClassEvidence && !isHandOccluded) {
        this.seedFusionEvidence(observations, frameQuality, frameNow);
      }
    }

    const recognizerMs = nowMsForTiming() - recognizerStart;

    const currentFen = this.recognizer.getFen();
    const turn = this.recognizer.getTurn();
    const recognitionStatus = this.recognizer.getStatus();

    // 4. Derive the user-visible position lifecycle. Transient missing-square
    // conditions are stabilization/reacquisition, never a failed position.
    let positionState: LivePositionState;
    if (trackingState === 'lost' || trackingState === 'uncalibrated') {
      positionState = 'REACQUIRING_BOARD';
    } else if (isHandOccluded) {
      positionState = 'STABILIZING';
    } else if (trackingState === 'holding') {
      const holdReason = this.tracker.getHoldReason();
      // A blurred or misaligned frame is a transient stabilization problem;
      // only a pose discontinuity / missing board requires reacquisition.
      positionState =
        holdReason === 'motion blur' || holdReason === 'grid misaligned'
          ? 'STABILIZING'
          : 'REACQUIRING_BOARD';
    } else if (!this.positionConfirmed) {
      positionState =
        this.startVerified || this.adoptionStreak > 0
          ? 'POSITION_CANDIDATE'
          : 'COLLECTING_EVIDENCE';
    } else if (recognitionResult?.provisionalMove) {
      positionState = 'POSITION_CANDIDATE';
    } else if (hasClassEvidence && !this.lastFusionFrameAccepted) {
      positionState = 'STABILIZING';
    } else if (recognitionStatus === 'occluded') {
      positionState = 'REACQUIRING_BOARD';
    } else if (recognitionStatus === 'uncertain' || recognitionStatus === 'stabilizing') {
      positionState = 'STABILIZING';
    } else {
      positionState = 'POSITION_CONFIRMED';
    }
    this.positionState = positionState;

    // Engine analysis, evaluation and arrows are only exposed while the
    // position is confirmed AND the current coordinate mapping is fresh.
    // The legacy occupancy-only path (no classifier evidence, position known
    // by the caller) keeps its historical gating so offline callers are not
    // forced through the camera-evidence lifecycle.
    const legacyConfirmed = !this.requirePositionEvidence && !hasClassEvidence;
    const recognitionUsable =
      this.positionConfirmed &&
      (legacyConfirmed
        ? recognitionStatus !== 'occluded' && recognitionStatus !== 'uncertain'
        : positionState === 'POSITION_CONFIRMED' &&
          recognitionStatus !== 'occluded' &&
          recognitionStatus !== 'uncertain');
    // Legacy offline callers have always been allowed to keep showing the
    // best move for their known position even while an occupancy delta is
    // being resolved. Camera-evidence callers only get it when confirmed.
    const analysisEligible = legacyConfirmed ? this.positionConfirmed : recognitionUsable;

    // 5. Trigger local engine analysis if the confirmed position changed
    // (asynchronous with stale result suppression). No analysis is allowed
    // while the board is stabilizing, reacquiring or unverified.
    if (
      analysisEligible &&
      currentFen !== this.analyzedFen &&
      !this.recognizer.getGame().isGameOver()
    ) {
      // Never show an evaluation or arrow belonging to an earlier position.
      this.currentAnalysis = null;
      this.analyzedFen = currentFen;
      this.engine.stop();
      this.searchGeneration += 1;
      const currentToken = this.searchGeneration;
      this.isAnalyzing = true;
      const engineStart = nowMsForTiming();

      this.engine
        .analyze(currentFen, { depth: 8, maxTimeMs: 150 })
        .then((result) => {
          if (
            this.positionConfirmed &&
            currentToken === this.searchGeneration &&
            result.fen === this.recognizer.getFen()
          ) {
            this.currentAnalysis = result;
            this.lastEngineLatencyMs = nowMsForTiming() - engineStart;
            this.engineResultCount += 1;
          }
        })
        .catch(() => {
          // Engine search stopped or cancelled safely
        })
        .finally(() => {
          if (currentToken === this.searchGeneration) {
            this.isAnalyzing = false;
          }
        });
    }

    // 6. Construct human instruction text
    let instruction = 'Point camera at the board.';
    const bestMove = analysisEligible ? this.currentAnalysis?.bestMove ?? null : null;

    if (trackingState === 'lost' || trackingState === 'uncalibrated') {
      instruction = 'Locating board... Keep all 4 corners in view.';
    } else if (trackingState === 'holding') {
      instruction = 'Reacquiring board... Hold steady.';
    } else if (!this.positionConfirmed) {
      instruction = this.recognitionReason ?? 'Board found. Recognizing all 32 pieces...';
    } else if (isHandOccluded) {
      instruction = 'Hand over board. Complete move and clear hand.';
    } else if (recognitionStatus === 'occluded') {
      instruction = 'Hand over board. Complete move and clear hand.';
    } else if (recognitionResult?.provisionalMove) {
      instruction = `Position candidate ${recognitionResult.provisionalMove.san} (${recognitionResult.provisionalStreak ?? 0}/${this.recognizer.getProvisionalConfirmations()}). Hold steady.`;
    } else if (positionState === 'STABILIZING') {
      instruction = 'Stabilizing evidence. Hold the board steady...';
    } else if (recognitionStatus === 'uncertain') {
      instruction = 'Unrecognized change. Tap Review to verify.';
    } else if (this.recognizer.getGame().isGameOver()) {
      instruction = this.recognizer.getGame().isCheckmate() ? 'Checkmate!' : 'Game drawn.';
    } else if (bestMove) {
      const pieceName = this.getPieceNameAtSquare(bestMove.from);
      instruction = `Suggested: Move ${pieceName} from ${bestMove.from} to ${bestMove.to} (${bestMove.san})`;
    } else if (recognitionStatus === 'stabilizing') {
      instruction = 'Waiting for board to settle...';
    } else {
      instruction = `${turn === 'w' ? 'White' : 'Black'} to move.`;
    }

    // 5. Compute perspective geometry if corners are available.
    // Confident overlays (source/destination/arrow) render only while the
    // fix is fresh (state === 'tracking') AND the piece position is
    // confirmed. While holding or recognizing, the board boundary and grid
    // render but the move arrow and square highlights are suppressed so a
    // stale or unverified fix can never look authoritative.
    let sourceQuad: PerspectiveQuad | null = null;
    let destinationQuad: PerspectiveQuad | null = null;
    let arrow: ArrowGeometry | null = null;
    let gridLines: GridLine[] | null = null;
    const geometryStart = nowMsForTiming();
    const overlayTrusted =
      Boolean(stabilizedCorners) && trackingState === 'tracking' && recognitionUsable;

    if (stabilizedCorners) {
      gridLines = computePerspectiveGridLines(stabilizedCorners);

      if (bestMove && overlayTrusted) {
        sourceQuad = computeSquarePerspectiveQuad(
          stabilizedCorners,
          bestMove.from,
          this.orientation,
        );
        destinationQuad = computeSquarePerspectiveQuad(
          stabilizedCorners,
          bestMove.to,
          this.orientation,
        );
        arrow = computeMoveArrowGeometry(
          stabilizedCorners,
          bestMove.from,
          bestMove.to,
          this.orientation,
        );
      }
    }
    const geometryMs = nowMsForTiming() - geometryStart;

    return {
      trackingState,
      trackingHoldReason: this.tracker.getHoldReason(),
      trackingSource,
      trackingConfidence,
      trackInliers: trackedProposal?.inliers ?? null,
      positionState,
      recognizerStatus: recognitionStatus,
      turn,
      fen: currentFen,
      instruction,
      bestMove: bestMove
        ? {
            from: bestMove.from as any,
            to: bestMove.to as any,
            san: bestMove.san,
            promotion: bestMove.promotion,
          }
        : null,
      evaluationScore: recognitionUsable ? this.currentAnalysis?.score ?? null : null,
      engineName: this.engineSelection.engineName,
      isEngineAnalyzing: this.isAnalyzing && analysisEligible,
      pendingPromotion: this.recognizer.getPendingPromotion(),
      provisionalMove: this.recognizer.getProvisionalMove(),
      provisionalStreak: this.recognizer.getProvisionalStreak(),
      positionConfirmed: this.positionConfirmed,
      startReady: this.startVerified && !this.positionConfirmed,
      initializationMode: this.initializationMode,
      recognition: {
        positionConfirmed: this.positionConfirmed,
        adoptionStreak: this.adoptionStreak,
        occupiedSquares: this.recognitionSummary.occupiedSquares,
        averageConfidence: this.recognitionSummary.averageConfidence,
        uncertainSquares: this.recognitionSummary.uncertainSquares,
        reason: this.recognitionReason,
        candidateFen: this.suggestedFen,
        classifiedFrames: this.classifiedFrames,
        adoptionFrames: this.adoptionFrames.length,
        adoptionWindow: this.adoptionWindow,
        adoptionConfirmations: this.adoptionConfirmations,
        inferenceMs: this.lastInferenceMs,
        cropPrepMs: this.lastCropPrepMs,
        diagnosticSquares: this.recognitionSummary.diagnosticSquares,
        startUsedPrior: this.startUsedPrior,
        startPriorResolvedSquares: this.startPriorResolvedSquares,
      },
      fusion: {
        acceptedFrames: this.fusionSnapshot?.acceptedFrames ?? 0,
        rejectedFrames: this.fusionSnapshot?.rejectedFrames ?? 0,
        lastRejection: this.fusionSnapshot?.lastRejection ?? null,
        uncertainSquares: this.fusionSnapshot?.uncertainSquares ?? [],
        missingSquares: this.fusionSnapshot?.missingSquares ?? [],
        hasFullCoverage: this.fusionSnapshot?.hasFullCoverage ?? false,
        averageFrameWeight: this.fusionSnapshot?.averageFrameWeight ?? 0,
        transitionResets: this.fusionSnapshot?.transitionResets ?? 0,
        sharpness: this.lastFrameSharpness,
        alignmentConfidence: this.lastAlignmentConfidence,
        frameAccepted: this.lastFusionFrameAccepted,
      },
      stageTimings: {
        trackerMs,
        recognizerMs,
        geometryMs,
        totalMs: nowMsForTiming() - frameStart,
      },
      engineLatencyMs: this.lastEngineLatencyMs,
      engineResultCount: this.engineResultCount,
      perspectiveGeometry: {
        corners: stabilizedCorners,
        sourceQuad,
        destinationQuad,
        arrow,
        gridLines,
      },
    };
  }

  private getPieceNameAtSquare(square: string): string {
    const piece = this.recognizer.getGame().pieceAt(square as any);
    if (!piece) return 'piece';
    switch (piece.type) {
      case 'p':
        return 'pawn';
      case 'n':
        return 'knight';
      case 'b':
        return 'bishop';
      case 'r':
        return 'rook';
      case 'q':
        return 'queen';
      case 'k':
        return 'king';
      default:
        return 'piece';
    }
  }

  private updateRecognitionSummary(observations: SquareObservation[]): void {
    const summary = summarizeSquareEvidence(observations);
    this.recognitionSummary = {
      occupiedSquares: summary.occupiedSquares,
      averageConfidence: summary.averageConfidence,
      uncertainSquares: summary.uncertainSquares,
      diagnosticSquares: summary.diagnosticSquares,
    };
  }

  private clearAdoptionEvidence(): void {
    this.adoptionFrames = [];
    this.adoptionStreak = 0;
    this.adoptionCandidateFen = null;
    this.suggestedFen = null;
    this.startVerified = false;
    this.startUsedPrior = false;
    this.startPriorResolvedSquares = [];
    this.resetFusionEvidence();
  }

  private resetFusionEvidence(): void {
    this.fusion.reset();
    this.fusionSnapshot = null;
    this.lastFusionFrameAccepted = false;
  }

  private seedFusionEvidence(
    observations: SquareObservation[],
    quality: FrameQuality,
    nowMs: number,
  ): void {
    const added = this.fusion.addFrame(observations, quality, nowMs);
    this.lastFusionFrameAccepted = added.accepted;
    this.fusionSnapshot = this.fusion.snapshot(nowMs);
  }

  getTracker(): BoardTracker {
    return this.tracker;
  }

  getRecognizer(): MoveRecognizer {
    return this.recognizer;
  }

  reset(fen?: string): void {
    this.tracker.reset();
    this.recognizer.reset(fen);
    this.currentAnalysis = null;
    this.analyzedFen = null;
    this.isAnalyzing = false;
    this.searchGeneration += 1;
    this.lastEngineLatencyMs = null;
    this.engineResultCount = 0;
    this.engine.stop();
    this.positionConfirmed = Boolean(fen);
    this.clearAdoptionEvidence();
    this.recognitionReason = null;
    this.recognitionSummary = {
      occupiedSquares: 0,
      averageConfidence: 0,
      uncertainSquares: [],
      diagnosticSquares: [],
    };
    this.classifiedFrames = 0;
    this.lastInferenceMs = null;
    this.lastCropPrepMs = null;
    this.resetFusionEvidence();
    this.lastFrameSharpness = null;
    this.lastAlignmentConfidence = null;
    this.positionState = fen ? 'POSITION_CONFIRMED' : 'REACQUIRING_BOARD';
  }
}

/** Monotonic clock for stage timings, with a wall-clock fallback. */
function nowMsForTiming(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}
