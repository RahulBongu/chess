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

export type TrackingSource = 'detector' | 'motion' | 'hold' | 'none';

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

export interface LiveGuidanceModel {
  trackingState: TrackingState;
  /** Which input produced the current fix, or 'hold'/'none' when degraded. */
  trackingSource: TrackingSource;
  /** BoardTracker confidence (0..1). */
  trackingConfidence: number;
  /** RANSAC inliers from the motion tracker for this frame, if any. */
  trackInliers: number | null;
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

  constructor(options: LiveGuidanceOptions = {}) {
    this.orientation = options.orientation ?? 'white_bottom';
    this.tracker = new BoardTracker();
    this.recognizer = new MoveRecognizer({ initialFen: options.initialFen });

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
    this.orientation = orientation;
  }

  getOrientation(): BoardOrientation {
    return this.orientation;
  }

  /**
   * Seeds the board tracker with known corners from assisted calibration.
   */
  seedCorners(corners: Point2D[], nowMs?: number): boolean {
    return this.tracker.seed(corners, nowMs);
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
  ): Promise<LiveGuidanceModel> {
    const frameStart = nowMsForTiming();

    // 1. Update temporal corner tracker (detection preferred, motion-tracked fallback)
    const trackingState = this.tracker.update(
      detectedCorners,
      rawConfidence,
      nowMs,
      trackedProposal?.corners ?? null,
      trackedProposal?.confidence ?? 0,
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

    // 2. Update move recognizer ONLY on trustworthy frames. When the board
    // is holding/lost, sampled squares are misaligned and must never be
    // allowed to trigger a move. The confirmed position is preserved.
    let recognitionResult: MoveRecognitionResult | null = null;
    const recognizerTrusted = trackingState === 'tracking';
    const recognizerStart = nowMsForTiming();
    if (observations && observations.length === 64 && recognizerTrusted) {
      recognitionResult = this.recognizer.processFrame(observations, isHandOccluded);
    }
    const recognizerMs = nowMsForTiming() - recognizerStart;

    const currentFen = this.recognizer.getFen();
    const turn = this.recognizer.getTurn();

    // 3. Trigger local engine analysis if position changed (asynchronous with stale result suppression)
    if (currentFen !== this.analyzedFen && !this.recognizer.getGame().isGameOver()) {
      this.analyzedFen = currentFen;
      this.engine.stop();
      this.searchGeneration += 1;
      const currentToken = this.searchGeneration;
      this.isAnalyzing = true;
      const engineStart = nowMsForTiming();

      this.engine
        .analyze(currentFen, { depth: 8, maxTimeMs: 150 })
        .then((result) => {
          if (currentToken === this.searchGeneration && result.fen === this.recognizer.getFen()) {
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

    // 4. Construct human instruction text
    let instruction = 'Point camera at the board.';
    const bestMove = this.currentAnalysis?.bestMove ?? null;

    if (trackingState === 'lost' || trackingState === 'uncalibrated') {
      instruction = 'Locating board... Keep all 4 corners in view.';
    } else if (trackingState === 'holding') {
      instruction = 'Holding board tracking during movement...';
    } else if (this.recognizer.getStatus() === 'occluded') {
      instruction = 'Hand over board. Complete move and clear hand.';
    } else if (this.recognizer.getStatus() === 'uncertain') {
      instruction = 'Unrecognized change. Tap Review to verify.';
    } else if (this.recognizer.getGame().isGameOver()) {
      instruction = this.recognizer.getGame().isCheckmate() ? 'Checkmate!' : 'Game drawn.';
    } else if (bestMove) {
      const pieceName = this.getPieceNameAtSquare(bestMove.from);
      instruction = `Suggested: Move ${pieceName} from ${bestMove.from} to ${bestMove.to} (${bestMove.san})`;
    } else if (this.recognizer.getStatus() === 'stabilizing') {
      instruction = 'Waiting for board to settle...';
    } else {
      instruction = `${turn === 'w' ? 'White' : 'Black'} to move.`;
    }

    // 5. Compute perspective geometry if corners are available.
    // Confident overlays (source/destination/arrow) render only while the
    // fix is fresh (state === 'tracking'). While holding, the board boundary
    // and grid render but the move arrow and square highlights are
    // suppressed so a stale fix can never look authoritative.
    let sourceQuad: PerspectiveQuad | null = null;
    let destinationQuad: PerspectiveQuad | null = null;
    let arrow: ArrowGeometry | null = null;
    let gridLines: GridLine[] | null = null;
    const geometryStart = nowMsForTiming();
    const overlayTrusted = stabilizedCorners && trackingState === 'tracking';

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
      trackingSource,
      trackingConfidence,
      trackInliers: trackedProposal?.inliers ?? null,
      recognizerStatus: this.recognizer.getStatus(),
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
      evaluationScore: this.currentAnalysis?.score ?? null,
      engineName: this.engineSelection.engineName,
      isEngineAnalyzing: this.isAnalyzing,
      pendingPromotion: this.recognizer.getPendingPromotion(),
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
  }
}

/** Monotonic clock for stage timings, with a wall-clock fallback. */
function nowMsForTiming(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}
