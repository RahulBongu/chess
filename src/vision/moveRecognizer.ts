/**
 * Temporal Move Recognizer for Handheld Chess Navigation.
 *
 * Implements legal-move-constrained temporal differential tracking:
 * - Detects hand occlusion and pauses evaluation during transient movement.
 * - Requires K consecutive stable frames post-movement before evaluating.
 * - Compares stable state against pre-move baseline to isolate candidate square deltas.
 * - Reconciles captures using reliable occupant-change evidence (color inversion or physical clearing), not luminance alone.
 * - Strictly enforces chess rules for quiet moves, captures, castling, and en-passant.
 * - Requires explicit promotion piece confirmation; bans unverified automatic Queen promotion.
 * - Suppresses duplicate move events once a move is confirmed.
 * - Provides an explicit UNCERTAIN state with safe manual recovery.
 */

import { ChessGame, type ChessSquare, type PromotionPiece } from '../chess/ChessGame.ts';
import {
  PIECE_CLASSES,
  boardClassesFromGame,
  classProbabilitiesToTop,
  observationsFromFen,
  reconcileLegalMove,
  type PieceClass,
} from './pieceRecognition.ts';

export type RecognizerStatus =
  | 'watching'
  | 'occluded'
  | 'stabilizing'
  | 'move_detected'
  | 'uncertain'
  | 'game_over';

export type PieceColor = 'w' | 'b' | null;

export interface SquareObservation {
  square: ChessSquare;
  occupied: boolean;
  color?: PieceColor;
  luminance?: number;
  confidence: number;
  /**
   * Fraction of the square's sample points that fall inside the decoded
   * frame (0..1). Defaults to 1 when not provided by the sampler.
   */
  visibility?: number;
  /**
   * 13-class model output for this square, ordered exactly like
   * `PIECE_CLASSES`. When all 64 observations carry this field the recognizer
   * uses piece-aware reconciliation instead of occupancy-only deltas.
   */
  classProbabilities?: number[];
  /**
   * True when temporal fusion could not assign a confident class. The
   * probabilities are still present; callers must not treat the top class as
   * a confirmed piece identity.
   */
  uncertain?: boolean;
  /** Where the accepted value came from. */
  provenance?: 'observed' | 'inferred';
  /** Fused probability that the square is occupied (1 - P(empty)). */
  occupiedProbability?: number;
  /** Fused colour mass conditional on the square being occupied. */
  whiteProbability?: number;
  blackProbability?: number;
  /** Number of retained frames contributing evidence to this square. */
  evidenceFrames?: number;
  /** Timestamp of the most recent frame that observed this square. */
  lastObservedMs?: number;
}

export interface RecognizedMove {
  from: ChessSquare;
  to: ChessSquare;
  san: string;
  promotion?: PromotionPiece;
}

export interface PendingPromotion {
  from: ChessSquare;
  to: ChessSquare;
  candidates: PromotionPiece[];
}

export interface MoveRecognitionResult {
  status: RecognizerStatus;
  move: RecognizedMove | null;
  fen: string;
  turn: 'w' | 'b';
  message: string;
  stableFrames: number;
  pendingPromotion?: PendingPromotion | null;
  /**
   * A legal, uniquely-supported move that is not yet strong enough to commit
   * because one or more changed squares are still uncertain. The confirmed
   * FEN is unchanged while this is present.
   */
  provisionalMove?: RecognizedMove | null;
  provisionalStreak?: number;
  unmatchedDeltas?: {
    emptied: ChessSquare[];
    filled: ChessSquare[];
    captured: ChessSquare[];
  };
}

export interface MoveRecognizerOptions {
  /** Required consecutive stable frames after hand departure before evaluating. Default 3 (~750ms). */
  requiredStableFrames?: number;
  /** Maximum number of square flips between adjacent frames allowed to consider the frame stationary. Default 2. */
  maxInterFrameFluctuation?: number;
  /** Minimum observation confidence required for a changed square to count as evidence. Default 0.60. */
  minDeltaConfidence?: number;
  /** Minimum visibility (0..1) required for a changed square to count as evidence. Default 0.60. */
  minDeltaVisibility?: number;
  /** Consecutive unanimous provisional candidates required before committing. Default 2. */
  provisionalConfirmations?: number;
  initialFen?: string;
}

export class MoveRecognizer {
  private game: ChessGame;
  private status: RecognizerStatus = 'watching';
  private baseline: Map<ChessSquare, boolean> | null = null;
  private baselineObs: Map<ChessSquare, SquareObservation> | null = null;
  private previousFrame: Map<ChessSquare, boolean> | null = null;
  private stableFrameCount = 0;
  private lastMove: RecognizedMove | null = null;
  private lastEvaluatedFen: string;
  private pendingPromotion: PendingPromotion | null = null;
  private pendingObservations: SquareObservation[] | null = null;
  private previousTopClasses: (PieceClass | null)[] | null = null;
  private previousUncertain: boolean[] | null = null;
  private provisionalCandidate: RecognizedMove | null = null;
  private provisionalStreak = 0;
  private positionConfirmed: boolean;

  private readonly requiredStableFrames: number;
  private readonly maxInterFrameFluctuation: number;
  private readonly minDeltaConfidence: number;
  private readonly minDeltaVisibility: number;
  private readonly provisionalConfirmations: number;

  constructor(options: MoveRecognizerOptions = {}) {
    this.requiredStableFrames = options.requiredStableFrames ?? 3;
    this.maxInterFrameFluctuation = options.maxInterFrameFluctuation ?? 2;
    this.minDeltaConfidence = options.minDeltaConfidence ?? 0.60;
    this.minDeltaVisibility = options.minDeltaVisibility ?? 0.60;
    this.provisionalConfirmations = Math.max(1, Math.floor(options.provisionalConfirmations ?? 2));
    this.game = new ChessGame(options.initialFen);
    this.lastEvaluatedFen = this.game.fen();
    // A caller-supplied FEN is an explicit position claim (assisted setup,
    // Board Review, deterministic playback) and counts as confirmed. Without
    // one, the camera must prove the position before classified deltas apply.
    this.positionConfirmed = Boolean(options.initialFen);
    this.initializeBaselineFromGame();
  }

  /**
   * Initializes baseline occupancy and color expectations from the current authoritative ChessGame state.
   */
  private initializeBaselineFromGame(): void {
    this.baseline = new Map();
    this.baselineObs = new Map();
    this.previousTopClasses = null;
    this.previousUncertain = null;
    this.clearProvisionalCandidate();
    const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    for (const file of files) {
      for (let rank = 1; rank <= 8; rank += 1) {
        const sq = `${file}${rank}` as ChessSquare;
        const piece = this.game.pieceAt(sq);
        const occupied = Boolean(piece);
        this.baseline.set(sq, occupied);
        this.baselineObs.set(sq, {
          square: sq,
          occupied,
          color: piece ? (piece.color as PieceColor) : null,
          luminance: piece ? (piece.color === 'w' ? 180 : 50) : 128,
          confidence: 1.0,
        });
      }
    }
  }

  /**
   * Directly sets the baseline occupancy and observations.
   */
  setBaseline(observations: SquareObservation[]): void {
    this.baseline = new Map();
    this.baselineObs = new Map();
    for (const obs of observations) {
      this.baseline.set(obs.square, obs.occupied);
      this.baselineObs.set(obs.square, { ...obs });
    }
    this.previousFrame = new Map(this.baseline);
    this.previousTopClasses = this.topClassesOf(observations);
    this.previousUncertain = new Array(64).fill(false);
    this.stableFrameCount = 0;
    this.pendingPromotion = null;
    this.pendingObservations = null;
    this.clearProvisionalCandidate();
    this.status = 'watching';
    this.positionConfirmed = true;
  }

  private clearProvisionalCandidate(): void {
    this.provisionalCandidate = null;
    this.provisionalStreak = 0;
  }

  getProvisionalMove(): RecognizedMove | null {
    return this.provisionalCandidate;
  }

  getProvisionalStreak(): number {
    return this.provisionalStreak;
  }

  getProvisionalConfirmations(): number {
    return this.provisionalConfirmations;
  }

  /**
   * Pauses temporal evaluation without consuming square evidence. Blurred,
   * misaligned or otherwise unusable frames call this so a stable-frame or
   * provisional streak cannot advance from evidence the pipeline rejected.
   */
  pause(): void {
    this.stableFrameCount = 0;
    this.clearProvisionalCandidate();
    if (this.positionConfirmed && this.status !== 'game_over') {
      this.status = 'stabilizing';
    }
  }

  /**
   * Adopts a legally validated position produced by piece recognition.
   * Rejects any FEN that chess.js does not round-trip exactly.
   */
  adoptPosition(fen: string, observations?: SquareObservation[]): boolean {
    try {
      const game = new ChessGame(fen);
      if (game.fen() !== fen) return false;
      const baseline =
        observations && observations.length === 64
          ? observations
          : observationsFromFen(fen);
      if (!baseline || baseline.length !== 64) return false;
      this.game = game;
      this.lastEvaluatedFen = fen;
      this.lastMove = null;
      this.positionConfirmed = true;
      this.setBaseline(baseline);
      return true;
    } catch {
      return false;
    }
  }

  isPositionConfirmed(): boolean {
    return this.positionConfirmed;
  }

  private topClassesOf(observations: SquareObservation[]): (PieceClass | null)[] | null {
    return this.analyzeTopClasses(observations)?.classes ?? null;
  }

  /**
   * Returns the per-square top class plus the fusion uncertainty mask. The
   * uncertainty mask lets stability tracking ignore a square whose top class
   * is genuinely ambiguous instead of resetting the stable-frame counter on
   * every flicker.
   */
  private analyzeTopClasses(
    observations: SquareObservation[],
  ): { classes: (PieceClass | null)[]; uncertain: boolean[] } | null {
    if (observations.length !== 64) return null;
    const classes: (PieceClass | null)[] = new Array(64).fill(null);
    const uncertain: boolean[] = new Array(64).fill(false);
    for (const observation of observations) {
      if (!observation.classProbabilities) return null;
      const top = classProbabilitiesToTop(observation.classProbabilities);
      if (!top) return null;
      const file = observation.square.charCodeAt(0) - 97;
      const rank = Number(observation.square[1]);
      if (file < 0 || file > 7 || rank < 1 || rank > 8) return null;
      const index = (8 - rank) * 8 + file;
      classes[index] = PIECE_CLASSES[top.index];
      uncertain[index] = observation.uncertain === true;
    }
    return { classes, uncertain };
  }

  /**
   * Ingests a new video frame's square observations and evaluates temporal state.
   */
  processFrame(
    observations: SquareObservation[],
    isHandOccluded = false,
  ): MoveRecognitionResult {
    const currentFen = this.game.fen();
    const turn = this.game.turn();

    if (this.game.isGameOver()) {
      this.status = 'game_over';
      return {
        status: 'game_over',
        move: null,
        fen: currentFen,
        turn,
        message: this.game.isCheckmate() ? 'Game over by checkmate.' : 'Game over by draw.',
        stableFrames: this.stableFrameCount,
      };
    }

    if (observations.length !== 64) {
      return {
        status: this.status,
        move: null,
        fen: currentFen,
        turn,
        message: `Incomplete square observations: expected 64, got ${observations.length}`,
        stableFrames: this.stableFrameCount,
      };
    }

    if (
      observations.every(
        (observation) =>
          Array.isArray(observation.classProbabilities) &&
          observation.classProbabilities.length === PIECE_CLASSES.length,
      )
    ) {
      return this.processClassifiedFrame(observations, isHandOccluded);
    }

    const currentMap = new Map<ChessSquare, boolean>();
    const currentObsMap = new Map<ChessSquare, SquareObservation>();
    for (const obs of observations) {
      currentMap.set(obs.square, obs.occupied);
      currentObsMap.set(obs.square, obs);
    }

    // 1. Explicit Hand Occlusion Check
    if (isHandOccluded) {
      this.status = 'occluded';
      this.stableFrameCount = 0;
      this.previousFrame = currentMap;
      return {
        status: 'occluded',
        move: null,
        fen: currentFen,
        turn,
        message: 'Hand or occlusion over board. Pausing move detection...',
        stableFrames: 0,
      };
    }

    // 2. Inter-Frame Stability Check
    if (this.previousFrame) {
      let fluctuations = 0;
      for (const [sq, occ] of currentMap.entries()) {
        if (this.previousFrame.get(sq) !== occ) {
          fluctuations += 1;
        }
      }

      if (fluctuations > 0) {
        this.stableFrameCount = 0;
        if (fluctuations > this.maxInterFrameFluctuation) {
          this.status = 'occluded';
          this.previousFrame = currentMap;
          return {
            status: 'occluded',
            move: null,
            fen: currentFen,
            turn,
            message: `Movement in progress (${fluctuations} squares changing)...`,
            stableFrames: 0,
          };
        }
      }
    }

    this.previousFrame = currentMap;

    // 3. Increment Stable Post-Motion Frames
    this.stableFrameCount += 1;

    // If board matches baseline exactly, we are stationary in the current position
    const isIdenticalToBaseline =
      this.baseline &&
      [...currentMap.entries()].every(([sq, occ]) => {
        if (this.baseline!.get(sq) !== occ) return false;
        if (occ && this.baselineObs) {
          const base = this.baselineObs.get(sq);
          const curr = currentObsMap.get(sq);
          if (base?.color && curr?.color && base.color !== curr.color) return false;
        }
        return true;
      });

    if (isIdenticalToBaseline) {
      this.status = 'watching';
      return {
        status: 'watching',
        move: null,
        fen: currentFen,
        turn,
        message: 'Board steady. Waiting for move.',
        stableFrames: this.stableFrameCount,
      };
    }

    if (this.stableFrameCount < this.requiredStableFrames) {
      this.status = 'stabilizing';
      return {
        status: 'stabilizing',
        move: null,
        fen: currentFen,
        turn,
        message: `Board stabilizing (${this.stableFrameCount}/${this.requiredStableFrames})...`,
        stableFrames: this.stableFrameCount,
      };
    }

    // 4. Stable State Reached: Compare Against Baseline
    if (!this.baseline || !this.baselineObs) {
      this.setBaseline(observations);
      this.status = 'watching';
      return {
        status: 'watching',
        move: null,
        fen: currentFen,
        turn,
        message: 'Baseline initialized. Watching for next move.',
        stableFrames: this.stableFrameCount,
      };
    }

    const emptied: ChessSquare[] = [];
    const filled: ChessSquare[] = [];
    const capturedTargets: ChessSquare[] = [];

    for (const [sq, currentOcc] of currentMap.entries()) {
      const baseObs = this.baselineObs.get(sq);
      const currObs = currentObsMap.get(sq);
      const baseOcc = baseObs?.occupied ?? false;

      if (baseOcc && !currentOcc) {
        emptied.push(sq);
      } else if (!baseOcc && currentOcc) {
        filled.push(sq);
      } else if (baseOcc && currentOcc) {
        // Both occupied: check for verified piece capture (occupant color change)
        // Does NOT rely on raw luminance shift alone, preventing shadows from triggering captures
        const colorFlipped =
          baseObs?.color && currObs?.color && baseObs.color !== currObs.color;

        if (colorFlipped) {
          capturedTargets.push(sq);
        }
      }
    }

    // 4.5 Low-Evidence Guard: every changed square must be well-observed.
    // A half-visible square that flipped occupancy (edge-clamped pixels)
    // must never be allowed to trigger a confident move.
    const changedSquares = [...emptied, ...filled, ...capturedTargets];
    const insufficient: ChessSquare[] = [];
    for (const sq of changedSquares) {
      const obs = currentObsMap.get(sq);
      const confidence = obs?.confidence ?? 0;
      const visibility = obs?.visibility ?? 1;
      if (confidence < this.minDeltaConfidence || visibility < this.minDeltaVisibility) {
        insufficient.push(sq);
      }
    }
    if (insufficient.length > 0) {
      this.status = 'uncertain';
      return {
        status: 'uncertain',
        move: null,
        fen: currentFen,
        turn,
        message: `Low-evidence board change on ${insufficient.join(', ')}; keeping confirmed position.`,
        stableFrames: this.stableFrameCount,
        unmatchedDeltas: { emptied, filled, captured: capturedTargets },
      };
    }

    // 5. Total Changed Squares Guard: In chess, no single legal move can change > 4 squares.
    const totalDeltas = emptied.length + filled.length + capturedTargets.length;
    if (totalDeltas > 4) {
      this.status = 'occluded';
      this.stableFrameCount = 0;
      return {
        status: 'occluded',
        move: null,
        fen: currentFen,
        turn,
        message: `Too many square changes (${totalDeltas}). Possible hand or board occlusion.`,
        stableFrames: 0,
        unmatchedDeltas: { emptied, filled, captured: capturedTargets },
      };
    }

    // 6. No Net Change: Same position confirmed
    if (totalDeltas === 0) {
      this.status = 'watching';
      return {
        status: 'watching',
        move: null,
        fen: currentFen,
        turn,
        message: 'Board steady. Waiting for move.',
        stableFrames: this.stableFrameCount,
      };
    }

    // 7. Reconcile Deltas Against Legal Moves
    const legalMoves = this.game.allLegalMoves();
    const candidateMatches: RecognizedMove[] = [];

    for (const move of legalMoves) {
      const fromSq = move.from as ChessSquare;
      const toSq = move.to as ChessSquare;

      // Special Move 1: Castling (involves King moving 2 squares and Rook jumping)
      const isCastling =
        (move.piece === 'k' && Math.abs(fromSq.charCodeAt(0) - toSq.charCodeAt(0)) === 2) ||
        move.san === 'O-O' ||
        move.san === 'O-O-O';

      if (isCastling) {
        let expectedEmptied: ChessSquare[] = [];
        let expectedFilled: ChessSquare[] = [];

        if (move.san === 'O-O') {
          if (fromSq === 'e1') {
            expectedEmptied = ['e1', 'h1'];
            expectedFilled = ['g1', 'f1'];
          } else {
            expectedEmptied = ['e8', 'h8'];
            expectedFilled = ['g8', 'f8'];
          }
        } else if (move.san === 'O-O-O') {
          if (fromSq === 'e1') {
            expectedEmptied = ['e1', 'a1'];
            expectedFilled = ['c1', 'd1'];
          } else {
            expectedEmptied = ['e8', 'a8'];
            expectedFilled = ['c8', 'd8'];
          }
        }

        const matchesEmptied =
          expectedEmptied.every((sq) => emptied.includes(sq)) && emptied.length === 2;
        const matchesFilled =
          expectedFilled.every((sq) => filled.includes(sq)) && filled.length === 2;

        if (matchesEmptied && matchesFilled) {
          candidateMatches.push({
            from: fromSq,
            to: toSq,
            san: move.san,
            promotion: move.promotion as PromotionPiece | undefined,
          });
        }
        continue;
      }

      // Special Move 2: En Passant
      const isEnPassant = move.flags?.includes('e') || (move.piece === 'p' && move.captured && !this.baseline.get(toSq));
      if (isEnPassant) {
        const capturedPawnSq = `${toSq[0]}${fromSq[1]}` as ChessSquare;
        const epEmptied = emptied.includes(fromSq) && emptied.includes(capturedPawnSq);
        const epFilled = filled.includes(toSq);

        if (epEmptied && epFilled) {
          candidateMatches.push({
            from: fromSq,
            to: toSq,
            san: move.san,
            promotion: move.promotion as PromotionPiece | undefined,
          });
        }
        continue;
      }

      // Standard Move or Standard Capture
      // A capture MUST present occupant-change evidence (a verified colour
      // inversion at the destination). Luminance shifts or an empty→occupied
      // flip at the destination never prove a capture on their own.
      const isRegularCandidate =
        emptied.length === 1 &&
        (move.captured
          ? capturedTargets.includes(toSq)
          : filled.length === 1 && filled.includes(toSq));

      const sourceMatches = emptied.includes(fromSq);

      if (sourceMatches && isRegularCandidate) {
        candidateMatches.push({
          from: fromSq,
          to: toSq,
          san: move.san,
          promotion: move.promotion as PromotionPiece | undefined,
        });
      }
    }

    // 8. Handle Promotion Candidates — Ban automatic Queen promotion
    // If a pawn reaches the 8th rank, require explicit user confirmation of the promoted piece.
    const promotionCandidates = candidateMatches.filter((m) => m.promotion !== undefined);
    if (promotionCandidates.length > 0) {
      const distinctPairs = new Set(promotionCandidates.map((m) => `${m.from}->${m.to}`));
      if (distinctPairs.size === 1) {
        const promoFrom = promotionCandidates[0].from;
        const promoTo = promotionCandidates[0].to;
        this.pendingPromotion = {
          from: promoFrom,
          to: promoTo,
          candidates: ['q', 'r', 'b', 'n'],
        };
        this.pendingObservations = observations;
        this.status = 'uncertain';

        return {
          status: 'uncertain',
          move: null,
          fen: currentFen,
          turn,
          message: `Pawn promotion detected at ${promoTo}. Please confirm piece (Queen, Rook, Bishop, Knight).`,
          stableFrames: this.stableFrameCount,
          pendingPromotion: this.pendingPromotion,
        };
      }
    }

    // 9. Unique Legal Move Found: Accept and Execute
    if (candidateMatches.length === 1) {
      const match = candidateMatches[0];
      const executed = this.game.tryMove({
        from: match.from,
        to: match.to,
        promotion: match.promotion ?? 'q',
      });

      if (executed) {
        const nextFen = this.game.fen();
        this.lastMove = {
          from: match.from,
          to: match.to,
          san: executed.san,
          promotion: executed.promotion as PromotionPiece | undefined,
        };
        this.lastEvaluatedFen = nextFen;
        this.setBaseline(observations);
        this.status = this.game.isGameOver() ? 'game_over' : 'move_detected';

        return {
          status: this.status,
          move: this.lastMove,
          fen: nextFen,
          turn: this.game.turn(),
          message: `Move recognized: ${executed.san}`,
          stableFrames: this.stableFrameCount,
        };
      }
    }

    // 10. Ambiguous or Illegal Delta: Transition to UNCERTAIN
    this.status = 'uncertain';
    return {
      status: 'uncertain',
      move: null,
      fen: currentFen,
      turn,
      message:
        candidateMatches.length > 1
          ? `Ambiguous move: ${candidateMatches.map((m) => m.san).join(' or ')}`
          : `Unrecognized board change: emptied=[${emptied.join(',')}], filled=[${filled.join(',')}], captured=[${capturedTargets.join(',')}]`,
      stableFrames: this.stableFrameCount,
      unmatchedDeltas: { emptied, filled, captured: capturedTargets },
    };
  }

  /**
   * Piece-aware path. A move is accepted only when the classified board
   * differs from the confirmed position on squares that exactly match one
   * legal move, with enough probability margin to exclude the runner-up.
   */
  private processClassifiedFrame(
    observations: SquareObservation[],
    isHandOccluded: boolean,
  ): MoveRecognitionResult {
    const currentFen = this.game.fen();
    const turn = this.game.turn();

    if (!this.positionConfirmed) {
      return {
        status: 'watching',
        move: null,
        fen: currentFen,
        turn,
        message: 'Waiting for a legal, confident board position.',
        stableFrames: this.stableFrameCount,
      };
    }

    if (isHandOccluded) {
      this.status = 'occluded';
      this.stableFrameCount = 0;
      this.previousTopClasses = this.topClassesOf(observations);
      return {
        status: 'occluded',
        move: null,
        fen: currentFen,
        turn,
        message: 'Hand or occlusion over board. Pausing move detection...',
        stableFrames: 0,
      };
    }

    const current = this.analyzeTopClasses(observations);
    if (!current) {
      this.clearProvisionalCandidate();
      this.status = 'uncertain';
      return {
        status: 'uncertain',
        move: null,
        fen: currentFen,
        turn,
        message: 'Class probabilities are incomplete; keeping confirmed position.',
        stableFrames: this.stableFrameCount,
      };
    }

    // More than three genuinely ambiguous squares invalidates the frame as
    // move evidence. Never let a wide class/colour ambiguity be silently
    // resolved by top-1 values: keep the confirmed position and wait.
    const uncertainCount = current.uncertain.reduce(
      (count, uncertain) => count + (uncertain ? 1 : 0),
      0,
    );
    if (uncertainCount > 3) {
      this.stableFrameCount = 0;
      this.clearProvisionalCandidate();
      this.status = 'uncertain';
      return {
        status: 'uncertain',
        move: null,
        fen: currentFen,
        turn,
        message: `${uncertainCount} squares are too uncertain; keeping confirmed position.`,
        stableFrames: 0,
      };
    }
    const currentTop = current.classes;

    if (this.previousTopClasses && this.previousUncertain) {
      let changes = 0;
      for (let index = 0; index < 64; index += 1) {
        if (currentTop[index] === this.previousTopClasses[index]) continue;
        // A genuinely ambiguous square flipping its top class is not a board
        // transition. Only confident changes reset stability, otherwise two
        // uncertain squares could prevent a valid move from ever settling.
        if (current.uncertain[index] || this.previousUncertain[index]) continue;
        changes += 1;
      }
      if (changes > 0) {
        this.stableFrameCount = 0;
      }
      // A single legal move can change at most four squares. More than that
      // between frames is a hand, a shadow, or a camera jump, not a move.
      if (changes > 4) {
        this.previousTopClasses = currentTop;
        this.previousUncertain = current.uncertain;
        this.clearProvisionalCandidate();
        this.status = 'occluded';
        return {
          status: 'occluded',
          move: null,
          fen: currentFen,
          turn,
          message: `Board changing rapidly (${changes} squares); pausing.`,
          stableFrames: 0,
        };
      }
    }
    this.previousTopClasses = currentTop;
    this.previousUncertain = current.uncertain;
    this.stableFrameCount += 1;

    const expected = boardClassesFromGame(this.game);
    const differences: {
      square: ChessSquare;
      expected: PieceClass;
      observed: PieceClass | null;
      confidence: number;
      uncertain: boolean;
    }[] = [];
    for (const observation of observations) {
      const file = observation.square.charCodeAt(0) - 97;
      const rank = Number(observation.square[1]);
      const boardIndex = (8 - rank) * 8 + file;
      const top = observation.classProbabilities
        ? classProbabilitiesToTop(observation.classProbabilities)
        : null;
      const observed = top ? PIECE_CLASSES[top.index] : null;
      if (observed !== expected[boardIndex]) {
        differences.push({
          square: observation.square,
          expected: expected[boardIndex],
          observed,
          confidence: top?.probability ?? 0,
          uncertain: observation.uncertain === true,
        });
      }
    }

    if (differences.length === 0) {
      this.clearProvisionalCandidate();
      this.status = 'watching';
      return {
        status: 'watching',
        move: null,
        fen: currentFen,
        turn,
        message: 'Board steady. Waiting for move.',
        stableFrames: this.stableFrameCount,
      };
    }

    const reliableDifferences = differences.filter((difference) => !difference.uncertain);
    if (reliableDifferences.length > 4) {
      this.stableFrameCount = 0;
      this.clearProvisionalCandidate();
      this.status = 'uncertain';
      return {
        status: 'uncertain',
        move: null,
        fen: currentFen,
        turn,
        message: `${reliableDifferences.length} confident squares conflict with the confirmed position; keeping it.`,
        stableFrames: 0,
      };
    }

    if (this.stableFrameCount < this.requiredStableFrames) {
      this.clearProvisionalCandidate();
      this.status = 'stabilizing';
      return {
        status: 'stabilizing',
        move: null,
        fen: currentFen,
        turn,
        message: `Board stabilizing (${this.stableFrameCount}/${this.requiredStableFrames})...`,
        stableFrames: this.stableFrameCount,
      };
    }

    const reconciliation = reconcileLegalMove(this.game, observations, {
      maxUncertainSquares: 3,
    });
    if (reconciliation.reason === 'promotion' && reconciliation.best) {
      this.clearProvisionalCandidate();
      this.pendingPromotion = {
        from: reconciliation.best.move.from,
        to: reconciliation.best.move.to,
        candidates: ['q', 'r', 'b', 'n'],
      };
      this.pendingObservations = observations;
      this.status = 'uncertain';
      return {
        status: 'uncertain',
        move: null,
        fen: currentFen,
        turn,
        message: `Pawn promotion detected at ${reconciliation.best.move.to}. Please confirm piece (Queen, Rook, Bishop, Knight).`,
        stableFrames: this.stableFrameCount,
        pendingPromotion: this.pendingPromotion,
      };
    }

    if (!reconciliation.valid || !reconciliation.best) {
      this.clearProvisionalCandidate();
      this.status = 'uncertain';
      return {
        status: 'uncertain',
        move: null,
        fen: currentFen,
        turn,
        message: reconciliation.reason ?? 'No legal move matches the observed board.',
        stableFrames: this.stableFrameCount,
      };
    }

    const best = reconciliation.best;
    const changed = new Set(best.changedSquares);
    const unexplained = differences.filter(
      (difference) =>
        !changed.has(difference.square) &&
        !difference.uncertain &&
        difference.confidence >= 0.62,
    );
    if (unexplained.length > 0) {
      this.clearProvisionalCandidate();
      this.status = 'uncertain';
      return {
        status: 'uncertain',
        move: null,
        fen: currentFen,
        turn,
        message: `Confident changes outside ${best.move.san}: ${unexplained
          .map((difference) => difference.square)
          .join(', ')}.`,
        stableFrames: this.stableFrameCount,
        unmatchedDeltas: {
          emptied: [],
          filled: unexplained.map((difference) => difference.square),
          captured: [],
        },
      };
    }

    // A move whose changed squares include fused-uncertain evidence is a
    // provisional candidate, not a commit. It must be the unique legal move
    // for several consecutive frames before it may advance the confirmed FEN.
    const isProvisional = reconciliation.provisional === true || best.provisional === true;
    if (isProvisional) {
      const candidateKey = `${best.move.from}${best.move.to}${best.move.promotion ?? ''}`;
      const previousKey = this.provisionalCandidate
        ? `${this.provisionalCandidate.from}${this.provisionalCandidate.to}${this.provisionalCandidate.promotion ?? ''}`
        : null;
      if (previousKey === candidateKey) {
        this.provisionalStreak += 1;
      } else {
        this.provisionalCandidate = { ...best.move };
        this.provisionalStreak = 1;
      }

      if (this.provisionalStreak < this.provisionalConfirmations) {
        this.status = 'stabilizing';
        return {
          status: 'stabilizing',
          move: null,
          fen: currentFen,
          turn,
          message: `Position candidate ${best.move.san} (${this.provisionalStreak}/${this.provisionalConfirmations}); holding confirmed position.`,
          stableFrames: this.stableFrameCount,
          provisionalMove: this.provisionalCandidate,
          provisionalStreak: this.provisionalStreak,
        };
      }
    } else {
      this.clearProvisionalCandidate();
    }

    // The scored move may have come from an ep-augmented FEN. Apply it from
    // that exact source so an inferred en-passant capture is not lost.
    if (best.sourceFen !== this.game.fen()) {
      try {
        const source = new ChessGame(best.sourceFen);
        if (source.fen() !== best.sourceFen) {
          throw new Error('Non-canonical source FEN.');
        }
        this.game = source;
      } catch {
        this.clearProvisionalCandidate();
        this.status = 'uncertain';
        return {
          status: 'uncertain',
          move: null,
          fen: currentFen,
          turn,
          message: `Legal reconciliation proposed ${best.move.san}, but its source position was rejected.`,
          stableFrames: this.stableFrameCount,
        };
      }
    }

    const executed = this.game.tryMove({
      from: best.move.from,
      to: best.move.to,
      promotion: best.move.promotion,
    });
    if (!executed) {
      this.clearProvisionalCandidate();
      this.status = 'uncertain';
      return {
        status: 'uncertain',
        move: null,
        fen: currentFen,
        turn,
        message: `Legal reconciliation proposed ${best.move.san}, but the move could not be applied.`,
        stableFrames: this.stableFrameCount,
      };
    }

    const nextFen = this.game.fen();
    this.lastMove = {
      from: best.move.from,
      to: best.move.to,
      san: executed.san,
      promotion: executed.promotion as PromotionPiece | undefined,
    };
    this.lastEvaluatedFen = nextFen;
    // Seed the post-move baseline from the confirmed FEN, not from the noisy
    // frame that triggered the commit. The FEN is the authoritative belief.
    this.setBaseline(observationsFromFen(nextFen) ?? observations);
    this.status = this.game.isGameOver() ? 'game_over' : 'move_detected';
    return {
      status: this.status,
      move: this.lastMove,
      fen: nextFen,
      turn: this.game.turn(),
      message: `Move recognized: ${executed.san}`,
      stableFrames: this.stableFrameCount,
    };
  }

  /**
   * Confirms a pending pawn promotion with an explicit piece selection ('q' | 'r' | 'b' | 'n').
   */
  confirmPromotion(piece: PromotionPiece): boolean {
    if (!this.pendingPromotion) {
      return false;
    }

    const { from, to } = this.pendingPromotion;
    const executed = this.game.tryMove({ from, to, promotion: piece });
    if (!executed) {
      return false;
    }

    this.lastMove = {
      from,
      to,
      san: executed.san,
      promotion: piece,
    };
    this.lastEvaluatedFen = this.game.fen();

    const confirmedBaseline = observationsFromFen(this.lastEvaluatedFen);
    if (confirmedBaseline) {
      this.setBaseline(confirmedBaseline);
    } else if (this.pendingObservations) {
      this.setBaseline(this.pendingObservations);
    } else {
      this.initializeBaselineFromGame();
    }

    this.pendingPromotion = null;
    this.pendingObservations = null;
    this.status = this.game.isGameOver() ? 'game_over' : 'move_detected';
    this.stableFrameCount = 0;
    return true;
  }

  getPendingPromotion(): PendingPromotion | null {
    return this.pendingPromotion;
  }

  /**
   * Safe manual recovery: user or test confirms a specific move.
   */
  confirmManualMove(
    from: ChessSquare,
    to: ChessSquare,
    promotion?: PromotionPiece,
  ): boolean {
    const executed = this.game.tryMove({ from, to, promotion: promotion ?? 'q' });
    if (!executed) {
      return false;
    }

    this.lastMove = {
      from,
      to,
      san: executed.san,
      promotion: executed.promotion as PromotionPiece | undefined,
    };
    this.lastEvaluatedFen = this.game.fen();
    this.initializeBaselineFromGame();
    this.status = this.game.isGameOver() ? 'game_over' : 'watching';
    this.stableFrameCount = 0;
    this.pendingPromotion = null;
    this.pendingObservations = null;
    return true;
  }

  getGame(): ChessGame {
    return this.game;
  }

  getFen(): string {
    return this.game.fen();
  }

  getTurn(): 'w' | 'b' {
    return this.game.turn();
  }

  /** Changes side to move without discarding the confirmed placement. */
  setTurn(turn: 'w' | 'b'): boolean {
    const parts = this.game.fen().split(' ');
    if (parts.length < 2 || parts[1] === turn) return true;
    parts[1] = turn;
    const fen = parts.join(' ');
    try {
      const game = new ChessGame(fen);
      if (game.fen() !== fen) return false;
      this.game = game;
      this.lastEvaluatedFen = fen;
      this.stableFrameCount = 0;
      this.previousTopClasses = null;
      this.previousUncertain = null;
      this.clearProvisionalCandidate();
      this.status = this.game.isGameOver() ? 'game_over' : 'watching';
      return true;
    } catch {
      return false;
    }
  }

  getStatus(): RecognizerStatus {
    return this.status;
  }

  getLastMove(): RecognizedMove | null {
    return this.lastMove;
  }

  reset(fen?: string): void {
    this.game = new ChessGame(fen);
    this.status = 'watching';
    this.lastMove = null;
    this.lastEvaluatedFen = this.game.fen();
    this.stableFrameCount = 0;
    this.previousFrame = null;
    this.previousTopClasses = null;
    this.previousUncertain = null;
    this.pendingPromotion = null;
    this.pendingObservations = null;
    this.clearProvisionalCandidate();
    this.positionConfirmed = Boolean(fen);
    this.initializeBaselineFromGame();
  }
}
