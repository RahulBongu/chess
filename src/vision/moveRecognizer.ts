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

  private readonly requiredStableFrames: number;
  private readonly maxInterFrameFluctuation: number;
  private readonly minDeltaConfidence: number;
  private readonly minDeltaVisibility: number;

  constructor(options: MoveRecognizerOptions = {}) {
    this.requiredStableFrames = options.requiredStableFrames ?? 3;
    this.maxInterFrameFluctuation = options.maxInterFrameFluctuation ?? 2;
    this.minDeltaConfidence = options.minDeltaConfidence ?? 0.60;
    this.minDeltaVisibility = options.minDeltaVisibility ?? 0.60;
    this.game = new ChessGame(options.initialFen);
    this.lastEvaluatedFen = this.game.fen();
    this.initializeBaselineFromGame();
  }

  /**
   * Initializes baseline occupancy and color expectations from the current authoritative ChessGame state.
   */
  private initializeBaselineFromGame(): void {
    this.baseline = new Map();
    this.baselineObs = new Map();
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
    this.stableFrameCount = 0;
    this.pendingPromotion = null;
    this.pendingObservations = null;
    this.status = 'watching';
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

    if (this.pendingObservations) {
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
    this.pendingPromotion = null;
    this.pendingObservations = null;
    this.initializeBaselineFromGame();
  }
}
