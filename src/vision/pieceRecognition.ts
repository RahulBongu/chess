/**
 * Piece-aware board adoption and legal-move reconciliation.
 *
 * The detector proposes geometry. This module is the evidence gate: a board is
 * not trusted until the 13-class square model produces a legal, stable
 * position. Move candidates are then scored against the observed class
 * probabilities rather than against luminance changes.
 */

import { ChessGame, STARTING_FEN, type ChessSquare, type PromotionPiece } from '../chess/ChessGame.ts';
import type { PieceType } from './types.ts';
import type { SquareObservation } from './moveRecognizer.ts';
import { colRowToAlgebraic, type BoardOrientation } from './perspectiveOverlay.ts';

export const PIECE_CLASSES = [
  'empty',
  'white_pawn',
  'white_knight',
  'white_bishop',
  'white_rook',
  'white_queen',
  'white_king',
  'black_pawn',
  'black_knight',
  'black_bishop',
  'black_rook',
  'black_queen',
  'black_king',
] as const;

export type PieceClass = (typeof PIECE_CLASSES)[number];
export const PIECE_CLASS_INDEX: Readonly<Record<PieceClass, number>> = Object.fromEntries(
  PIECE_CLASSES.map((name, index) => [name, index]),
) as Record<PieceClass, number>;

const PIECE_TYPE_TO_NAME: Readonly<Record<PieceType, string>> = {
  pawn: 'pawn',
  knight: 'knight',
  bishop: 'bishop',
  rook: 'rook',
  queen: 'queen',
  king: 'king',
};

const PIECE_SYMBOL_TO_NAME: Readonly<Record<string, string>> = {
  p: 'pawn',
  n: 'knight',
  b: 'bishop',
  r: 'rook',
  q: 'queen',
  k: 'king',
};

const FEN_CHAR: Readonly<Record<PieceClass, string>> = {
  empty: '',
  white_pawn: 'P',
  white_knight: 'N',
  white_bishop: 'B',
  white_rook: 'R',
  white_queen: 'Q',
  white_king: 'K',
  black_pawn: 'p',
  black_knight: 'n',
  black_bishop: 'b',
  black_rook: 'r',
  black_queen: 'q',
  black_king: 'k',
};

export function pieceClassFromType(type: PieceType, color: 'w' | 'b'): PieceClass {
  const name = PIECE_TYPE_TO_NAME[type];
  return `${color === 'w' ? 'white' : 'black'}_${name}` as PieceClass;
}

export function pieceClassFromFenChar(char: string): PieceClass {
  if (!char) return 'empty';
  const type = char.toLowerCase();
  const name =
    type === 'p'
      ? 'pawn'
      : type === 'n'
        ? 'knight'
        : type === 'b'
          ? 'bishop'
          : type === 'r'
            ? 'rook'
            : type === 'q'
              ? 'queen'
              : 'king';
  return `${char === char.toUpperCase() ? 'white' : 'black'}_${name}` as PieceClass;
}

export function classToPiece(
  pieceClass: PieceClass,
): { type: PieceType; color: 'w' | 'b' } | null {
  if (pieceClass === 'empty') return null;
  const [colorName, ...rest] = pieceClass.split('_');
  const typeName = rest.join('_');
  return {
    type: typeName as PieceType,
    color: colorName === 'white' ? 'w' : 'b',
  };
}

export function classProbabilitiesToTop(
  probabilities: ArrayLike<number>,
): { index: number; probability: number; secondProbability: number } | null {
  if (!probabilities || probabilities.length !== PIECE_CLASSES.length) return null;
  let best = -1;
  let bestIndex = -1;
  let second = -1;
  for (let index = 0; index < probabilities.length; index += 1) {
    const value = Number(probabilities[index]);
    if (!Number.isFinite(value)) return null;
    if (value > best) {
      second = best;
      best = value;
      bestIndex = index;
    } else if (value > second) {
      second = value;
    }
  }
  return {
    index: bestIndex,
    probability: best,
    secondProbability: second,
  };
}

/** Stable softmax for one 13-class square vector of raw model logits. */
export function softmaxSquareScores(scores: ArrayLike<number>): number[][] | null {
  const classCount = PIECE_CLASSES.length;
  if (!scores || scores.length !== 64 * classCount) {
    return null;
  }
  const result: number[][] = [];
  for (let square = 0; square < 64; square += 1) {
    const offset = square * classCount;
    let maximum = -Infinity;
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const value = Number(scores[offset + classIndex]);
      if (!Number.isFinite(value)) return null;
      if (value > maximum) maximum = value;
    }
    let total = 0;
    const probabilities = new Array<number>(classCount);
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const value = Math.exp(Number(scores[offset + classIndex]) - maximum);
      probabilities[classIndex] = value;
      total += value;
    }
    if (!(total > 0) || !Number.isFinite(total)) return null;
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      probabilities[classIndex] /= total;
    }
    result.push(probabilities);
  }
  return result;
}

/**
 * Converts the classifier's board-view row-major output into algebraic
 * observations. Row 0 is the top edge of the board as seen by the camera.
 */
export function observationsFromSquareProbabilities(
  probabilities: number[][],
  orientation: BoardOrientation,
): SquareObservation[] {
  if (probabilities.length !== 64) {
    throw new Error(`Expected 64 square probability vectors, got ${probabilities.length}.`);
  }
  const observations: SquareObservation[] = [];
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      const square = colRowToAlgebraic(column, row, orientation) as ChessSquare;
      const vector = probabilities[row * 8 + column];
      const top = classProbabilitiesToTop(vector);
      if (!top) {
        throw new Error(`Invalid class probabilities for ${square}.`);
      }
      const pieceClass = PIECE_CLASSES[top.index];
      const isOccupied = pieceClass !== 'empty' && top.probability >= 0.5;
      const [, ...typeParts] = pieceClass.split('_');
      observations.push({
        square,
        occupied: isOccupied,
        color: isOccupied ? (pieceClass.startsWith('white_') ? 'w' : 'b') : null,
        confidence: top.probability,
        visibility: 1,
        classProbabilities: vector.slice(),
        uncertain: false,
        provenance: 'observed',
      });
    }
  }
  return observations;
}

/**
 * Rotates an 8x8 board-grid probability map from sensor orientation into the
 * upright display orientation. CameraX analysis buffers are not rotated even
 * when the preview is, so a portrait phone delivers a landscape board image
 * rotated by 90/270 degrees. Mapping those rows directly to ranks would put
 * every pawn on a file and make legal adoption fail.
 */
export function rotateSquareProbabilities(
  probabilities: number[][],
  rotationDegrees: number,
): number[][] | null {
  if (probabilities.length !== 64) return null;
  const rotation = ((Math.round(rotationDegrees) % 360) + 360) % 360;
  if (rotation !== 0 && rotation !== 90 && rotation !== 180 && rotation !== 270) {
    return null;
  }
  if (rotation === 0) return probabilities;

  const rotated: number[][] = new Array(64);
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      let sourceRow = row;
      let sourceColumn = column;
      if (rotation === 90) {
        // Clockwise: the old bottom-left becomes the new top-left.
        sourceRow = 7 - column;
        sourceColumn = row;
      } else if (rotation === 180) {
        sourceRow = 7 - row;
        sourceColumn = 7 - column;
      } else {
        // 270 clockwise (90 counter-clockwise).
        sourceRow = column;
        sourceColumn = 7 - row;
      }
      const source = probabilities[sourceRow * 8 + sourceColumn];
      if (!Array.isArray(source) || source.length !== PIECE_CLASSES.length) {
        return null;
      }
      rotated[row * 8 + column] = source;
    }
  }
  return rotated;
}

export interface DiagnosticSquareDetail {
  square: ChessSquare;
  topClass: PieceClass;
  probability: number;
}

export interface SquareEvidenceSummary {
  occupiedSquares: number;
  averageConfidence: number;
  uncertainSquares: ChessSquare[];
  diagnosticSquares: DiagnosticSquareDetail[];
}

/**
 * Averages several complete classified frames in algebraic-square space.
 *
 * Initial adoption happens while the board is static; a single frame's noisy
 * square is uncorrelated with the next frame's noise. Move detection does NOT
 * use this path because averaging across a piece transition would smear the
 * evidence the move scorer needs.
 */
export function averageSquareObservations(
  frames: SquareObservation[][],
): SquareObservation[] | null {
  if (frames.length === 0) return null;
  const sums = new Map<ChessSquare, number[]>();
  for (const frame of frames) {
    if (frame.length !== 64) return null;
    for (const observation of frame) {
      const probabilities = observation.classProbabilities;
      if (!probabilities || probabilities.length !== PIECE_CLASSES.length) return null;
      const existing = sums.get(observation.square);
      if (!existing) {
        sums.set(observation.square, Array.from(probabilities));
      } else {
        for (let index = 0; index < probabilities.length; index += 1) {
          existing[index] += probabilities[index];
        }
      }
    }
  }
  if (sums.size !== 64) return null;

  const result: SquareObservation[] = [];
  for (const [square, sum] of sums) {
    const probabilities = sum.map((value) => value / frames.length);
    const top = classProbabilitiesToTop(probabilities);
    if (!top) return null;
    const pieceClass = PIECE_CLASSES[top.index];
    const occupied = pieceClass !== 'empty' && top.probability >= 0.5;
    result.push({
      square,
      occupied,
      color: occupied ? (pieceClass.startsWith('white_') ? 'w' : 'b') : null,
      confidence: top.probability,
      visibility: 1,
      classProbabilities: probabilities,
    });
  }
  return result;
}

/**
 * Builds the observation set implied by a FEN.
 *
 * A confirmed position is a belief, not a frame. Seeding the move recogniser
 * from the noisy frame that happened to pass the gate would immediately treat
 * that noise as the baseline and make the next cleaner frame look like a move.
 * Synthesising the baseline from the accepted FEN keeps the tracker anchored to
 * the position the user is actually analysing.
 */
export function observationsFromFen(fen: string): SquareObservation[] | null {
  let game: ChessGame;
  try {
    game = new ChessGame(fen);
  } catch {
    return null;
  }
  const result: SquareObservation[] = [];
  for (let rank = 8; rank >= 1; rank -= 1) {
    for (let file = 0; file < 8; file += 1) {
      const square = `${String.fromCharCode(97 + file)}${rank}` as ChessSquare;
      const piece = game.pieceAt(square);
      const pieceClass: PieceClass = piece
        ? (`${piece.color === 'w' ? 'white' : 'black'}_${PIECE_SYMBOL_TO_NAME[piece.type]}` as PieceClass)
        : 'empty';
      const probabilities = new Array<number>(PIECE_CLASSES.length).fill(0);
      probabilities[PIECE_CLASS_INDEX[pieceClass]] = 1;
      result.push({
        square,
        occupied: pieceClass !== 'empty',
        color: pieceClass === 'empty' ? null : pieceClass.startsWith('white_') ? 'w' : 'b',
        confidence: 1,
        visibility: 1,
        classProbabilities: probabilities,
        uncertain: false,
        provenance: 'inferred',
      });
    }
  }
  return result;
}

const DIAGNOSTIC_WATCH_SQUARES: readonly ChessSquare[] = [
  'e1',
  'e8',
  'd1',
  'd8',
  'e4',
  'd4',
];

/** Lightweight HUD summary of the same evidence the adoption gate uses. */
export function summarizeSquareEvidence(observations: SquareObservation[]): SquareEvidenceSummary {
  let occupiedSquares = 0;
  let confidenceSum = 0;
  const uncertainSquares: ChessSquare[] = [];
  const diagnosticSquares: DiagnosticSquareDetail[] = [];

  for (const observation of observations) {
    const top = observation.classProbabilities
      ? classProbabilitiesToTop(observation.classProbabilities)
      : null;
    if (!top) continue;
    const pieceClass = PIECE_CLASSES[top.index];

    if (DIAGNOSTIC_WATCH_SQUARES.includes(observation.square)) {
      diagnosticSquares.push({
        square: observation.square,
        topClass: pieceClass,
        probability: Math.round(top.probability * 100) / 100,
      });
    }

    if (observation.uncertain === true) {
      uncertainSquares.push(observation.square);
      continue;
    }
    if (pieceClass === 'empty') continue;
    if (top.probability >= 0.34 && top.probability - top.secondProbability >= 0.06) {
      occupiedSquares += 1;
      confidenceSum += top.probability;
    } else {
      uncertainSquares.push(observation.square);
    }
  }
  return {
    occupiedSquares,
    averageConfidence: occupiedSquares > 0 ? confidenceSum / occupiedSquares : 0,
    uncertainSquares,
    diagnosticSquares,
  };
}

export interface OrientationRecognition {
  orientation: BoardOrientation;
  position: RecognizedPosition;
}

/**
 * A still image cannot prove whose turn it is. Testing both board
 * orientations helps because the wrong one usually places pawns on a back
 * rank or produces an implausible piece count, which the legal-position gate
 * rejects. When both are legal, the caller's active orientation wins.
 */
export function recognizeBothOrientations(
  probabilities: number[][],
  turn: 'w' | 'b' = 'w',
  preferred: BoardOrientation = 'white_bottom',
): OrientationRecognition[] {
  const candidates: OrientationRecognition[] = (['white_bottom', 'black_bottom'] as const).map(
    (orientation) => ({
      orientation,
      position: recognizePositionFromObservations(
        observationsFromSquareProbabilities(probabilities, orientation),
        { turn },
      ),
    }),
  );
  return candidates.sort((a, b) => {
    if (a.position.ok !== b.position.ok) return a.position.ok ? -1 : 1;
    if (a.orientation === preferred) return -1;
    if (b.orientation === preferred) return 1;
    return b.position.averageConfidence - a.position.averageConfidence;
  });
}

function squareFileRank(square: ChessSquare): { file: number; rank: number } {
  return { file: square.charCodeAt(0) - 97, rank: Number(square[1]) };
}

function squareName(file: number, rank: number): ChessSquare {
  return `${String.fromCharCode(97 + file)}${rank}` as ChessSquare;
}

export interface StartingPositionVerification {
  ok: boolean;
  fen: string;
  reason: string | null;
  matchingSquares: number;
  whiteHomeCount: number;
  blackHomeCount: number;
  centerOccupiedCount: number;
  occupiedCenterSquares: ChessSquare[];
  missingWhiteHomeSquares: ChessSquare[];
  missingBlackHomeSquares: ChessSquare[];
  /** Home-rank squares occupied by the opposite colour (orientation error). */
  wrongColorHomeErrors: number;
  totalOccupied: number;
  /** True when genuinely uncertain squares were resolved from the expected layout. */
  usedPrior: boolean;
  /** Squares whose class came from the expected layout, not the classifier. */
  priorResolvedSquares: ChessSquare[];
}

export interface StartingPositionVerificationOptions {
  /**
   * Genuinely uncertain squares that may be resolved from the expected layout.
   * Default 0 (strict): no prior resolution. Callers should require repeated
   * fused evidence before allowing more than zero; the useful maximum is 3.
   */
  maxUncertainSquares?: number;
  /**
   * Expected layout used only for uncertain squares. Only the standard
   * starting FEN is accepted: this option must never be used to push a
   * mid-game board back to the start.
   */
  priorFen?: string;
}

function expectedClassAt(game: ChessGame, square: ChessSquare): PieceClass {
  const piece = game.pieceAt(square);
  return piece
    ? pieceClassFromFenChar(
        piece.color === 'w' ? piece.type.toUpperCase() : piece.type,
      )
    : 'empty';
}

/** True when a square is too ambiguous to act as proof, but not contradictory. */
function isConfidentContradiction(
  observation: SquareObservation,
  expected: PieceClass,
): boolean {
  const top = observation.classProbabilities
    ? classProbabilitiesToTop(observation.classProbabilities)
    : null;
  if (!top) return false;
  const observed = PIECE_CLASSES[top.index];
  return (
    observed !== expected &&
    top.probability >= 0.8 &&
    top.probability - top.secondProbability >= 0.4
  );
}

function failedStartingVerification(reason: string): StartingPositionVerification {
  return {
    ok: false,
    fen: '',
    reason,
    matchingSquares: 0,
    whiteHomeCount: 0,
    blackHomeCount: 0,
    centerOccupiedCount: 0,
    occupiedCenterSquares: [],
    missingWhiteHomeSquares: [],
    missingBlackHomeSquares: [],
    wrongColorHomeErrors: 0,
    totalOccupied: 0,
    usedPrior: false,
    priorResolvedSquares: [],
  };
}

/**
 * Conclusively verifies whether the observed board matches the untouched 32-piece
 * chess starting position without relying on brittle minor-piece type classification.
 *
 * Strict requirements (defaults):
 * - Central ranks 3, 4, 5, 6 (all 32 squares) must be completely empty (centerOccupied === 0).
 * - White home ranks 1 and 2 (all 16 squares) must be occupied by White pieces (whiteHome === 16).
 * - Black home ranks 7 and 8 (all 16 squares) must be occupied by Black pieces (blackHome === 16).
 * - Zero inverted-color pieces on home ranks.
 * - Total occupied count must be exactly 32.
 * - All 64 squares must match the starting template (matchingSquares === 64).
 *
 * The starting layout may be used as a *prior* only when the caller explicitly
 * allows a bounded number of genuinely uncertain squares. Confident
 * contradictions still block, and the caller must still require explicit user
 * confirmation before adopting STARTING_FEN.
 */
export function verifyStartingPosition(
  observations: SquareObservation[],
  orientation: BoardOrientation = 'white_bottom',
  options: StartingPositionVerificationOptions = {},
): StartingPositionVerification {
  const maxUncertainSquares = Math.max(0, Math.floor(options.maxUncertainSquares ?? 0));
  const priorFen = options.priorFen ?? STARTING_FEN;

  if (observations.length !== 64) {
    return failedStartingVerification(
      `Expected 64 square observations, got ${observations.length}`,
    );
  }

  let priorGame: ChessGame | null = null;
  if (maxUncertainSquares > 0) {
    if (priorFen !== STARTING_FEN) {
      return failedStartingVerification(
        'Only the standard starting layout may be used as a prior.',
      );
    }
    try {
      priorGame = new ChessGame(priorFen);
      if (priorGame.fen() !== priorFen) {
        return failedStartingVerification('The expected starting layout is not a legal position.');
      }
    } catch {
      return failedStartingVerification('The expected starting layout is not a legal position.');
    }
  }

  const uncertainSquares = observations
    .filter((observation) => observation.uncertain === true)
    .map((observation) => observation.square);
  if (uncertainSquares.length > maxUncertainSquares) {
    return {
      ...failedStartingVerification(
        `${uncertainSquares.length} square${uncertainSquares.length === 1 ? '' : 's'} too uncertain to verify a starting position.`,
      ),
    };
  }

  if (priorGame) {
    for (const observation of observations) {
      if (observation.uncertain !== true) continue;
      const expected = expectedClassAt(priorGame, observation.square);
      if (isConfidentContradiction(observation, expected)) {
        return {
          ...failedStartingVerification(
            `Confident evidence on ${observation.square} contradicts the expected starting layout.`,
          ),
        };
      }
    }
  }

  let whiteHomeCount = 0;
  let blackHomeCount = 0;
  let centerEmptyCount = 0;
  let centerOccupiedCount = 0;
  let wrongColorHomeErrors = 0;
  let totalOccupied = 0;
  let matchingSquares = 0;

  const occupiedCenterSquares: ChessSquare[] = [];
  const missingWhiteHomeSquares: ChessSquare[] = [];
  const missingBlackHomeSquares: ChessSquare[] = [];
  const priorResolvedSquares: ChessSquare[] = [];

  for (const obs of observations) {
    const { rank } = squareFileRank(obs.square);
    const top = obs.classProbabilities
      ? classProbabilitiesToTop(obs.classProbabilities)
      : null;
    const observedClass = top ? PIECE_CLASSES[top.index] : 'empty';
    const flaggedUncertain = obs.uncertain === true;
    let effectiveClass = observedClass;

    if (flaggedUncertain && priorGame) {
      effectiveClass = expectedClassAt(priorGame, obs.square);
      priorResolvedSquares.push(obs.square);
    }

    const useObservedColor = !(flaggedUncertain && priorGame);
    const isWhite =
      effectiveClass.startsWith('white_') || (useObservedColor && obs.color === 'w');
    const isBlack =
      effectiveClass.startsWith('black_') || (useObservedColor && obs.color === 'b');
    const isEmpty =
      effectiveClass === 'empty' && (flaggedUncertain && priorGame ? true : !obs.occupied);

    if (!isEmpty) {
      totalOccupied += 1;
    }

    if (rank === 1 || rank === 2) {
      // Expected White home ranks
      if (isWhite) {
        whiteHomeCount += 1;
        matchingSquares += 1;
      } else {
        missingWhiteHomeSquares.push(obs.square);
        if (isBlack) {
          wrongColorHomeErrors += 1;
        }
      }
    } else if (rank === 7 || rank === 8) {
      // Expected Black home ranks
      if (isBlack) {
        blackHomeCount += 1;
        matchingSquares += 1;
      } else {
        missingBlackHomeSquares.push(obs.square);
        if (isWhite) {
          wrongColorHomeErrors += 1;
        }
      }
    } else {
      // Central empty ranks: 3, 4, 5, 6 (32 squares)
      if (isEmpty) {
        centerEmptyCount += 1;
        matchingSquares += 1;
      } else {
        centerOccupiedCount += 1;
        occupiedCenterSquares.push(obs.square);
      }
    }
  }

  const ok =
    centerOccupiedCount === 0 &&
    whiteHomeCount === 16 &&
    blackHomeCount === 16 &&
    wrongColorHomeErrors === 0 &&
    totalOccupied === 32 &&
    matchingSquares === 64;

  let reason: string | null = null;
  if (!ok) {
    if (centerOccupiedCount > 0) {
      reason = `Piece detected in center (${occupiedCenterSquares.join(', ')} occupied in ranks 3-6); not an untouched starting position.`;
    } else if (wrongColorHomeErrors > 0) {
      reason = `Inverted piece color detected on home rank (${wrongColorHomeErrors} pieces). Check White/Black bottom toggle.`;
    } else if (missingWhiteHomeSquares.length > 0) {
      reason = `White home square ${missingWhiteHomeSquares.join(', ')} is not occupied by White (expected White piece in starting setup).`;
    } else if (missingBlackHomeSquares.length > 0) {
      reason = `Black home square ${missingBlackHomeSquares.join(', ')} is not occupied by Black (expected Black piece in starting setup).`;
    } else if (totalOccupied !== 32) {
      reason = `Detected piece count is ${totalOccupied}/32; not the 32-piece starting setup.`;
    } else {
      reason = `Board layout matches ${matchingSquares}/64 starting squares.`;
    }
  }

  return {
    ok,
    fen: ok ? STARTING_FEN : '',
    reason,
    matchingSquares,
    whiteHomeCount,
    blackHomeCount,
    centerOccupiedCount,
    occupiedCenterSquares,
    missingWhiteHomeSquares,
    missingBlackHomeSquares,
    wrongColorHomeErrors,
    totalOccupied,
    usedPrior: priorResolvedSquares.length > 0,
    priorResolvedSquares,
  };
}

export interface RecognizedPosition {
  ok: boolean;
  fen: string;
  reason: string | null;
  occupied: number;
  averageConfidence: number;
  uncertainSquares: ChessSquare[];
  whiteKingConfidence: number;
  blackKingConfidence: number;
}

export interface AdoptionOptions {
  /** Minimum top-1 probability for an occupied square to be written to the FEN. */
  minPieceProbability?: number;
  /** Minimum top-1 minus top-2 margin for an occupied square. */
  minPieceMargin?: number;
  /** Maximum number of low-evidence occupied-looking squares tolerated. */
  maxUncertainSquares?: number;
  /** Side to move cannot be inferred from a still image; caller supplies it. */
  turn?: 'w' | 'b';
}

/**
 * Best-effort placement for the assisted board editor.
 *
 * This intentionally ignores per-square confidence but keeps the structural
 * checks (both kings, piece counts, no pawns on the back rank). The result is
 * never adopted automatically: it is a starting point the user must review and
 * correct square by square.
 */
export function suggestPositionFromObservations(
  observations: SquareObservation[],
  options: Pick<AdoptionOptions, 'turn'> = {},
): RecognizedPosition {
  return recognizePositionFromObservations(observations, {
    turn: options.turn,
    minPieceProbability: 0,
    minPieceMargin: 0,
    maxUncertainSquares: 64,
  });
}

/**
 * Builds a FEN from the square model and rejects it unless it is a plausible,
 * legal chess position. This is deliberately conservative: preserving the
 * previous confirmed position is better than inventing one.
 */
export function recognizePositionFromObservations(
  observations: SquareObservation[],
  options: AdoptionOptions = {},
): RecognizedPosition {
  const minPieceProbability = options.minPieceProbability ?? 0.34;
  const minPieceMargin = options.minPieceMargin ?? 0.06;
  const maxUncertainSquares = options.maxUncertainSquares ?? 0;
  const turn = options.turn ?? 'w';

  const bySquare = new Map<ChessSquare, SquareObservation>();
  for (const observation of observations) {
    bySquare.set(observation.square, observation);
  }
  if (bySquare.size !== 64) {
    return emptyRecognizedPosition('A full 64-square observation set is required.');
  }

  const board: (PieceClass | null)[][] = Array.from({ length: 8 }, () =>
    Array.from({ length: 8 }, () => null),
  );
  const uncertainSquares: ChessSquare[] = [];
  let occupied = 0;
  let confidenceSum = 0;
  let whiteKingConfidence = 0;
  let blackKingConfidence = 0;

  for (const observation of observations) {
    const top = observation.classProbabilities
      ? classProbabilitiesToTop(observation.classProbabilities)
      : null;
    if (!top) {
      return emptyRecognizedPosition(`No class probabilities for ${observation.square}.`);
    }
    const pieceClass = PIECE_CLASSES[top.index];
    const { file, rank } = squareFileRank(observation.square);
    const row = 8 - rank;
    const flaggedUncertain = observation.uncertain === true;
    const occupiedCandidate =
      !flaggedUncertain &&
      pieceClass !== 'empty' &&
      top.probability >= minPieceProbability &&
      top.probability - top.secondProbability >= minPieceMargin;

    if (flaggedUncertain) {
      // Never invent an identity (or an empty square) from evidence fusion has
      // already declared ambiguous. The structural gate still decides whether
      // the surrounding board is plausible, and the caller's uncertainty
      // budget decides whether it may be adopted at all.
      uncertainSquares.push(observation.square);
      board[row][file] = 'empty';
    } else if (pieceClass !== 'empty' && !occupiedCandidate) {
      uncertainSquares.push(observation.square);
      board[row][file] = 'empty';
    } else {
      board[row][file] = pieceClass;
    }

    if (occupiedCandidate) {
      occupied += 1;
      confidenceSum += top.probability;
      if (pieceClass === 'white_king') {
        whiteKingConfidence = top.probability;
      } else if (pieceClass === 'black_king') {
        blackKingConfidence = top.probability;
      }
    }
  }

  if (uncertainSquares.length > maxUncertainSquares) {
    return {
      ...emptyRecognizedPosition(
        `${uncertainSquares.length} squares are too uncertain to confirm a position.`,
      ),
      uncertainSquares,
    };
  }
  if (whiteKingConfidence <= 0 || blackKingConfidence <= 0) {
    return {
      ...emptyRecognizedPosition(
        'Both kings were not recognized with enough confidence; board evidence rejected.',
      ),
      uncertainSquares,
      whiteKingConfidence,
      blackKingConfidence,
    };
  }

  const counts = new Map<PieceClass, number>();
  for (const row of board) {
    for (const pieceClass of row) {
      if (!pieceClass || pieceClass === 'empty') continue;
      counts.set(pieceClass, (counts.get(pieceClass) ?? 0) + 1);
    }
  }
  if (occupied < 2 || occupied > 32) {
    return emptyRecognizedPosition(`Implausible occupied-square count: ${occupied}.`);
  }
  for (const color of ['white', 'black'] as const) {
    let colorCount = 0;
    let pawnCount = 0;
    for (const [pieceClass, count] of counts.entries()) {
      if (!pieceClass.startsWith(color)) continue;
      colorCount += count;
      if (pieceClass.endsWith('_pawn')) pawnCount += count;
    }
    if (colorCount > 16) {
      return emptyRecognizedPosition(`${color} has ${colorCount} pieces; maximum is 16.`);
    }
    if (pawnCount > 8) {
      return emptyRecognizedPosition(`${color} has ${pawnCount} pawns; maximum is 8.`);
    }
    if ((counts.get(`${color}_king` as PieceClass) ?? 0) !== 1) {
      return emptyRecognizedPosition(`${color} does not have exactly one recognized king.`);
    }
  }

  for (let rank = 1; rank <= 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const pieceClass = board[8 - rank][file];
      if (pieceClass && pieceClass !== 'empty' && pieceClass.endsWith('_pawn') && (rank === 1 || rank === 8)) {
        return emptyRecognizedPosition('A pawn was recognized on the first or eighth rank.');
      }
    }
  }

  const placement = board
    .map((row) => {
      let result = '';
      let emptyRun = 0;
      for (const pieceClass of row) {
        if (!pieceClass || pieceClass === 'empty') {
          emptyRun += 1;
        } else {
          if (emptyRun > 0) result += String(emptyRun);
          result += FEN_CHAR[pieceClass];
          emptyRun = 0;
        }
      }
      if (emptyRun > 0) result += String(emptyRun);
      return result;
    })
    .join('/');

  const castling = inferCastlingRights(board);
  const fen = `${placement} ${turn} ${castling} - 0 1`;
  try {
    const game = new ChessGame(fen);
    // chess.js normalizes some placements (for example by filling in
    // castling or en-passant defaults). A mismatch means the placement is
    // not an exact, legal position and must not be adopted.
    if (game.fen() !== fen) {
      return emptyRecognizedPosition('The recognized placement is not a legal chess position.');
    }
  } catch {
    return emptyRecognizedPosition('The recognized placement is not a legal chess position.');
  }

  return {
    ok: true,
    fen,
    reason: null,
    occupied,
    averageConfidence: occupied > 0 ? confidenceSum / occupied : 0,
    uncertainSquares,
    whiteKingConfidence,
    blackKingConfidence,
  };
}

function inferCastlingRights(board: (PieceClass | null)[][]): string {
  let rights = '';
  if (board[7][4] === 'white_king') {
    if (board[7][7] === 'white_rook') rights += 'K';
    if (board[7][0] === 'white_rook') rights += 'Q';
  }
  if (board[0][4] === 'black_king') {
    if (board[0][7] === 'black_rook') rights += 'k';
    if (board[0][0] === 'black_rook') rights += 'q';
  }
  return rights || '-';
}

function emptyRecognizedPosition(reason: string): RecognizedPosition {
  return {
    ok: false,
    fen: '',
    reason,
    occupied: 0,
    averageConfidence: 0,
    uncertainSquares: [],
    whiteKingConfidence: 0,
    blackKingConfidence: 0,
  };
}

export function boardClassesFromGame(game: ChessGame): PieceClass[] {
  const classes: PieceClass[] = [];
  for (let rank = 8; rank >= 1; rank -= 1) {
    for (let file = 0; file < 8; file += 1) {
      const square = squareName(file, rank);
      const piece = game.pieceAt(square);
      classes.push(piece ? pieceClassFromFenChar(piece.color === 'w' ? piece.type.toUpperCase() : piece.type) : 'empty');
    }
  }
  return classes;
}

function probability(
  observation: SquareObservation | undefined,
  pieceClass: PieceClass,
): number {
  const probs = observation?.classProbabilities;
  if (!probs || probs.length !== PIECE_CLASSES.length) return 0;
  const value = Number(probs[PIECE_CLASS_INDEX[pieceClass]]);
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
}

function topClass(observation: SquareObservation | undefined): PieceClass | null {
  if (!observation?.classProbabilities) return null;
  const top = classProbabilitiesToTop(observation.classProbabilities);
  return top ? PIECE_CLASSES[top.index] : null;
}

/**
 * Finds en-passant targets implied by the observed transition. A candidate
 * requires: a newly arrived pawn on the fifth/sixth rank, an empty destination
 * in the confirmed position, an enemy pawn directly behind it, and a friendly
 * pawn on an adjacent file that could have captured it.
 */
function inferEnPassantTargets(
  game: ChessGame,
  bySquare: Map<ChessSquare, SquareObservation>,
): ChessSquare[] {
  const mover = game.turn();
  const pawnClass: PieceClass = mover === 'w' ? 'white_pawn' : 'black_pawn';
  const enemyPawnClass: PieceClass = mover === 'w' ? 'black_pawn' : 'white_pawn';
  const before = boardClassesFromGame(game);
  const targets: ChessSquare[] = [];

  for (let index = 0; index < 64; index += 1) {
    const rank = 8 - Math.floor(index / 8);
    const file = index % 8;
    const square = squareName(file, rank);
    if (topClass(bySquare.get(square)) !== pawnClass) continue;
    if (before[index] !== 'empty') continue;
    if (mover === 'w' && rank !== 6) continue;
    if (mover === 'b' && rank !== 3) continue;

    const capturedRank = mover === 'w' ? rank - 1 : rank + 1;
    const capturedIndex = (8 - capturedRank) * 8 + file;
    if (before[capturedIndex] !== enemyPawnClass) continue;
    if (topClass(bySquare.get(squareName(file, capturedRank))) !== 'empty') continue;

    const sourceRank = mover === 'w' ? rank - 1 : rank + 1;
    let sourceExists = false;
    for (const delta of [-1, 1]) {
      const sourceFile = file + delta;
      if (sourceFile < 0 || sourceFile > 7) continue;
      if (before[(8 - sourceRank) * 8 + sourceFile] === pawnClass) {
        sourceExists = true;
        break;
      }
    }
    if (sourceExists) targets.push(square);
  }
  return targets;
}

export interface MoveEvidenceScore {
  move: {
    from: ChessSquare;
    to: ChessSquare;
    san: string;
    promotion?: PromotionPiece;
  };
  /** Exact FEN used to generate this move (may carry an inferred ep target). */
  sourceFen: string;
  score: number;
  coreProbability: number;
  changedSquares: ChessSquare[];
  /** Changed squares whose fused evidence is confident. */
  reliableChangedSquares: ChessSquare[];
  /** Changed squares resolved by fusion as genuinely uncertain. */
  uncertainChangedSquares: ChessSquare[];
  sourceEmptyProbability: number;
  destinationPieceProbability: number;
  unexpectedChanges: ChessSquare[];
  /** True when the candidate relies on at least one uncertain changed square. */
  provisional: boolean;
  valid: boolean;
  reason: string | null;
}

export interface PieceAwareReconciliation {
  best: MoveEvidenceScore | null;
  second: MoveEvidenceScore | null;
  promotionCandidates: MoveEvidenceScore[];
  valid: boolean;
  reason: string | null;
  /** All fused-uncertain squares supplied with this observation set. */
  uncertainSquares: ChessSquare[];
  /** True when the best candidate needs provisional confirmation. */
  provisional: boolean;
}

export interface ReconcileOptions {
  /** Minimum geometric mean probability over the squares the move changes. */
  minCoreProbability?: number;
  /** Minimum log-score margin over the second distinct move. */
  minLogMargin?: number;
  /** Maximum high-confidence observed changes outside the candidate move. */
  maxUnexpectedChanges?: number;
  /** Maximum fused-uncertain squares tolerated in the observation set. Default 0. */
  maxUncertainSquares?: number;
  /**
   * Minimum number of confident changed squares a candidate needs. Default 1;
   * a provisional candidate must still have at least one reliable arrival.
   */
  minReliableChangedSquares?: number;
}

/**
 * Scores every legal move by the probability mass the model assigns to the
 * exact board it would produce. Captures are never inferred from brightness.
 */
export function reconcileLegalMove(
  game: ChessGame,
  observations: SquareObservation[],
  options: ReconcileOptions = {},
): PieceAwareReconciliation {
  const minCoreProbability = options.minCoreProbability ?? 0.28;
  const minLogMargin = options.minLogMargin ?? 0.22;
  const maxUnexpectedChanges = options.maxUnexpectedChanges ?? 1;
  const maxUncertainSquares = Math.max(0, Math.floor(options.maxUncertainSquares ?? 0));
  const minReliableChangedSquares = Math.max(
    1,
    Math.floor(options.minReliableChangedSquares ?? 1),
  );

  if (observations.length !== 64) {
    return {
      best: null,
      second: null,
      promotionCandidates: [],
      valid: false,
      reason: 'Expected 64 observations.',
      uncertainSquares: [],
      provisional: false,
    };
  }
  const bySquare = new Map<ChessSquare, SquareObservation>();
  for (const observation of observations) bySquare.set(observation.square, observation);

  const uncertainSquares = new Set<ChessSquare>(
    observations
      .filter((observation) => observation.uncertain === true)
      .map((observation) => observation.square),
  );
  const uncertainList = [...uncertainSquares];
  if (uncertainSquares.size > maxUncertainSquares) {
    return {
      best: null,
      second: null,
      promotionCandidates: [],
      valid: false,
      reason: `${uncertainSquares.size} uncertain square${uncertainSquares.size === 1 ? '' : 's'} exceed the ${maxUncertainSquares}-square limit.`,
      uncertainSquares: uncertainList,
      provisional: false,
    };
  }

  const before = boardClassesFromGame(game);
  const bySquareForEp = new Map<ChessSquare, SquareObservation>();
  for (const observation of observations) bySquareForEp.set(observation.square, observation);

  // A still image cannot carry en-passant rights: the ep field is normalized
  // away when a position is adopted. Reconstruct the specific ep target that
  // the observed board implies, so a physical en-passant capture reconciles
  // instead of being reported as an unexplained change. The scoring below
  // still has to prefer this move over every other legal candidate.
  const games = [game];
  for (const target of inferEnPassantTargets(game, bySquareForEp)) {
    const parts = game.fen().split(' ');
    parts[3] = target;
    const fen = parts.join(' ');
    try {
      const augmented = new ChessGame(fen);
      if (augmented.fen() === fen) games.push(augmented);
    } catch {
      // chess.js rejected the inferred ep target; ignore it.
    }
  }

  const legalMoves: { source: ChessGame; move: ReturnType<ChessGame['allLegalMoves']>[number] }[] = [];
  const seenMoves = new Set<string>();
  for (const source of games) {
    for (const move of source.allLegalMoves()) {
      const key = `${move.from}${move.to}${move.promotion ?? ''}`;
      if (seenMoves.has(key)) continue;
      seenMoves.add(key);
      legalMoves.push({ source, move });
    }
  }
  if (legalMoves.length === 0) {
    return {
      best: null,
      second: null,
      promotionCandidates: [],
      valid: false,
      reason: 'No legal moves.',
      uncertainSquares: uncertainList,
      provisional: false,
    };
  }

  const scores: MoveEvidenceScore[] = [];
  for (const { source, move } of legalMoves) {
    const from = move.from as ChessSquare;
    const to = move.to as ChessSquare;
    const child = new ChessGame(source.fen());
    const executed = child.tryMove({
      from,
      to,
      promotion: move.promotion as PromotionPiece | undefined,
    });
    if (!executed) continue;
    const after = boardClassesFromGame(child);

    const changedSquares: ChessSquare[] = [];
    const changedIndices: number[] = [];
    const reliableChangedSquares: ChessSquare[] = [];
    const uncertainChangedSquares: ChessSquare[] = [];
    const coreLogs: number[] = [];
    let sourceEmptyProbability = 0;
    let destinationPieceProbability = 0;
    let coreProbability = 0;
    for (let index = 0; index < 64; index += 1) {
      if (before[index] === after[index]) continue;
      const rank = 8 - Math.floor(index / 8);
      const file = index % 8;
      const square = squareName(file, rank);
      const expected = after[index];
      const p = probability(bySquare.get(square), expected);
      changedSquares.push(square);
      changedIndices.push(index);
      if (uncertainSquares.has(square)) {
        // Fusion already declared this square ambiguous. Do not let its top
        // class masquerade as positive proof for the candidate.
        uncertainChangedSquares.push(square);
      } else {
        reliableChangedSquares.push(square);
        coreLogs.push(Math.log(Math.max(p, 1e-6)));
      }
      if (square === from) sourceEmptyProbability = p;
      if (square === to) destinationPieceProbability = p;
    }
    if (changedSquares.length === 0) continue;
    coreProbability =
      coreLogs.length > 0
        ? Math.exp(coreLogs.reduce((sum, value) => sum + value, 0) / coreLogs.length)
        : 0;

    const changedSet = new Set(changedSquares);
    const unexpectedChanges: ChessSquare[] = [];
    for (let index = 0; index < 64; index += 1) {
      const rank = 8 - Math.floor(index / 8);
      const file = index % 8;
      const square = squareName(file, rank);
      if (changedSet.has(square)) continue;
      const observation = bySquare.get(square);
      const top = topClass(observation);
      if (!top || top === before[index]) continue;
      const topInfo = observation?.classProbabilities
        ? classProbabilitiesToTop(observation.classProbabilities)
        : null;
      // A confident contradiction blocks even when the square is flagged
      // uncertain: an uncertain *empty* square cannot excuse a confident
      // opposite-colour piece elsewhere.
      if (topInfo && topInfo.probability >= 0.72 && topInfo.probability - topInfo.secondProbability >= 0.3) {
        unexpectedChanges.push(square);
      }
    }

    const provisional = uncertainChangedSquares.length > 0;
    const fromUncertain = uncertainSquares.has(from);
    const toUncertain = uncertainSquares.has(to);
    const hasReliableArrival = reliableChangedSquares.some((square) => {
      const index =
        (8 - Number(square[1])) * 8 + (square.charCodeAt(0) - 97);
      return after[index] !== 'empty';
    });

    let valid = true;
    let reason: string | null = null;
    if (!fromUncertain && changedSet.has(from) && sourceEmptyProbability < 0.25) {
      valid = false;
      reason = `Source ${from} was not observed as empty.`;
    } else if (!toUncertain && changedSet.has(to) && destinationPieceProbability < 0.2) {
      valid = false;
      reason = `Destination ${to} does not match the moving piece.`;
    } else if (reliableChangedSquares.length < minReliableChangedSquares) {
      valid = false;
      reason = 'Too few confident changed squares support this move.';
    } else if (provisional && !hasReliableArrival) {
      valid = false;
      reason = 'Only an emptied source was reliably observed; the arriving piece is unknown.';
    } else if (coreProbability < minCoreProbability) {
      valid = false;
      reason = `Changed-square evidence too weak (${(coreProbability * 100).toFixed(1)}%).`;
    } else if (unexpectedChanges.length > maxUnexpectedChanges) {
      valid = false;
      reason = `Unexpected changes on ${unexpectedChanges.join(', ')}.`;
    }

    scores.push({
      move: {
        from,
        to,
        san: executed.san,
        promotion: executed.promotion as PromotionPiece | undefined,
      },
      sourceFen: source.fen(),
      score:
        coreLogs.length > 0
          ? coreLogs.reduce((sum, value) => sum + value, 0) / coreLogs.length
          : Number.NEGATIVE_INFINITY,
      coreProbability,
      changedSquares,
      reliableChangedSquares,
      uncertainChangedSquares,
      sourceEmptyProbability,
      destinationPieceProbability,
      unexpectedChanges,
      provisional,
      valid,
      reason,
    });
  }

  const validScores = scores.filter((score) => score.valid).sort((a, b) => b.score - a.score);
  if (validScores.length === 0) {
    return {
      best: null,
      second: null,
      promotionCandidates: [],
      valid: false,
      reason: scores[0]?.reason ?? 'No legal move matches the observed board.',
      uncertainSquares: uncertainList,
      provisional: false,
    };
  }
  const best = validScores[0];
  const second = validScores[1] ?? null;

  // A promotion is never executed silently, even when the model prefers one
  // promoted piece. The user must confirm the piece explicitly.
  if (best.move.promotion !== undefined) {
    const promotionCandidates = validScores.filter(
      (score) => score.move.from === best.move.from && score.move.to === best.move.to,
    );
    return {
      best,
      second: validScores.find((score) => score !== best) ?? null,
      promotionCandidates,
      valid: false,
      reason: 'promotion',
      uncertainSquares: uncertainList,
      provisional: best.provisional,
    };
  }

  if (second && best.score - second.score < minLogMargin) {
    return {
      best,
      second,
      promotionCandidates: [],
      valid: false,
      reason: `Ambiguous between ${best.move.san} and ${second.move.san}.`,
      uncertainSquares: uncertainList,
      provisional: best.provisional || second.provisional,
    };
  }
  return {
    best,
    second,
    promotionCandidates: [],
    valid: true,
    reason: null,
    uncertainSquares: uncertainList,
    provisional: best.provisional,
  };
}
