import { ChessGame } from '../chess/ChessGame.ts';

export type ChessPieceSymbol =
  | 'P'
  | 'N'
  | 'B'
  | 'R'
  | 'Q'
  | 'K'
  | 'p'
  | 'n'
  | 'b'
  | 'r'
  | 'q'
  | 'k';

export type SquareEntry = ChessPieceSymbol | null | undefined | '' | '.';

export type SquareMap = Record<string, string | null | undefined>;

export interface SquareArrayEntry {
  square: string;
  piece?: string | null | undefined;
}

export type PlacementInput =
  | SquareMap
  | (string | null | undefined | SquareArrayEntry)[];

export interface CastlingRightsFlags {
  K?: boolean;
  Q?: boolean;
  k?: boolean;
  q?: boolean;
}

export interface PositionMetadataInput {
  /** Side to move: 'w' (White) or 'b' (Black). Default: 'w'. */
  turn?: 'w' | 'b';
  /**
   * Castling rights: either standard FEN string (e.g. 'KQkq', 'Kq', '-')
   * or structured object { K?: boolean; Q?: boolean; k?: boolean; q?: boolean }.
   * Default: '-' (none).
   */
  castling?: string | CastlingRightsFlags;
  /**
   * En-passant target square (e.g. 'e3', 'd6') or '-' / null.
   * Default: '-' (none).
   */
  enPassant?: string | null;
  /**
   * Halfmove clock: number of halfmoves since the last pawn advance or capture.
   * Default: 0.
   */
  halfmoveClock?: number;
  /**
   * Fullmove number: starting at 1, incremented after Black's move.
   * Default: 1.
   */
  fullmoveNumber?: number;
}

export interface FenBuilderOptions {
  /**
   * Display orientation used when a raw 64-element array is supplied.
   * - 'white-bottom': array index 0 is a8 (top-left), index 63 is h1 (bottom-right).
   * - 'black-bottom': array index 0 is h1 (top-left), index 63 is a8 (bottom-right).
   * Default: 'white-bottom'.
   */
  orientation?: 'white-bottom' | 'black-bottom';
  metadata?: PositionMetadataInput;
}

export interface FenValidationDetails {
  whiteKingSquare: string | null;
  blackKingSquare: string | null;
  whitePieceCount: number;
  blackPieceCount: number;
  whitePawnCount: number;
  blackPawnCount: number;
  isCheck: boolean;
  isCheckmate: boolean;
  isStalemate: boolean;
  legalMoveCount: number;
}

export interface FenBuilderResult {
  /** Full six-field FEN string if structurally valid, or candidate FEN. */
  fen: string | null;
  /** Piece placement component of FEN (the first field with '/' separators). */
  placementFen: string | null;
  /** True if the position satisfies all structural chess invariants and chess.js rules. */
  valid: boolean;
  /** Specific validation error messages. */
  errors: string[];
  /** Informational validation warnings. */
  warnings: string[];
  /** Resolved metadata components with explicit defaults documented. */
  metadata: {
    turn: 'w' | 'b';
    castling: string;
    enPassant: string;
    halfmoveClock: number;
    fullmoveNumber: number;
  };
  /** Extended chess details if basic structural validation passed. */
  details?: FenValidationDetails;
  /** Disclaimer clarifying that structural validity does not prove historical reachability. */
  reachabilityDisclaimer: string;
}

const VALID_PIECES = new Set<string>([
  'P', 'N', 'B', 'R', 'Q', 'K',
  'p', 'n', 'b', 'r', 'q', 'k',
]);

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;
const RANKS = ['1', '2', '3', '4', '5', '6', '7', '8'] as const;

export const REACHABILITY_DISCLAIMER =
  'Structural validity confirms standard chess piece placement and rule invariants, but does not prove the position is historically reachable via legal play from the initial position.';

/**
 * Checks whether an algebraic square string is syntactically valid (a1..h8).
 */
export function isValidSquareName(square: string): boolean {
  if (typeof square !== 'string' || square.length !== 2) {
    return false;
  }
  const file = square[0].toLowerCase();
  const rank = square[1];
  return file >= 'a' && file <= 'h' && rank >= '1' && rank <= '8';
}

/**
 * Normalizes input placement (dictionary or 64-element array) into an authoritative
 * 64-square dictionary keyed by lowercase algebraic square ('a1'..'h8').
 */
export function normalizePlacementInput(
  input: PlacementInput,
  orientation: 'white-bottom' | 'black-bottom' = 'white-bottom',
): {
  squares: Record<string, ChessPieceSymbol | null>;
  errors: string[];
} {
  const squares: Record<string, ChessPieceSymbol | null> = {};
  const errors: string[] = [];

  for (const file of FILES) {
    for (const rank of RANKS) {
      squares[`${file}${rank}`] = null;
    }
  }

  if (Array.isArray(input)) {
    if (input.length !== 64) {
      errors.push(`Expected array of exactly 64 squares, received ${input.length}.`);
      return { squares, errors };
    }

    // Check if entries are { square, piece } objects or raw piece symbols
    const hasExplicitSquares = input.some(
      (entry) => entry && typeof entry === 'object' && 'square' in entry,
    );

    if (hasExplicitSquares) {
      for (let i = 0; i < input.length; i += 1) {
        const item = input[i];
        if (!item || typeof item !== 'object' || !('square' in item)) {
          errors.push(`Array index ${i} is missing a valid square property.`);
          continue;
        }
        const sq = String(item.square).toLowerCase();
        if (!isValidSquareName(sq)) {
          errors.push(`Invalid square name "${item.square}" at array index ${i}.`);
          continue;
        }
        const symbol = item.piece;
        if (!symbol || symbol === '.') {
          squares[sq] = null;
        } else if (VALID_PIECES.has(symbol)) {
          squares[sq] = symbol as ChessPieceSymbol;
        } else {
          errors.push(`Invalid piece symbol "${symbol}" at square ${sq}.`);
        }
      }
    } else {
      // 64 raw entries in display row-major order:
      // Row r in [0..7] (top to bottom), Col c in [0..7] (left to right)
      for (let r = 0; r < 8; r += 1) {
        for (let c = 0; c < 8; c += 1) {
          const index = r * 8 + c;
          const symbol = input[index] as string | null | undefined;

          // Map grid (r, c) to algebraic square
          const file = orientation === 'white-bottom' ? c : 7 - c;
          const rank = orientation === 'white-bottom' ? 7 - r : r;
          const sq = `${FILES[file]}${RANKS[rank]}`;

          if (!symbol || symbol === '.') {
            squares[sq] = null;
          } else if (VALID_PIECES.has(symbol)) {
            squares[sq] = symbol as ChessPieceSymbol;
          } else {
            errors.push(`Invalid piece symbol "${symbol}" at index ${index} (${sq}).`);
          }
        }
      }
    }
  } else if (input && typeof input === 'object') {
    for (const [key, value] of Object.entries(input)) {
      const sq = key.toLowerCase();
      if (!isValidSquareName(sq)) {
        errors.push(`Unrecognized square key "${key}" in placement dictionary.`);
        continue;
      }
      if (!value || value === '.') {
        squares[sq] = null;
      } else if (VALID_PIECES.has(value)) {
        squares[sq] = value as ChessPieceSymbol;
      } else {
        errors.push(`Invalid piece symbol "${value}" at square ${sq}.`);
      }
    }
  } else {
    errors.push('Placement input must be either a 64-element array or a square dictionary.');
  }

  return { squares, errors };
}

/**
 * Encodes an authoritative 64-square placement into standard FEN piece placement string.
 * Ranks are processed from Rank 8 down to Rank 1; files from a to h.
 */
export function buildPlacementFen(
  squares: Record<string, ChessPieceSymbol | null>,
): string {
  const rankStrings: string[] = [];

  for (let rankIdx = 7; rankIdx >= 0; rankIdx -= 1) {
    const rank = RANKS[rankIdx];
    let emptyCount = 0;
    let rankStr = '';

    for (let fileIdx = 0; fileIdx < 8; fileIdx += 1) {
      const file = FILES[fileIdx];
      const sq = `${file}${rank}`;
      const piece = squares[sq];

      if (!piece) {
        emptyCount += 1;
      } else {
        if (emptyCount > 0) {
          rankStr += String(emptyCount);
          emptyCount = 0;
        }
        rankStr += piece;
      }
    }

    if (emptyCount > 0) {
      rankStr += String(emptyCount);
    }

    rankStrings.push(rankStr);
  }

  return rankStrings.join('/');
}

/**
 * Normalizes and validates castling rights input.
 */
export function normalizeCastling(
  castling: string | CastlingRightsFlags | undefined,
): { castlingStr: string; errors: string[] } {
  const errors: string[] = [];

  if (castling === undefined || castling === null || castling === '' || castling === '-') {
    return { castlingStr: '-', errors };
  }

  if (typeof castling === 'string') {
    const trimmed = castling.trim();
    if (trimmed === '-') {
      return { castlingStr: '-', errors };
    }

    let hasK = false;
    let hasQ = false;
    let hasSmallK = false;
    let hasSmallQ = false;

    for (const char of trimmed) {
      if (char === 'K') hasK = true;
      else if (char === 'Q') hasQ = true;
      else if (char === 'k') hasSmallK = true;
      else if (char === 'q') hasSmallQ = true;
      else {
        errors.push(`Invalid character "${char}" in castling rights string.`);
      }
    }

    let result = '';
    if (hasK) result += 'K';
    if (hasQ) result += 'Q';
    if (hasSmallK) result += 'k';
    if (hasSmallQ) result += 'q';

    return { castlingStr: result.length > 0 ? result : '-', errors };
  }

  if (typeof castling === 'object') {
    let result = '';
    if (castling.K) result += 'K';
    if (castling.Q) result += 'Q';
    if (castling.k) result += 'k';
    if (castling.q) result += 'q';
    return { castlingStr: result.length > 0 ? result : '-', errors };
  }

  errors.push('Castling rights must be a string or structured flags object.');
  return { castlingStr: '-', errors };
}

/**
 * Builds a deterministic, validated six-field FEN from user-supplied square placement
 * and optional position metadata.
 */
export function buildPositionFen(
  placement: PlacementInput,
  options: FenBuilderOptions = {},
): FenBuilderResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  const orientation = options.orientation ?? 'white-bottom';
  const metadata = options.metadata ?? {};

  // 1. Normalize square placement
  const { squares, errors: placementErrors } = normalizePlacementInput(placement, orientation);
  errors.push(...placementErrors);

  // 2. Validate metadata
  const turn = metadata.turn ?? 'w';
  if (turn !== 'w' && turn !== 'b') {
    errors.push(`Side to move must be "w" or "b", received "${metadata.turn}".`);
  }

  const { castlingStr, errors: castlingErrors } = normalizeCastling(metadata.castling);
  errors.push(...castlingErrors);

  let enPassant = '-';
  if (metadata.enPassant && metadata.enPassant !== '-') {
    const epSquare = metadata.enPassant.toLowerCase().trim();
    if (!isValidSquareName(epSquare)) {
      errors.push(`Invalid en-passant square format: "${metadata.enPassant}".`);
    } else {
      enPassant = epSquare;
    }
  }

  const halfmoveClock = metadata.halfmoveClock ?? 0;
  if (!Number.isInteger(halfmoveClock) || halfmoveClock < 0) {
    errors.push(`Halfmove clock must be a non-negative integer, received ${halfmoveClock}.`);
  }

  const fullmoveNumber = metadata.fullmoveNumber ?? 1;
  if (!Number.isInteger(fullmoveNumber) || fullmoveNumber < 1) {
    errors.push(`Fullmove number must be a positive integer (>= 1), received ${fullmoveNumber}.`);
  }

  // 3. Piece count and square constraint validation
  let whiteKings: string[] = [];
  let blackKings: string[] = [];
  let whitePawns = 0;
  let blackPawns = 0;
  let whitePieces = 0;
  let blackPieces = 0;
  let whiteQueens = 0;
  let blackQueens = 0;
  let whiteRooks = 0;
  let blackRooks = 0;
  let whiteBishops = 0;
  let blackBishops = 0;
  let whiteKnights = 0;
  let blackKnights = 0;

  for (const [sq, piece] of Object.entries(squares)) {
    if (!piece) continue;

    const rank = sq[1];

    if (piece === 'K') {
      whiteKings.push(sq);
      whitePieces += 1;
    } else if (piece === 'k') {
      blackKings.push(sq);
      blackPieces += 1;
    } else if (piece === 'P') {
      whitePawns += 1;
      whitePieces += 1;
      if (rank === '1' || rank === '8') {
        errors.push(`White pawn on ${sq} is illegal (pawns cannot exist on rank 1 or 8).`);
      }
    } else if (piece === 'p') {
      blackPawns += 1;
      blackPieces += 1;
      if (rank === '1' || rank === '8') {
        errors.push(`Black pawn on ${sq} is illegal (pawns cannot exist on rank 1 or 8).`);
      }
    } else if (piece === 'Q') {
      whiteQueens += 1;
      whitePieces += 1;
    } else if (piece === 'q') {
      blackQueens += 1;
      blackPieces += 1;
    } else if (piece === 'R') {
      whiteRooks += 1;
      whitePieces += 1;
    } else if (piece === 'r') {
      blackRooks += 1;
      blackPieces += 1;
    } else if (piece === 'B') {
      whiteBishops += 1;
      whitePieces += 1;
    } else if (piece === 'b') {
      blackBishops += 1;
      blackPieces += 1;
    } else if (piece === 'N') {
      whiteKnights += 1;
      whitePieces += 1;
    } else if (piece === 'n') {
      blackKnights += 1;
      blackPieces += 1;
    }
  }

  // King counts
  if (whiteKings.length === 0) {
    errors.push('Missing White king (must have exactly 1).');
  } else if (whiteKings.length > 1) {
    errors.push(`Found ${whiteKings.length} White kings on ${whiteKings.join(', ')} (must have exactly 1).`);
  }

  if (blackKings.length === 0) {
    errors.push('Missing Black king (must have exactly 1).');
  } else if (blackKings.length > 1) {
    errors.push(`Found ${blackKings.length} Black kings on ${blackKings.join(', ')} (must have exactly 1).`);
  }

  // Non-adjacent kings
  if (whiteKings.length === 1 && blackKings.length === 1) {
    const wSq = whiteKings[0];
    const bSq = blackKings[0];
    const wFile = wSq.charCodeAt(0) - 97;
    const wRank = parseInt(wSq[1], 10) - 1;
    const bFile = bSq.charCodeAt(0) - 97;
    const bRank = parseInt(bSq[1], 10) - 1;

    if (Math.abs(wFile - bFile) <= 1 && Math.abs(wRank - bRank) <= 1) {
      errors.push(`White king on ${wSq} and Black king on ${bSq} are adjacent (kings cannot touch).`);
    }
  }

  // Piece counts
  if (whitePawns > 8) {
    errors.push(`White has ${whitePawns} pawns (maximum allowed is 8).`);
  }
  if (blackPawns > 8) {
    errors.push(`Black has ${blackPawns} pawns (maximum allowed is 8).`);
  }
  if (whitePieces > 16) {
    errors.push(`White has ${whitePieces} pieces (maximum allowed is 16).`);
  }
  if (blackPieces > 16) {
    errors.push(`Black has ${blackPieces} pieces (maximum allowed is 16).`);
  }

  // Promoted pieces bounds
  const whitePromoted =
    Math.max(0, whiteQueens - 1) +
    Math.max(0, whiteRooks - 2) +
    Math.max(0, whiteBishops - 2) +
    Math.max(0, whiteKnights - 2);
  const maxWhitePromotions = Math.max(0, 8 - whitePawns);
  if (whitePromoted > maxWhitePromotions) {
    errors.push(
      `White piece count exceeds legal limits: ${whitePromoted} extra pieces require more than ${maxWhitePromotions} promoted pawns.`,
    );
  }

  const blackPromoted =
    Math.max(0, blackQueens - 1) +
    Math.max(0, blackRooks - 2) +
    Math.max(0, blackBishops - 2) +
    Math.max(0, blackKnights - 2);
  const maxBlackPromotions = Math.max(0, 8 - blackPawns);
  if (blackPromoted > maxBlackPromotions) {
    errors.push(
      `Black piece count exceeds legal limits: ${blackPromoted} extra pieces require more than ${maxBlackPromotions} promoted pawns.`,
    );
  }

  // 4. Castling rights placement consistency
  if (castlingStr.includes('K')) {
    if (squares['e1'] !== 'K' || squares['h1'] !== 'R') {
      errors.push(
        `White kingside castling ("K") is invalid: King must be on e1 (found ${squares['e1'] ?? 'empty'}) and Rook on h1 (found ${squares['h1'] ?? 'empty'}).`,
      );
    }
  }
  if (castlingStr.includes('Q')) {
    if (squares['e1'] !== 'K' || squares['a1'] !== 'R') {
      errors.push(
        `White queenside castling ("Q") is invalid: King must be on e1 (found ${squares['e1'] ?? 'empty'}) and Rook on a1 (found ${squares['a1'] ?? 'empty'}).`,
      );
    }
  }
  if (castlingStr.includes('k')) {
    if (squares['e8'] !== 'k' || squares['h8'] !== 'r') {
      errors.push(
        `Black kingside castling ("k") is invalid: King must be on e8 (found ${squares['e8'] ?? 'empty'}) and Rook on h8 (found ${squares['h8'] ?? 'empty'}).`,
      );
    }
  }
  if (castlingStr.includes('q')) {
    if (squares['e8'] !== 'k' || squares['a8'] !== 'r') {
      errors.push(
        `Black queenside castling ("q") is invalid: King must be on e8 (found ${squares['e8'] ?? 'empty'}) and Rook on a8 (found ${squares['a8'] ?? 'empty'}).`,
      );
    }
  }

  if (castlingStr !== '-') {
    warnings.push(
      'Castling rights require that King and Rook have never moved historically; piece presence on starting squares alone does not establish historical castling rights.',
    );
  }

  // 5. En-passant target consistency
  if (enPassant !== '-') {
    const file = enPassant[0];
    const rank = enPassant[1];

    if (turn === 'w') {
      // White to move: Black just pushed pawn from rank 7 to rank 5
      if (rank !== '6') {
        errors.push(`En-passant target for White to move must be on rank 6, received "${enPassant}".`);
      } else {
        const passedPawnSq = `${file}5`;
        const originSq = `${file}7`;
        if (squares[passedPawnSq] !== 'p') {
          errors.push(`En-passant square ${enPassant} is invalid: expected Black pawn on ${passedPawnSq} (found ${squares[passedPawnSq] ?? 'empty'}).`);
        }
        if (squares[enPassant] !== null) {
          errors.push(`En-passant target square ${enPassant} must be empty (found ${squares[enPassant]}).`);
        }
        if (squares[originSq] !== null) {
          errors.push(`En-passant square ${enPassant} is invalid: origin square ${originSq} must be empty (found ${squares[originSq]}).`);
        }
      }
    } else {
      // Black to move: White just pushed pawn from rank 2 to rank 4
      if (rank !== '3') {
        errors.push(`En-passant target for Black to move must be on rank 3, received "${enPassant}".`);
      } else {
        const passedPawnSq = `${file}4`;
        const originSq = `${file}2`;
        if (squares[passedPawnSq] !== 'P') {
          errors.push(`En-passant square ${enPassant} is invalid: expected White pawn on ${passedPawnSq} (found ${squares[passedPawnSq] ?? 'empty'}).`);
        }
        if (squares[enPassant] !== null) {
          errors.push(`En-passant target square ${enPassant} must be empty (found ${squares[enPassant]}).`);
        }
        if (squares[originSq] !== null) {
          errors.push(`En-passant square ${enPassant} is invalid: origin square ${originSq} must be empty (found ${squares[originSq]}).`);
        }
      }
    }
  }

  // Construct FEN strings
  const placementFen = buildPlacementFen(squares);
  const fullFen = `${placementFen} ${turn} ${castlingStr} ${enPassant} ${halfmoveClock} ${fullmoveNumber}`;

  const resolvedMetadata = {
    turn,
    castling: castlingStr,
    enPassant,
    halfmoveClock,
    fullmoveNumber,
  };

  // 6. Chess engine validation (active check & inactive check checks)
  let details: FenValidationDetails | undefined;

  if (whiteKings.length === 1 && blackKings.length === 1 && errors.length === 0) {
    const game = new ChessGame();
    const loaded = game.load(fullFen);

    if (!loaded) {
      errors.push(`Chess core rejected FEN: "${fullFen}".`);
    } else {
      // Check whether the opponent (inactive player) is in check
      const opponentTurn = turn === 'w' ? 'b' : 'w';
      const opponentProbeFen = `${placementFen} ${opponentTurn} - - 0 1`;
      const opponentGame = new ChessGame();

      if (opponentGame.load(opponentProbeFen) && opponentGame.isCheck()) {
        const opponentColor = turn === 'w' ? 'Black' : 'White';
        errors.push(
          `${opponentColor} king is in check, but it is ${turn === 'w' ? 'White' : 'Black'} to move (illegal position).`,
        );
      }

      details = {
        whiteKingSquare: whiteKings[0],
        blackKingSquare: blackKings[0],
        whitePieceCount: whitePieces,
        blackPieceCount: blackPieces,
        whitePawnCount: whitePawns,
        blackPawnCount: blackPawns,
        isCheck: game.isCheck(),
        isCheckmate: game.isCheckmate(),
        isStalemate: game.isStalemate(),
        legalMoveCount: game.allLegalMoves().length,
      };
    }
  }

  const isValid = errors.length === 0;

  return {
    fen: isValid ? fullFen : null,
    placementFen,
    valid: isValid,
    errors,
    warnings,
    metadata: resolvedMetadata,
    details,
    reachabilityDisclaimer: REACHABILITY_DISCLAIMER,
  };
}
