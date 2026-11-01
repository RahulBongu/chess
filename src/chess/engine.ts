import { ChessGame } from './ChessGame.ts';

// A replaceable chess-analysis interface. The Study UI depends on this
// interface, not on any concrete engine implementation.

export type EngineName = 'heuristic-v1';

export interface EngineMove {
  from: string;
  to: string;
  promotion?: 'q' | 'r' | 'b' | 'n';
  san: string;
}

export interface EngineAnalysisOptions {
  // Requested search depth. The heuristic implementation evaluates legal moves
  // one ply deep and documents this limitation; a future native engine may use
  // this value directly.
  depth?: number;
  // Requested time budget in milliseconds. The heuristic implementation is
  // intentionally shallow and returns promptly; this is retained for interface
  // compatibility.
  maxTimeMs?: number;
}

export interface EngineAnalysisResult {
  fen: string;
  bestMove: EngineMove | null;
  // Centipawns from the side-to-move perspective before the move: positive means
  // the side to move is better. Terminal mates are represented as very large
  // values rather than a separate mate score.
  score: number;
  perspective: 'side-to-move';
  terminal: 'none' | 'checkmate' | 'stalemate' | 'draw';
  pv: string[];
  nodes: number;
  depth: number;
  engine: EngineName;
}

export class EngineCancelledError extends Error {
  constructor() {
    super('Chess engine analysis was cancelled.');
    this.name = 'EngineCancelledError';
  }
}

export interface ChessEngine {
  analyze(fen: string, options?: EngineAnalysisOptions): Promise<EngineAnalysisResult>;
  stop(): void;
}

const PIECE_VALUES: Record<string, number> = {
  p: 100,
  n: 320,
  b: 330,
  r: 500,
  q: 900,
  k: 0,
};

// Piece-square tables are intentionally simple and are expressed for White.
// Black uses the vertically mirrored square.
const PAWN_TABLE = [
  0, 0, 0, 0, 0, 0, 0, 0,
  50, 50, 50, 50, 50, 50, 50, 50,
  10, 10, 20, 30, 30, 20, 10, 10,
  5, 5, 10, 25, 25, 10, 5, 5,
  0, 0, 0, 20, 20, 0, 0, 0,
  5, -5, -10, 0, 0, -10, -5, 5,
  5, 10, 10, -20, -20, 10, 10, 5,
  0, 0, 0, 0, 0, 0, 0, 0,
];

const KNIGHT_TABLE = [
  -50, -40, -30, -30, -30, -30, -40, -50,
  -40, -20, 0, 0, 0, 0, -20, -40,
  -30, 0, 10, 15, 15, 10, 0, -30,
  -30, 5, 15, 20, 20, 15, 5, -30,
  -30, 0, 15, 20, 20, 15, 0, -30,
  -30, 5, 10, 15, 15, 10, 5, -30,
  -40, -20, 0, 5, 5, 0, -20, -40,
  -50, -40, -30, -30, -30, -30, -40, -50,
];

const BISHOP_TABLE = [
  -20, -10, -10, -10, -10, -10, -10, -20,
  -10, 0, 0, 0, 0, 0, 0, -10,
  -10, 0, 5, 10, 10, 5, 0, -10,
  -10, 5, 5, 10, 10, 5, 5, -10,
  -10, 0, 10, 10, 10, 10, 0, -10,
  -10, 10, 10, 10, 10, 10, 10, -10,
  -10, 5, 0, 0, 0, 0, 5, -10,
  -20, -10, -10, -10, -10, -10, -10, -20,
];

const ROOK_TABLE = [
  0, 0, 0, 0, 0, 0, 0, 0,
  5, 10, 10, 10, 10, 10, 10, 5,
  -5, 0, 0, 0, 0, 0, 0, -5,
  -5, 0, 0, 0, 0, 0, 0, -5,
  -5, 0, 0, 0, 0, 0, 0, -5,
  -5, 0, 0, 0, 0, 0, 0, -5,
  -5, 0, 0, 0, 0, 0, 0, -5,
  0, 0, 0, 5, 5, 0, 0, 0,
];

const QUEEN_TABLE = [
  -20, -10, -10, -5, -5, -10, -10, -20,
  -10, 0, 0, 0, 0, 0, 0, -10,
  -10, 0, 5, 5, 5, 5, 0, -10,
  -5, 0, 5, 5, 5, 5, 0, -5,
  0, 0, 5, 5, 5, 5, 0, -5,
  -10, 5, 5, 5, 5, 5, 0, -10,
  -10, 0, 5, 0, 0, 0, 0, -10,
  -20, -10, -10, -5, -5, -10, -10, -20,
];

const KING_TABLE = [
  -30, -40, -40, -50, -50, -40, -40, -30,
  -30, -40, -40, -50, -50, -40, -40, -30,
  -30, -40, -40, -50, -50, -40, -40, -30,
  -30, -40, -40, -50, -50, -40, -40, -30,
  -20, -30, -30, -40, -40, -30, -30, -20,
  -10, -20, -20, -20, -20, -20, -20, -10,
  20, 20, 0, 0, 0, 0, 20, 20,
  20, 30, 10, 0, 0, 10, 30, 20,
];

const PIECE_TABLES: Record<string, number[]> = {
  p: PAWN_TABLE,
  n: KNIGHT_TABLE,
  b: BISHOP_TABLE,
  r: ROOK_TABLE,
  q: QUEEN_TABLE,
  k: KING_TABLE,
};

const MATE_SCORE = 100000;

function pieceSquareValue(pieceType: string, squareIndex: number, color: 'w' | 'b'): number {
  const table = PIECE_TABLES[pieceType];
  if (!table) {
    return 0;
  }
  const index = color === 'w' ? squareIndex : 63 - squareIndex;
  return table[index];
}

function evaluateWhite(game: ChessGame): number {
  let score = 0;
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;

  for (let rank = 1; rank <= 8; rank += 1) {
    for (const file of files) {
      const square = `${file}${rank}` as Parameters<ChessGame['pieceAt']>[0];
      const piece = game.pieceAt(square);
      if (!piece) {
        continue;
      }

      const squareIndex = (rank - 1) * 8 + (file.charCodeAt(0) - 97);
      const material = PIECE_VALUES[piece.type] ?? 0;
      const positional = pieceSquareValue(piece.type, squareIndex, piece.color);
      const value = material + positional;

      if (piece.color === 'w') {
        score += value;
      } else {
        score -= value;
      }
    }
  }

  return score;
}

function terminalResult(game: ChessGame): EngineAnalysisResult['terminal'] {
  if (game.isCheckmate()) {
    return 'checkmate';
  }
  if (game.isStalemate()) {
    return 'stalemate';
  }
  if (game.isDraw()) {
    return 'draw';
  }
  return 'none';
}

function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

export class HeuristicChessEngine implements ChessEngine {
  private currentToken = 0;

  stop(): void {
    this.currentToken += 1;
  }

  async analyze(fen: string, options: EngineAnalysisOptions = {}): Promise<EngineAnalysisResult> {
    const token = ++this.currentToken;

    // Yield before doing any work so a newer request or an explicit stop() can
    // cancel this analysis without blocking the UI thread.
    await yieldToEventLoop();
    if (token !== this.currentToken) {
      throw new EngineCancelledError();
    }

    const game = new ChessGame();
    if (!game.load(fen)) {
      throw new Error(`Invalid FEN: ${fen}`);
    }

    const terminal = terminalResult(game);
    const legalMoves = game.allLegalMoves();

    if (legalMoves.length === 0) {
      const score = terminal === 'checkmate' ? -MATE_SCORE : 0;
      return {
        fen: game.fen(),
        bestMove: null,
        score,
        perspective: 'side-to-move',
        terminal,
        pv: [],
        nodes: 0,
        depth: 1,
        engine: 'heuristic-v1',
      };
    }

    let bestMove: EngineMove | null = null;
    let bestScore = Number.NEGATIVE_INFINITY;
    let nodes = 0;

    const orderedMoves = [...legalMoves].sort((left, right) => {
      const leftKey = `${left.from}${left.to}${left.promotion ?? ''}`;
      const rightKey = `${right.from}${right.to}${right.promotion ?? ''}`;
      return leftKey.localeCompare(rightKey);
    });

    for (const move of orderedMoves) {
      if (token !== this.currentToken) {
        throw new EngineCancelledError();
      }

      const candidate = new ChessGame(game.fen());
      const applied = candidate.tryMove({
        from: move.from,
        to: move.to,
        promotion: move.promotion ? (move.promotion as 'q' | 'r' | 'b' | 'n') : undefined,
      });

      if (!applied) {
        continue;
      }

      nodes += 1;
      // Score the resulting position from the opponent's (side-to-move) view,
      // then negate it: a candidate is good when it leaves the opponent worse.
      const whiteEval = evaluateWhite(candidate);
      const opponentSideScore = candidate.turn() === 'w' ? whiteEval : -whiteEval;
      const candidateScore = -opponentSideScore;

      if (candidateScore > bestScore) {
        bestScore = candidateScore;
        bestMove = {
          from: move.from,
          to: move.to,
          promotion: move.promotion ? (move.promotion as 'q' | 'r' | 'b' | 'n') : undefined,
          san: move.san,
        };
      }
    }

    if (token !== this.currentToken) {
      throw new EngineCancelledError();
    }

    return {
      fen: game.fen(),
      bestMove,
      score: Number.isFinite(bestScore) ? bestScore : 0,
      perspective: 'side-to-move',
      terminal: 'none',
      pv: bestMove ? [bestMove.san] : [],
      nodes,
      depth: 1,
      engine: 'heuristic-v1',
    };
  }
}

export function createHeuristicEngine(): ChessEngine {
  return new HeuristicChessEngine();
}

// HeuristicChessEngine limitations:
// - It evaluates legal moves only one ply deep.
// - It uses material and simple piece-square tables; no search, tactics,
//   king safety, pawn structure, or endgame knowledge.
// - Scores are rough centipawn estimates, not calibrated engine evaluations.
// - It is deterministic for a given FEN and option set.
// A future native Stockfish implementation can replace this class without
// changing the ChessEngine interface or Study UI.
