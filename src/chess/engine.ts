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
  positionsEvaluated: number;
  depth: number;
  engine: EngineName;
  /** Legacy alias for backward compatibility */
  nodes: number;
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

export function pieceSquareIndex(fileIdx: number, rank: number, color: 'w' | 'b'): number {
  // Row 0 in PIECE_TABLES represents Rank 8; Row 7 represents Rank 1.
  // White advances from Rank 1 (row 7) to Rank 8 (row 0).
  // Black advances from Rank 8 (row 7) to Rank 1 (row 0).
  // Files ('a'-'h', cols 0-7) are preserved without horizontal flipping.
  return color === 'w' ? (8 - rank) * 8 + fileIdx : (rank - 1) * 8 + fileIdx;
}

export function pieceSquareValue(pieceType: string, fileIdx: number, rank: number, color: 'w' | 'b'): number {
  const table = PIECE_TABLES[pieceType];
  if (!table) {
    return 0;
  }
  const index = pieceSquareIndex(fileIdx, rank, color);
  return table[index];
}

export function evaluateWhite(game: ChessGame): number {
  let score = 0;
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;

  for (let rank = 1; rank <= 8; rank += 1) {
    for (const file of files) {
      const square = `${file}${rank}` as Parameters<ChessGame['pieceAt']>[0];
      const piece = game.pieceAt(square);
      if (!piece) {
        continue;
      }

      const fileIdx = file.charCodeAt(0) - 97;
      const positional = pieceSquareValue(piece.type, fileIdx, rank, piece.color);
      const material = PIECE_VALUES[piece.type] ?? 0;
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

    const rootTurn = game.turn();
    const rootTerminal = terminalResult(game);
    const rootLegalMoves = game.allLegalMoves();

    // Handle terminal root positions or positions with no legal moves.
    if (rootLegalMoves.length === 0 || game.isGameOver()) {
      const score = rootTerminal === 'checkmate' ? -MATE_SCORE : 0;
      return {
        fen: game.fen(),
        bestMove: null,
        score,
        perspective: 'side-to-move',
        terminal: rootTerminal,
        pv: [],
        positionsEvaluated: 1,
        nodes: 1,
        depth: 0,
        engine: 'heuristic-v1',
      };
    }

    let bestMove: EngineMove | null = null;
    let bestReplySan: string | null = null;
    let bestScore = Number.NEGATIVE_INFINITY;
    let positionsEvaluated = 0;

    const orderedRootMoves = [...rootLegalMoves].sort((left, right) => {
      const leftKey = `${left.from}${left.to}${left.promotion ?? ''}`;
      const rightKey = `${right.from}${right.to}${right.promotion ?? ''}`;
      return leftKey.localeCompare(rightKey);
    });

    for (let rootIndex = 0; rootIndex < orderedRootMoves.length; rootIndex += 1) {
      if (token !== this.currentToken) {
        throw new EngineCancelledError();
      }

      // Yield every 4 root moves to keep the JavaScript event loop responsive.
      if (rootIndex > 0 && rootIndex % 4 === 0) {
        await yieldToEventLoop();
        if (token !== this.currentToken) {
          throw new EngineCancelledError();
        }
      }

      const rootMove = orderedRootMoves[rootIndex];
      const candidate1 = new ChessGame(game.fen());
      const applied1 = candidate1.tryMove({
        from: rootMove.from,
        to: rootMove.to,
        promotion: rootMove.promotion ? (rootMove.promotion as 'q' | 'r' | 'b' | 'n') : undefined,
      });

      if (!applied1) {
        continue;
      }

      let candidateScore: number;
      let worstReplyForMover: string | null = null;

      // 1. Did the root move deliver checkmate to the opponent?
      if (candidate1.isCheckmate()) {
        positionsEvaluated += 1;
        candidateScore = MATE_SCORE;
      }
      // 2. Did the root move deliver a draw (stalemate, insufficient material, repetition)?
      else if (candidate1.isDraw() || candidate1.isStalemate()) {
        positionsEvaluated += 1;
        candidateScore = 0;
      }
      // 3. Game continues: evaluate opponent's immediate replies (2-ply minimax).
      else {
        const replies = candidate1.allLegalMoves();
        if (replies.length === 0) {
          positionsEvaluated += 1;
          candidateScore = candidate1.isCheck() ? MATE_SCORE : 0;
        } else {
          let minReplyScore = Number.POSITIVE_INFINITY;

          const orderedReplies = [...replies].sort((left, right) => {
            const leftKey = `${left.from}${left.to}${left.promotion ?? ''}`;
            const rightKey = `${right.from}${right.to}${right.promotion ?? ''}`;
            return leftKey.localeCompare(rightKey);
          });

          for (const reply of orderedReplies) {
            if (token !== this.currentToken) {
              throw new EngineCancelledError();
            }

            positionsEvaluated += 1;
            const applied2 = candidate1.tryMove({
              from: reply.from,
              to: reply.to,
              promotion: reply.promotion ? (reply.promotion as 'q' | 'r' | 'b' | 'n') : undefined,
            });

            if (!applied2) {
              continue;
            }

            let leafScore: number;
            // Did opponent's reply deliver checkmate to the root mover?
            if (candidate1.isCheckmate()) {
              leafScore = -MATE_SCORE;
            } else if (candidate1.isDraw() || candidate1.isStalemate()) {
              leafScore = 0;
            } else {
              const whiteEval = evaluateWhite(candidate1);
              leafScore = rootTurn === 'w' ? whiteEval : -whiteEval;
            }

            candidate1.undo();

            if (leafScore < minReplyScore) {
              minReplyScore = leafScore;
              worstReplyForMover = reply.san;
            }
          }

          candidateScore = Number.isFinite(minReplyScore) ? minReplyScore : 0;
        }
      }

      if (candidateScore > bestScore) {
        bestScore = candidateScore;
        bestMove = {
          from: rootMove.from,
          to: rootMove.to,
          promotion: rootMove.promotion ? (rootMove.promotion as 'q' | 'r' | 'b' | 'n') : undefined,
          san: rootMove.san,
        };
        bestReplySan = worstReplyForMover;
      }
    }

    if (token !== this.currentToken) {
      throw new EngineCancelledError();
    }

    const pv: string[] = [];
    if (bestMove) {
      pv.push(bestMove.san);
      if (bestReplySan) {
        pv.push(bestReplySan);
      }
    }

    return {
      fen: game.fen(),
      bestMove,
      score: Number.isFinite(bestScore) ? bestScore : 0,
      perspective: 'side-to-move',
      terminal: 'none',
      pv,
      positionsEvaluated,
      nodes: positionsEvaluated,
      depth: 2,
      engine: 'heuristic-v1',
    };
  }
}

export function createHeuristicEngine(): ChessEngine {
  return new HeuristicChessEngine();
}

// HeuristicChessEngine characteristics:
// - Evaluates root candidate moves and the opponent's best immediate replies (2-ply minimax).
// - Recognizes immediate checkmates (+MATE_SCORE) and drawn positions (0).
// - Punishes moves that allow the opponent an immediate checkmate (-MATE_SCORE).
// - Uses material and vertically-mirrored piece-square tables.
// - Scores are centipawn estimates from the side-to-move perspective.
// - It is deterministic for a given FEN and options.
// A future native Stockfish implementation can replace this class without
// changing the ChessEngine interface or Study UI.
