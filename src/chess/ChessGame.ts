import { Chess, type Color, type Move as ChessMove, type Piece, type PieceSymbol, type Square } from 'chess.js';

export type PlayerColor = Color;
export type ChessSquare = Square;
export type PromotionPiece = 'q' | 'r' | 'b' | 'n';

export interface GameSnapshot {
  fen: string;
  turn: PlayerColor;
  isCheck: boolean;
  isCheckmate: boolean;
  isStalemate: boolean;
  isDraw: boolean;
  isGameOver: boolean;
  history: string[];
  moveCount: number;
}

export interface MoveAttempt {
  from: ChessSquare;
  to: ChessSquare;
  promotion?: PromotionPiece;
}

/**
 * A single authoritative wrapper around chess.js.
 *
 * The UI never implements chess rules. Every legality check, special move,
 * terminal-state calculation, and history entry comes from this object.
 */
export class ChessGame {
  private chess: Chess;

  constructor(fen?: string) {
    this.chess = fen ? new Chess(fen) : new Chess();
  }

  fen(): string {
    return this.chess.fen();
  }

  turn(): PlayerColor {
    return this.chess.turn();
  }

  pieceAt(square: ChessSquare): Piece | undefined {
    return this.chess.get(square);
  }

  legalDestinations(square: ChessSquare): ChessSquare[] {
    const piece = this.chess.get(square);
    if (!piece || piece.color !== this.chess.turn()) {
      return [];
    }
    return this.chess.moves({ square, verbose: true }).map((move) => move.to as ChessSquare);
  }

  allLegalMoves(): ChessMove[] {
    return this.chess.moves({ verbose: true }) as ChessMove[];
  }

  tryMove({ from, to, promotion }: MoveAttempt): ChessMove | null {
    const candidate = this.chess.moves({ square: from, verbose: true }).find((move) => move.to === to);
    if (!candidate) {
      return null;
    }

    const promotionPiece = promotion ?? (candidate.promotion ? 'q' : undefined);
    try {
      return this.chess.move({ from, to, promotion: promotionPiece }) as ChessMove;
    } catch {
      return null;
    }
  }

  isPromotionMove(from: ChessSquare, to: ChessSquare): boolean {
    return this.chess
      .moves({ square: from, verbose: true })
      .some((move) => move.to === to && Boolean(move.promotion));
  }

  undo(): ChessMove | null {
    return this.chess.undo() as ChessMove | null;
  }

  reset(): void {
    this.chess.reset();
  }

  load(fen: string): boolean {
    try {
      this.chess.load(fen);
      return true;
    } catch {
      return false;
    }
  }

  loadPgn(pgn: string): boolean {
    try {
      this.chess.loadPgn(pgn);
      return true;
    } catch {
      return false;
    }
  }

  history(): string[] {
    return this.chess.history();
  }

  verboseHistory(): ChessMove[] {
    return this.chess.history({ verbose: true }) as ChessMove[];
  }

  isCheck(): boolean {
    return this.chess.isCheck();
  }

  isCheckmate(): boolean {
    return this.chess.isCheckmate();
  }

  isStalemate(): boolean {
    return this.chess.isStalemate();
  }

  isDraw(): boolean {
    return this.chess.isDraw();
  }

  isGameOver(): boolean {
    return this.chess.isGameOver();
  }

  moveCount(): number {
    return this.chess.history().length;
  }

  snapshot(): GameSnapshot {
    return {
      fen: this.fen(),
      turn: this.turn(),
      isCheck: this.isCheck(),
      isCheckmate: this.isCheckmate(),
      isStalemate: this.isStalemate(),
      isDraw: this.isDraw(),
      isGameOver: this.isGameOver(),
      history: this.history(),
      moveCount: this.moveCount(),
    };
  }
}

export function createInitialGame(): ChessGame {
  return new ChessGame();
}

export const STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';