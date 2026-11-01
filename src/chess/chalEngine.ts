import {
  EngineCancelledError,
  type ChessEngine,
  type EngineAnalysisOptions,
  type EngineAnalysisResult,
} from './engine.ts';
import { ChessGame, type ChessSquare, type PromotionPiece } from './ChessGame.ts';
import {
  ChalEngine,
  type ChalEngineNative,
  type ChalSearchResult,
} from '../../modules/chal-engine/src/index.ts';

/**
 * Adapter for the in-process Chal engine.
 *
 * The native bridge returns UCI moves and scores from the side-to-move
 * perspective. This class validates the move against chess.js, converts the
 * principal variation to SAN, and presents the result through ChessEngine.
 */
export class ChalChessEngine implements ChessEngine {
  private currentToken = 0;
  private initialized = false;
  private readonly native: ChalEngineNative | null;

  constructor(native: ChalEngineNative | null = ChalEngine) {
    this.native = native;
  }

  async analyze(
    fen: string,
    options: EngineAnalysisOptions = {},
  ): Promise<EngineAnalysisResult> {
    const native = this.native;
    if (!native) {
      throw new Error('Chal native engine module is not available in this environment.');
    }

    const token = ++this.currentToken;
    const game = new ChessGame();
    if (!game.load(fen)) {
      throw new Error(`Invalid FEN: ${fen}`);
    }

    // Preserve the analyzed root position. Validating the returned move must
    // not leak the post-move position into the result.
    const rootFen = game.fen();

    if (game.isGameOver() || game.allLegalMoves().length === 0) {
      const isMate = game.isCheckmate();
      const terminal = isMate ? 'checkmate' : (game.isStalemate() ? 'stalemate' : 'draw');
      return {
        fen: rootFen,
        bestMove: null,
        score: isMate ? -100000 : 0,
        perspective: 'side-to-move',
        terminal,
        pv: [],
        positionsEvaluated: 1,
        depth: 0,
        engine: 'chal-2.0.0',
        nodes: 1,
        mate: isMate ? 0 : null,
      };
    }

    if (!this.initialized) {
      const ready = await native.init();
      if (!ready) {
        throw new Error('Chal native engine failed to initialize.');
      }
      this.initialized = true;
    }

    if (token !== this.currentToken) {
      throw new EngineCancelledError();
    }

    const depth = options.depth ?? 10;
    const maxTime = options.maxTimeMs ?? 1000;

    const raw: ChalSearchResult = await native.search(rootFen, depth, maxTime);
    if (token !== this.currentToken) {
      throw new EngineCancelledError();
    }

    if (raw.error) {
      throw new Error(raw.error);
    }

    // Validate the returned UCI move against the exact root position without
    // mutating the game that owns the reported FEN.
    const validatedMove = parseChalUciMove(new ChessGame(rootFen), raw.bestMove);

    return {
      fen: rootFen,
      bestMove: validatedMove,
      score: raw.score,
      perspective: 'side-to-move',
      terminal: 'none',
      pv: formatChalPv(rootFen, raw.pv),
      positionsEvaluated: raw.nodes,
      depth: raw.depth,
      engine: 'chal-2.0.0',
      nodes: raw.nodes,
      mate: raw.mate,
    };
  }

  stop(): void {
    this.currentToken += 1;
    const pendingStop = this.native?.stop();
    if (pendingStop) {
      // A teardown race can reject the native promise; cancellation must remain
      // a best-effort request from the caller's perspective.
      void pendingStop.catch(() => {});
    }
  }
}

export function parseChalUciMove(
  game: ChessGame,
  uciMove: string,
): { from: string; to: string; promotion?: PromotionPiece; san: string } | null {
  if (!uciMove || uciMove.length < 4) {
    return null;
  }
  const from = uciMove.substring(0, 2) as ChessSquare;
  const to = uciMove.substring(2, 4) as ChessSquare;
  const promoChar = uciMove.length >= 5 ? uciMove[4].toLowerCase() : undefined;
  const promotion = (promoChar === 'q' || promoChar === 'r' || promoChar === 'b' || promoChar === 'n')
    ? (promoChar as PromotionPiece)
    : undefined;

  const applied = game.tryMove({ from, to, promotion });
  if (!applied) {
    return null;
  }
  return { from, to, promotion, san: applied.san };
}

/** Convert a UCI principal variation into SAN, stopping at the first illegal move. */
export function formatChalPv(fen: string, uciMoves: string[]): string[] {
  const game = new ChessGame(fen);
  const san: string[] = [];

  for (const uciMove of uciMoves) {
    const move = parseChalUciMove(game, uciMove);
    if (!move) {
      break;
    }
    san.push(move.san);
  }

  return san;
}
