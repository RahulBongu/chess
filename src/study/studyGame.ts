import { ChessGame } from '../chess/ChessGame.ts';

export interface StudyPosition {
  fen: string;
  moveSan?: string;
}

export interface StudyImportResult {
  positions: StudyPosition[];
  moves: string[];
}

export function importFen(fen: string): StudyPosition[] | null {
  const trimmed = fen.trim();
  if (trimmed === '') {
    return null;
  }

  const game = new ChessGame();
  if (!game.load(trimmed)) {
    return null;
  }

  return [{ fen: game.fen() }];
}

export function importPgn(pgn: string): StudyImportResult | null {
  const trimmed = pgn.trim();
  if (trimmed === '') {
    return null;
  }

  const parsed = new ChessGame();
  if (!parsed.loadPgn(trimmed)) {
    return null;
  }

  const verboseMoves = parsed.verboseHistory();
  const game = new ChessGame();
  const positions: StudyPosition[] = [{ fen: game.fen() }];
  const moves: string[] = [];

  for (const move of verboseMoves) {
    const applied = game.tryMove({
      from: move.from,
      to: move.to,
      promotion: move.promotion ? (move.promotion as 'q' | 'r' | 'b' | 'n') : undefined,
    });

    if (!applied) {
      return null;
    }

    positions.push({ fen: game.fen(), moveSan: move.san });
    moves.push(move.san);
  }

  return { positions, moves };
}

export function importStudyInput(input: string): StudyImportResult | null {
  const trimmed = input.trim();
  if (trimmed === '') {
    return null;
  }

  const fenResult = importFen(trimmed);
  if (fenResult) {
    return { positions: fenResult, moves: [] };
  }

  return importPgn(trimmed);
}