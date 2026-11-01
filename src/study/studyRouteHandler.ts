import { importStudyInput, type StudyImportResult, type StudyPosition } from './studyGame.ts';

export interface StudyRouteState {
  input: string;
  positions: StudyPosition[];
  moves: string[];
  currentIndex: number;
  error: string | null;
  lastImportedFen: string | null;
}

export function createInitialStudyRouteState(): StudyRouteState {
  return {
    input: '',
    positions: [],
    moves: [],
    currentIndex: 0,
    error: null,
    lastImportedFen: null,
  };
}

/**
 * Handles incoming route FEN parameters into Study state.
 *
 * Requirements:
 * - A new valid FEN replaces current position and resets currentIndex to 0.
 * - Navigating from FEN A -> FEN B -> FEN A replaces position on each change.
 * - Rerenders with the same FEN do NOT reset the board.
 * - Invalid FEN parameters do NOT overwrite current position or update lastImportedFen.
 */
export function handleStudyRouteFen(
  current: StudyRouteState,
  routeFen: string | undefined | null,
): { state: StudyRouteState; updated: boolean } {
  if (!routeFen || typeof routeFen !== 'string') {
    return { state: current, updated: false };
  }

  const trimmed = routeFen.trim();
  if (trimmed === '') {
    return { state: current, updated: false };
  }

  // Deduplicate: if route parameter matches the currently active imported FEN, do not reset!
  if (trimmed === current.lastImportedFen) {
    return { state: current, updated: false };
  }

  const result: StudyImportResult | null = importStudyInput(trimmed);
  if (!result || result.positions.length === 0) {
    // Malformed FEN: record error, do NOT update lastImportedFen, keep existing board intact
    return {
      state: {
        ...current,
        error: 'Invalid FEN passed in route parameter.',
      },
      updated: false,
    };
  }

  // Valid new FEN: replace position state and update lastImportedFen
  return {
    state: {
      input: trimmed,
      positions: result.positions,
      moves: result.moves,
      currentIndex: 0,
      error: null,
      lastImportedFen: trimmed,
    },
    updated: true,
  };
}
