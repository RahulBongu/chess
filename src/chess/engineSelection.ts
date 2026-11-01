import { ChalEngine, type ChalEngineNative } from '../../modules/chal-engine/src/index.ts';
import { ChalChessEngine } from './chalEngine.ts';
import { createHeuristicEngine, type ChessEngine, type EngineName } from './engine.ts';

export interface EngineSelection {
  engine: ChessEngine;
  engineName: EngineName;
  /** Null when the preferred native engine is selected. */
  fallbackReason: string | null;
}

/**
 * Select Chal when the native module is registered, otherwise fall back to the
 * built-in heuristic engine. The caller must surface fallbackReason so a
 * fallback is never silent.
 */
export function createDefaultEngine(
  nativeEngine: ChalEngineNative | null = ChalEngine,
): EngineSelection {
  if (nativeEngine) {
    return {
      engine: new ChalChessEngine(nativeEngine),
      engineName: 'chal-2.0.0',
      fallbackReason: null,
    };
  }

  return {
    engine: createHeuristicEngine(),
    engineName: 'heuristic-v1',
    fallbackReason:
      'Chal 2.0.0 native engine is not available in this build. Analysis is using the built-in heuristic 2-ply fallback.',
  };
}
