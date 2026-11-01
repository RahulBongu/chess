export interface ChalSearchResult {
  bestMove: string;
  score: number;
  depth: number;
  nodes: number;
  pv: string[];
  mate: number | null;
  error?: string;
}

export interface ChalEngineNative {
  init(): Promise<boolean>;
  search(fen: string, maxDepth: number, maxTimeMs: number): Promise<ChalSearchResult>;
  stop(): Promise<boolean>;
  teardown(): Promise<boolean>;
}

function getNativeChalEngine(): ChalEngineNative | null {
  try {
    const { requireOptionalNativeModule } = require('expo-modules-core');
    return requireOptionalNativeModule('ChalEngine');
  } catch {
    return null;
  }
}

export const ChalEngine: ChalEngineNative | null = getNativeChalEngine();
