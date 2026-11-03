/// <reference types="node" />
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { LiveGuidanceCoordinator } from '../src/vision/liveGuidance.ts';
import type { Point2D } from '../src/vision/perspectiveOverlay.ts';
import type { SquareObservation } from '../src/vision/moveRecognizer.ts';
import type { ChessSquare } from '../src/chess/ChessGame.ts';
import type { ChessEngine, EngineAnalysisResult } from '../src/chess/engine.ts';

const PERSPECTIVE_CORNERS: Point2D[] = [
  { x: 100, y: 50 },  // top-left
  { x: 300, y: 50 },  // top-right
  { x: 380, y: 350 }, // bottom-right
  { x: 20, y: 350 },  // bottom-left
];

class MockEngine implements ChessEngine {
  private stopped = false;

  async analyze(fen: string): Promise<EngineAnalysisResult> {
    return {
      fen,
      bestMove: {
        from: 'e2',
        to: 'e4',
        san: 'e4',
      },
      score: 35,
      perspective: 'side-to-move',
      terminal: 'none',
      pv: ['e2e4', 'e7e5'],
      positionsEvaluated: 100,
      depth: 6,
      engine: 'chal-2.0.0',
      nodes: 100,
    };
  }

  stop(): void {
    this.stopped = true;
  }
}

function createDummyObservations(): SquareObservation[] {
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const obs: SquareObservation[] = [];
  for (const file of files) {
    for (let rank = 1; rank <= 8; rank += 1) {
      obs.push({
        square: `${file}${rank}` as ChessSquare,
        occupied: rank <= 2 || rank >= 7,
        confidence: 0.9,
      });
    }
  }
  return obs;
}

test('LiveGuidanceCoordinator produces perspective geometry and instructions', async () => {
  const mockEngine = new MockEngine();
  const coordinator = new LiveGuidanceCoordinator({
    engine: mockEngine,
    orientation: 'white_bottom',
  });

  const obs = createDummyObservations();
  // Frame 1 launches asynchronous engine analysis
  await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  // Yield to allow mock engine promise to resolve
  await new Promise((resolve) => setTimeout(resolve, 5));
  // Frame 2 receives the resolved engine guidance
  const model = await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);

  assert.equal(model.trackingState, 'tracking');
  assert.equal(model.turn, 'w');
  assert.ok(model.perspectiveGeometry.corners !== null);
  assert.ok(model.perspectiveGeometry.gridLines !== null);
  assert.equal(model.perspectiveGeometry.gridLines.length, 14);

  // Engine best move is e2 -> e4
  // Source quad (e2) and destination quad (e4) must be strictly convex
  assert.ok(model.perspectiveGeometry.sourceQuad !== null);
  assert.ok(model.perspectiveGeometry.destinationQuad !== null);
  assert.ok(model.perspectiveGeometry.arrow !== null);

  // Verify instruction text
  assert.ok(
    model.instruction.includes('e2') && model.instruction.includes('e4'),
    `Unexpected instruction: "${model.instruction}"`,
  );
});

test('LiveGuidanceCoordinator reflects orientation flip in projected coordinates', async () => {
  const mockEngine = new MockEngine();
  const coordinatorWhite = new LiveGuidanceCoordinator({
    engine: mockEngine,
    orientation: 'white_bottom',
  });
  const coordinatorBlack = new LiveGuidanceCoordinator({
    engine: mockEngine,
    orientation: 'black_bottom',
  });

  const obs = createDummyObservations();
  await coordinatorWhite.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  await coordinatorBlack.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  await new Promise((resolve) => setTimeout(resolve, 5));

  const modelWhite = await coordinatorWhite.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  const modelBlack = await coordinatorBlack.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);

  // In white_bottom, e2 is in row 6 (near bottom, y > 200)
  // In black_bottom, e2 is in row 1 (near top, y < 150)
  const whiteSourceY = modelWhite.perspectiveGeometry.sourceQuad![0].y;
  const blackSourceY = modelBlack.perspectiveGeometry.sourceQuad![0].y;

  assert.ok(
    whiteSourceY > blackSourceY + 50,
    `whiteSourceY (${whiteSourceY}) should be significantly greater than blackSourceY (${blackSourceY})`,
  );
});

test('LiveGuidanceCoordinator handles lost tracking state gracefully', async () => {
  const coordinator = new LiveGuidanceCoordinator();
  // No corners passed
  const model = await coordinator.processFrame(null, null);

  assert.equal(model.trackingState, 'uncalibrated');
  assert.equal(model.perspectiveGeometry.corners, null);
  assert.equal(model.perspectiveGeometry.arrow, null);
  assert.ok(model.instruction.includes('Locating board') || model.instruction.includes('Point camera'));
});

test('LiveGuidanceCoordinator suppresses stale engine analysis from an earlier position', async () => {
  let resolveFirstSearch: (res: EngineAnalysisResult) => void;
  const delayedEngine: ChessEngine = {
    analyze: (fen: string) => {
      if (fen.includes('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR')) {
        // Initial position search: delay artificially
        return new Promise<EngineAnalysisResult>((resolve) => {
          resolveFirstSearch = resolve;
        });
      }
      // Subsequent position search: resolve immediately
      return Promise.resolve({
        fen,
        bestMove: { from: 'e7', to: 'e5', san: 'e5' },
        score: -20,
        perspective: 'side-to-move',
        terminal: 'none',
        pv: ['e7e5'],
        positionsEvaluated: 50,
        depth: 5,
        engine: 'chal-2.0.0',
        nodes: 50,
      });
    },
    stop: () => {},
  };

  const coordinator = new LiveGuidanceCoordinator({ engine: delayedEngine });
  const obs = createDummyObservations();

  // Position 1: start position triggers delayed search
  const model1 = await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  assert.equal(model1.bestMove, null); // search still in flight

  // Position 2: White plays e4, so position advances to Black to move
  coordinator.getRecognizer().confirmManualMove('e2', 'e4');
  await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  await new Promise((resolve) => setTimeout(resolve, 5));
  const model2 = await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  assert.equal(model2.bestMove?.san, 'e5'); // Position 2's immediate search accepted

  // Now, delayed search for Position 1 resolves late!
  resolveFirstSearch!({
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    bestMove: { from: 'd2', to: 'd4', san: 'd4' },
    score: 40,
    perspective: 'side-to-move',
    terminal: 'none',
    pv: ['d2d4'],
    positionsEvaluated: 100,
    depth: 6,
    engine: 'chal-2.0.0',
    nodes: 100,
  });

  // Verify next frame does NOT overwrite Position 2 with stale Position 1 result!
  const model3 = await coordinator.processFrame(PERSPECTIVE_CORNERS, obs, false, 0.9);
  assert.equal(model3.bestMove?.san, 'e5');
  assert.notEqual(model3.bestMove?.san, 'd4');
});
