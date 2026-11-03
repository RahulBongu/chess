/**
 * Tests for `src/vision/liveDiagnostics.ts` (Milestone 5.4F).
 *
 * These are pure unit tests of counter/rate/latency bookkeeping. They do not
 * prove that any camera frame reached the phone's JavaScript runtime.
 */

import { test } from 'node:test';
import nodeAssert from 'node:assert/strict';

import {
  LiveDiagnostics,
  createEmptySnapshot,
  describePipelineStage,
  summarizeLatency,
} from '../src/vision/liveDiagnostics.ts';

test('summarizeLatency reports unavailable stats when a stage never ran', () => {
  const summary = summarizeLatency([]);
  nodeAssert.deepEqual(summary, { count: 0, medianMs: null, p95Ms: null, maxMs: null });
});

test('summarizeLatency computes odd/even medians and nearest-rank p95', () => {
  nodeAssert.deepEqual(summarizeLatency([5]), {
    count: 1,
    medianMs: 5,
    p95Ms: 5,
    maxMs: 5,
  });

  nodeAssert.deepEqual(summarizeLatency([1, 2, 3, 4]), {
    count: 4,
    medianMs: 2.5,
    p95Ms: 4,
    maxMs: 4,
  });

  const twenty = Array.from({ length: 20 }, (_, index) => index + 1);
  nodeAssert.deepEqual(summarizeLatency(twenty), {
    count: 20,
    medianMs: 10.5,
    p95Ms: 19,
    maxMs: 20,
  });
});

test('summarizeLatency ignores non-finite and negative samples', () => {
  const summary = summarizeLatency([2, Number.NaN, -1, Number.POSITIVE_INFINITY, 4]);
  nodeAssert.deepEqual(summary, { count: 2, medianMs: 3, p95Ms: 4, maxMs: 4 });
});

test('snapshot rates use the measured interval, not configured intervals', () => {
  const diagnostics = new LiveDiagnostics();
  diagnostics.reset(1_000);
  diagnostics.counters.trackingEventsReceived = 5;
  diagnostics.counters.detectionEventsReceived = 2;
  diagnostics.counters.motionTrackerAttempts = 4;

  const first = diagnostics.snapshot(2_000);
  nodeAssert.equal(first.intervalMs, 1_000);
  nodeAssert.equal(first.rates.trackingReceivedPerSec, 5);
  nodeAssert.equal(first.rates.detectionReceivedPerSec, 2);
  nodeAssert.equal(first.rates.motionTrackerUpdatesPerSec, 4);
  nodeAssert.equal(first.rates.trackingNativeAnalyzedPerSec, null);

  diagnostics.counters.trackingEventsReceived = 8;
  diagnostics.counters.detectionEventsReceived = 3;
  diagnostics.counters.motionTrackerAttempts = 6;

  const second = diagnostics.snapshot(2_500);
  nodeAssert.equal(second.intervalMs, 500);
  nodeAssert.equal(second.rates.trackingReceivedPerSec, 6);
  nodeAssert.equal(second.rates.detectionReceivedPerSec, 2);
  nodeAssert.equal(second.rates.motionTrackerUpdatesPerSec, 4);
});

test('native counters stay null until the bridge reports them', () => {
  const diagnostics = new LiveDiagnostics();
  diagnostics.reset(0);
  diagnostics.noteNativeTracking(120, 35);
  diagnostics.noteNativeDetection(undefined, 9);

  const snapshot = diagnostics.snapshot(1_000);
  nodeAssert.equal(snapshot.counters.nativeTrackingAnalyzed, 120);
  nodeAssert.equal(snapshot.counters.nativeTrackingDropped, 35);
  nodeAssert.equal(snapshot.counters.nativeDetectionAnalyzed, null);
  nodeAssert.equal(snapshot.counters.nativeDetectionDropped, 9);
  nodeAssert.equal(snapshot.rates.trackingNativeAnalyzedPerSec, 120);
  nodeAssert.equal(snapshot.rates.detectionNativeAnalyzedPerSec, null);
});

test('latency windows are bounded and snapshot summarizes the retained samples', () => {
  const diagnostics = new LiveDiagnostics(3);
  diagnostics.reset(0);
  diagnostics.record('tracker', 1);
  diagnostics.record('tracker', 2);
  diagnostics.record('tracker', 3);
  diagnostics.record('tracker', 4);

  const snapshot = diagnostics.snapshot(100);
  nodeAssert.equal(snapshot.latency.tracker.count, 3);
  nodeAssert.equal(snapshot.latency.tracker.medianMs, 3);
  nodeAssert.equal(snapshot.latency.tracker.maxMs, 4);
  nodeAssert.equal(snapshot.latency.detection.count, 0);
  nodeAssert.equal(snapshot.latency.detection.medianMs, null);
});

test('reset clears counters, windows and previous snapshot state', () => {
  const diagnostics = new LiveDiagnostics();
  diagnostics.reset(0);
  diagnostics.counters.trackingEventsReceived = 10;
  diagnostics.record('decode', 3);
  diagnostics.snapshot(1_000);
  diagnostics.reset(2_000);

  const snapshot = diagnostics.snapshot(2_500);
  nodeAssert.equal(snapshot.counters.trackingEventsReceived, 0);
  nodeAssert.equal(snapshot.latency.decode.count, 0);
  nodeAssert.equal(snapshot.intervalMs, 500);
});

test('describePipelineStage distinguishes each failure point', () => {
  const snapshot = createEmptySnapshot();
  nodeAssert.equal(describePipelineStage(snapshot, null, false).code, 'no_frames');

  snapshot.counters.trackingEventsReceived = 40;
  nodeAssert.equal(describePipelineStage(snapshot, null, false).code, 'frames_no_board');

  snapshot.counters.candidatesAccepted = 1;
  nodeAssert.equal(describePipelineStage(snapshot, null, false).code, 'board_not_tracked');

  snapshot.counters.motionTrackerValid = 20;
  nodeAssert.equal(describePipelineStage(snapshot, 'tracking', false).code, 'tracking_no_samples');

  snapshot.counters.squareSamplingOps = 5;
  nodeAssert.equal(describePipelineStage(snapshot, 'tracking', false).code, 'tracking_active');

  snapshot.counters.uncertainObservations = 3;
  nodeAssert.equal(describePipelineStage(snapshot, 'tracking', false).code, 'tracking_uncertain');

  snapshot.counters.trustedOverlayFrames = 10;
  nodeAssert.equal(describePipelineStage(snapshot, 'tracking', true).code, 'guidance_active');
});

test('describePipelineStage reports the current tracker state over stale history', () => {
  const snapshot = createEmptySnapshot();
  snapshot.counters.trackingEventsReceived = 40;
  snapshot.counters.candidatesAccepted = 1;
  snapshot.counters.motionTrackerValid = 20;
  snapshot.counters.squareSamplingOps = 5;
  snapshot.counters.trustedOverlayFrames = 10;

  // Cumulative counters say a trusted overlay existed, but the tracker is
  // currently holding/lost: the HUD must not keep claiming guidance is active.
  nodeAssert.equal(describePipelineStage(snapshot, 'holding', true).code, 'tracking_holding');
  nodeAssert.equal(describePipelineStage(snapshot, 'lost', true).code, 'tracking_lost');
  nodeAssert.equal(
    describePipelineStage(snapshot, 'uninitialized', true).code,
    'tracking_lost',
  );
  // BoardTracker clears its corners after the grace window and reports
  // 'uncalibrated' again; that must still read as lost, not "awaiting guidance".
  nodeAssert.equal(describePipelineStage(snapshot, 'uncalibrated', true).code, 'tracking_lost');

  // Holding is itself evidence that tracking started, even before any
  // overlay was rendered or the motion tracker logged a valid fix.
  const fresh = createEmptySnapshot();
  fresh.counters.trackingEventsReceived = 10;
  fresh.counters.candidatesAccepted = 1;
  nodeAssert.equal(describePipelineStage(fresh, 'holding', false).code, 'tracking_holding');
  nodeAssert.equal(describePipelineStage(snapshot, 'tracking', true).code, 'guidance_active');
});

test('session rates divide totals by elapsed time since reset', () => {
  const diagnostics = new LiveDiagnostics();
  diagnostics.reset(1_000);
  diagnostics.counters.trackingEventsReceived = 30;
  diagnostics.counters.detectionEventsReceived = 10;
  diagnostics.counters.motionTrackerAttempts = 25;
  diagnostics.noteNativeTracking(60, 300);

  const snapshot = diagnostics.snapshot(11_000);
  nodeAssert.equal(snapshot.sessionElapsedMs, 10_000);
  nodeAssert.equal(snapshot.sessionRates.trackingReceivedPerSec, 3);
  nodeAssert.equal(snapshot.sessionRates.detectionReceivedPerSec, 1);
  nodeAssert.equal(snapshot.sessionRates.motionTrackerUpdatesPerSec, 2.5);
  nodeAssert.equal(snapshot.sessionRates.trackingNativeAnalyzedPerSec, 6);
  nodeAssert.equal(snapshot.sessionRates.trackingDroppedPerSec, 30);
  nodeAssert.equal(snapshot.sessionRates.detectionNativeAnalyzedPerSec, null);

  // The instantaneous interval rate is still available separately.
  nodeAssert.equal(snapshot.rates.trackingReceivedPerSec, 3);
});
