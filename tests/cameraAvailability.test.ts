import { test } from 'node:test';
import assert from 'node:assert/strict';

import { resolveCameraAvailability } from '../src/camera/availability.ts';

test('native platforms never call the web-only availability probe', async () => {
  let calls = 0;
  const probe = async () => {
    calls += 1;
    return true;
  };

  assert.equal(await resolveCameraAvailability('android', probe), null);
  assert.equal(await resolveCameraAvailability('ios', probe), null);
  assert.equal(calls, 0);
});

test('web probes camera availability', async () => {
  assert.equal(await resolveCameraAvailability('web', async () => true), true);
  assert.equal(await resolveCameraAvailability('web', async () => false), false);
});

test('a failed web probe is reported as unavailable', async () => {
  assert.equal(
    await resolveCameraAvailability('web', async () => {
      throw new Error('camera probe failed');
    }),
    false,
  );
});
