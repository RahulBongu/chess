import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { base64ToUint8Array, validateGrayscaleFrame } from '../src/vision/grayscaleFrame.ts';

describe('grayscale frame ingestion', () => {
  it('decodes a valid base64 payload to bytes', () => {
    const input = new Uint8Array([0, 1, 2, 255]);
    const base64 = Buffer.from(input).toString('base64');
    const output = base64ToUint8Array(base64);
    assert.equal(output.length, input.length);
    for (let i = 0; i < input.length; i += 1) {
      assert.equal(output[i], input[i]);
    }
  });

  it('accepts a payload whose length matches width*height', () => {
    const width = 4;
    const height = 3;
    const data = new Uint8Array(width * height).fill(7);
    const base64 = Buffer.from(data).toString('base64');
    const result = validateGrayscaleFrame(base64, width, height);
    assert.equal(result.ok, true);
    if (result.ok) {
      assert.equal(result.buffer.length, width * height);
      assert.equal(result.buffer[0], 7);
    }
  });

  it('rejects a payload whose length does not match dimensions', () => {
    const data = new Uint8Array(5);
    const base64 = Buffer.from(data).toString('base64');
    const result = validateGrayscaleFrame(base64, 2, 2);
    assert.equal(result.ok, false);
    if (!result.ok) {
      assert.match(result.message, /does not match/);
    }
  });

  it('rejects non-positive or non-integer dimensions', () => {
    const data = new Uint8Array(4);
    const base64 = Buffer.from(data).toString('base64');
    assert.equal(validateGrayscaleFrame(base64, 0, 2).ok, false);
    assert.equal(validateGrayscaleFrame(base64, 2, 0).ok, false);
    assert.equal(validateGrayscaleFrame(base64, 1.5, 2).ok, false);
    assert.equal(validateGrayscaleFrame(base64, 2, 2.5).ok, false);
  });

  it('rejects empty or invalid base64', () => {
    assert.equal(validateGrayscaleFrame('', 2, 2).ok, false);
    assert.equal(validateGrayscaleFrame('!!!not-base64!!!', 2, 2).ok, false);
  });

  it('rejects empty base64 in decoder', () => {
    assert.throws(() => base64ToUint8Array(''), /empty/);
  });
});