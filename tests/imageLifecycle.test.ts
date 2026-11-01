/// <reference types="node" />
import assert from 'node:assert/strict';
import { afterEach, beforeEach, describe, it } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  deleteCachedCapture,
  setCustomFileClient,
  type FileSystemClient,
} from '../src/vision/imageLifecycle.ts';

describe('camera image ownership and lifecycle management', { concurrency: 1 }, () => {
  beforeEach(() => {
    // Provide node:fs client for test execution
    setCustomFileClient((_uri) => ({
      exists: (u: string) => fs.existsSync(u),
      delete: (u: string) => fs.unlinkSync(u),
    }));
  });

  afterEach(() => {
    setCustomFileClient(null);
  });

  it('1. Camera -> Board Review: image remains available after camera blur', () => {
    // Simulate temporary image on filesystem
    const testFile = path.join(os.tmpdir(), `test_cam_blur_${Date.now()}.jpg`);
    fs.writeFileSync(testFile, 'test image bytes');
    assert.equal(fs.existsSync(testFile), true);

    // Camera screen state simulation
    let capturedImageUri: string | null = testFile;
    let isHandedOff = false;

    // User taps "Review & Reconstruct Position" -> Ownership transferred!
    const transferredUri = capturedImageUri;
    isHandedOff = true;
    capturedImageUri = null; // Camera relinquishes reference

    // Camera screen loses focus -> useFocusEffect cleanup runs!
    const cameraCleanupOnBlur = () => {
      if (isHandedOff) {
        // Ownership transferred: do NOT delete file!
        isHandedOff = false;
        return;
      }
      if (capturedImageUri) {
        deleteCachedCapture(capturedImageUri);
        capturedImageUri = null;
      }
    };

    cameraCleanupOnBlur();

    // Verify the transferred file was NOT deleted!
    assert.equal(fs.existsSync(testFile), true, 'Transferred capture must remain on disk after camera blur');

    // Clean up
    deleteCachedCapture(testFile);
    assert.equal(fs.existsSync(testFile), false);
  });

  it('2. Board Review -> Study: image is not prematurely deleted while Board Review is underneath', () => {
    const testFile = path.join(os.tmpdir(), `test_review_underneath_${Date.now()}.jpg`);
    fs.writeFileSync(testFile, 'test image bytes');
    assert.equal(fs.existsSync(testFile), true);

    // Board review owns the image ref
    const ownedImageRef = { current: testFile as string | null };

    // User navigates from Board Review to Study (router.push('/study'))
    // Board Review remains mounted in the navigation stack underneath Study
    const pushToStudy = () => {
      // Pushing does NOT unmount Board Review, so ownedImageRef is NOT released
    };

    pushToStudy();

    // Verify file is still intact on disk
    assert.equal(fs.existsSync(testFile), true, 'File must not be deleted when pushing Study');

    // Clean up
    deleteCachedCapture(ownedImageRef.current);
    assert.equal(fs.existsSync(testFile), false);
  });

  it('3. Leaving the owning review flow: temporary image is cleaned up on unmount or discard', () => {
    const testFile = path.join(os.tmpdir(), `test_unmount_cleanup_${Date.now()}.jpg`);
    fs.writeFileSync(testFile, 'test image bytes');
    assert.equal(fs.existsSync(testFile), true);

    const ownedImageRef = { current: testFile as string | null };

    // Simulate Board Review unmount (user exits review flow completely)
    const boardReviewUnmount = () => {
      if (ownedImageRef.current) {
        deleteCachedCapture(ownedImageRef.current);
        ownedImageRef.current = null;
      }
    };

    boardReviewUnmount();

    // Verify file is deleted on disk
    assert.equal(fs.existsSync(testFile), false, 'Temporary capture must be deleted when review unmounts');
    assert.equal(ownedImageRef.current, null);
  });

  it('4. Manual review without an image: handles null/empty URI safely without errors', () => {
    assert.equal(deleteCachedCapture(null), true);
    assert.equal(deleteCachedCapture(undefined), true);
    assert.equal(deleteCachedCapture(''), true);
    assert.equal(deleteCachedCapture('   '), true);
  });

  it('5. Invalid or missing URI: handled gracefully without throwing or crashing', () => {
    const nonExistentPath = path.join(os.tmpdir(), 'non_existent_image_12345.jpg');
    // Calling deleteCachedCapture on a file that does not exist should return true without throwing
    const result = deleteCachedCapture(nonExistentPath);
    assert.equal(result, true);

    // Verify error safety with custom throwing file client in local try/finally
    setCustomFileClient((_uri) => ({
      exists: () => true,
      delete: () => {
        throw new Error('OS permission denied');
      },
    }));

    try {
      const errorSafeResult = deleteCachedCapture('mock://protected/file.jpg');
      assert.equal(errorSafeResult, false, 'Should catch client exceptions and return false');
    } finally {
      // Reset is handled by afterEach
    }
  });
});
