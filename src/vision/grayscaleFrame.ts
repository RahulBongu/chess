// Helpers for the live camera's raw grayscale (Y-plane) frames.
//
// The native analyzer emits Base64-encoded single-channel 8-bit pixels, NOT a
// JPEG/PNG image. OpenCV's createFromBase64 decodes encoded images, so it must
// not be used for these frames. Use Mat.createFromBuffer('uint8', height, width, 1, buffer).

export interface GrayscaleFrameOk {
  ok: true;
  buffer: Uint8Array;
}

export interface GrayscaleFrameError {
  ok: false;
  message: string;
}

export type GrayscaleFrameValidation = GrayscaleFrameOk | GrayscaleFrameError;

export function base64ToUint8Array(base64: string): Uint8Array {
  if (typeof base64 !== 'string' || base64.trim() === '') {
    throw new Error('Grayscale frame payload is empty.');
  }

  let binary: string;
  try {
    binary = atob(base64);
  } catch {
    throw new Error('Grayscale frame payload is not valid Base64.');
  }

  const buffer = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    buffer[index] = binary.charCodeAt(index) & 0xff;
  }
  return buffer;
}

export function validateGrayscaleFrame(
  base64: string,
  width: number,
  height: number,
): GrayscaleFrameValidation {
  if (!Number.isInteger(width) || width <= 0) {
    return { ok: false, message: 'Grayscale frame width must be a positive integer.' };
  }
  if (!Number.isInteger(height) || height <= 0) {
    return { ok: false, message: 'Grayscale frame height must be a positive integer.' };
  }

  let buffer: Uint8Array;
  try {
    buffer = base64ToUint8Array(base64);
  } catch (error) {
    return {
      ok: false,
      message: error instanceof Error ? error.message : 'Grayscale frame payload is invalid.',
    };
  }

  const expectedLength = width * height;
  if (buffer.length !== expectedLength) {
    return {
      ok: false,
      message: `Grayscale frame payload length ${buffer.length} does not match ${width}x${height} (${expectedLength}).`,
    };
  }

  return { ok: true, buffer };
}