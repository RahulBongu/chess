export interface FileSystemClient {
  delete(uri: string): void;
  exists(uri: string): boolean;
}

let customFileClient: ((uri: string) => FileSystemClient) | null = null;

export function setCustomFileClient(client: ((uri: string) => FileSystemClient) | null): void {
  customFileClient = client;
}

/**
 * Safely deletes a cached capture file without throwing unhandled exceptions.
 * Uses injected FileSystemClient if provided (e.g. in tests) or expo-file-system
 * on React Native.
 */
export function deleteCachedCapture(uri: string | null | undefined): boolean {
  if (!uri || typeof uri !== 'string' || uri.trim() === '') {
    return true;
  }

  const trimmed = uri.trim();

  // Custom or test client
  if (customFileClient) {
    try {
      const client = customFileClient(trimmed);
      if (client.exists(trimmed)) {
        client.delete(trimmed);
      }
      return true;
    } catch {
      return false;
    }
  }

  // React Native / Expo environment
  try {
    const { File } = require('expo-file-system');
    const file = new File(trimmed);
    if (file.exists) {
      file.delete();
    }
    return true;
  } catch {
    return false;
  }
}
