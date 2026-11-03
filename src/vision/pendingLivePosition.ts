/**
 * Session-scoped handoff for a position the user has reviewed in the board
 * editor. The editor runs as a separate route, so a tiny module store avoids
 * stuffing a FEN (and its side-to-move) through navigation back-stack params.
 *
 * This is deliberately in-memory: a confirmed position is not persisted, never
 * leaves the device, and is cleared as soon as the live screen consumes it.
 */

let pendingFen: string | null = null;

export function setPendingLivePosition(fen: string): void {
  pendingFen = fen;
}

export function takePendingLivePosition(): string | null {
  const value = pendingFen;
  pendingFen = null;
  return value;
}

export function clearPendingLivePosition(): void {
  pendingFen = null;
}
