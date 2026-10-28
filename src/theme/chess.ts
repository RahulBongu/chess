export const ChessColors = {
  lightSquare: '#F0D9B5',
  darkSquare: '#B58863',
  selected: 'rgba(20, 85, 30, 0.45)',
  legalMove: 'rgba(20, 85, 30, 0.28)',
  legalCapture: 'rgba(180, 40, 40, 0.32)',
  lastMove: 'rgba(255, 215, 0, 0.35)',
  check: 'rgba(220, 40, 40, 0.45)',
  whitePiece: '#FFFFFF',
  blackPiece: '#111111',
} as const;

export const PIECE_GLYPHS: Record<string, string> = {
  wp: '♙',
  wn: '♘',
  wb: '♗',
  wr: '♖',
  wq: '♕',
  wk: '♔',
  bp: '♟',
  bn: '♞',
  bb: '♝',
  br: '♜',
  bq: '♛',
  bk: '♚',
};