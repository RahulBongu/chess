import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { ChessColors, PIECE_GLYPHS } from '@/theme/chess';
import type { ChessGame, ChessSquare } from '@/chess';

interface ChessBoardProps {
  game: ChessGame;
  selectedSquare: ChessSquare | null;
  legalDestinations: ChessSquare[];
  onSquarePress: (square: ChessSquare) => void;
  boardSize: number;
}

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const;

export function ChessBoard({
  game,
  selectedSquare,
  legalDestinations,
  onSquarePress,
  boardSize,
}: ChessBoardProps) {
  const squareSize = Math.floor(boardSize / 8);
  const legalSet = new Set(legalDestinations);

  return (
    <View style={[styles.board, { width: squareSize * 8, height: squareSize * 8 }]}>
      {RANKS.map((rank) => (
        <View key={rank} style={styles.row}>
          {FILES.map((file) => {
            const square = `${file}${rank}` as ChessSquare;
            const piece = game.pieceAt(square);
            const isLight = (file.charCodeAt(0) - 97 + rank) % 2 === 0;
            const isSelected = selectedSquare === square;
            const isLegal = legalSet.has(square);
            const isKingInCheck =
              piece?.type === 'k' && piece.color === game.turn() && game.isCheck() && square === findKingSquare(game, piece.color);

            const backgroundColor = isSelected
              ? ChessColors.selected
              : isKingInCheck
                ? ChessColors.check
                : isLight
                  ? ChessColors.lightSquare
                  : ChessColors.darkSquare;

            return (
              <Pressable
                key={square}
                onPress={() => onSquarePress(square)}
                style={[
                  styles.square,
                  { width: squareSize, height: squareSize, backgroundColor },
                ]}
                accessibilityRole="button"
                accessibilityLabel={`${square}${piece ? ` ${piece.color} ${piece.type}` : ''}`}>
                {isLegal && !piece ? <View style={styles.legalDot} /> : null}
                {isLegal && piece ? <View style={styles.legalCaptureRing} /> : null}
                {piece ? (
                  <Text
                    style={[
                      styles.piece,
                      {
                        fontSize: Math.round(squareSize * 0.72),
                        color: piece.color === 'w' ? ChessColors.whitePiece : ChessColors.blackPiece,
                        textShadowColor: piece.color === 'w' ? '#000000' : '#FFFFFF',
                        textShadowOffset: { width: 0, height: 0 },
                        textShadowRadius: 2,
                      },
                    ]}>
                    {PIECE_GLYPHS[`${piece.color}${piece.type}`]}
                  </Text>
                ) : null}
              </Pressable>
            );
          })}
        </View>
      ))}
    </View>
  );
}

function findKingSquare(game: ChessGame, color: 'w' | 'b'): ChessSquare | null {
  for (let r = 1; r <= 8; r++) {
    for (const f of FILES) {
      const square = `${f}${r}` as ChessSquare;
      const piece = game.pieceAt(square);
      if (piece?.type === 'k' && piece.color === color) {
        return square;
      }
    }
  }
  return null;
}

const styles = StyleSheet.create({
  board: {
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: '#333333',
  },
  row: {
    flexDirection: 'row',
  },
  square: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  piece: {
    fontWeight: '700',
    lineHeight: undefined,
  },
  legalDot: {
    position: 'absolute',
    width: '28%',
    height: '28%',
    borderRadius: 999,
    backgroundColor: ChessColors.legalMove,
  },
  legalCaptureRing: {
    position: 'absolute',
    width: '88%',
    height: '88%',
    borderRadius: 999,
    borderWidth: 4,
    borderColor: ChessColors.legalCapture,
  },
});