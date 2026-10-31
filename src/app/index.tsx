import { router } from 'expo-router';
import React, { useRef, useState } from 'react';
import {
  Alert,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ChessBoard } from '@/components/ChessBoard';
import { ChessGame, type ChessSquare, type PromotionPiece } from '@/chess';
import { ChessColors, PIECE_GLYPHS } from '@/theme/chess';

const PROMOTION_OPTIONS: PromotionPiece[] = ['q', 'r', 'b', 'n'];

export default function ChessScreen() {
  const { width } = useWindowDimensions();
  const gameRef = useRef(new ChessGame());
  const game = gameRef.current;
  const [, setRevision] = useState(0);
  const [selectedSquare, setSelectedSquare] = useState<ChessSquare | null>(null);
  const [legalDestinations, setLegalDestinations] = useState<ChessSquare[]>([]);
  const [pendingPromotion, setPendingPromotion] = useState<{ from: ChessSquare; to: ChessSquare } | null>(null);

  const bump = () => setRevision((value) => value + 1);
  const boardSize = Math.min(width - 24, 420);

  const resetSelection = () => {
    setSelectedSquare(null);
    setLegalDestinations([]);
  };

  const handleSquarePress = (square: ChessSquare) => {
    if (pendingPromotion) {
      return;
    }

    if (selectedSquare && legalDestinations.includes(square)) {
      if (game.isPromotionMove(selectedSquare, square)) {
        setPendingPromotion({ from: selectedSquare, to: square });
        return;
      }

      const move = game.tryMove({ from: selectedSquare, to: square });
      if (move) {
        resetSelection();
        bump();
      }
      return;
    }

    const piece = game.pieceAt(square);
    if (piece && piece.color === game.turn()) {
      setSelectedSquare(square);
      setLegalDestinations(game.legalDestinations(square));
    } else {
      resetSelection();
    }
  };

  const handleUndo = () => {
    if (game.undo()) {
      resetSelection();
      bump();
    }
  };

  const handleRestart = () => {
    game.reset();
    resetSelection();
    bump();
  };

  const handleOpenLiveGame = () => {
    try {
      router.push('/live-game');
    } catch (error) {
      Alert.alert(
        'Navigation failed',
        error instanceof Error ? error.message : 'Unable to open Live Game.',
      );
    }
  };

  const handlePromotion = (piece: PromotionPiece) => {
    if (!pendingPromotion) {
      return;
    }
    const move = game.tryMove({ from: pendingPromotion.from, to: pendingPromotion.to, promotion: piece });
    if (move) {
      resetSelection();
      bump();
    }
    setPendingPromotion(null);
  };

  const turnLabel = game.turn() === 'w' ? 'White' : 'Black';
  const statusLabel = game.isCheckmate()
    ? `Checkmate — ${game.turn() === 'w' ? 'Black' : 'White'} wins`
    : game.isStalemate()
      ? 'Stalemate'
      : game.isDraw()
        ? 'Draw'
        : game.isCheck()
          ? `${turnLabel} is in check`
          : `${turnLabel} to move`;

  const history = game.history();

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Text style={styles.title}>AI Chess Navigator</Text>
          <Text style={styles.subtitle}>Local chessboard — Stockfish analysis coming later</Text>
        </View>

        <View style={styles.statusCard}>
          <Text style={styles.statusText}>{statusLabel}</Text>
          <Text style={styles.statusMeta}>Move {Math.ceil(game.moveCount() / 2) || 1} · {history.length} plies</Text>
        </View>

        <View style={styles.boardWrapper}>
          <ChessBoard
            game={game}
            selectedSquare={selectedSquare}
            legalDestinations={legalDestinations}
            onSquarePress={handleSquarePress}
            boardSize={boardSize}
          />
        </View>

        <View style={styles.controls}>
          <Pressable style={styles.button} onPress={handleUndo}>
            <Text style={styles.buttonText}>Undo</Text>
          </Pressable>
          <Pressable style={[styles.button, styles.secondaryButton]} onPress={handleRestart}>
            <Text style={styles.buttonText}>Restart</Text>
          </Pressable>
          <Pressable
            style={[styles.button, styles.setupButton]}
            onPress={() => router.push('/camera-setup')}>
            <Text style={styles.buttonText}>Camera / Board Setup</Text>
          </Pressable>
          <Pressable
            style={[styles.button, styles.liveButton]}
            onPress={handleOpenLiveGame}>
            <Text style={styles.buttonText}>Live Game</Text>
          </Pressable>
          <Pressable
            style={[styles.button, styles.setupButton]}
            onPress={() => router.push('/camera-lab')}>
            <Text style={styles.buttonText}>Camera Lab</Text>
          </Pressable>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Move history</Text>
          {history.length === 0 ? (
            <Text style={styles.emptyText}>No moves yet.</Text>
          ) : (
            <View style={styles.historyList}>
              {history.map((san, index) => (
                <Text key={`${san}-${index}`} style={styles.historyItem}>
                  {Math.floor(index / 2) + 1}
                  {index % 2 === 0 ? '.' : '...'} {san}
                </Text>
              ))}
            </View>
          )}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Analysis</Text>
          <Text style={styles.unavailableText}>Unavailable until Stockfish is integrated.</Text>
          <Text style={styles.cardBody}>
            This panel will later show the recommended move, source and destination squares, an arrow overlay, and a concise explanation.
          </Text>
        </View>
      </ScrollView>

      <Modal visible={Boolean(pendingPromotion)} transparent animationType="fade">
        <View style={styles.modalBackdrop}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Promote pawn</Text>
            <View style={styles.promotionRow}>
              {PROMOTION_OPTIONS.map((piece) => (
                <Pressable
                  key={piece}
                  style={styles.promotionButton}
                  onPress={() => handlePromotion(piece)}>
                  <Text style={styles.promotionGlyph}>
                    {PIECE_GLYPHS[`w${piece}`]}
                  </Text>
                </Pressable>
              ))}
            </View>
            <Pressable style={styles.cancelButton} onPress={() => setPendingPromotion(null)}>
              <Text style={styles.cancelText}>Cancel</Text>
            </Pressable>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  content: {
    padding: 12,
    gap: 12,
    alignItems: 'center',
  },
  header: {
    width: '100%',
    alignItems: 'center',
    gap: 4,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
  },
  subtitle: {
    fontSize: 13,
    opacity: 0.7,
    textAlign: 'center',
  },
  statusCard: {
    width: '100%',
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#EAF1F8',
    alignItems: 'center',
    gap: 2,
  },
  statusText: {
    fontSize: 16,
    fontWeight: '600',
  },
  statusMeta: {
    fontSize: 12,
    opacity: 0.7,
  },
  boardWrapper: {
    width: '100%',
    alignItems: 'center',
  },
  controls: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    justifyContent: 'center',
  },
  button: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 10,
    backgroundColor: '#208AEF',
  },
  secondaryButton: {
    backgroundColor: '#6C757D',
  },
  setupButton: {
    backgroundColor: '#7B5EA7',
  },
  liveButton: {
    backgroundColor: '#1B8A5A',
  },
  buttonText: {
    color: '#FFFFFF',
    fontWeight: '600',
    fontSize: 14,
  },
  card: {
    width: '100%',
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#F4F4F6',
    gap: 6,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '700',
  },
  cardBody: {
    fontSize: 13,
    opacity: 0.8,
  },
  emptyText: {
    fontSize: 13,
    opacity: 0.6,
  },
  historyList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  historyItem: {
    fontSize: 13,
    fontFamily: 'monospace',
  },
  unavailableText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#B00020',
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.45)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalCard: {
    width: '80%',
    padding: 20,
    borderRadius: 16,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    gap: 12,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
  },
  promotionRow: {
    flexDirection: 'row',
    gap: 12,
  },
  promotionButton: {
    width: 56,
    height: 56,
    borderRadius: 12,
    backgroundColor: '#EAF1F8',
    alignItems: 'center',
    justifyContent: 'center',
  },
  promotionGlyph: {
    fontSize: 34,
    color: '#111111',
  },
  cancelButton: {
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  cancelText: {
    color: '#208AEF',
    fontWeight: '600',
  },
});