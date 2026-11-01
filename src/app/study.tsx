import React, { useEffect, useRef, useState } from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ChessBoard } from '@/components/ChessBoard';
import { ChessGame, type ChessSquare } from '@/chess';
import { createHeuristicEngine, type EngineAnalysisResult } from '@/chess/engine';
import { importStudyInput, type StudyPosition } from '@/study/studyGame';

export default function StudyScreen() {
  const { width } = useWindowDimensions();
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [positions, setPositions] = useState<StudyPosition[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [moves, setMoves] = useState<string[]>([]);

  const engineRef = useRef(createHeuristicEngine());
  const analysisTokenRef = useRef(0);
  const [analysis, setAnalysis] = useState<EngineAnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const boardSize = Math.min(width - 24, 420);

  const handleImport = () => {
    const result = importStudyInput(input);
    if (!result) {
      setError('Invalid FEN or PGN. Check the text and try again.');
      setPositions([]);
      setMoves([]);
      setCurrentIndex(0);
      return;
    }

    setError(null);
    setPositions(result.positions);
    setMoves(result.moves);
    setCurrentIndex(0);
  };

  const handleClear = () => {
    setInput('');
    setError(null);
    setPositions([]);
    setMoves([]);
    setCurrentIndex(0);
  };

  const handleAnalyze = async () => {
    if (!current) {
      return;
    }

    const token = ++analysisTokenRef.current;
    setIsAnalyzing(true);
    setAnalysisError(null);

    try {
      const result = await engineRef.current.analyze(current.fen);
      if (token !== analysisTokenRef.current) {
        return;
      }
      setAnalysis(result);
    } catch (error) {
      if (token !== analysisTokenRef.current) {
        return;
      }
      setAnalysisError(error instanceof Error ? error.message : 'Analysis failed.');
    } finally {
      if (token === analysisTokenRef.current) {
        setIsAnalyzing(false);
      }
    }
  };

  const current = positions[currentIndex];
  const game = current ? new ChessGame(current.fen) : null;

  useEffect(() => {
    engineRef.current.stop();
    setAnalysis(null);
    setAnalysisError(null);
  }, [currentIndex, positions]);

  useEffect(() => {
    return () => {
      engineRef.current.stop();
    };
  }, []);

  const canGoBack = currentIndex > 0;
  const canGoForward = currentIndex < positions.length - 1;

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <Text style={styles.title}>Board Study</Text>
        <Text style={styles.body}>
          Paste a FEN position or PGN game. Everything is parsed locally and offline.
        </Text>

        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Paste FEN or PGN here"
          multiline
          autoCapitalize="none"
          autoCorrect={false}
        />

        <View style={styles.controls}>
          <Pressable style={styles.primaryButton} onPress={handleImport}>
            <Text style={styles.buttonText}>Import</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={handleClear}>
            <Text style={styles.buttonText}>Clear</Text>
          </Pressable>
        </View>

        {error ? <Text style={styles.error}>{error}</Text> : null}

        {game ? (
          <View style={styles.controls}>
            <Pressable
              style={isAnalyzing ? styles.disabledButton : styles.primaryButton}
              onPress={handleAnalyze}
              disabled={isAnalyzing}>
              <Text style={styles.buttonText}>{isAnalyzing ? 'Analyzing...' : 'Analyze'}</Text>
            </Pressable>
          </View>
        ) : null}

        {analysis ? (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Heuristic suggestion — preliminary analysis</Text>
            <Text style={styles.body}>
              {analysis.bestMove
                ? `Suggested move: ${analysis.bestMove.san} (${analysis.bestMove.from} to ${analysis.bestMove.to})`
                : 'No legal move available.'}
            </Text>
            <Text style={styles.body}>
              Evaluation: {analysis.score} centipawns ({analysis.perspective}) · depth {analysis.depth} · positions evaluated {analysis.positionsEvaluated ?? analysis.nodes}
            </Text>
            {analysis.pv.length > 1 ? (
              <Text style={styles.body}>Predicted line: {analysis.pv.join(' ')}</Text>
            ) : null}
            {analysis.terminal !== 'none' ? (
              <Text style={styles.body}>Terminal: {analysis.terminal}</Text>
            ) : null}
            <Text style={styles.body}>
              This is a shallow heuristic prototype, not Stockfish and not grandmaster-strength analysis.
            </Text>
          </View>
        ) : null}

        {analysisError ? <Text style={styles.error}>{analysisError}</Text> : null}

        {game ? (
          <>
            <View style={styles.boardWrapper}>
              <ChessBoard
                game={game}
                selectedSquare={null}
                legalDestinations={[]}
                onSquarePress={() => {}}
                boardSize={boardSize}
              />
            </View>

            <Text style={styles.status}>
              Position {currentIndex + 1} of {positions.length}
              {current?.moveSan ? ` — ${current.moveSan}` : ''}
            </Text>

            <View style={styles.controls}>
              <Pressable
                style={canGoBack ? styles.secondaryButton : styles.disabledButton}
                onPress={() => canGoBack && setCurrentIndex(0)}
                disabled={!canGoBack}>
                <Text style={styles.buttonText}>First</Text>
              </Pressable>
              <Pressable
                style={canGoBack ? styles.secondaryButton : styles.disabledButton}
                onPress={() => canGoBack && setCurrentIndex((value) => value - 1)}
                disabled={!canGoBack}>
                <Text style={styles.buttonText}>Prev</Text>
              </Pressable>
              <Pressable
                style={canGoForward ? styles.secondaryButton : styles.disabledButton}
                onPress={() => canGoForward && setCurrentIndex((value) => value + 1)}
                disabled={!canGoForward}>
                <Text style={styles.buttonText}>Next</Text>
              </Pressable>
              <Pressable
                style={canGoForward ? styles.secondaryButton : styles.disabledButton}
                onPress={() => canGoForward && setCurrentIndex(positions.length - 1)}
                disabled={!canGoForward}>
                <Text style={styles.buttonText}>Last</Text>
              </Pressable>
            </View>

            {moves.length > 0 ? (
              <View style={styles.card}>
                <Text style={styles.cardTitle}>Moves</Text>
                <Text style={styles.body}>{moves.join(' ')}</Text>
              </View>
            ) : null}
          </>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1 },
  content: { padding: 16, gap: 12, alignItems: 'center' },
  title: { fontSize: 22, fontWeight: '700', textAlign: 'center' },
  body: { fontSize: 13, textAlign: 'center', opacity: 0.8 },
  status: { fontSize: 14, textAlign: 'center', fontWeight: '600' },
  error: { fontSize: 13, textAlign: 'center', color: '#B00020', fontWeight: '600' },
  input: {
    width: '100%',
    minHeight: 120,
    borderWidth: 1,
    borderColor: '#B0BEC5',
    borderRadius: 10,
    padding: 10,
    fontSize: 13,
    fontFamily: 'monospace',
  },
  boardWrapper: { width: '100%', alignItems: 'center' },
  controls: { width: '100%', flexDirection: 'row', gap: 8, justifyContent: 'center', flexWrap: 'wrap' },
  primaryButton: { paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, backgroundColor: '#208AEF', minWidth: 120, alignItems: 'center' },
  secondaryButton: { paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, backgroundColor: '#6C757D', minWidth: 90, alignItems: 'center' },
  disabledButton: { paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, backgroundColor: '#B0BEC5', minWidth: 90, alignItems: 'center' },
  buttonText: { color: '#FFFFFF', fontWeight: '600', fontSize: 14 },
  card: { width: '100%', padding: 12, borderRadius: 12, backgroundColor: '#F4F4F6', gap: 6 },
  cardTitle: { fontSize: 16, fontWeight: '700' },
});
