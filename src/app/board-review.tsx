import { router, useLocalSearchParams } from 'expo-router';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { deleteCachedCapture } from '@/vision/imageLifecycle';

import {
  buildPositionFen,
  type ChessPieceSymbol,
  type FenBuilderResult,
  type SquareEntry,
} from '@/vision/fenBuilder';
import {
  computeRectifiedBoardGeometry,
  rectifyMat,
} from '@/vision/rectify';
import {
  classifySquareFeatures,
  type AnyImageData,
} from '@/vision/classifySquare';
import {
  recognizeBoardFromRectifiedImage,
  SYNTHETIC_MODEL_DISCLOSURE,
  type BoardRecognitionResult,
} from '@/vision/recognizeBoard';
import type { BoardViewOrientation, DetectedCorner } from '@/vision/types';

function getOpenCV(): any | null {
  try {
    return require('react-native-fast-opencv');
  } catch {
    return null;
  }
}

const PIECE_GLYPHS: Record<ChessPieceSymbol, string> = {
  K: '♔',
  Q: '♕',
  R: '♖',
  B: '♗',
  N: '♘',
  P: '♙',
  k: '♚',
  q: '♛',
  r: '♜',
  b: '♝',
  n: '♞',
  p: '♟',
};

const PALETTE_OPTIONS: { label: string; symbol: ChessPieceSymbol | null; glyph: string }[] = [
  { label: 'Empty', symbol: null, glyph: '·' },
  { label: 'White Pawn', symbol: 'P', glyph: '♙' },
  { label: 'White Knight', symbol: 'N', glyph: '♘' },
  { label: 'White Bishop', symbol: 'B', glyph: '♗' },
  { label: 'White Rook', symbol: 'R', glyph: '♖' },
  { label: 'White Queen', symbol: 'Q', glyph: '♕' },
  { label: 'White King', symbol: 'K', glyph: '♔' },
  { label: 'Black Pawn', symbol: 'p', glyph: '♟' },
  { label: 'Black Knight', symbol: 'n', glyph: '♞' },
  { label: 'Black Bishop', symbol: 'b', glyph: '♝' },
  { label: 'Black Rook', symbol: 'r', glyph: '♜' },
  { label: 'Black Queen', symbol: 'q', glyph: '♛' },
  { label: 'Black King', symbol: 'k', glyph: '♚' },
];

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;
const RANKS = ['1', '2', '3', '4', '5', '6', '7', '8'] as const;

export default function BoardReviewScreen() {
  const { width } = useWindowDimensions();
  const params = useLocalSearchParams<{
    imageUri?: string;
    corners?: string;
    imageWidth?: string;
    imageHeight?: string;
    orientation?: string;
  }>();

  const [attachedImageUri, setAttachedImageUri] = useState<string | null>(params.imageUri ?? null);
  const ownedImageUriRef = useRef<string | null>(params.imageUri ?? null);

  const initialOrientation: BoardViewOrientation =
    params.orientation === 'black-bottom' ? 'black-bottom' : 'white-bottom';

  const [orientation, setOrientation] = useState<BoardViewOrientation>(initialOrientation);
  const [selectedSquare, setSelectedSquare] = useState<string | null>('e4');

  const [recognitionResult, setRecognitionResult] = useState<BoardRecognitionResult | null>(null);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [recognitionStatus, setRecognitionStatus] = useState<string | null>(null);

  // Relinquish and delete temporary capture only when the review flow completely unmounts
  useEffect(() => {
    return () => {
      if (ownedImageUriRef.current) {
        deleteCachedCapture(ownedImageUriRef.current);
        ownedImageUriRef.current = null;
      }
    };
  }, []);

  const handleDiscardImage = () => {
    if (ownedImageUriRef.current) {
      deleteCachedCapture(ownedImageUriRef.current);
      ownedImageUriRef.current = null;
    }
    setAttachedImageUri(null);
    setRecognitionResult(null);
    setRecognitionStatus(null);
  };

  // Authoritative 64-square placement state
  const [squares, setSquares] = useState<Record<string, ChessPieceSymbol | null>>(() => {
    const map: Record<string, ChessPieceSymbol | null> = {};
    for (const f of FILES) {
      for (const r of RANKS) {
        map[`${f}${r}`] = null;
      }
    }
    return map;
  });

  // Track which squares have been confirmed by the user
  const [confirmedSquares, setConfirmedSquares] = useState<Set<string>>(new Set());

  // Automatic piece recognition on mount when capture and corners are provided
  useEffect(() => {
    let active = true;

    async function runAutoRecognition() {
      if (!params.imageUri || !params.corners) {
        return;
      }

      let parsedCorners: { x: number; y: number }[] = [];
      try {
        parsedCorners = JSON.parse(params.corners);
        if (!Array.isArray(parsedCorners) || parsedCorners.length !== 4) {
          throw new Error('Expected 4 corner points.');
        }
      } catch {
        if (active) setRecognitionStatus('Board corners unavailable. Manual reconstruction active.');
        return;
      }

      setIsRecognizing(true);
      setRecognitionStatus('Processing image and generating piece suggestions...');

      try {
        const opencv = getOpenCV();
        if (!opencv) {
          if (active) {
            setRecognitionStatus('OpenCV native module unavailable. Manual placement active.');
            setIsRecognizing(false);
          }
          return;
        }

        const { File } = require('expo-file-system');
        const file = new File(params.imageUri);
        const base64 = await file.base64();
        if (!base64 || !active) return;

        const { Mat } = opencv;
        let src: any = null;
        let releaseRectified: (() => void) | null = null;

        try {
          src = Mat.createFromBase64(base64);

          // Support both normalized [0, 1] coordinates and pixel coordinates
          const isNormalized = parsedCorners.every(
            (pt) => pt.x >= 0 && pt.x <= 1.05 && pt.y >= 0 && pt.y <= 1.05,
          );
          const pixelCorners: DetectedCorner[] = parsedCorners.map((pt) => ({
            x: isNormalized ? pt.x * src.cols : pt.x,
            y: isNormalized ? pt.y * src.rows : pt.y,
          }));

          const { rectified, release } = rectifyMat(src, pixelCorners, 800);
          releaseRectified = release;

          const buf = rectified.toBuffer('uint8');
          const isColor = buf.channels === 3;
          const imgData: AnyImageData = isColor
            ? { width: 800, height: 800, channels: 3, data: buf.buffer }
            : { width: 800, height: 800, data: buf.buffer };

          const geometry = computeRectifiedBoardGeometry(pixelCorners, { view: orientation });
          const result = recognizeBoardFromRectifiedImage(imgData, geometry);

          if (active) {
            setRecognitionResult(result);
            setSquares(result.suggestedPlacements);
            // All suggestions begin UNCONFIRMED to enforce user review
            setConfirmedSquares(new Set());
            setRecognitionStatus(
              `Generated ${result.summary.suggestedCount} piece suggestions. All squares with ? require your confirmation.`,
            );
          }
        } finally {
          releaseRectified?.();
          try {
            src?.release?.();
          } catch {
            // Safe release
          }
        }
      } catch (err: any) {
        if (active) {
          setRecognitionStatus(
            err instanceof Error
              ? `Automatic recognition unavailable: ${err.message}. Manual placement active.`
              : 'Automatic recognition unavailable. Manual placement active.',
          );
        }
      } finally {
        if (active) {
          setIsRecognizing(false);
        }
      }
    }

    runAutoRecognition();

    return () => {
      active = false;
    };
  }, [params.imageUri, params.corners, orientation]);

  const handleConfirmAll = () => {
    const all = new Set<string>();
    for (const f of FILES) {
      for (const r of RANKS) {
        all.add(`${f}${r}`);
      }
    }
    setConfirmedSquares(all);
  };

  // Metadata controls
  const [turn, setTurn] = useState<'w' | 'b'>('w');
  const [castlingK, setCastlingK] = useState(false);
  const [castlingQ, setCastlingQ] = useState(false);
  const [castlingSmallK, setCastlingSmallK] = useState(false);
  const [castlingSmallQ, setCastlingSmallQ] = useState(false);
  const [enPassantInput, setEnPassantInput] = useState('-');
  const [halfmoveInput, setHalfmoveInput] = useState('0');
  const [fullmoveInput, setFullmoveInput] = useState('1');

  const boardSize = Math.min(width - 24, 380);
  const cellSize = boardSize / 8;

  // Build castling flags
  const castlingFlags = useMemo(
    () => ({
      K: castlingK,
      Q: castlingQ,
      k: castlingSmallK,
      q: castlingSmallQ,
    }),
    [castlingK, castlingQ, castlingSmallK, castlingSmallQ],
  );

  // Compute live validation result through fenBuilder
  const validationResult: FenBuilderResult = useMemo(() => {
    const halfmove = parseInt(halfmoveInput, 10);
    const fullmove = parseInt(fullmoveInput, 10);

    return buildPositionFen(squares, {
      orientation,
      metadata: {
        turn,
        castling: castlingFlags,
        enPassant: enPassantInput.trim() === '' ? '-' : enPassantInput.trim(),
        halfmoveClock: Number.isNaN(halfmove) ? 0 : halfmove,
        fullmoveNumber: Number.isNaN(fullmove) ? 1 : fullmove,
      },
    });
  }, [squares, orientation, turn, castlingFlags, enPassantInput, halfmoveInput, fullmoveInput]);

  const handleSquarePress = (squareName: string) => {
    setSelectedSquare(squareName);
  };

  const handlePaletteSelect = (symbol: ChessPieceSymbol | null) => {
    if (!selectedSquare) {
      return;
    }
    setSquares((prev) => ({
      ...prev,
      [selectedSquare]: symbol,
    }));
    setConfirmedSquares((prev) => {
      const next = new Set(prev);
      next.add(selectedSquare);
      return next;
    });
  };

  const handleResetDraft = () => {
    const cleared: Record<string, ChessPieceSymbol | null> = {};
    for (const f of FILES) {
      for (const r of RANKS) {
        cleared[`${f}${r}`] = null;
      }
    }
    setSquares(cleared);
    setConfirmedSquares(new Set());
    setSelectedSquare('e4');
  };

  const handleLoadStartingPosition = () => {
    const starting: Record<string, ChessPieceSymbol | null> = {
      a8: 'r', b8: 'n', c8: 'b', d8: 'q', e8: 'k', f8: 'b', g8: 'n', h8: 'r',
      a7: 'p', b7: 'p', c7: 'p', d7: 'p', e7: 'p', f7: 'p', g7: 'p', h7: 'p',
      a2: 'P', b2: 'P', c2: 'P', d2: 'P', e2: 'P', f2: 'P', g2: 'P', h2: 'P',
      a1: 'R', b1: 'N', c1: 'B', d1: 'Q', e1: 'K', f1: 'B', g1: 'N', h1: 'R',
    };
    for (const f of FILES) {
      for (const r of RANKS) {
        if (!( `${f}${r}` in starting)) {
          starting[`${f}${r}`] = null;
        }
      }
    }
    setSquares(starting);
    setConfirmedSquares(new Set(Object.keys(starting)));
    setCastlingK(true);
    setCastlingQ(true);
    setCastlingSmallK(true);
    setCastlingSmallQ(true);
    setTurn('w');
    setEnPassantInput('-');
    setHalfmoveInput('0');
    setFullmoveInput('1');
  };

  const handleToggleOrientation = () => {
    setOrientation((prev) => (prev === 'white-bottom' ? 'black-bottom' : 'white-bottom'));
  };

  const handleSendToStudy = () => {
    if (!validationResult.valid || !validationResult.fen) {
      return;
    }
    router.push({
      pathname: '/study',
      params: { fen: validationResult.fen },
    });
  };

  // Render grid cells in display row/column order
  const gridRows = useMemo(() => {
    const rows: { squareName: string; isLight: boolean }[][] = [];
    for (let r = 0; r < 8; r += 1) {
      const rowCells: { squareName: string; isLight: boolean }[] = [];
      for (let c = 0; c < 8; c += 1) {
        const fileIdx = orientation === 'white-bottom' ? c : 7 - c;
        const rankIdx = orientation === 'white-bottom' ? 7 - r : r;
        const squareName = `${FILES[fileIdx]}${RANKS[rankIdx]}`;
        const isLight = (fileIdx + rankIdx) % 2 !== 0;
        rowCells.push({ squareName, isLight });
      }
      rows.push(rowCells);
    }
    return rows;
  }, [orientation]);

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <Text style={styles.title}>Position Review</Text>
        <Text style={styles.body}>
          Review and confirm each square. Automatic suggestions are generated from camera captures and require your confirmation.
        </Text>

        {/* Source image banner or manual mode badge */}
        {attachedImageUri ? (
          <View style={styles.imageCard}>
            <View style={styles.imageHeaderRow}>
              <Text style={styles.sectionHeader}>Attached Camera Capture (Raw Preview)</Text>
              <Pressable style={styles.discardButton} onPress={handleDiscardImage}>
                <Text style={styles.discardButtonText}>Discard Image</Text>
              </Pressable>
            </View>
            <Text style={styles.rawCaptureSubtitle}>
              Raw camera capture. Rectification and perspective warping are in-memory.
            </Text>
            <Image source={{ uri: attachedImageUri }} style={styles.sourceImage} resizeMode="contain" />
          </View>
        ) : (
          <View style={styles.manualCard}>
            <Text style={styles.manualHeader}>Manual Position Reconstruction</Text>
            <Text style={styles.manualText}>
              No camera capture attached. Set up pieces manually on the digital board below.
            </Text>
          </View>
        )}

        {/* Synthetic Model Disclosure & Status Banner */}
        {attachedImageUri ? (
          <View style={styles.disclosureCard}>
            <Text style={styles.disclosureTitle}>Notice: Experimental Piece Recognition</Text>
            <Text style={styles.disclosureBody}>{SYNTHETIC_MODEL_DISCLOSURE}</Text>
            {recognitionStatus ? (
              <View style={styles.recognitionStatusBox}>
                {isRecognizing ? <ActivityIndicator size="small" color="#2B6CB0" /> : null}
                <Text style={styles.recognitionStatusText}>{recognitionStatus}</Text>
              </View>
            ) : null}
          </View>
        ) : null}

        {/* Orientation and Reset Controls */}
        <View style={styles.controlsRow}>
          <Pressable style={styles.secondaryButton} onPress={handleToggleOrientation}>
            <Text style={styles.secondaryButtonText}>
              {orientation === 'white-bottom' ? 'View: White at Bottom' : 'View: Black at Bottom'}
            </Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={handleResetDraft}>
            <Text style={styles.secondaryButtonText}>Clear Board</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={handleLoadStartingPosition}>
            <Text style={styles.secondaryButtonText}>Initial Setup</Text>
          </Pressable>
          {confirmedSquares.size < 64 ? (
            <Pressable style={styles.secondaryButton} onPress={handleConfirmAll}>
              <Text style={styles.secondaryButtonText}>Confirm All</Text>
            </Pressable>
          ) : null}
        </View>

        {/* Confirmation Stats */}
        <View style={styles.statsBadge}>
          <Text style={styles.statsText}>
            User-Confirmed: {confirmedSquares.size} / 64 squares · Selected:{' '}
            {selectedSquare ? selectedSquare.toUpperCase() : 'None'} (
            {selectedSquare && squares[selectedSquare]
              ? `${squares[selectedSquare] === squares[selectedSquare]?.toUpperCase() ? 'White' : 'Black'} ${squares[selectedSquare]}`
              : 'Empty'}
            )
          </Text>
        </View>

        {/* 8x8 Interactive Digital Chess Board */}
        <View style={[styles.boardWrapper, { width: boardSize, height: boardSize }]}>
          {gridRows.map((row, rIdx) => (
            <View key={`row-${rIdx}`} style={styles.boardRow}>
              {row.map(({ squareName, isLight }) => {
                const piece = squares[squareName];
                const isSelected = selectedSquare === squareName;
                const isConfirmed = confirmedSquares.has(squareName);

                return (
                  <Pressable
                    key={squareName}
                    style={[
                      styles.cell,
                      {
                        width: cellSize,
                        height: cellSize,
                        backgroundColor: isLight ? '#EAF1F8' : '#8CA4C4',
                      },
                      isSelected ? styles.cellSelected : null,
                    ]}
                    onPress={() => handleSquarePress(squareName)}>
                    <Text
                      style={[
                        styles.pieceGlyph,
                        {
                          fontSize: cellSize * 0.7,
                          color: piece && piece === piece.toUpperCase() ? '#FFFFFF' : '#1C1C1C',
                        },
                      ]}>
                      {piece ? PIECE_GLYPHS[piece] : ''}
                    </Text>
                    {/* Unconfirmed badge */}
                    {!isConfirmed && piece ? (
                      <View style={styles.unconfirmedBadge}>
                        <Text style={styles.unconfirmedText}>?</Text>
                      </View>
                    ) : null}
                    <Text style={styles.coordinateLabel}>{squareName}</Text>
                  </Pressable>
                );
              })}
            </View>
          ))}
        </View>

        {/* Selected Square AI Suggestion Details */}
        {selectedSquare && recognitionResult?.squares[selectedSquare] ? (
          <View style={styles.suggestionCard}>
            <Text style={styles.suggestionHeader}>
              AI Recognition for {selectedSquare.toUpperCase()}:
            </Text>
            {(() => {
              const rec = recognitionResult.squares[selectedSquare];
              if (rec.occupancy === 'empty') {
                return (
                  <Text style={styles.suggestionBody}>
                    Predicted Empty ({(rec.occupancyConfidence * 100).toFixed(0)}% confidence)
                  </Text>
                );
              }
              if (rec.suggestedPiece) {
                const colorLabel = rec.pieceColor === 'white' ? 'White' : 'Black';
                return (
                  <>
                    <Text style={styles.suggestionBody}>
                      Suggested: {colorLabel} {rec.predictedType} (confidence: {(rec.confidence * 100).toFixed(0)}%, margin: {(rec.margin * 100).toFixed(0)}%)
                    </Text>
                    {rec.candidates.length > 1 ? (
                      <Text style={styles.candidatesText}>
                        Top candidates: {rec.candidates.slice(0, 3).map((c) => `${c.type} (${(c.probability * 100).toFixed(0)}%)`).join(' · ')}
                      </Text>
                    ) : null}
                  </>
                );
              }
              return (
                <Text style={styles.suggestionBodyUncertain}>
                  Status: Ambiguous piece type or color ({rec.reason ?? 'select from palette below'})
                </Text>
              );
            })()}
          </View>
        ) : null}

        {/* Piece Selection Palette */}
        <View style={styles.paletteContainer}>
          <Text style={styles.sectionHeader}>
            Assign Piece to {selectedSquare ? selectedSquare.toUpperCase() : 'selected square'}:
          </Text>
          <View style={styles.paletteRow}>
            {PALETTE_OPTIONS.map((item) => {
              const isSelected = selectedSquare && squares[selectedSquare] === item.symbol;
              const isSuggested =
                selectedSquare &&
                recognitionResult?.squares[selectedSquare]?.suggestedPiece === item.symbol &&
                item.symbol !== null;

              return (
                <Pressable
                  key={item.label}
                  style={[
                    styles.paletteButton,
                    isSelected ? styles.paletteButtonActive : null,
                    isSuggested && !isSelected ? styles.paletteButtonSuggested : null,
                  ]}
                  onPress={() => handlePaletteSelect(item.symbol)}>
                  <Text style={styles.paletteGlyph}>{item.glyph}</Text>
                  <Text style={styles.paletteLabel}>
                    {item.label.replace('White ', 'W ').replace('Black ', 'B ')}
                    {isSuggested ? ' ★' : ''}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        </View>

        {/* Metadata Controls */}
        <View style={styles.card}>
          <Text style={styles.sectionHeader}>Position Metadata</Text>

          {/* Turn selector */}
          <Text style={styles.inputLabel}>Side to Move:</Text>
          <View style={styles.toggleRow}>
            <Pressable
              style={[styles.toggleButton, turn === 'w' ? styles.toggleButtonActive : null]}
              onPress={() => setTurn('w')}>
              <Text style={turn === 'w' ? styles.toggleTextActive : styles.toggleText}>White to move</Text>
            </Pressable>
            <Pressable
              style={[styles.toggleButton, turn === 'b' ? styles.toggleButtonActive : null]}
              onPress={() => setTurn('b')}>
              <Text style={turn === 'b' ? styles.toggleTextActive : styles.toggleText}>Black to move</Text>
            </Pressable>
          </View>

          {/* Castling rights */}
          <Text style={styles.inputLabel}>Castling Rights:</Text>
          <View style={styles.toggleRow}>
            <Pressable
              style={[styles.checkbox, castlingK ? styles.checkboxActive : null]}
              onPress={() => setCastlingK((v) => !v)}>
              <Text style={castlingK ? styles.checkboxTextActive : styles.checkboxText}>K (White O-O)</Text>
            </Pressable>
            <Pressable
              style={[styles.checkbox, castlingQ ? styles.checkboxActive : null]}
              onPress={() => setCastlingQ((v) => !v)}>
              <Text style={castlingQ ? styles.checkboxTextActive : styles.checkboxText}>Q (White O-O-O)</Text>
            </Pressable>
            <Pressable
              style={[styles.checkbox, castlingSmallK ? styles.checkboxActive : null]}
              onPress={() => setCastlingSmallK((v) => !v)}>
              <Text style={castlingSmallK ? styles.checkboxTextActive : styles.checkboxText}>k (Black O-O)</Text>
            </Pressable>
            <Pressable
              style={[styles.checkbox, castlingSmallQ ? styles.checkboxActive : null]}
              onPress={() => setCastlingSmallQ((v) => !v)}>
              <Text style={castlingSmallQ ? styles.checkboxTextActive : styles.checkboxText}>q (Black O-O-O)</Text>
            </Pressable>
          </View>

          {/* En-Passant & Counters */}
          <View style={styles.metaInputGrid}>
            <View style={styles.metaInputBox}>
              <Text style={styles.inputLabel}>En-passant target:</Text>
              <TextInput
                style={styles.textInput}
                value={enPassantInput}
                onChangeText={setEnPassantInput}
                placeholder="'-' or square"
                autoCapitalize="none"
              />
            </View>
            <View style={styles.metaInputBox}>
              <Text style={styles.inputLabel}>Halfmove clock:</Text>
              <TextInput
                style={styles.textInput}
                value={halfmoveInput}
                onChangeText={setHalfmoveInput}
                keyboardType="numeric"
              />
            </View>
            <View style={styles.metaInputBox}>
              <Text style={styles.inputLabel}>Fullmove count:</Text>
              <TextInput
                style={styles.textInput}
                value={fullmoveInput}
                onChangeText={setFullmoveInput}
                keyboardType="numeric"
              />
            </View>
          </View>
        </View>

        {/* Live FEN Preview and Validation Status */}
        <View style={styles.card}>
          <Text style={styles.sectionHeader}>FEN Preview & Rule Validation</Text>
          <View style={styles.fenBox}>
            <Text style={styles.fenText}>{validationResult.fen ?? validationResult.placementFen ?? 'Invalid placement'}</Text>
          </View>

          {validationResult.errors.length > 0 ? (
            <View style={styles.errorContainer}>
              <Text style={styles.errorHeader}>Position Errors (Must resolve to send to Study):</Text>
              {validationResult.errors.map((err, i) => (
                <Text key={`err-${i}`} style={styles.errorItem}>
                  • {err}
                </Text>
              ))}
            </View>
          ) : (
            <View style={styles.validContainer}>
              <Text style={styles.validText}>
                ✓ Position is structurally valid ({validationResult.details?.legalMoveCount ?? 0} legal moves)
              </Text>
              {validationResult.details?.isCheck ? <Text style={styles.validSub}>Position is in CHECK.</Text> : null}
              {validationResult.details?.isCheckmate ? <Text style={styles.validSub}>Position is CHECKMATE.</Text> : null}
              {validationResult.details?.isStalemate ? <Text style={styles.validSub}>Position is STALEMATE.</Text> : null}
            </View>
          )}

          {validationResult.warnings.map((warn, i) => (
            <Text key={`warn-${i}`} style={styles.warningItem}>
              ⚠ {warn}
            </Text>
          ))}

          <Text style={styles.disclaimerText}>{validationResult.reachabilityDisclaimer}</Text>
        </View>

        {/* Handoff to Study Button */}
        <View style={styles.handoffContainer}>
          <Pressable
            style={validationResult.valid ? styles.primaryButton : styles.disabledButton}
            onPress={handleSendToStudy}
            disabled={!validationResult.valid}>
            <Text style={styles.primaryButtonText}>
              {validationResult.valid ? 'Send Position to Board Study' : 'Resolve Errors to Send to Study'}
            </Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F4F7FB' },
  content: { padding: 12, alignItems: 'center', gap: 12 },
  title: { fontSize: 22, fontWeight: '700', color: '#102A43', textAlign: 'center' },
  body: { fontSize: 13, color: '#486581', textAlign: 'center', marginBottom: 4 },
  sectionHeader: { fontSize: 14, fontWeight: '700', color: '#102A43', marginBottom: 6 },
  imageCard: {
    width: '100%',
    backgroundColor: '#FFFFFF',
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#D9E2EC',
    gap: 4,
  },
  imageHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
  },
  discardButton: {
    paddingVertical: 3,
    paddingHorizontal: 8,
    backgroundColor: '#FED7D7',
    borderRadius: 4,
  },
  discardButtonText: { fontSize: 11, fontWeight: '600', color: '#C53030' },
  rawCaptureSubtitle: { fontSize: 11, color: '#627D98', marginBottom: 4 },
  sourceImage: { width: '100%', height: 180, borderRadius: 6 },
  manualCard: {
    width: '100%',
    backgroundColor: '#EBF8FF',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#BEE3F8',
    alignItems: 'center',
  },
  manualHeader: { fontSize: 14, fontWeight: '700', color: '#2B6CB0' },
  manualText: { fontSize: 12, color: '#4A5568', textAlign: 'center', marginTop: 2 },
  controlsRow: {
    flexDirection: 'row',
    gap: 8,
    width: '100%',
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  secondaryButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    backgroundColor: '#E2E8F0',
    borderRadius: 6,
  },
  secondaryButtonText: { fontSize: 12, fontWeight: '600', color: '#2D3748' },
  statsBadge: {
    backgroundColor: '#EDF2F7',
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 6,
  },
  statsText: { fontSize: 12, color: '#4A5568', fontWeight: '500' },
  boardWrapper: {
    borderWidth: 2,
    borderColor: '#334E68',
    borderRadius: 4,
    overflow: 'hidden',
  },
  boardRow: { flexDirection: 'row', flex: 1 },
  cell: {
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  cellSelected: {
    borderWidth: 2.5,
    borderColor: '#0052CC',
  },
  pieceGlyph: {
    fontWeight: '700',
    textAlign: 'center',
  },
  coordinateLabel: {
    position: 'absolute',
    bottom: 1,
    right: 2,
    fontSize: 9,
    color: 'rgba(0,0,0,0.4)',
    fontWeight: '700',
  },
  unconfirmedBadge: {
    position: 'absolute',
    top: 2,
    left: 2,
    width: 13,
    height: 13,
    borderRadius: 6.5,
    backgroundColor: '#DD6B20',
    alignItems: 'center',
    justifyContent: 'center',
  },
  unconfirmedText: { fontSize: 9, fontWeight: '700', color: '#FFFFFF' },
  paletteContainer: {
    width: '100%',
    backgroundColor: '#FFFFFF',
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#D9E2EC',
    alignItems: 'center',
  },
  paletteRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    justifyContent: 'center',
  },
  paletteButton: {
    width: 48,
    height: 52,
    backgroundColor: '#F0F4F8',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#BCCCDC',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 2,
  },
  paletteButtonActive: {
    borderColor: '#0052CC',
    backgroundColor: '#E0EEFF',
    borderWidth: 2,
  },
  paletteButtonSuggested: {
    borderColor: '#16A34A',
    backgroundColor: '#F0FDF4',
    borderWidth: 1.5,
  },
  disclosureCard: {
    width: '100%',
    backgroundColor: '#FFFBEB',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#FDE68A',
    gap: 3,
  },
  disclosureTitle: { fontSize: 12, fontWeight: '700', color: '#92400E' },
  disclosureBody: { fontSize: 11, color: '#78350F' },
  recognitionStatusBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 4,
  },
  recognitionStatusText: { fontSize: 11, fontWeight: '600', color: '#2B6CB0' },
  suggestionCard: {
    width: '100%',
    backgroundColor: '#F0FDF4',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#BBF7D0',
    gap: 2,
  },
  suggestionHeader: { fontSize: 12, fontWeight: '700', color: '#166534' },
  suggestionBody: { fontSize: 12, color: '#15803D' },
  suggestionBodyUncertain: { fontSize: 12, color: '#B45309' },
  candidatesText: { fontSize: 11, color: '#4B5563', fontStyle: 'italic', marginTop: 2 },
  paletteGlyph: { fontSize: 24, color: '#102A43' },
  paletteLabel: { fontSize: 9, color: '#486581', textAlign: 'center' },
  card: {
    width: '100%',
    backgroundColor: '#FFFFFF',
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#D9E2EC',
    gap: 8,
  },
  inputLabel: { fontSize: 12, fontWeight: '600', color: '#334E68' },
  toggleRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap' },
  toggleButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#BCCCDC',
    backgroundColor: '#F0F4F8',
  },
  toggleButtonActive: {
    backgroundColor: '#0052CC',
    borderColor: '#0052CC',
  },
  toggleText: { fontSize: 12, color: '#334E68', fontWeight: '600' },
  toggleTextActive: { fontSize: 12, color: '#FFFFFF', fontWeight: '600' },
  checkbox: {
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#BCCCDC',
    backgroundColor: '#F0F4F8',
  },
  checkboxActive: {
    backgroundColor: '#2B6CB0',
    borderColor: '#2B6CB0',
  },
  checkboxText: { fontSize: 12, color: '#334E68' },
  checkboxTextActive: { fontSize: 12, color: '#FFFFFF', fontWeight: '600' },
  metaInputGrid: { flexDirection: 'row', gap: 8, width: '100%' },
  metaInputBox: { flex: 1 },
  textInput: {
    borderWidth: 1,
    borderColor: '#BCCCDC',
    borderRadius: 6,
    padding: 6,
    fontSize: 13,
    backgroundColor: '#FAFCFE',
    color: '#102A43',
  },
  fenBox: {
    backgroundColor: '#F0F4F8',
    padding: 8,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#D9E2EC',
  },
  fenText: { fontFamily: 'monospace', fontSize: 11, color: '#102A43' },
  errorContainer: {
    backgroundColor: '#FFF5F5',
    padding: 8,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#FEB2B2',
  },
  errorHeader: { fontSize: 12, fontWeight: '700', color: '#C53030', marginBottom: 2 },
  errorItem: { fontSize: 11, color: '#9B2C2C', marginTop: 1 },
  validContainer: {
    backgroundColor: '#F0FFF4',
    padding: 8,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#9AE6B4',
  },
  validText: { fontSize: 12, fontWeight: '700', color: '#276749' },
  validSub: { fontSize: 11, color: '#2F855A', marginTop: 2 },
  warningItem: { fontSize: 11, color: '#C05621' },
  disclaimerText: { fontSize: 10, color: '#829AB1', fontStyle: 'italic', marginTop: 4 },
  handoffContainer: { width: '100%', marginBottom: 16 },
  primaryButton: {
    backgroundColor: '#0052CC',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#CBD5E0',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  primaryButtonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '700' },
});
