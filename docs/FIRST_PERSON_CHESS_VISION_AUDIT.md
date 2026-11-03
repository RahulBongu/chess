# Milestone 5.4 — Handheld First-Person Chess Vision and Live Guidance Feasibility & License Audit

**Milestone:** 5.4  
**Date:** 2026-11-03  
**Target Architecture:** Handheld Android ARM64, offline, real-time rear-camera preview  
**Authoritative Engine:** Chal v2.0.0 (commit `c5410b5`, MIT License)  
**Game State Engine:** `chess.js` (MIT License)  
**Image Processing:** `react-native-fast-opencv` (Apache 2.0 / MIT)  

---

## 1. Executive Summary & Problem Formulation

Milestone 5.4 addresses the primary product goal: an **offline Android AI Chess Navigator that operates while a player holds their phone in their hand and points its rear camera at a real chessboard from a normal seated or standing playing position**.

Reference real-world systems (such as wearable Meta glasses HUD experiments by reverse.dev / pjchess and AiFiTech) establish the intended user experience:
1. The user points their camera at the board; the live rear camera remains the main interaction surface.
2. The board is recognized and tracked across normal hand tremor and slight angle changes.
3. The app tracks the legal chess position across moves.
4. When a player moves a piece and their hand clears the board, the app detects the move, updates the position against legal chess rules, and runs the local chess engine (Chal v2).
5. The live camera preview overlays a perspective-correct visual guide: the source square and destination square are highlighted directly on the physical board, connected by an arrow with a concise text instruction (e.g., *"Move knight from g1 to f3"*).
6. The user makes the move; the app detects the completed move and continues tracking.

### Critical Engineering Findings from Prior Milestones
- **The Failure of Independent 64-Square Classification:** Attempting to independently classify all 64 squares from scratch on every video frame fails in real time on mobile devices. Even if an individual square classifier had 85% accuracy, $0.85^{64} \approx 0.00004$ (less than 0.01% chance of a clean board). Moreover, players' hands passing over the board trigger wild classification noise.
- **The Power of Legal-Move-Constrained Temporal Tracking:** Chess is a sequential state machine. From any confirmed position $S_t$, there are typically only 20–35 legal moves ($M_{\text{legal}}$). Every legal move alters at most 2 squares (or 4 for castling, 3 for en-passant); 60 to 62 squares remain untouched. By filtering temporal occupancy deltas against $M_{\text{legal}}$, move detection becomes robust, immune to false piece classifications on static squares, and computationally lightweight.
- **Projective Homography vs. Linear Interpolation:** Naive linear interpolation (`lerp`) across quadrilateral edges distorts square boundaries under perspective (foreshortening causes nearer squares to appear larger than distant squares). A true projective homography $H_{\text{square} \to \text{quad}}$ is mathematically required to project squares and arrows onto the camera preview without distortion.

---

## 2. Technical Feasibility Audit

### 2.1 Handheld Camera Reality
In a handheld playing posture:
- The camera is oblique (30°–65° from horizontal), tilted, off-center, and subjected to continuous micro-movement (physiological tremor).
- Distance changes as the user leans in or shifts posture.
- A player's hand enters the frame, partially or wholly occluding squares during a move.

**Architecture Decisions:**
1. **Adaptive Temporal Filtering (EMA):** Smoothes corner coordinates ($\hat{C}_t = (1-\alpha)\hat{C}_{t-1} + \alpha C_t$, $\alpha \approx 0.35$) to eliminate tremor while tracking intentional panning.
2. **Outlier Gating:** Large single-frame corner jumps ($\Delta > \Delta_{\text{max}}$) are dampened to prevent transient occlusions from corrupting grid registration.
3. **Occlusion Grace Period:** When board detection drops a frame during hand movement, the tracker holds the last verified stable geometry for up to $N_{\text{grace}}$ frames rather than dropping the overlay.
4. **Assisted Calibration Fallback:** The user can seed or lock the four board corners with a single tap sequence; once seeded, tracking maintains calibration without requiring re-calibration after each move.

### 2.2 Move Detection & Temporal State Machine
Move recognition operates as an explicit multi-state automaton:
```text
[IDLE / UNCALIBRATED] 
        │ (Corners established)
        ▼
    [WATCHING] ──(Hand enters / delta spike)──► [OCCLUDED]
        ▲                                            │
        │ (Revert / timeout)                         │ (Hand clears / variance drops)
        │                                            ▼
        │                                      [STABILIZING]
        │                                            │ (K stable frames)
        │                                            ▼
   [UNCERTAIN] ◄──(No legal move match)────── [EVALUATING]
        ▲                                            │ (Unique legal move match)
        │                                            ▼
        └───────────────────────────────────── [MOVE_CONFIRMED]
                                                     │ (Execute move & query engine)
                                                     ▼
                                              [GUIDANCE_ACTIVE]
```
- **Transient Occlusion Suppression:** Frames during hand movement are classified as `OCCLUDED`. No move is evaluated while high variance persists.
- **Stability Gate:** Move evaluation triggers only after $K$ consecutive stable post-move frames (e.g., $K=3$, $\approx 750\text{ ms}$).
- **Delta-to-Legality Matching:** The difference between pre-move and post-move states isolates the source square (emptied) and destination square (occupied). Matching against `ChessGame.allLegalMoves()` ensures that only strictly legal chess moves are accepted.
- **Duplicate Suppression:** Subsequent identical frames are recognized as the active position and ignored to avoid repeated move announcements.
- **Zero Silent Invention:** If observed changes do not match any legal move, the system transitions to `UNCERTAIN` and offers a one-tap recovery rather than fabricating a false move.

### 2.3 Perspective Projection Mathematics
Let unit-square board coordinates be $(u, v) \in [0, 1]^2$.
The planar projective homography $H$ maps $(u, v)$ to preview coordinates $(x, y)$:
$$x = \frac{a \cdot u + b \cdot v + c}{g \cdot u + h \cdot v + 1}, \quad y = \frac{d \cdot u + e \cdot v + f}{g \cdot u + h \cdot v + 1}$$
For any square defined by column index $c \in [0, 7]$ and row index $r \in [0, 7]$:
- The four quad corners are computed by evaluating $H$ at $(c/8, r/8)$, $((c+1)/8, r/8)$, $((c+1)/8, (r+1)/8)$, and $(c/8, (r+1)/8)$.
- The move arrow starts at $P_{\text{src}} = H((c_{\text{from}}+0.5)/8, (r_{\text{from}}+0.5)/8)$ and ends at $P_{\text{dst}} = H((c_{\text{to}}+0.5)/8, (r_{\text{to}}+0.5)/8)$.
- Arrowhead wings are calculated in screen space directed back from $P_{\text{dst}}$ along the vector $P_{\text{dst}} - P_{\text{src}}$.

---

## 3. Dependency & License Compliance Audit

| Component | Repository Implementation | License | Compliance Status |
| :--- | :--- | :---: | :---: |
| **Chess Engine** | Chal v2.0.0 (`libchal-engine.so`, commit `c5410b5`) | **MIT** | Fully compliant. Embedded 49 KB NNUE weights; zero external network or disk dependencies. |
| **Chess Rules & Move Gen** | `chess.js` v1.4.0 | **MIT** | Fully compliant. Authoritative legal move generation and FEN handling. |
| **Computer Vision** | `react-native-fast-opencv` v1.0.1 / OpenCV 4.x | **Apache 2.0 / BSD** | Fully compliant. In-process C++ homography and matrix operations. |
| **Camera Framework** | `modules/live-camera` (CameraX ImageAnalysis) | Project-native | Fully compliant. Emits in-memory Y-plane frames; no frames stored. |
| **Stockfish Consideration** | Excluded | **GPL v3** | Excluded due to GPL copyleft restrictions. Chal v2 (MIT) remains the authoritative local engine. |

**Privacy & Offline Invariant:**
- `android.permission.INTERNET`: **STRICTLY ABSENT** in release builds.
- `android.permission.RECORD_AUDIO`: **STRICTLY ABSENT**.
- Camera frames are processed in-memory and immediately released. Zero cloud or telemetry dependencies.

---

## 4. Verification & Implementation Roadmap

1. **`src/vision/perspectiveOverlay.ts`**: Pure mathematical projective overlay engine (homography mapping, square quads, perspective arrow geometry, grid line generation).
2. **`src/vision/boardTracker.ts`**: Temporal corner stabilizer with EMA smoothing, outlier dampening, convexity enforcement, and occlusion grace period.
3. **`src/vision/moveRecognizer.ts`**: Temporal move recognizer with hand-occlusion detection, stability gating, legal-move reconciliation, duplicate event suppression, and uncertainty recovery.
4. **`src/vision/liveGuidance.ts`**: High-level coordinator linking `ChessGame`, `BoardTracker`, `MoveRecognizer`, and `ChalChessEngine` into a cohesive live guidance model.
5. **`src/app/(tabs)/live-game.tsx`**: Handheld live camera HUD screen integrating camera preview, perspective overlay, guidance banner, assisted calibration, and recovery workflows.
6. **Automated Unit & Integration Test Suites**: Deterministic tests verifying all projective calculations, corner stabilization, temporal move states, and error recovery paths.
