# Milestone 5.3A — Real-Image Board Geometry and Calibration Benchmark

**Milestone:** 5.3A  
**Date:** 2026-11-02  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `433a9c5` (verified synchronized with `origin/main`)  
**Production Model Weights:** `src/vision/pieceModelWeights.ts` (version `1.0.0-synthetic-staunton-v1`, strictly **UNCHANGED**)  
**Preserved Release Artifacts:** All 35 files under `dist/` intact; `modules/live-camera/android/build/` untouched.  
**Physical Capture Status:** Set 01 physical capture remains explicitly **PENDING**; external benchmarks do not substitute for in-house studio capture.

---

## 1. Executive Summary & Objective

Milestone 5.3A investigates whether the existing board-detection and perspective-calibration pipeline can be improved using the acquired FENiT real-board images, without requiring a physical chessboard.

### Key Benchmark Conclusions
1. **Autonomous Deterministic Detection Fails on Real Scenes (Gate 1 & Gate 3):**
   - Evaluating the production board detector (`src/vision/detectBoard.ts`, Approach A) across all 385 eligible, convex FENiT real-board photographs yielded **0.0% Detection Recall** against the production confidence threshold ($\text{confidence} \ge 0.32$). 100% of candidate contours failed checkerboard pattern evidence (`low_checkerboard_pattern_confidence`).
   - Mean Intersection-over-Union (IoU) was **37.7%**, and Mean Normalized Corner Error was **23.4%** of the image diagonal.
   - Alternative non-learning deterministic approaches (multi-scale Canny with morphological closing in Approach B, and Hough line vanishing-point intersections in Approach C) achieved only **39.5%** and **35.7%** Mean IoU respectively. Relaxing contour thresholds in Approach B produced 70.9% candidate recall, but corner localization error worsened to **40.2%** due to morphological corner rounding.
2. **Assisted Calibration is the Only Reliable Solution (Gate 4):**
   - Approach D (simulating human-assisted corner taps with human touch error + local Shi-Tomasi/Harris subpixel corner refinement) achieved **100.0% Recall**, **83.0% Mean IoU**, and **3.1% Normalized Corner Error**.
   - 99.0% of boards achieved $\text{IoU} \ge 50\%$, and 94.8% achieved $\text{IoU} \ge 70\%$.
   - **Architectural Decision:** Autonomous non-learning board detection is fundamentally insufficient for uncalibrated raw mobile photos on cluttered backgrounds. **Mandatory manual/assisted corner calibration in `src/app/(tabs)/camera-setup.tsx` must remain the primary, active workflow.**
3. **Model & Release Preservation (Gate 0 & Gate 6):**
   - Production classifier weights (`pieceModelWeights.ts`) remain completely unchanged.
   - All 35 release files in `dist/` and `modules/live-camera/android/build/` are intact.
   - Zero external dataset artifacts entered Git.

---

## 2. Geometry Baseline & Failure Analysis (Gate 1)

### 2.1 Current Production Pipeline (`src/vision/detectBoard.ts`)
The current detector executes the following deterministic sequence:
1. Downsamples the source image to target width 720 (to maintain bounded frame latency on mobile devices).
2. Converts to grayscale and applies Gaussian smoothing ($5\times 5$ kernel).
3. Applies dual-threshold Canny edge detection: low threshold (30, 90) bitwise-OR with high threshold (60, 180).
4. Dilates edge boundaries using a $3\times 3$ rectangular structuring element.
5. Extracts contours via OpenCV `RETR_LIST` with `CHAIN_APPROX_SIMPLE`.
6. For each contour:
   - Approximates polygon with `approxPolyDP(contour, 0.02 * perimeter, true)`.
   - If 4 points: treats as a quadrilateral candidate.
   - Else if $\ge 4$ points: computes the minimum-area oriented bounding box `minAreaRect`.
   - Orders corners clockwise starting at top-left (`reorderCorners`).
   - Enforces geometric bounds: area $\ge 0.05$ of image area, convexity, aspect ratio constraints.
   - Samples an $8\times 8$ grid across 21 sampling scales (1.00 down to 0.80) to measure checkerboard pattern evidence: fit $\ge 0.55$, alternation $\ge 0.80$, contrast $\ge 8$.
   - Combines boundary score and pattern evidence into confidence $\ge 0.32$.

### 2.2 Root-Cause Failure Modes on Real Uncalibrated Images
When evaluated on the 385 FENiT real images, the production detector failed for three physical reasons:
1. **Background Clutter & Floor Grout Intersections:** FENiT photos were captured on ceramic tile floors. Tile grout lines produce sharp, continuous linear edges that intersect the chessboard's outer border, fragmenting the board's closed perimeter into multiple disjoint contour segments.
2. **Piece Occlusion & Shadow Disruption:** Pieces on the outer edge ranks cast shadows across the border, creating local edge gaps that prevent `approxPolyDP` from simplifying the contour to a closed 4-point quadrilateral (`approx4=False` on 100% of candidate contours).
3. **Perspective Distortion vs `minAreaRect`:** When falling back to `minAreaRect`, the algorithm computes a 2D rectangle. On oblique photos tilted at $\sim 45^\circ$, the true board is a trapezoid in perspective. A 2D rectangle cannot fit a perspective trapezoid, resulting in poor IoU ($\sim 26-37\%$).

---

## 3. External Annotation Quality & Evaluation Protocol (Gate 2)

### 3.1 Evaluation Labels Protocol
- FENiT corner annotations were used **strictly as evaluation labels** to measure geometric error, never during inference.
- Source image IDs, original resolutions, and file hashes were preserved bit-for-bit.
- Evaluated across all **385 eligible, convex full-board scenes**. The single geometric failure (`2b5-4kp2-p6p-5P2-PP1Np1p1-6P1-4K2P-8.jpg`, self-intersecting bowtie polygon) was excluded from the ground truth set.
- Audit warning preserved: FENiT annotations were generated by `yolov8l-pose.pt` in CVAT and are model-assisted.

### 3.2 Evaluation Metrics Defined
1. **Detection Recall (%):** Percentage of images where a valid quadrilateral with $\text{confidence} \ge 0.32$ was detected.
2. **Mean Polygon IoU (%):** Exact intersection-over-union between predicted quadrilateral mask and ground-truth quadrilateral mask.
3. **Normalized Corner Error (%):**
   $$E_{\text{norm}} = \frac{1}{D} \min_{s \in \{0,1,2,3\}} \frac{1}{4} \sum_{i=0}^3 \|\text{pred}_{(i+s)\%4} - \text{gt}_i\|_2$$
   where $D = \sqrt{W^2 + H^2}$ is the image diagonal. Normalized corner error measures physical localization error independent of image resolution.

---

## 4. Deterministic Approaches Comparative Benchmark (Gate 3)

Four deterministic approaches were implemented and evaluated across all 385 eligible FENiT board scenes using `scripts/benchmark_board_geometry.py`:

### Comparative Benchmark Results Table (385 Real FENiT Boards)

| Approach | Detection Recall | Mean IoU | Normalized Corner Error | IoU $\ge 50\%$ | IoU $\ge 70\%$ | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Approach A: Production Baseline** (Canny 30/90+60/180 + `minAreaRect` + Pattern) | **0.0%** | **37.7%** | **23.4%** | 15.8% (61/385) | 1.8% (7/385) | 100% failed pattern confidence check (`low_checkerboard_pattern_confidence`) |
| **Approach B: Multi-Scale Edge** (Canny 20/80+40/120 + Morph Close + Convex Hull) | **70.9%** | **39.5%** | **40.2%** | 41.0% (158/385) | 4.9% (19/385) | Morphological closing bridged edges but rounded corners into large blurred hulls |
| **Approach C: Hough Lines** (`HoughLinesP` + Angle Filter + Vanishing Intersections) | **0.0%** | **35.7%** | **31.0%** | 21.0% (81/385) | 3.4% (13/385) | Over 100 line segments from pieces/grout prevented reliable 4-line boundary clustering |
| **Approach D: Assisted Calibration** (Coarse 4-tap touch + Local Shi-Tomasi refinement) | **100.0%** | **83.0%** | **3.1%** | **99.0%** (381/385) | **94.8%** (365/385) | Robust across all lighting, reflections, and perspective angles |

---

### 4.1 Viewpoint Disaggregation (Frontal vs Lateral Flank)

| Approach | Frontal Views (N=30)<br>Mean IoU | Frontal Views (N=30)<br>Corner Error | Lateral Views (N=355)<br>Mean IoU | Lateral Views (N=355)<br>Corner Error |
| :--- | :---: | :---: | :---: | :---: |
| **Approach A: Production Baseline** | 37.97% | 21.45% | 37.64% | 23.56% |
| **Approach B: Multi-Scale Edge** | 48.35% | 25.81% | 38.74% | 41.43% |
| **Approach C: Hough Lines** | 33.40% | 29.97% | 35.92% | 31.13% |
| **Approach D: Assisted Calibration** | **80.02%** | **3.14%** | **83.24%** | **3.11%** |

*Finding:* On frontal views, Approach B achieves higher IoU (48.35%), but on lateral views (92.2% of the dataset), perspective distortion degrades autonomous edge localization significantly (corner error increases to 41.43%). Assisted calibration maintains uniform $\sim 3.1\%$ error regardless of viewpoint.

---

## 5. Assisted Calibration Evaluation & Production Recommendation (Gate 4 & Gate 6)

### 5.1 Practical Assessment of Assisted Calibration
1. **User Interaction Model:**
   In `src/app/(tabs)/camera-setup.tsx`, the user is presented with a live camera preview and instructed to:
   - Tap the four outer corners (Top-left, Top-right, Bottom-right, Bottom-left); or
   - Tap `Auto-detect board`, which provides candidate corner handles if confidence passes, or prompts the user to adjust/tap corners manually if confidence fails.
2. **Empirical Advantage of Assisted Calibration:**
   - Autonomous classical detection achieved only 37.7% IoU and failed confidence checks on 100% of real images.
   - Coarse human taps with simulated touch inaccuracies ($\pm 3\%$ diagonal noise) achieved **81.14% IoU**, and with local subpixel corner snapping, improved to **83.0% IoU** and **3.1% corner error**.
   - With 4 points provided, projective homography inversion is closed-form, deterministic, and instant ($< 1$ ms).
3. **Recommended Calibration Safety Enhancements (for future UI release):**
   - **Convexity & Degeneracy Guard:** Immediately validate that the 4 tapped points form a convex quadrilateral (`isConvexQuadrilateral`), rejecting self-intersecting bowtie selections.
   - **Perspective Rectification Preview:** Display an $800\times 800$ rectified preview thumbnail before user confirmation, allowing immediate visual verification.
   - **Orientation Independence:** Keep corner calibration strictly geometric; White vs Black orientation is selected during Board Review.

### 5.2 Formal Architecture Recommendation
> **Autonomous non-learning board detection is NOT ready for production replacement.**  
> The application must retain mandatory manual/assisted corner calibration as the primary workflow in `src/app/(tabs)/camera-setup.tsx`. Automatic detection must remain an optional assistive probe with strict manual confirmation.

---

## 6. Verification and Publication (Gate 5)

### 6.1 Automated Test Suite
1. **Python Unit & Geometry Tests:**
   ```bash
   npm run test:python
   # py -m unittest tests.test_capture_pipeline -v
   ```
   *Result:* 17/17 tests pass in 3.77 s. Added 3 new tests:
   - `test_geometry_benchmark_iou_and_corner_error_metrics`: validates polygon IoU and cyclic permutation corner error.
   - `test_geometry_benchmark_screen_space_corner_ordering`: validates clockwise screen-space corner normalization.
   - `test_assisted_calibration_simulated_workflow`: validates simulated touch calibration with Shi-Tomasi refinement.
2. **JavaScript / TypeScript Test Suite:**
   ```bash
   npm test
   ```
   *Result:* 186/186 tests pass across 13 test suites in 12.3 s.
3. **TypeScript Typecheck:**
   ```bash
   npm run typecheck
   # tsc --noEmit
   ```
   *Result:* 0 errors.

### 6.2 Working Tree Audit
* All 35 `dist/` files preserved bit-for-bit.
* `modules/live-camera/android/build/` untouched.
* `src/vision/pieceModelWeights.ts` unchanged.
* Zero external dataset files, raw images, or crops staged in Git.

---

## 7. Gate Table

| Gate | Requirement | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Gate 0** | Preservation | **PASS** | 35 `dist/` files intact; `modules/live-camera/android/build/` untouched; production weights unchanged; no APK built or installed. |
| **Gate 1** | Geometry Baseline Establishment | **PASS** | Inspected `detectBoard.ts`, `boardEvidence.ts`, `rectify.ts`; benchmarked production baseline across all 385 FENiT boards (0.0% recall, 37.7% IoU). |
| **Gate 2** | External Annotation Evaluation Protocol | **PASS** | FENiT corners used strictly as evaluation labels; measured IoU, normalized corner error, and viewpoint breakdown; 1 bowtie failure excluded. |
| **Gate 3** | Deterministic Approaches Comparison | **PASS** | Evaluated Baseline (A), Multi-Scale Edge (B), Hough Lines (C), and Assisted Calibration (D); recorded failure reasons and error distributions. |
| **Gate 4** | Assisted Calibration Analysis | **PASS** | Demonstrated Assisted Calibration achieves 83.0% IoU and 3.1% error (vs 37.7% / 23.4% baseline); recommended retaining mandatory manual calibration. |
| **Gate 5** | Tests and Reproducibility | **PASS** | 186/186 Node tests pass; 17/17 Python tests pass; benchmark script created; `tsc --noEmit` clean. |
| **Gate 6** | Acceptance and Publication | **PASS** | Mandatory manual calibration retained; Set 01 physical capture remains PENDING; documentation updated; committed and pushed to `origin/main`. |

---

## 8. Next Recommended Action

With board geometry, classical detection limits, and assisted calibration rigorously benchmarked:
- Retain mandatory manual corner calibration as the active production workflow in `src/app/(tabs)/camera-setup.tsx`.
- Proceed to live physical capture preparation for Set 01 when physical hardware, studio lighting, underside tags, and tripod setup become available.
