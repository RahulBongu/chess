# Milestone 5.3C — Geometry Metric Reconciliation and Calibration Robustness

**Milestone:** 5.3C  
**Date:** 2026-11-03  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `3290082` (verified synchronized with `origin/main`)  
**Production Model Weights:** `src/vision/pieceModelWeights.ts` (version `1.0.0-synthetic-staunton-v1`, strictly **UNCHANGED**)  
**Preserved Release Artifacts:** All 35 files under `dist/` intact; `modules/live-camera/android/build/` untouched.  
**Physical Capture Status:** Set 01 physical capture remains explicitly **PENDING**; external benchmarks do not substitute for in-house studio capture.

---

## 1. Executive Summary & Objective

Milestone 5.3C resolves the metric reconciliation between proposal-level and accepted-level geometry figures, strengthens corner-ordering and coordinate-boundary test coverage, characterizes the ground-truth-seeded assisted calibration simulation with comprehensive distributional metrics (mean, median, P90, P95), and establishes a fully reproducible benchmark baseline.

### Key Audit Reconciliations & Robustness Gates
1. **Approach B Metrics Reconciled (Gate 1):**
   - **Discrepancy Explained:** In Milestone 5.3B, Approach B reported $55.7\%$ accepted-level IoU alongside $39.5\%$ proposal-level IoU despite having identical sample counts ($273/385$).
   - **Root Cause & Mathematical Proof:**
     - The accepted-level metric evaluated **only the 273 images where a candidate proposal was found** ($N=273$), yielding a conditional mean IoU of $\mathbf{55.68\%}$ and mean corner error of $\mathbf{15.68\%}$.
     - The proposal-level diagnostic evaluated **all 385 images in the dataset** ($N=385$), assigning $0\%$ IoU and $100\%$ corner error penalty to the 112 images where no candidate was found:
       $$\text{Proposal Mean IoU} = 55.68\% \times \frac{273}{385} = \mathbf{39.48\%}$$
       $$\text{Proposal Mean Corner Error} = 15.68\% \times \frac{273}{385} + 100.0\% \times \frac{112}{385} = 11.12\% + 29.09\% = \mathbf{40.21\%}$$
     - The candidate population was identical ($273$ images), but the denominators differed ($273$ vs $385$). Both figures are now explicitly defined and mathematically verified.
   - **Relaxed Threshold Disclosure:** In Approach B, the 273 candidates were "accepted" under a relaxed experimental threshold ($\text{confidence} = 0.35$ assigned to any convex quad) that completely bypassed the production checkerboard pattern evidence check. Under the true production acceptance gate (pattern fit $\ge 0.55$, alternation $\ge 0.80$, contrast $\ge 8$), **0 / 385** Approach B candidates pass.
2. **Ground-Truth-Seeded Assisted-Calibration Characterization (Gate 2):**
   - Formally designated as a **Ground-Truth-Seeded Assisted-Calibration Simulation** (evaluated with fixed random seed `42` on 385 FENiT annotations with Gaussian noise $\sigma = 0.03 \times \text{diagonal}$).
   - **Initial Perturbed Taps (Before Refinement):**
     - Mean IoU: **81.36%** (Median: **81.75%**, P90: **87.83%**)
     - Mean Corner Error: **3.71%** (Median: **3.66%**, P90: **4.93%**, P95: **5.35%**)
   - **Post-Refinement Snapped Corners (Local Shi-Tomasi in 5% window):**
     - Mean IoU: **83.09%** (Median: **83.81%**, P90: **92.60%**)
     - Mean Corner Error: **3.10%** (Median: **2.99%**, P90: **4.98%**, P95: **5.53%**)
   - **Convexity Quality Gate:** Local snapping corrupted the quadrilateral into a non-convex shape on **1 of 385 boards (0.26%)**, which was safely rejected by the convexity check (`isContourConvex`).
   - **Disclosure:** This evaluates local feature snapping on perturbed ground-truth; it is **NOT** a measured human-operator device result.
3. **Corner-Ordering Invariance & Boundary Test Coverage (Gate 3 & Gate 4):**
   - Added automated tests proving that `reorderCalibrationPoints` correctly normalizes **all 24 permutations ($4! = 24$)** of known convex quadrilaterals without mutating $(x, y)$ coordinates.
   - Verified that steep perspective trapezoids and rotated diamond quadrilaterals reorder cleanly to valid clockwise orientation.
   - Quantified aspect-ratio display quantization error: integer display pixel rounding ($h_{\text{disp}} = \text{round}(w_{\text{disp}} \cdot H / W)$) introduces $\le 0.5 \text{ px}$ rounding, resulting in a strictly bounded relative aspect-ratio error of **$< 0.35\%$** across displays $\ge 320 \text{ px}$ width. Extreme boundary points $[0.0, 1.0]$ map exactly to sensor boundaries.

---

## 2. Reconciled Geometry Benchmark Baseline (Gate 1 & Gate 2)

### 2.1 Reconciled Comparative Summary Table (385 Real FENiT Boards)

The table below presents the verified benchmark baseline, strictly separating accepted detections from candidate proposals with explicit denominators:

| Approach | Accepted Count | Accepted Recall | Accepted Mean IoU ($N_{\text{acc}}$) | Accepted Corner Error ($N_{\text{acc}}$) | Candidate Proposals ($N_{\text{prop}}$) | Proposal Mean IoU ($N=385$) | Proposal Corner Error ($N=385$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approach A: Production Baseline** (Canny + `minAreaRect` + Pattern) | **0 / 385** | **0.0%** | **N/A (0 det)** | **N/A (0 det)** | **385 / 385** (100.0%) | **37.7%** | **23.4%** |
| **Approach B: Multi-Scale Edge** (Canny + Morph Close + Convex Hull) | **273 / 385** | **70.9%** | **55.7%** | **15.7%** | **273 / 385** (70.9%) | **39.5%** | **40.2%** |
| **Approach C: Hough Lines** (`HoughLinesP` + Vanishing Point Intersections) | **0 / 385** | **0.0%** | **N/A (0 det)** | **N/A (0 det)** | **344 / 385** (89.3%) | **35.7%** | **31.0%** |
| **Approach D: Ground-Truth-Seeded Assisted Calibration Simulation** | **384 / 385** | **99.7%** | **83.1%** | **3.1%** | **385 / 385** (100.0%) | **83.0%** | **3.1%** |

*Threshold Disclosures:*
- **Approach A & C Acceptance Gate:** Production threshold ($\text{confidence} \ge 0.32$), requiring strict checkerboard pattern evidence (fit $\ge 0.55$, alternation $\ge 0.80$, contrast $\ge 8$). Both failed 100% on raw uncalibrated scenes.
- **Approach B Acceptance Gate:** Relaxed experimental threshold ($\text{confidence} = 0.35$ assigned to any convex hull quad), completely bypassing checkerboard pattern verification. If evaluated under production pattern evidence, Approach B acceptance is **0 / 385 (0.0%)**.
- **Approach D Simulation:** 1 quad out of 385 was rejected due to non-convexity after local snapping, yielding 384/385 accepted.

---

### 2.2 Detailed Statistical Distribution of Approach D Simulation
Evaluated from the identical 385 FENiT board corner annotations with Gaussian touch noise ($\sigma = 0.03 \times D$, seed `42`):

| Evaluation Stage | Mean IoU | Median IoU | P90 IoU | Mean Corner Error | Median Corner Error | P90 Corner Error | P95 Corner Error | Rejections |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Initial Coarse Taps (Before Refinement)** | 81.36% | 81.75% | 87.83% | 3.71% | 3.66% | 4.93% | 5.35% | 0 |
| **Post-Refinement Snapped Corners** | **83.09%** | **83.81%** | **92.60%** | **3.10%** | **2.99%** | **4.98%** | **5.53%** | 1 (0.26%) |

---

## 3. Corner Ordering Invariance & Coordinate Quality Gates (Gate 3 & Gate 4)

### 3.1 24-Permutation Ordering Invariance
An automated test suite verified that `reorderCalibrationPoints` guarantees coordinate invariance across all $4! = 24$ permutations:
- **No Coordinate Mutation:** For every permutation $\sigma(P)$, the set of resulting coordinate values equals the original set:
  $$\{P_{\sigma(0)}, P_{\sigma(1)}, P_{\sigma(2)}, P_{\sigma(3)}\} = \{P_0, P_1, P_2, P_3\}$$
- **Deterministic Starting Point:** The first corner is uniquely defined as $P_{\text{TL}} = \arg\min_{P \in \text{quad}} (x + y)$ (the top-leftmost vertex).
- **Clockwise Traversal:** Angular sorting around the centroid ensures consecutive vertices traverse clockwise in screen coordinates.
- **Convexity Invariance:** All 24 permutations of any convex quadrilateral pass `isValidCalibration(reordered).valid === true`.

### 3.2 Degenerate and Ambiguous Input Rejection
Geometric validation (`isValidCalibration`) reliably detects and rejects:
1. **Self-Intersecting Bowties:** Opposite edge intersection (`segmentsIntersect`) flags crossing edges.
2. **Collinear / Zero-Area Points:** Enclosed area $< 0.02$ flags flattened or degenerate selections.
3. **Nearly Coincident Points:** Squared distance $< 0.05^2$ flags duplicate touches.
4. **Out-of-Bounds Touches:** Coordinates outside $[0.0, 1.0]$ are blocked.

### 3.3 Display Normalization and Aspect-Ratio Rounding Error Bounds
- In `src/app/(tabs)/camera-setup.tsx`, preview height is calculated as:
  $$h_{\text{disp}} = \text{round}\left(w_{\text{disp}} \cdot \frac{H_{\text{orig}}}{W_{\text{orig}}}\right)$$
- Because $h_{\text{disp}}$ is an integer pixel value, a quantization rounding difference of $|h_{\text{disp}} - w_{\text{disp}} \cdot (H/W)| \le 0.5 \text{ px}$ occurs.
- **Verified Error Bound:** Across standard mobile display widths ($w_{\text{disp}} \ge 320 \text{ px}$) and sensor formats (4:3, 16:9, 3:4, 9:16, 1:1, 5712:4284), the relative aspect-ratio error is strictly bounded below **$0.35\%$** (less than $1/280\text{th}$).
- Extreme boundary touches ($[0.0, 0.0]$ to $[1.0, 1.0]$) map exactly to $(0, 0)$ and $(W_{\text{orig}}, H_{\text{orig}})$ without clipping.

---

## 4. Assisted Calibration UX Implementation (Gate 3)

The following UI improvements are active in `src/app/(tabs)/camera-setup.tsx`:
1. **Actionable Validation Warning Banner:**
   When 4 corners are selected or detected but fail geometric validation (`isValidCalibration === false`), an explicit warning banner renders `validation.reason` directly above the controls (e.g., *"The selected corners form a self-intersecting quadrilateral"* or *"Select the corners in top-left, top-right, bottom-right, bottom-left order"*).
2. **Clockwise Corner Normalization Button (`reorderCalibrationPoints`):**
   A *"Sort corners clockwise"* button normalizes unaligned or counter-clockwise corner selections into canonical clockwise order without requiring the user to reset all 4 points.
3. **Primary Workflow Preservation:**
   Manual corner calibration remains the mandatory primary workflow. Autonomous detection remains an optional assistive probe, and user confirmation is required before any geometry is committed.

---

## 5. Verification and Publication (Gate 5)

### 5.1 Test Suites and Typecheck
1. **Python Unit & Geometry Tests:**
   ```bash
   npm run test:python
   # py -m unittest tests.test_capture_pipeline -v
   ```
   *Result:* 17/17 tests pass in 3.93 s.
2. **JavaScript / TypeScript Test Suite:**
   ```bash
   npm test
   ```
   *Result:* 192/192 tests pass across 13 test suites in 12.9 s (includes 5 new comprehensive calibration geometry and aspect-ratio quantization tests in `tests/calibration.test.ts`).
3. **TypeScript Typecheck:**
   ```bash
   npm run typecheck
   # tsc --noEmit
   ```
   *Result:* 0 errors.

### 5.2 Working Tree Audit
* All 35 `dist/` files preserved bit-for-bit.
* `modules/live-camera/android/build/` untouched.
* `src/vision/pieceModelWeights.ts` unchanged (`1.0.0-synthetic-staunton-v1`).
* Zero external dataset files, raw images, or crops staged in Git.

---

## 6. Gate Table

| Gate | Requirement | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Gate 0** | Mandatory Preservation Constraints | **PASS** | 35 `dist/` files intact; `modules/live-camera/android/build/` untouched; production weights unchanged; no APK built or installed. |
| **Gate 1** | Reconcile Approach B Metrics | **PASS** | Proven mathematically why accepted IoU is 55.7% ($N=273$) and proposal IoU is 39.5% ($N=385$ with 0 penalty for 112 failures); documented relaxed threshold bypass. |
| **Gate 2** | Characterize Approach D Simulation | **PASS** | Formally characterized as Ground-Truth-Seeded Assisted Calibration Simulation; measured initial vs post-refinement error (3.71% $\to$ 3.10%) with median, P90, P95 distributions. |
| **Gate 3** | Corner Ordering & Geometric Validation | **PASS** | Proved 24-permutation coordinate invariance; validated rejection of collinear, coincident, and bowtie quads; added "Sort corners clockwise" UX action. |
| **Gate 4** | Coordinate & Aspect-Ratio Error Bounds | **PASS** | Proved display aspect-ratio quantization error is strictly bounded $< 0.35\%$; verified exact boundary scaling $[0.0, 1.0] \to [0, W]$. |
| **Gate 5** | Regression Verification & Publication | **PASS** | 192/192 Node tests pass; 17/17 Python tests pass; `tsc --noEmit` clean; documentation updated; committed and pushed to `origin/main`. |

---

## 7. Next Recommended Action

With geometry metrics reconciled, coordinate quantization bounds proven, 24-permutation invariance tested, and the assisted calibration UX hardened:
- **Readiness Recommendation:** The assisted manual calibration workflow is mathematically robust, verified, and ready for physical on-device field testing.
- **Next Technical Step:** Proceed to live physical capture preparation for Set 01 when physical hardware, studio lighting, underside tags, and tripod setup become available.
