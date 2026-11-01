# Milestone 5.1 & 5.1A: Dataset-to-Runtime Alignment, Color-Channel Verification, and CNN Training Audit

**Milestones:** 5.1 & 5.1A  
**Date:** 2026-11-01  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `ca439b1`  
**Production Model:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, UNCHANGED)  
**Evaluated Candidates:**
1. Synthetic HOG Baseline (`src/vision/pieceModelWeights.ts`)
2. Phase 4.3E Linear HOG Adapted Candidate (`src/vision/experimentalPieceModelWeights.ts`)
3. Milestone 5.0 Direct-Resize MicroChessCNN (`models/micro_chess_cnn_candidate.pt`)
4. Milestone 5.1A Mobile-Matched MicroChessCNN (standard ITU-R BT.601 BGR luminance)  
**External Dataset:** Zenodo Record 6656212 (Campos & Berlanga, Universidad Carlos III de Madrid, CC BY 4.0)  

---

## 1. Executive Summary & Objective

In Milestone 5.0, a clean-room compact convolutional neural network (`MicroChessCNN`, 40,038 parameters, 160 KB raw binary) trained from scratch on Zenodo Sets 1–3 achieved only **20.40% Mean Leave-One-Physical-Set-Out (LOSO) Top-1** accuracy and **27.08% Top-1** on held-out Set 4.

Milestones 5.1 and 5.1A executed a thorough, root-cause audit of the mobile runtime pipeline, OpenCV buffer format, color channel ordering, dataset preprocessing, and training/evaluation harness:

### Key Audit Findings & Resolutions
1. **Gate 1 Discrepancy Resolved (1.5625 Step vs. Center Crop):**
   * In `src/vision/classifyPiece.ts` (lines 71–76), fallback logic exists for isolated $100\times 100$ images that crops the central $64\times 64$ sub-window (`x: 18..82, y: 18..82`, with `xStep = 1.0`).
   * In the production pipeline (`src/vision/recognizeBoard.ts:156`), `options.roi` is explicitly supplied as `square.bounds` ($100\times 100$).
   * Consequently, `classifyPiece.ts` (lines 78–79) sets `xStep = yStep = 100 / 64 = 1.5625`.
   * The live runtime pipeline **resamples the full $100\times 100$ board square down to 64 samples at step 1.5625**, retaining the entire piece footprint, rather than cropping a 64×64 center.
2. **OpenCV Buffer Format & BGR Channel Correction (Milestone 5.1A):**
   * Traced `Mat.createFromBase64` to `react-native-fast-opencv/cpp/ConvertImage.cpp:158`, which invokes `cv::imdecode(..., IMREAD_UNCHANGED)`. This decodes 3-channel photos into native **BGR** format.
   * `rectifyMat` applies `OpenCV.warpPerspective` preserving BGR order, and `toBuffer('uint8')` outputs raw bytes: byte 0 is Blue ($B$), byte 1 is Green ($G$), and byte 2 is Red ($R$).
   * In `src/vision/classifySquare.ts` (prior to M5.1A), byte 0 was assigned to `r` and byte 2 to `b`, inverting Red and Blue in the luminance formula ($\text{lum} = \text{round}(0.299 B + 0.587 G + 0.114 R)$).
   * **Resolution:** Corrected `getPixelLuminanceAndRgb` to respect `colorOrder: 'bgr' | 'rgb'`: byte 2 is read as Red, byte 1 as Green, and byte 0 as Blue for standard OpenCV BGR buffers, restoring standard ITU-R BT.601 luminance ($Y = \text{round}(0.299 R + 0.587 G + 0.114 B)$).
   * Validated on pure color test fixtures: Pure Red yields $Y=76$, Pure Green yields $Y=150$, Pure Blue yields $Y=29$.
3. **Cross-Language Preprocessing Parity (Gate 2):**
   * Implemented `scripts/verify_preprocessing_parity.py` and `tests/classifyPiece.test.ts` (Test 15) using a deterministic $100\times 100$ BGR fixture.
   * Verified that TypeScript and Python HOG feature extraction and linear classifier inference match to **$5.96 \times 10^{-8}$ maximum difference** (IEEE 754 single-precision float limit).
4. **Controlled Diagnosis of Input Transforms & Fold Breakdown (Gate 3 & 4):**
   * **Mean LOSO Top-1:** Exactly identical at **20.40%** for both Direct Resize and Mobile-Matched with standard luminance ($0.00$ pp mean difference).
   * **Individual Fold Breakdown:**
     * Fold 1 (Set 1 held out): $24.87\% \to 24.87\%$ ($0.00$ pp)
     * Fold 2 (Set 2 held out): $17.06\% \to 16.41\%$ ($-0.65$ pp)
     * Fold 3 (Set 3 held out): $19.27\% \to 19.92\%$ ($+0.65$ pp)
   * **Set 4 (Historical Holdout):** Direct Resize achieves **27.08%**; Mobile-Matched achieves **27.99%** ($+0.91$ pp, $+7$ samples).
   * **Cross-Transform Robustness:** Evaluating the Direct-Resize trained model on Mobile-Matched data achieves **22.01%** ($-5.07$ pp distribution shift).
   * **Center Crop Amputation:** Resizing to $100\times 100$ and center-cropping $[18:82, 18:82]$ truncates Queen/King crowns and piece bases, causing accuracy to collapse to $19.66\%$ on Linear and $21.88\%$ on CNN.
5. **Root Cause of the 20% LOSO Ceiling:**
   * The poor cross-set generalization is **not** caused by a preprocessing bug or channel-order artifact.
   * A 40,000-parameter CNN trained from scratch on only 3 physical chess sets ($N=2,304$) suffers from a fundamental **sample diversity deficit**: it memorizes low-level wood grain and lathe patterns (scoring $64.28\%$ in-domain), but cannot learn 3D shape invariance across unobserved sets ($20.40\%$ LOSO).
   * Handcrafted HOG features with L2-norm block normalization provide an inductive bias that suppresses texture and lighting, allowing linear HOG to reach $35.33\%$ LOSO Top-1.
6. **Production Decision:**
   * CNN fails all production gates ($\text{Top-1 } 27.99\% \ll 88.0\%$).
   * Production weights in `src/vision/pieceModelWeights.ts` remain **100% UNCHANGED**.
   * Mandatory user confirmation (amber `?` badges) and the manual piece editor remain the truthful, safe production fail-safes.

---

## 2. Gate 0: Preservation Verification

* **Git HEAD:** Verified synchronized with `origin/main`.
* **Artifact Preservation:** All 33 files in `dist/` preserved untouched.
* **Build Tree Preservation:** `modules/live-camera/android/build/` preserved untouched.
* **No APK Builds:** Zero builds, packaging, or Android compilation commands executed.
* **Storage Isolation:** Raw datasets, image caches, and binary checkpoints reside strictly in `C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\` outside Git.

---

## 3. Gate 1: Actual OpenCV Buffer Format & Channel Order Verification

We audited the concrete C++ and TypeScript implementation from decode to pixel access:

```
takePictureAsync() ──► Base64 JPEG
       │
Mat.createFromBase64() [MatFactory.cpp:52 -> ConvertImage.cpp:158]
       │
cv::imdecode(data, IMREAD_UNCHANGED) ──► 3-Channel BGR cv::Mat (CV_8UC3)
       │
rectifyMat() [rectify.ts:521] ──► OpenCV.warpPerspective() ──► 800x800 BGR cv::Mat
       │
rectified.toBuffer('uint8') [MatDelegate.cpp:53] ──► Uint8Array(1,920,000) raw BGR bytes
       │
imgData: { width: 800, height: 800, channels: 3, data: buf.buffer, colorOrder: 'bgr' }
```

### 3.1 Proven Byte Layout
For any pixel $(x, y)$ in the canonical $800\times 800$ rectified buffer:
$$\text{idx} = (y \times 800 + x) \times 3$$
* $\text{data}[\text{idx}] = \text{Blue } (B)$
* $\text{data}[\text{idx} + 1] = \text{Green } (G)$
* $\text{data}[\text{idx} + 2] = \text{Red } (R)$

### 3.2 Correction in `src/vision/classifySquare.ts`
Updated `getPixelLuminanceAndRgb` to support `colorOrder: 'bgr' | 'rgb'` (defaulting to `'bgr'`):
```ts
export function getPixelLuminanceAndRgb(
  image: AnyImageData,
  x: number,
  y: number,
): { lum: number; r: number; g: number; b: number } {
  const clampedX = Math.max(0, Math.min(image.width - 1, Math.round(x)));
  const clampedY = Math.max(0, Math.min(image.height - 1, Math.round(y)));

  if (isRgbImage(image)) {
    const idx = (clampedY * image.width + clampedX) * 3;
    const isRgb = image.colorOrder === 'rgb';
    const r = isRgb ? image.data[idx] : image.data[idx + 2];
    const g = image.data[idx + 1];
    const b = isRgb ? image.data[idx + 2] : image.data[idx];
    const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    return { lum, r, g, b };
  }

  const idx = clampedY * image.width + clampedX;
  const lum = image.data[idx];
  return { lum, r: lum, g: lum, b: lum };
}
```

### 3.3 Unit Test Verification on Pure Colors (tests/classifySquare.test.ts)

| Pixel Test Case | Raw BGR Bytes in Buffer | Extracted $R, G, B$ | Expected Luminance ($Y$) | Actual Computed Luminance | Test Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure Red** | `[0, 0, 255]` | $R=255, G=0, B=0$ | $\text{round}(0.299 \times 255) = 76$ | $76$ | **PASS** |
| **Pure Green** | `[0, 255, 0]` | $R=0, G=255, B=0$ | $\text{round}(0.587 \times 255) = 150$ | $150$ | **PASS** |
| **Pure Blue** | `[255, 0, 0]` | $R=0, G=0, B=255$ | $\text{round}(0.114 \times 255) = 29$ | $29$ | **PASS** |
| **Pure White** | `[255, 255, 255]` | $R=255, G=255, B=255$ | $\text{round}(1.000 \times 255) = 255$ | $255$ | **PASS** |
| **Pure Black** | `[0, 0, 0]` | $R=0, G=0, B=0$ | $0$ | $0$ | **PASS** |
| **Mid-Gray** | `[128, 128, 128]` | $R=128, G=128, B=128$ | $128$ | $128$ | **PASS** |

*Regression Guard:* Verified that if Red and Blue are swapped, Red yields 29 and Blue yields 76. The test asserts `pRed.lum > pBlue.lum` (76 > 29), permanently blocking channel inversion regressions.

---

## 4. Gate 2: Preprocessing and Sampling Parity Verification

We created a deterministic $100\times 100$ BGR test fixture with:
* Top-left: Pure Red (`[0, 0, 255]`)
* Top-right: Pure Green (`[0, 255, 0]`)
* Bottom-left: Pure Blue (`[255, 0, 0]`)
* Bottom-right: Wood tone (`[140, 180, 210]`)
* Center: Circular piece silhouette (`[30, 30, 30]`, radius 20)

### 4.1 Cross-Language Numerical Comparison

Evaluated on the exact mobile sampling path ($100\times 100$ cell, step $1.5625$, nearest-neighbor integer coordinate rounding, 288-bin HOG extraction):

```
Spot-Check Comparison (TypeScript vs. Python):
  Index   0: TS = 0.000000, PY = 0.000000, diff = 0.00e+00
  Index   8: TS = 0.592673, PY = 0.592673, diff = 5.96e-08
  Index  32: TS = 0.288792, PY = 0.288792, diff = 2.98e-08
  Index  64: TS = 0.320126, PY = 0.320127, diff = 2.98e-08
  Index 128: TS = 0.120523, PY = 0.120523, diff = 0.00e+00
  Index 200: TS = 0.130177, PY = 0.130177, diff = 0.00e+00
  Index 256: TS = 0.365449, PY = 0.365449, diff = 2.98e-08
```

* **Maximum Absolute Feature Difference:** **$5.96 \times 10^{-8}$** (well within the $1.0 \times 10^{-4}$ tolerance).
* **Mean Absolute Feature Difference:** **$2.72 \times 10^{-9}$**.
* **End-to-End Inference Parity (Production Model):**
  * TypeScript: `knight`, confidence = $0.977434$, margin = $0.955119$
  * Python:     `knight`, confidence = $0.977434$, margin = $0.955119$
* **End-to-End Inference Parity (Experimental Model):**
  * TypeScript: `knight`, confidence = $0.726998$, margin = $0.516422$
  * Python:     `knight`, confidence = $0.726998$, margin = $0.516421$

Parity is established at IEEE 754 float precision across both the feature extractor and inference engine.

---

## 5. Gate 3 & 4: Controlled Diagnosis and Experiment Consistency

We executed `scripts/diagnose_preprocessing.py` under strictly controlled conditions:
* Architecture: `MicroChessCNN` (40,038 float32 parameters, 160 KB).
* Optimizer: AdamW ($lr=10^{-3}$, weight_decay=$10^{-4}$, Cosine Annealing, 8 epochs).
* Seeds: $42 + \text{test\_set}$ for LOSO folds; $42$ for candidate fit.
* Historical Set 4: Evaluated as a historical holdout from Milestone 4.3D/5.0.

### 5.1 Controlled Diagnosis Summary

| Metric | Direct Resize (Milestone 5.0 Baseline) | Mobile-Matched (Milestone 5.1A Standard Lum) | Preprocessing Effect ($\Delta$) | M5.0 Model on Mobile-Matched Set 4 (Robustness Shift) |
| :--- | :---: | :---: | :---: | :---: |
| **Training Input** | Direct Resize ($181\to 64$) | Mobile-Matched ($100\to 64$) | Matched Pipeline | Direct Resize ($181\to 64$) |
| **Evaluation Input** | Direct Resize ($181\to 64$) | Mobile-Matched ($100\to 64$) | Matched Pipeline | Mobile-Matched ($100\to 64$) |
| **Mean LOSO Top-1** | **20.40%** | **20.40%** | **+0.00 pp** | N/A |
| **Mean LOSO Top-2** | **37.11%** | **38.32%** | **+1.21 pp** | N/A |
| Fold 1 (Set 1 held out) | 24.87% (191/768) | 24.87% (191/768) | +0.00 pp | N/A |
| Fold 2 (Set 2 held out) | 17.06% (131/768) | 16.41% (126/768) | -0.65 pp | N/A |
| Fold 3 (Set 3 held out) | 19.27% (148/768) | 19.92% (153/768) | +0.65 pp | N/A |
| **In-Domain Sets 1–3 Top-1** | **63.37%** (1460/2304) | **64.28%** (1481/2304) | +0.91 pp | N/A |
| **Set 4 Top-1 (Historical)** | **27.08%** (208/768) | **27.99%** (215/768) | **+0.91 pp** | **22.01%** (169/768, -5.07 pp shift) |
| **Set 4 Top-2 (Historical)** | **51.17%** (393/768) | **50.78%** (390/768) | -0.39 pp | **47.53%** (365/768, -3.64 pp shift) |
| Pawn Recall | 30.47% (39/128) | 35.16% (45/128) | +4.69 pp | 26.56% (34/128) |
| Knight Recall | 45.31% (58/128) | 51.56% (66/128) | +6.25 pp | 32.81% (42/128) |
| Bishop Recall | 0.78% (1/128) | 0.78% (1/128) | +0.00 pp | 0.78% (1/128) |
| Rook Recall | 69.53% (89/128) | 68.75% (88/128) | -0.78 pp | 69.53% (89/128) |
| Queen Recall | 14.84% (19/128) | 11.72% (15/128) | -3.12 pp | 0.78% (1/128) |
| King Recall | 1.56% (2/128) | 0.00% (0/128) | -1.56 pp | 1.56% (2/128) |
| Abstention Rate ($\ge 80\%$) | 95.18% (731/768) | 94.14% (723/768) | -1.04 pp | 96.09% (738/768) |
| Coverage ($\le 20\%$) | 4.82% (37/768) | 5.86% (45/768) | +1.04 pp | 3.91% (30/768) |
| Accuracy on Accepted | 67.57% (25/37) | 64.44% (29/45) | -3.13 pp | 66.67% (20/30) |

### 5.2 Confusion Matrix (Mobile-Matched MicroChessCNN on Set 4)
```
              Pred: pawn   knig   bish   rook   quee   king    Total
True: pawn            45     52      0     25      6      0      128
True: knight          21     66      2     28      9      2      128
True: bishop          11     48      1     47     19      2      128
True: rook             7     20      0     88     11      2      128
True: queen            4     31      0     74     15      4      128
True: king             4     22      2     77     23      0      128
--------------------------------------------------------------------
Total Pred:           92    239      5    339     83     10      768
```

---

## 6. Gate 5: Production Acceptance Gates & Safeguards

```
========================================================================================================
MILESTONE 5.1A PRODUCTION ACCEPTANCE GATES
========================================================================================================
Requirement                         Gate Standard      Mobile-Matched MicroChessCNN   Gate Decision
--------------------------------------------------------------------------------------------------------
Top-1 Accuracy (Physical Set 4):    >= 88.0%           27.99%                         FAIL
Top-2 Accuracy (Physical Set 4):    >= 96.0%           50.78%                         FAIL
Queen Recall (Physical Set 4):      >= 90.0%           11.72%                         FAIL
King Recall (Physical Set 4):       >= 90.0%           0.00%                          FAIL
Abstention Rate (>= 80%, 90% tgt):  >= 80.0%           94.14%                         PASS (Target met)
Zero-Silent-Error Policy:           100.0%             100.0%                         PASS
Production Weight Isolation:        pieceModelWeights  UNCHANGED                      PASS
Safety Guarantees Maintained:       Manual Review      Mandatory confirmation active  PASS
========================================================================================================
OVERALL DECISION: DO NOT PROMOTE ANY CNN CANDIDATE TO PRODUCTION.
PRESERVE PRODUCTION WEIGHTS UNCHANGED AT src/vision/pieceModelWeights.ts.
PRESERVE MANDATORY MANUAL CONFIRMATION IN BOARD REVIEW.
========================================================================================================
```

---

## 7. Changed Files and Test Verification

* **TypeScript Typecheck (`npx tsc --noEmit`):** 0 errors.
* **Test Suite (`pnpm test`):** 155 / 155 tests passing across all 11 suites.
* **Tracked Changes:**
  - `src/vision/types.ts`: Added `colorOrder?: 'bgr' | 'rgb'` to `RgbImage` interface.
  - `src/vision/classifySquare.ts`: Corrected `getPixelLuminanceAndRgb` to read byte 2 as Red and byte 0 as Blue for standard BGR buffers, restoring standard ITU-R BT.601 luminance.
  - `src/app/board-review.tsx`: Explicitly tagged rectified buffer as `colorOrder: 'bgr'`.
  - `tests/classifySquare.test.ts`: Added BGR synthetic fixtures and pure-color unit tests (Red 76, Green 150, Blue 29).
  - `tests/classifyPiece.test.ts`: Added Test 15 verifying 100x100 BGR mobile sampling parity with Python reference.
  - `scripts/verify_preprocessing_parity.py`: Dedicated Python parity verification script.
  - `scripts/diagnose_preprocessing.py`: Updated with fold differences and cross-transform robustness evaluation.
  - `docs/DATASET_TO_RUNTIME_ALIGNMENT_AUDIT.md`: Complete audit and benchmark documentation.
* **Zero APK Rebuilds:** The previously verified release APK (`dist/ai-chess-navigator-1.0.0-arm64-milestone4-opencv-fix-20261101.apk`) remains the authoritative release artifact.
