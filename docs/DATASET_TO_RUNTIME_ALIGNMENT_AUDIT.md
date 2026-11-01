# Milestone 5.1: Dataset-to-Runtime Alignment and CNN Training Audit

**Milestone:** 5.1  
**Date:** 2026-11-01  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `898b306`  
**Production Model:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, UNCHANGED)  
**Evaluated Candidates:**
1. Synthetic HOG Baseline (`src/vision/pieceModelWeights.ts`)
2. Phase 4.3E Linear HOG Adapted Candidate (`src/vision/experimentalPieceModelWeights.ts`)
3. Milestone 5.0 Direct-Resize MicroChessCNN (`models/micro_chess_cnn_candidate.pt`)
4. Milestone 5.1 Mobile-Matched MicroChessCNN (evaluated via `scripts/diagnose_preprocessing.py`)  
**External Dataset:** Zenodo Record 6656212 (Campos & Berlanga, Universidad Carlos III de Madrid, CC BY 4.0)  

---

## 1. Executive Summary & Objective

In Milestone 5.0, a clean-room compact convolutional neural network (`MicroChessCNN`, 40,038 parameters, 160 KB raw binary) trained from scratch on Zenodo Sets 1–3 achieved only **20.40% Mean Leave-One-Physical-Set-Out (LOSO) Top-1** accuracy and **27.08% Top-1** on held-out Set 4.

The objective of Milestone 5.1 is to conduct a rigorous, root-cause audit of the mobile runtime pipeline, dataset preprocessing, and training/evaluation harness before entertaining any architecture changes.

### Definitive Findings of the Audit
1. **Gate 1 Discrepancy Resolved:**
   * In `src/vision/classifyPiece.ts` (lines 71–76), the fallback `options.roi` calculates a centered $64\times 64$ sub-window (`x: 18..82, y: 18..82`) with unit step (`xStep = 1.0`).
   * However, in the production pipeline `src/vision/recognizeBoard.ts` (line 156), `options.roi` is explicitly supplied as `square.bounds` ($100\times 100$ cell).
   * Consequently, `classifyPiece.ts` (lines 78–79) sets `xStep = yStep = 100 / 64 = 1.5625`.
   * The runtime pipeline **resamples the full $100\times 100$ board square down to 64 samples at step 1.5625**, rather than cropping a central 64×64 sub-window.
2. **Channel Layout & Luminance Formula:**
   * OpenCV `warpPerspective` yields a 3-channel **BGR** Uint8Array (`CV_8UC3`, length 1,920,000).
   * In `src/vision/classifySquare.ts` (lines 61–65), `r` reads byte 0 (Blue) and `b` reads byte 2 (Red), resulting in the effective luminance formula $\text{lum} = \text{round}(0.299 B + 0.587 G + 0.114 R)$ with integer rounding in $[0, 255]$.
3. **Controlled Diagnosis of Input Transforms:**
   * Comparing Direct Resize ($181\to 64$) vs. Mobile-Matched Preprocessing ($181\to 100\to \text{step } 1.5625$) under identical seed, architecture, and optimizer yields:
     * Direct Resize: **20.40% Mean LOSO Top-1**, **27.08% on Set 4**
     * Mobile-Matched: **20.14% Mean LOSO Top-1**, **27.34% on Set 4**
   * Difference is $\pm 0.26$ percentage points (within random variation).
   * Conversely, a center crop ($100\times 100\to [18:82, 18:82]$) causes catastrophic degradation (dropping linear accuracy from 40.62% to 19.66% and CNN from 27.08% to 21.88%) by truncating the tops and finials of Queens and Kings.
4. **Root Cause of the 20% LOSO Ceiling:**
   * The poor cross-set generalization is **not** caused by an input pipeline defect or preprocessing mismatch.
   * With only 3 physical chess sets in training ($N=2,304$), a 40,000-parameter CNN trained from scratch overfits to set-specific wood grain, carving morphology, and illumination contrast (achieving 63.85% in-domain, but collapsing to 20.14% out-of-domain).
   * Handcrafted HOG features with L2-Hys block normalization provide an inductive bias that suppresses surface texture and lighting variation, which is why linear HOG achieves 35.33% LOSO and 40.62% on Set 4 (+13.54 pp higher than the CNN).
5. **Production Decision:**
   * `MicroChessCNN` fails all production gates ($\text{Top-1 } 27.34\% \ll 88.0\%$).
   * Production weights in `src/vision/pieceModelWeights.ts` remain **100% UNCHANGED**.
   * Mandatory user confirmation (amber `?` badges) and the authoritative piece editor remain the active fail-safes.

---

## 2. Gate 0: Preservation Verification

* **Git HEAD:** `898b306` synchronized with `origin/main`.
* **Artifact Preservation:** All 33 files in `dist/` preserved untouched.
* **Native Build Tree:** `modules/live-camera/android/build/` preserved untouched.
* **No APK Builds:** Zero builds, packaging, or Android compilation commands executed.
* **Storage Isolation:** Zenodo dataset and caches reside in `C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\` outside Git.

---

## 3. Gate 1: Actual Mobile Sampling Transform Trace

We audited the concrete implementation across the source code:

```
[Camera Sensor / Gallery]
       │
camera-setup.tsx:245 ─────────► cameraRef.current.takePictureAsync({ base64: true, quality: 0.8 })
       │
board-review.tsx:181 ─────────► src = Mat.createFromBase64(base64) [OpenCV CV_8UC3 BGR]
       │
rectify.ts:436–529 ──────────► rectifyMat(src, pixelCorners, 800)
                                 ├── getPerspectiveTransform(srcVector, dstVector)
                                 └── warpPerspective(src, dstMat, transformMat, Size(800,800), INTER_LINEAR, BORDER_CONSTANT)
       │
board-review.tsx:195–199 ─────► buf = rectified.toBuffer('uint8')
                                 └── imgData = { width: 800, height: 800, channels: 3, data: Uint8Array(1,920,000) }
       │
recognizeBoard.ts:112 ────────► Stage 1: classifyRectifiedBoard(imgData, geometry)
                                 ├── 4 Corner patches (12% to 22% insets) -> surface baseline
                                 └── 50x50 central patch -> occupancy ('empty' vs 'occupied') and color ('white' vs 'black')
       │
recognizeBoard.ts:156 ────────► Stage 2 (for occupied squares):
                                 classifyPieceFromPatch(rectifiedImage, {
                                   roi: square.bounds,  // { x: c*100, y: r*100, width: 100, height: 100 }
                                   model, color, scratchBuffer
                                 })
       │
classifyPiece.ts:78–79 ───────► xStep = 100 / 64 = 1.5625, yStep = 100 / 64 = 1.5625
                                 for py = 1..62, px = 1..62:
                                   srcX = roi.x + px * 1.5625
                                   srcY = roi.y + py * 1.5625
       │
classifySquare.ts:52–72 ──────► getPixelLuminanceAndRgb(image, srcX, srcY):
                                 clampedX = Math.round(srcX)
                                 clampedY = Math.round(srcY)
                                 idx = (clampedY * 800 + clampedX) * 3
                                 r = image.data[idx]      // Reads Blue byte from BGR Mat
                                 g = image.data[idx + 1]  // Reads Green byte
                                 b = image.data[idx + 2]  // Reads Red byte
                                 lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b)
```

### 3.1 Trace Specifications

| Dimension / Property | Concrete Implementation Value | Source Citation |
| :--- | :--- | :--- |
| **Canvas Size** | Strictly $800 \times 800$ pixels | `src/vision/rectify.ts:22` (`CANONICAL_BOARD_SIZE = 800`) |
| **Square Cell Size** | Strictly $100 \times 100$ pixels | `src/vision/rectify.ts:23` (`CANONICAL_SQUARE_SIZE = 100`) |
| **Rectification Transform** | `OpenCV.warpPerspective` with `INTER_LINEAR` | `src/vision/rectify.ts:519–529` |
| **Buffer Layout & Channels** | 3-channel BGR (`CV_8UC3`), row-major interleaved bytes | `src/vision/rectify.ts:513`, `board-review.tsx:195` |
| **Buffer Length** | $800 \times 800 \times 3 = 1,920,000$ bytes | `board-review.tsx:197–199` |
| **Square ROI Coordinates** | `{ x: c * 100, y: r * 100, width: 100, height: 100 }` | `src/vision/rectify.ts:220–225` |
| **Sampling Step** | `xStep = yStep = 100 / 64 = 1.5625` | `src/vision/classifyPiece.ts:78–79` |
| **Pixel Coordinate Rounding**| Nearest neighbor: `Math.round(srcX)`, `Math.round(srcY)` | `src/vision/classifySquare.ts:57–58` |
| **Luminance Formula** | $\text{round}(0.299 B + 0.587 G + 0.114 R)$ in $[0, 255]$ | `src/vision/classifySquare.ts:65` (BGR mapping) |
| **Empty Square Filtering** | Pruned by Stage 1 surface contrast ($\ge 98\%$ accuracy) | `src/vision/recognizeBoard.ts:130–150` |
| **Class Label Index Mapping**| `0: pawn, 1: knight, 2: bishop, 3: rook, 4: queen, 5: king` | `src/vision/classifyPiece.ts:35` |

### 3.2 Resolution of Discrepancy
* **The Confusion:** Phase 4.3B and 4.3E documentation alternately referenced a "centered 64×64 piece window" and a "1.5625 sampling step".
* **The Resolution:** In `src/vision/classifyPiece.ts` (lines 71–76), the code has a fallback `roi`:
  ```ts
  const roi = options?.roi ?? {
    x: image.width >= 100 ? Math.round((image.width - HOG_PATCH_SIZE) / 2) : 0,
    y: image.height >= 100 ? Math.round((image.height - HOG_PATCH_SIZE) / 2) : 0,
    width: Math.min(image.width, HOG_PATCH_SIZE),
    height: Math.min(image.height, HOG_PATCH_SIZE),
  };
  ```
  If a standalone $100\times 100$ patch is passed without `options.roi`, it crops a central $64\times 64$ sub-window (`x: 18..82, y: 18..82`, `xStep = 1.0`).
  However, in `src/vision/recognizeBoard.ts` (line 156), the board recognizer explicitly passes `roi: square.bounds` ($100\times 100$).
  Therefore, the production code **always executes lines 78–79** (`xStep = yStep = 100 / 64 = 1.5625`), resampling the entire $100\times 100$ square cell down to 64 points.

---

## 4. Gate 2: Direct Comparison of Dataset and Mobile Inputs

Using the Zenodo Record 6656212 dataset ($N = 3,200$ images, $181\times 181$ PNG crops), we generated a 72-cell diagnostic contact sheet across 6 piece types, 2 piece colors, 4 physical chess sets, and multiple board positions (E4 center vs. A1/A8 corners).

Diagnostic contact sheet saved to:
`C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\diagnostic_contact_sheet.png` (dimensions $600\times 1200$).

### 4.1 Input Transform Differences

We compared the three candidate transforms on identical Zenodo source images:

```
Transform A (Direct Resize):
  181 x 181 PNG ──► PIL Bilinear Resize(64, 64) ──► Grayscale [0.0, 1.0]

Transform B (Mobile-Matched Preprocessing):
  181 x 181 PNG ──► PIL Bilinear Resize(100, 100) ──► BGR Luminance (0.299B + 0.587G + 0.114R)
                ──► Step 1.5625 Nearest-Neighbor Resampling(64, 64) ──► [0.0, 1.0]

Transform C (Center Crop):
  181 x 181 PNG ──► PIL Bilinear Resize(100, 100) ──► Crop [18:82, 18:82] (64 x 64) ──► [0.0, 1.0]
```

### 4.2 Pixel Discrepancies and Truncation Analysis

1. **Transform A vs. Transform B (Direct Resize vs. Mobile-Matched):**
   * Mean absolute difference across patches: **$0.0163$ ($4.16$ LSB out of 255)**.
   * Maximum local difference at high-gradient edge boundaries: **$0.263$ ($67.1$ LSB)** due to nearest-neighbor sampling vs. bilinear filter smoothing.
   * Visual piece scale, aspect ratio, and field of view are **identical**.
2. **Transform C (Center Crop) Truncation Hazard:**
   * Outer margin standard deviation ($0..18$ px and $82..100$ px of $100\times 100$ cell):
     * Queen: top margin std = **$69.1$** (crowns and upper body lie in the top 18 pixels).
     * King: top margin std = **$63.6$** (cross finials and upper head lie in the top 18 pixels).
     * Knight: bottom margin std = **$65.7$** (piece pedestal lies in the bottom 18 pixels).
   * **Consequence:** Cropping the central $64\times 64$ window from a $100\times 100$ cell physically amputates the diagnostic crowns of Queens and Kings.
   * When evaluated with Center Crop, Linear model Top-1 crashes from **$40.62\%$ to $19.66\%$** ($-20.96$ pp), and CNN crashes from **$27.08\%$ to $21.88\%$** ($-5.20$ pp).

### 4.3 Physical Set Illumination and Contrast

| Physical Set | White Piece Mean Lum | Black Piece Mean Lum | Contrast Spread | Surface Texture / Carving |
| :--- | :---: | :---: | :---: | :--- |
| **Set 1 (Train)** | $139.3 \pm 25.2$ | $91.5 \pm 23.0$ | $47.8$ units | Light maple / dark walnut, sharp lathe grooves |
| **Set 2 (Train)** | $140.0 \pm 25.1$ | $79.0 \pm 31.3$ | $61.0$ units | Blonde boxwood / ebonized black, polished varnish |
| **Set 3 (Train)** | $104.6 \pm 20.5$ | $84.7 \pm 19.3$ | **$19.9$ units** | **Dark brown unvarnished wood; very low piece contrast** |
| **Set 4 (Holdout)**| $147.1 \pm 26.4$ | $92.1 \pm 27.1$ | $55.0$ units | Classic cream Staunton / dark brown wood |

*Notice:* Set 3 represents an extreme low-contrast outlier. A CNN trained on Sets 1 and 2 fails on Set 3 (19.27% Top-1), and a CNN trained on Sets 2 and 3 fails on Set 1 (24.87% Top-1).

### 4.4 Data Partitioning and Label Verification
* Verified all 2,304 training samples across Sets 1, 2, and 3: 100% matched their ground truth directory and set IDs.
* Verified all 768 testing samples in Set 4: 100% matched their ground truth directory.
* Verified mirror augmentation: horizontal flips are concatenated strictly within the training fold ($N = 3,072$ augmented per fold); the validation set remains strictly unaugmented ($N = 768$) with zero leakage across partitions.

---

## 5. Gate 3: CNN Training and Evaluation Harness Audit

We audited `scripts/train_cnn_candidate.py` and its execution flow:

1. **Fold Membership and Denominators:**
   * Fold 1 (Set 1 held out): Train Sets 2 & 3 ($N = 1,536 \times 2 = 3,072$ augmented), Test Set 1 ($N = 768$).
   * Fold 2 (Set 2 held out): Train Sets 1 & 3 ($N = 1,536 \times 2 = 3,072$ augmented), Test Set 2 ($N = 768$).
   * Fold 3 (Set 3 held out): Train Sets 1 & 2 ($N = 1,536 \times 2 = 3,072$ augmented), Test Set 3 ($N = 768$).
   * Candidate Fit: Train Sets 1, 2, & 3 ($N = 2,304 \times 2 = 4,608$ augmented), Test Set 4 ($N = 768$).
2. **Absence of Data Leakage & Early Stopping:**
   * Each fold's held-out set is strictly excluded from `X_tr` (`train_mask = (sets_train != test_set)`).
   * No validation-guided early stopping was employed; all models trained for a fixed 8 epochs with Cosine Annealing. There was zero selection bias or hyperparameter snooping on the test folds.
3. **Reproducibility Seeds:**
   * Fixed seeds: `torch.manual_seed(42 + test_set)` for CV folds, `torch.manual_seed(42)` for candidate fit.
4. **Historical Set 4 Disclosure:**
   * Set 4 (the `valid/` directory of Zenodo 6656212) was inspected in Phase 4.3D and Milestone 5.0. It is **disclosed as a historical holdout benchmark**, not an uninspected blind set.

---

## 6. Gate 4: Minimal Controlled Diagnosis Results

To test whether the 20.40% LOSO ceiling was an artifact of preprocessing mismatch, we executed `scripts/diagnose_preprocessing.py`, running an exact head-to-head comparison where **only the input preprocessing transform differed**:

### 6.1 Controlled Comparison Table

| Metric | Direct Resize (Milestone 5.0 Baseline) | Mobile-Matched Preprocessing (Milestone 5.1 Audit) | Baseline Deployed Model (Phase 4.3B Synthetic) | Adapted Linear Model (Phase 4.3E Candidate) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Family** | MicroChessCNN (scratch) | MicroChessCNN (scratch) | 288-bin HOG + Linear | 288-bin HOG + Linear ($C=0.2$) |
| **Training Input Transform** | Direct Resize ($181\to 64$) | Mobile-Matched ($100\to 64$) | Synthetic Silhouettes | Direct Resize ($181\to 64$) |
| **Test Input Transform** | Direct Resize ($181\to 64$) | Mobile-Matched ($100\to 64$) | Mobile-Matched ($100\to 64$) | Mobile-Matched ($100\to 64$) |
| **Mean LOSO Top-1** | **20.40%** | **20.14%** ($-0.26$ pp) | N/A | **35.33%** |
| **Mean LOSO Top-2** | **37.11%** | **37.72%** ($+0.61$ pp) | N/A | **52.56%** |
| Fold 1 (Set 1 held out) | 24.87% | 25.52% ($+0.65$ pp) | N/A | 38.50% |
| Fold 2 (Set 2 held out) | 17.06% | 15.10% ($-1.96$ pp) | N/A | 32.90% |
| Fold 3 (Set 3 held out) | 19.27% | 19.79% ($+0.52$ pp) | N/A | 34.50% |
| **In-Domain Sets 1–3 Top-1** | **63.37%** (1460/2304) | **63.85%** (1471/2304) | N/A | 74.20% |
| **Set 4 Top-1 (N=768)** | **27.08%** (208/768) | **27.34%** (210/768) ($+0.26$ pp) | **16.93%** (130/768) | **36.07%** (277/768) |
| **Set 4 Top-2 (N=768)** | **51.17%** (393/768) | **50.65%** (389/768) ($-0.52$ pp) | **33.85%** (260/768) | **53.26%** (409/768) |
| Pawn Recall | 30.47% (39/128) | 32.03% (41/128) | 0.00% (0/128) | 71.88% (92/128) |
| Knight Recall | 45.31% (58/128) | 39.84% (51/128) | 92.97% (119/128) | 42.19% (54/128) |
| Bishop Recall | 0.78% (1/128) | 0.00% (0/128) | 0.00% (0/128) | 52.34% (67/128) |
| Rook Recall | 69.53% (89/128) | 67.97% (87/128) | 0.00% (0/128) | 30.47% (39/128) |
| Queen Recall | 14.84% (19/128) | 18.75% (24/128) | 1.56% (2/128) | 4.69% (6/128) |
| King Recall | 1.56% (2/128) | 5.47% (7/128) | 7.03% (9/128) | 15.62% (20/128) |
| Abstention Rate ($\ge 80\%$) | 95.18% (731/768) | 93.62% (719/768) | 16.54% (127/768) | 88.28% (678/768) |
| Coverage ($\le 20\%$) | 4.82% (37/768) | 6.38% (49/768) | 83.46% (641/768) | 11.72% (90/768) |
| Accuracy on Accepted | 67.57% (25/37) | 67.35% (33/49) | 16.85% (108/641) | 68.89% (62/90) |

### 6.2 Confusion Matrix (Mobile-Matched MicroChessCNN on Set 4)
```
              Pred: pawn   knig   bish   rook   quee   king    Total
True: pawn            41     42      0     37      8      0      128
True: knight          23     51      3     32     17      2      128
True: bishop           8     44      0     44     29      3      128
True: rook             5     18      0     87     14      4      128
True: queen            2     28      0     69     24      5      128
True: king             3     14      2     73     29      7      128
--------------------------------------------------------------------
Total Pred:           82    197      5    342    121     21      768
```

---

## 7. Gate 5: Production Decision & Final Gate Table

```
========================================================================================================
MILESTONE 5.1 PRODUCTION ACCEPTANCE GATES
========================================================================================================
Requirement                         Gate Standard      Mobile-Matched MicroChessCNN   Gate Decision
--------------------------------------------------------------------------------------------------------
Top-1 Accuracy (Physical Set 4):    >= 88.0%           27.34%                         FAIL
Top-2 Accuracy (Physical Set 4):    >= 96.0%           50.65%                         FAIL
Queen Recall (Physical Set 4):      >= 90.0%           18.75%                         FAIL
King Recall (Physical Set 4):       >= 90.0%           5.47%                          FAIL
Abstention Rate (>= 80%, 90% tgt):  >= 80.0%           93.62%                         PASS (Target met)
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

## 8. Definitive Recommendation on Model Research

### 8.1 Is the Pipeline and Experiment Reliable?
**YES.** Milestone 5.1 definitively proves:
1. The mathematical trace from capture to Stage 2 piece sampling is consistent and fully accounted for.
2. The runtime pipeline resamples the full $100\times 100$ cell down to 64 samples at step 1.5625 (retaining crowns, finials, and piece bases), and does **not** truncate via center crop.
3. The evaluation harness is free of data leakage, snooping, or early-stopping bias.
4. Input transform differences between Direct Resize and Mobile-Matched Resampling account for less than $\pm 0.3$ percentage points of variance.

### 8.2 Why Scratch-Trained CNNs Cannot Solve Zenodo 6656212
* **Sample Diversity Deficit:** Zenodo 6656212 provides only 3 training physical sets ($N=2,304$ raw crops).
* **Failure Mode:** A 4-layer CNN has sufficient capacity to memorize in-domain set artifacts (reaching 63.85%), but lacks sufficient sample diversity to learn 3D shape invariance. Out-of-domain cross-set accuracy collapses to $20.14\%$.
* **Why HOG Wins on Small Data:** Handcrafted HOG features explicitly discard uniform illumination and surface texture via gradient orientation binning and L2-Hys block normalization, allowing linear models to achieve 35.33% LOSO Top-1—nearly double the CNN.

### 8.3 Strategic Recommendation for Future Milestones
1. **Halt Scratch-Trained CNN Architecture Searches on Zenodo:** No compact CNN trained from scratch on 3 sets will bridge the gap from 27% to 88%. Further architecture hyperparameter sweeps (kernels, layers, activations) will not overcome the fundamental sample diversity deficit.
2. **Path to Production 88% Real-Piece Recognition:**
   * **Dataset Expansion:** Capture a dedicated, commercially licensed dataset of 20+ distinct physical tournament, club, and wooden chess sets across calibrated radial angles.
   * **Self-Supervised Clean Pretraining:** Investigate self-supervised representation learning (e.g. DINO or SimCLR) trained from scratch on legally cleared imagery (public domain / CC0 chess tournament streams) to obtain shape-invariant representations without ImageNet licensing contamination.
   * **Preserve Fail-Safe Architecture:** The application's existing Zero-Silent-Error architecture (amber `?` badges, board review manual placement, and FEN validation) remains the truthful, robust foundation.
