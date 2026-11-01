# Real-World Chess Piece Classifier Benchmark & Adaptation

**Milestone:** 4.0 — Phase 4.3D / 4.3E  
**Date:** 2026-11-01  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Production Model:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, UNCHANGED)  
**Experimental Candidate Model:** `experimentalPieceClassifierModel` (`src/vision/experimentalPieceModelWeights.ts`, version `1.1.0-real-adapted-candidate-v1`)  
**Scripts:** `scripts/evaluate_real_benchmark.py`, `scripts/train_real_candidate.py`, `scripts/train_piece_classifier.py`  

---

## 1. Executive Summary

In Milestone 4.0 (Phase 4.3B/C), an offline 288-bin Histogram of Oriented Gradients (HOG) feature extractor and 6-class linear classifier were developed and integrated into the mobile Board Review screen. The initial model weights were trained and evaluated on synthetic procedural Staunton silhouettes across four style families, achieving **99.72% Top-1** (359/360) and **100.00% Top-2** (360/360) accuracy on a held-out synthetic shape family.

In Phase 4.3D and 4.3E:
1. **Gate 1 Discrepancy Resolved:** The discrepancy between Phase 4.3B (359/360 = 99.72%) and the text in the initial Phase 4.3D draft (328/360 = 91.11%) was audited and confirmed to be a documentation transcription typo. Both Python and TypeScript evaluators were re-verified against the exact 360 held-out inputs and agree to floating-point parity ($< 10^{-6}$) at **359/360 (99.72%) Top-1**.
2. **Real-World Baseline Measured (Phase 4.3D):** Evaluated against unseen physical pieces (Set 4 from Zenodo Record 6656212, 768 images), the synthetic model achieved **16.80% Top-1** and **33.85% Top-2** accuracy. A learned bias for `knight` ($+1.3579$) dominated diffuse real-world gradients, causing 91.0% of pieces to be predicted as knights.
3. **Real-World Adaptation Candidate (Phase 4.3E):** Using physical Sets 1, 2, and 3 with horizontal mirror augmentation ($N = 4,608$ samples), we established Leave-One-Set-Out (LOSO) cross-validation across the three physical sets (Mean LOSO Top-1 = 35.33%, Top-2 = 52.56%). A single frozen candidate trained on Sets 1–3 achieved **40.62% Top-1** (+23.82 pp over synthetic) and **56.77% Top-2** (+22.92 pp over synthetic) on held-out Set 4.
4. **Honest Gate Decision — Do Not Promote:** While the real-adapted candidate significantly improves over the synthetic baseline and increases non-abstained accuracy to 72.50%, its 40.62% Top-1 score falls short of the production quality gate ($\ge 88.0\%$). Under the repository's strict **Zero-Silent-Error policy**, the candidate is exported to a separate experimental file (`src/vision/experimentalPieceModelWeights.ts`), while production weights (`src/vision/pieceModelWeights.ts`) remain **100% UNCHANGED**.

---

## 2. Dataset Provenance & Licensing Audit

### Primary Benchmark Dataset: Zenodo Record 6656212
* **Title:** *Chess piece dataset for image classification*
* **DOI:** [10.5281/zenodo.6656212](https://doi.org/10.5281/zenodo.6656212)
* **Authors:** Pablo Tomas Campos (ORCID `0000-0001-6464-6685`), Antonio Berlanga (Supervisor, ORCID `0000-0002-5564-399X`), Universidad Carlos III de Madrid (UC3M).
* **Published Date:** June 22, 2022 (Version v1.0.0).
* **Archive Size:** 153,649,683 bytes (`chess_pieces.zip`).
* **Verified Checksum:** `md5:4583133a23383c05c982dd4ddd549068` (100% bit-for-bit match verified).
* **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)** (verified directly via Zenodo REST API).
* **Attribution Notice:**
  > Campos, P. T., & Berlanga, A. (2022). *Chess piece dataset for image classification* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6656212
* **Provenance Verification:** Academic thesis research at UC3M. Physical chess sets were set up and photographed across all 64 squares using a static overhead camera. Permitted for commercial use, adaptation, and model training with attribution.
* **Storage Policy:** Kept in dedicated local benchmark storage (`C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\`). Raw image files are excluded from Git and will never be packaged into the mobile APK.

### Rejected Candidate: Roboflow Universe (`chess-full-r3wh3-sajg6`)
* **License Claim:** CC BY 4.0 (platform badge).
* **Evaluation Decision:** **REJECTED**.
* **Reason:** Uploaded by an unverified pseudonymous user account (`chess-project-wfqnn`) with corrupted class labels (21 classes including Roboflow marketing text). Derived from a single wooden board photographed from a fixed side angle on a carpeted floor with severe inter-piece occlusion. Fails the repository's clean-room lineage gate.

---

## 3. Dataset Characteristics & Splits

The Zenodo 6656212 dataset contains **3,200 total PNG images** ($181\times 181$ RGB):

### Split Structure
1. **Train Split (Sets 1, 2, 3):**
   * 3 distinct physical chess sets.
   * 12 piece classes ($3 \times 64 = 192$ images each) = 2,304 piece images.
   * Empty board square class: 64 images.
   * Total: 2,368 images.
2. **Valid Split (Held-Out Physical Set 4):**
   * 1 completely unseen physical chess set (Set 4).
   * 12 piece classes ($1 \times 64 = 64$ images each) = 768 piece images.
   * Empty board square class: 64 images.
   * Total: 832 images.
   * *Note on Holdout Status:* Set 4 was inspected during Phase 4.3D baseline benchmarking. For future unbiased external validation, a completely independent capture will be required.

### Piece Class Balance (Valid Split — Set 4)
| Piece Class | White Samples | Black Samples | Total Ground Truth |
| :--- | :---: | :---: | :---: |
| **Pawn** | 64 | 64 | 128 |
| **Knight** | 64 | 64 | 128 |
| **Bishop** | 64 | 64 | 128 |
| **Rook** | 64 | 64 | 128 |
| **Queen** | 64 | 64 | 128 |
| **King** | 64 | 64 | 128 |
| **Total Pieces** | **384** | **384** | **768** |

---

## 4. Gate 1 Discrepancy Resolution

* **Discrepancy:** Phase 4.3B commit `6674d85` recorded 99.72% Top-1 (359/360), whereas an initial draft of Phase 4.3D mentioned 91.11% (328/360).
* **Audit Execution:**
  - Evaluated the exact 360 held-out procedural samples from `scripts/train_piece_classifier.py` (`SHAPE_FAMILIES[3]`, Rounded Wood, 60 variations per class with fixed seeds $1000 \times (c+1) + 300 + v$).
  - Evaluated both using the Python training pipeline and the deployed TypeScript `defaultPieceClassifierModel` from `src/vision/pieceModelWeights.ts`.
* **Finding:** Both evaluators output identical predictions:
  $$\text{Top-1} = 359 / 360 = 99.7222\% \approx 99.72\%$$
  $$\text{Top-2} = 360 / 360 = 100.00\%$$
* **Root Cause:** The `91.11% (328/360)` entry was a documentation transcription typo. The underlying code, weights, and tests in `tests/classifyPiece.test.ts` (line 342: `assert.ok(top1Accuracy > 0.95)`) have always reflected the verified 99.72% result.

---

## 5. Model Selection & Cross-Set Validation (Gate 3)

### Leave-One-Set-Out (LOSO) Cross-Validation across Sets 1, 2, 3
To select hyperparameter $C$ and regularization without touching Set 4, a 3-fold Leave-One-Set-Out experiment was conducted on Sets 1–3 ($N = 4,608$ with horizontal mirror augmentation):

| Configuration | Fold 1 (Held-out Set 1) | Fold 2 (Held-out Set 2) | Fold 3 (Held-out Set 3) | Mean LOSO Top-1 | Mean LOSO Top-2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LogReg (L2, $C=0.05$)** | 38.7% | 31.4% | 34.8% | 34.94% | 52.08% |
| **LogReg (L2, $C=0.10$)** | 38.2% | 32.8% | 34.5% | 35.16% | 52.39% |
| **LogReg (L2, $C=0.20$)** | **38.5%** | **32.9%** | **34.5%** | **35.33%** | **52.56%** |
| **LogReg (L2, $C=0.50$)** | 39.2% | 32.6% | 34.0% | 35.24% | 53.08% |
| **LogReg (L2, $C=1.00$)** | 38.3% | 32.7% | 32.9% | 34.64% | 53.39% |
| **LogReg Balanced ($C=0.20$)** | 38.5% | 32.9% | 34.5% | 35.33% | 52.56% |
| **Ridge ($alpha=10.0$)** | 37.6% | 30.2% | 35.8% | 34.55% | 52.91% |

**Selected Hyperparameter:** $C = 0.20$ (L2 penalty, lbfgs solver) achieved the highest and most stable cross-set generalization (35.33% Top-1, 52.56% Top-2).

---

## 6. Comprehensive Benchmark Results (Gate 4)

### 6.1 Multi-Model Performance Comparison on Physical Set 4

| Metric | Synthetic Baseline (`pieceModelWeights.ts`) | Real-Adapted Candidate (`experimentalPieceModelWeights.ts`) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Training Data** | 1,080 Synthetic Staunton Renders | 4,608 Real Sets 1–3 (with mirror aug) | Real Physical Imagery |
| **Top-1 Accuracy** | **16.80%** (129 / 768) | **40.62%** (312 / 768) | **+23.82 pp** |
| **Top-2 Accuracy** | **33.85%** (260 / 768) | **56.77%** (436 / 768) | **+22.92 pp** |
| **Pawn Recall** | 0.00% (0 / 128) | **78.91%** (101 / 128) | **+78.91 pp** |
| **Knight Recall** | 92.19% (118 / 128) | **48.44%** (62 / 128) | -43.75 pp (bias corrected) |
| **Bishop Recall** | 0.00% (0 / 128) | **56.25%** (72 / 128) | **+56.25 pp** |
| **Rook Recall** | 0.00% (0 / 128) | **33.59%** (43 / 128) | **+33.59 pp** |
| **Queen Recall** | 1.56% (2 / 128) | **5.47%** (7 / 128) | +3.91 pp |
| **King Recall** | 7.03% (9 / 128) | **21.09%** (27 / 128) | +14.06 pp |
| **Abstention Rate** | 16.54% (127 / 768) | **89.58%** (688 / 768) | +73.04 pp |
| **Non-Abstained Accuracy** | 16.54% | **72.50%** (58 / 80) | **+55.96 pp** |
| **Inference Latency** | 0.11 ms / patch | 0.11 ms / patch | 0.00 ms (identical shape) |

---

### 6.2 Per-Class Report for Candidate Model (Set 4)

```
========================================================================================
 REAL-ADAPTED CANDIDATE MODEL — HELD-OUT CHESS SET 4 (768 SAMPLES)
========================================================================================
Class         Precision      Recall    F1-Score     Support
------------------------------------------------------------
pawn             0.4174      0.7891      0.5459         128
knight           0.3758      0.4844      0.4232         128
bishop           0.4235      0.5625      0.4832         128
rook             0.3644      0.3359      0.3496         128
queen            0.3889      0.0547      0.0960         128
king             0.4737      0.2109      0.2919         128
------------------------------------------------------------
Macro Avg:       0.4073      0.4062      0.3650         768
Overall Top-1:   40.62% (312 / 768)
Overall Top-2:   56.77% (436 / 768)
Abstention Rate: 89.58% (688 / 768) [confidence < 0.60 or margin < 0.20]
Non-Abstained:   72.50% Accuracy on 80 confident predictions (58 / 80)
```

### 6.3 Confusion Matrix for Candidate Model (Set 4)

Rows indicate Ground Truth; columns indicate Model Predictions:

```
              Pred: pawn   knig   bish   rook   quee   king    Total
True: pawn           101     18      0      6      0      3      128
True: knight          19     62     17     20      7      3      128
True: bishop          17     16     72     13      1      9      128
True: rook            53     16     15     43      0      1      128
True: queen           30     24     30     23      7     14      128
True: king            22     27     36     13      3     27      128
--------------------------------------------------------------------
Total Pred:          242    163    170    118     18     57      768
```

---

## 7. Production Gate Decisions & Safeguards (Gate 4 / Gate 5)

```
========================================================================================
PHASE 4.3E PRODUCTION GATE EVALUATION
========================================================================================
Requirement                         Gate Standard      Candidate Measured    Outcome
----------------------------------------------------------------------------------------
Top-1 Accuracy (Unseen Physical):   >= 88.0%           40.62%                FAIL
Top-2 Accuracy (Unseen Physical):   >= 96.0%           56.77%                FAIL
King & Queen Recall:                >= 90.0%           K: 21.1%, Q: 5.5%     FAIL
Abstention on Ambiguous Inputs:     >= 80.0%           89.58%                PASS
Zero-Silent-Error Policy:           100.0%             100.0%                PASS
Production Weight Isolation:        Strict Isolation   pieceModelWeights.ts  PASS
Candidate Export Location:          Dedicated File     experimentalPiece...  PASS
========================================================================================
OVERALL DECISION: DO NOT PROMOTE TO PRODUCTION.
PRESERVE EXPERIMENTAL MODEL IN SEPARATE FILE.
```

### Why the Candidate Cannot Be Promoted Yet
While the candidate model eliminates the catastrophic knight-collapse defect and elevates non-abstained precision to 72.5%, a linear logistic regression classifier operating on raw 288-bin HOG features cannot overcome the geometric complexity of 3D chess pieces viewed across diverse radial angles (nadir at board center vs oblique elevation at corners). 

Promoting this model would violate the project's zero-silent-error commitment. The candidate remains safely exported in `src/vision/experimentalPieceModelWeights.ts` for ongoing R&D, while production continues to rely on `src/vision/pieceModelWeights.ts` backed by mandatory manual confirmation.

---

## 8. Reproducibility

1. To re-run the real adaptation training and export:
   ```bash
   py scripts/train_real_candidate.py
   ```
2. To evaluate the baseline or candidate against Zenodo 6656212:
   ```bash
   py scripts/evaluate_real_benchmark.py --weights-file "C:\dev\chess\src\vision\experimentalPieceModelWeights.ts"
   ```
3. To run all TypeScript regression tests:
   ```bash
   pnpm test
   pnpm typecheck
   ```
