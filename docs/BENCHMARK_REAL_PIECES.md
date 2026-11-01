# Real-World Chess Piece Classifier Benchmark

**Milestone:** 4.0 — Phase 4.3D  
**Date:** 2026-11-01  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Model Under Test:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`)  
**Evaluator Tool:** `scripts/evaluate_real_benchmark.py`  

---

## 1. Executive Summary

In Milestone 4.0 (Phase 4.3B/C), an offline 288-bin Histogram of Oriented Gradients (HOG) feature extractor and 6-class linear classifier were developed and integrated into the mobile Board Review screen. The initial model weights were trained and evaluated exclusively on synthetic procedural Staunton silhouettes across four style families, achieving **91.11% Top-1** accuracy on a held-out synthetic shape family.

In Phase 4.3D, we benchmarked this **exact, unchanged synthetic model** against real-world physical chessboard piece photographs from a legally approved academic dataset (Zenodo Record 6656212).

### Key Empirical Findings
* **Severe Domain Gap (-74.31 percentage points):** On unseen physical pieces (Set 4, 768 images), the synthetic model achieves **16.80% Top-1** and **33.85% Top-2** accuracy, performing near random chance ($1/6 \approx 16.67\%$).
* **Knight Bias Dominance:** The linear model's learned bias for the `knight` class ($+1.3579$) overwhelms real-world gradients, causing **91.0% (699/768)** of all real pieces to be misclassified as knights.
* **Production Safeguards Validated:** Because AI Chess Navigator enforces a strict **Zero-Silent-Error policy**, all piece suggestions begin marked with amber `?` badges requiring explicit user confirmation. Under production thresholds ($\text{confidence} \ge 0.60$ and $\text{margin} \ge 0.20$), ambiguous predictions are flagged for manual review, and the authoritative manual editor prevents incorrect FENs from being sent to Study.

---

## 2. Dataset Provenance & Licensing Audit

### Primary Benchmark Dataset: Zenodo Record 6656212
* **Title:** *Chess piece dataset for image classification*
* **DOI:** [10.5281/zenodo.6656212](https://doi.org/10.5281/zenodo.6656212)
* **Authors:** Pablo Tomas Campos (ORCID `0000-0001-6464-6685`), Antonio Berlanga (Supervisor, ORCID `0000-0002-5564-399X`), Universidad Carlos III de Madrid (UC3M).
* **Published Date:** June 22, 2022 (Version v1.0.0).
* **Archive Size:** 153,649,683 bytes (`chess_pieces.zip`).
* **Verified Checksum:** `md5:4583133a23383c05c982dd4ddd549068` (100% bit-for-bit match verified).
* **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
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

The Zenodo 6656212 dataset contains **3,200 total PNG images** ($181\times 181$ RGB), organized into two splits:

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

### Piece Class Balance (Valid Split — Unseen Physical Set 4)
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

## 4. Benchmark Methodology

1. **Model Under Test:** The exact deployed weights and biases from `src/vision/pieceModelWeights.ts` ($W \in \mathbb{R}^{6 \times 288}$, $b \in \mathbb{R}^6$, $T = 1.0$) were loaded without retraining or calibration.
2. **Feature Extraction:** Evaluated with the deterministic 288-bin HOG extractor (`extract_hog_288` matching `classifyPiece.ts`).
3. **Preprocessing:** Each $181\times 181$ square image was converted to 8-bit luminance using ITU-R 601-2 ($0.299R + 0.587G + 0.114B$) and resized to $64\times 64$ pixels with bilinear interpolation.
4. **Scoring:** Logits $z_c = \sum_{j} W_{c, j} \cdot x_j + b_c$ normalized via Softmax.
5. **Thresholds:** Production defaults ($\text{confidence} \ge 0.60$, $\text{margin} \ge 0.20$).

---

## 5. Measured Baseline Results

### 5.1 Synthetic vs. Real Performance Comparison

| Metric | Synthetic Held-Out Family (Rounded Wood) | Real Held-Out Set 4 (Zenodo valid/) | Real Sets 1, 2, 3 (Zenodo train/) | Domain Gap (Synthetic $\to$ Real Set 4) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Samples** | 360 | 768 | 2,304 | +408 samples |
| **Top-1 Accuracy** | **91.11%** (328/360) | **16.80%** (129/768) | **16.32%** (376/2304) | **-74.31 pp** |
| **Top-2 Accuracy** | **98.33%** (354/360) | **33.85%** (260/768) | **31.64%** (729/2304) | **-64.48 pp** |
| **Abstention Rate** | 2.50% (9/360) | 16.54% (127/768) | 36.28% (836/2304) | +14.04 pp |
| **HOG Latency** | 13.0 ms / patch | 13.1 ms / patch | 14.4 ms / patch | Negligible |
| **Inference Latency** | 0.11 ms / patch | 0.11 ms / patch | 0.12 ms / patch | Negligible |
| **Projected 64-Board** | ~840 ms (Python) | 847 ms (Python) | 932 ms (Python) | Real-time |

---

### 5.2 Real-World Classification Report (Held-Out Set 4)

```
========================================================================================
 HELD-OUT REAL CHESS SET 4 (768 SAMPLES)
========================================================================================
Class         Precision      Recall    F1-Score     Support
------------------------------------------------------------
pawn             0.0000      0.0000      0.0000         128
knight           0.1688      0.9219      0.2854         128
bishop           0.0000      0.0000      0.0000         128
rook             0.0000      0.0000      0.0000         128
queen            0.2000      0.0156      0.0290         128
king             0.3750      0.0703      0.1184         128
------------------------------------------------------------
Overall Top-1:   16.80% (129 / 768)
Overall Top-2:   33.85% (260 / 768)
Abstentions:     16.54% (127 / 768) [confidence < 0.60 or margin < 0.20]
```

---

### 5.3 Real-World Confusion Matrix (Held-Out Set 4)

Rows indicate Ground Truth; columns indicate Model Predictions:

```
              Pred: pawn   knig   bish   rook   quee   king    Total
True: pawn             0    123      3      0      0      2      128
True: knight           4    118      2      0      1      3      128
True: bishop           2    120      0      0      4      2      128
True: rook             0    124      2      0      0      2      128
True: queen            4    116      0      0      2      6      128
True: king            14     98      4      0      3      9      128
--------------------------------------------------------------------
Total Pred:           24    699     11      0     10     24      768
```

---

### 5.4 Failure Mode Analysis

#### A. Bias Imbalance Collapse
* The synthetic model's learned bias vector is:
  $$\mathbf{b} = [-0.1232\ (\text{P}),\ +1.3579\ (\text{N}),\ +0.0859\ (\text{B}),\ -0.4676\ (\text{R}),\ +0.8419\ (\text{Q}),\ -1.6949\ (\text{K})]$$
* Because synthetic silhouettes had clean binary edges and zero background texture, linear weights were tuned to subtle gradient differences. In real photography, diffuse room lighting, wood grain, and non-uniform square backgrounds produce soft gradient magnitudes that are overwhelmed by the $+1.3579$ Knight bias, forcing 91.0% of all real inputs into the Knight class.

#### B. Color Disparity
* **White Pieces Accuracy:** 17.97% (69 / 384)
* **Black Pieces Accuracy:** 15.62% (60 / 384)
* Both piece colors suffer from the same bias collapse; dark pieces on dark squares exhibit slightly worse edge definition than white pieces on dark squares.

#### C. Perspective Angle Degradation Across Board Squares
* **Center Squares (D4, D5, E4, E5):** 14.58% (7 / 48)
* **Corner Squares (A1, A8, H1, H8):** 16.67% (8 / 48)
* **Remaining 56 Squares:** 16.96% (114 / 672)
* Overhead cameras exhibit natural radial perspective shifts: pieces on perimeter squares appear tilted away from the camera optical center, altering the visible silhouette profile compared to upright center pieces.

---

## 6. UI Safeguard & Zero-Silent-Error Audit

The benchmark proves that **the app's design choices in Phase 4.3C were essential**:
1. **No Silent Auto-Handoff:** Had the app automatically marked piece suggestions as confirmed, an invalid board with 16 knights would have been sent to Study. Instead, every suggested square requires explicit manual confirmation.
2. **Prominent UI Disclosure:** The app UI explicitly warns the user on screen:
   > *"Notice: Experimental Piece Recognition: Piece suggestions are generated by an experimental model trained on synthetic Staunton silhouettes (Phase 4.3B). Accuracy on real physical boards has not been validated. Every square with ? must be verified before handoff."*
3. **Fail-Safe Manual Editor:** When suggestions fail, the user can tap any square and select the authoritative piece in one tap, or tap `Clear Board` / `Initial Setup`.

---

## 7. Recommended Roadmap for Model Adaptation (Phase 4.3E)

With the baseline rigorously established and the licensing gate cleared via Zenodo 6656212 (CC BY 4.0):
1. **Real-Image Weight Training:** Train weights on physical Sets 1, 2, and 3 from Zenodo 6656212 combined with synthetic augmentations.
2. **Validation on Held-Out Set 4:** Evaluate only on the unseen Set 4 to prevent data leakage.
3. **Target Production Acceptance Criteria:** Achieve $\ge 88\%$ Top-1 and $\ge 96\%$ Top-2 accuracy on Set 4 before replacing `pieceModelWeights.ts`.

---

## 8. Reproducibility

To re-run the benchmark at any time:
```bash
py scripts/evaluate_real_benchmark.py --data-dir "C:\Users\rahul\AppData\Local\Temp\opencode\zenodo_6656212\extracted"
```
All dataset files reside outside the Git repository in local temporary storage. No source files or model weights were altered during this audit.
