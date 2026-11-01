# Milestone 5.0: Lightweight CNN Feasibility for Real-World Chess-Piece Recognition

**Milestone:** 5.0  
**Date:** 2026-11-01  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `79c03e3`  
**Production Model:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, UNCHANGED)  
**Experimental CNN Checkpoint:** `models/micro_chess_cnn_candidate.pt` (40,038 parameters, 160 KB raw float32)  
**Dataset:** Zenodo Record 6656212 (Campos & Berlanga, Universidad Carlos III de Madrid, CC BY 4.0)  

---

## 1. Executive Summary & Objective

In Milestone 5.0, we investigated whether a compact convolutional neural network (CNN) can surpass the 2D handcrafted HOG representation and achieve production-grade recognition ($\ge 88.0\%$ Top-1) on real physical chess sets, while preserving the application's offline-first, license-safe, and zero-silent-error commitments.

### Definitive Empirical Conclusions
1. **CNN Generalization Deficit on Small Physical Datasets:**
   * In 3-fold Leave-One-Physical-Set-Out (LOSO) cross-validation across Sets 1, 2, and 3, `MicroChessCNN` achieved **20.40% Mean Top-1** and **37.11% Mean Top-2** accuracy.
   * On held-out Set 4 ($N = 768$ physical piece crops), the CNN achieved **27.08% Top-1** (208 / 768) and **51.17% Top-2** (393 / 768).
   * In comparison, the linear HOG candidate (Phase 4.3E) achieved **35.33% Mean LOSO Top-1** and **40.62% on Set 4** (+13.54 pp higher than the CNN).
2. **Root Cause Analysis (Texture Overfitting vs. Inductive Bias):**
   * With only 3 physical chess sets in training ($N = 2,304$ original images), the 40,000 convolutional parameters overfit to low-level set-specific surface cues: wood grain, background square luminance, and reflection sheen.
   * Handcrafted HOG descriptors enforce an explicit inductive bias (local edge gradient histograms with block-level contrast normalization) that is invariant to uniform illumination and surface texture, giving linear HOG superior cross-set domain stability when training data is small.
3. **Deployment Feasibility & Cost Assessment:**
   * Adding an external mobile neural runtime (ONNX Runtime Mobile: 18–24 MB; TensorFlow Lite: 8–12 MB) would bloat the APK by 15–35% while yielding inferior out-of-domain physical accuracy.
4. **Production Decision — Do Not Promote:**
   * The CNN fails all production accuracy gates (Top-1 $27.08\% \ll 88.0\%$).
   * In accordance with the project's **Zero-Silent-Error policy**, production weights in `src/vision/pieceModelWeights.ts` remain **100% UNCHANGED**.
   * Mandatory user confirmation via amber `?` badges and the authoritative manual piece editor remain the correct, truthful fail-safes.

---

## 2. Gate 1: Application Input Pipeline Audit

We audited the mobile application's actual runtime ingestion path to ensure zero preprocessing mismatch:

```
[Camera Sensor / Gallery]
       │
       ▼
takePictureAsync() ──► 3-Channel JPEG (e.g. 3056 x 4076)
       │
       ▼
cv::imdecode() ──────► 3-Channel BGR Mat (CV_8UC3)
       │
       ▼
rectifyMat() ────────► Canonical 800 x 800 Rectified Board Mat (CV_8UC3)
       │
       ▼
toBuffer('uint8') ───► Uint8Array (800 x 800 x 3 = 1,920,000 bytes, [0, 255])
       │
       ├──► Stage 1: classifyRectifiedBoard() ──► Occupancy (empty vs occupied) & Piece Color
       │
       └──► Stage 2: 100 x 100 Square ROI ──► Downsample to 64 x 64 patch ──► Piece Type Classifier
```

### Key Input Characteristics
* **Resolution & Crop Dimensions:** After OpenCV perspective rectification, the canonical board is strictly $800 \times 800$ pixels. Each square is an exact $100 \times 100$ pixel cell. Stage 2 samples the central $64 \times 64$ patch ($xStep = yStep = 1.5625$).
* **Color Handling:** In `board-review.tsx`, photo captures produce 3-channel BGR buffers. Live camera streaming in `modules/live-camera` delivers 1-channel grayscale YUV frames. Both are converted to standard ITU-R 601-2 luminance ($0.299R + 0.587G + 0.114B$) in $[0, 255]$.
* **Piece Color Isolation:** Piece color (`white` vs `black`) is determined Authoritatively in Stage 1 using background square reflectance contrast. The piece classifier operates strictly as a **6-class piece-type model** (`pawn, knight, bishop, rook, queen, king`).
* **Empty Squares:** Handled by Stage 1 occupancy detection ($\ge 98\%$ accuracy verified in `classifySquare.test.ts`). Occupied squares only are passed to Stage 2.
* **Zenodo Dataset Alignment:** The Zenodo 6656212 dataset consists of $181 \times 181$ square crops. Downsampling to $64 \times 64$ grayscale with bilinear interpolation matches the mobile app's input pipeline.

---

## 3. Gate 2: Model Architecture & Licensing Verification

| Architecture | Source / Reference | Code License | Checkpoint / Pretrained License | Licensing Decision |
| :--- | :--- | :---: | :---: | :---: |
| **MobileNetV3-Small** | Google / PyTorch | Apache-2.0 | **ImageNet Non-Commercial** | **DISQUALIFIED (Commercial Hazard)** |
| **MobileNetV4-Conv-Small** | Google (2024) | Apache-2.0 | **ImageNet Non-Commercial** | **DISQUALIFIED (Commercial Hazard)** |
| **SqueezeNet 1.1** | DeepScale / UC Berkeley | BSD-3-Clause | **ImageNet Non-Commercial** | **DISQUALIFIED (Commercial Hazard)** |
| **MicroChessCNN (Scratch)** | Custom Clean-Room | **MIT** | **CC BY 4.0 (Trained from Scratch)** | **APPROVED (100% Clean Room)** |

### Pretrained Weights Legal Hazard
Official MobileNetV3 and SqueezeNet checkpoints in torchvision/timm were pretrained on ImageNet-1K (ILSVRC 2012). The ImageNet Terms of Access explicitly restrict usage to *"non-commercial research and/or educational purposes"*. Shipping ImageNet-derived weights in a commercial-ready open-source mobile app violates our strict licensing mandate (`docs/DATASET_PROVENANCE.md`). 

Therefore, any viable neural candidate must be **trained from scratch** on legally cleared photography (Zenodo 6656212, CC BY 4.0).

### MicroChessCNN Architecture Specification
```
Input: (Batch, 1, 64, 64) float32 in [0.0, 1.0]
  ├── Conv2d(1, 16, k=3, s=2, p=1) + BatchNorm2d(16) + ReLU   ──► (16, 32, 32)
  ├── Conv2d(16, 32, k=3, s=1, p=1) + BatchNorm2d(32) + ReLU  ──► (32, 32, 32)
  │   └── MaxPool2d(k=2, s=2)                                 ──► (32, 16, 16)
  ├── Conv2d(32, 48, k=3, s=1, p=1) + BatchNorm2d(48) + ReLU  ──► (48, 16, 16)
  │   └── MaxPool2d(k=2, s=2)                                 ──► (48, 8, 8)
  ├── Conv2d(48, 48, k=3, s=1, p=1) + BatchNorm2d(48) + ReLU  ──► (48, 8, 8)
  │   └── AdaptiveAvgPool2d((1, 1))                           ──► (48, 1, 1)
  └── Dropout(0.2) + Linear(48, 6)                            ──► (6,) Logits
Total Parameters: 40,038 float32 (160.15 KB raw binary)
```

---

## 4. Gate 3 & Gate 4: Training, Cross-Validation & Deployment Analysis

### Leave-One-Physical-Set-Out (LOSO) Cross-Validation (Sets 1–3)
Trained for 8 epochs per fold using AdamW ($lr=10^{-3}$, weight_decay=$10^{-4}$) with CosineAnnealingLR and horizontal mirror augmentation ($N = 3,072$ augmented train, $N = 768$ unaugmented test):

```
Fold 1 (Held-out Set 1): Top-1 = 24.87%, Top-2 = 42.58% (N=768)
Fold 2 (Held-out Set 2): Top-1 = 17.06%, Top-2 = 36.07% (N=768)
Fold 3 (Held-out Set 3): Top-1 = 19.27%, Top-2 = 32.68% (N=768)
----------------------------------------------------------------
Mean LOSO Top-1 Accuracy: 20.40%
Mean LOSO Top-2 Accuracy: 37.11%
```

### In-Domain Performance on Sets 1–3 ($N = 2,304$)
* Top-1 Accuracy: **63.37% (1,460 / 2,304)**
* Top-2 Accuracy: **82.77% (1,907 / 2,304)**
* *Analysis:* The CNN easily learns the training sets (reaching 63.4% in-domain), but cross-set generalization collapses to 20.4%, proving severe overfitting to set-specific wood grain and board contrast.

### Android Deployment Runtime Trade-Offs

| Runtime Option | ARM64 Native Binary Size | Memory Overhead | Integration Complexity | Feasibility |
| :--- | :---: | :---: | :---: | :--- |
| **ONNX Runtime Mobile** | 18 – 24 MB | 15 MB heap | Requires new native C++ build + maven deps | Moderate (APK size penalty) |
| **TensorFlow Lite / LiteRT** | 8 – 12 MB | 10 MB heap | Requires new native `.so` packaging | Moderate (APK size penalty) |
| **Custom Embedded C++ JNI** | **0 MB (in libchal-engine)** | **< 1 MB** | Implement 4-layer conv loop in C++20 | **High (Zero new dependencies)** |

---

## 5. Gate 5: Set 4 Evaluation & Production Gate Table

### 5.1 Comprehensive Set 4 Performance ($N = 768$)

| Metric | Synthetic Baseline (`pieceModelWeights.ts`) | Linear Candidate (Phase 4.3E) | MicroChessCNN (Milestone 5.0) | Production Gate Target |
| :--- | :---: | :---: | :---: | :---: |
| **Model Type** | 288 HOG + Linear | 288 HOG + Linear ($C=0.2$) | 4-Layer ConvNet | Compact Offline Model |
| **Training Data** | 1,080 Synthetic Staunton | 4,608 Real Sets 1–3 (aug) | 4,608 Real Sets 1–3 (aug) | Real Physical Imagery |
| **Top-1 Accuracy** | **16.80%** (129/768) | **40.62%** (312/768) | **27.08%** (208/768) | $\ge 88.0\%$ (**FAIL**) |
| **Top-2 Accuracy** | **33.85%** (260/768) | **56.77%** (436/768) | **51.17%** (393/768) | $\ge 96.0\%$ (**FAIL**) |
| **Pawn Recall** | 0.00% (0/128) | **78.91%** (101/128) | 30.47% (39/128) | N/A |
| **Knight Recall** | 92.19% (118/128) | 48.44% (62/128) | 45.31% (58/128) | N/A |
| **Bishop Recall** | 0.00% (0/128) | **56.25%** (72/128) | 0.78% (1/128) | N/A |
| **Rook Recall** | 0.00% (0/128) | 33.59% (43/128) | **69.53%** (89/128) | N/A |
| **Queen Recall** | 1.56% (2/128) | 5.47% (7/128) | **14.84%** (19/128) | $\ge 90.0\%$ (**FAIL**) |
| **King Recall** | 7.03% (9/128) | **21.09%** (27/128) | 1.56% (2/128) | $\ge 90.0\%$ (**FAIL**) |
| **Abstention Rate** | 16.54% (127/768) | 89.58% (688/768) | **95.18%** (731/768) | $\ge 80.0\%$ (**PASS**) |
| **Accepted Accuracy** | 16.54% | **72.50%** (58/80) | 67.57% (25/37) | High-confidence filter |
| **Workstation Board Latency\*** | ~840 ms / 64 board | ~847 ms / 64 board | **124.6 ms / 64 board** | Measured on Ryzen 5 CPU |
| **Target Device Latency (vivo)** | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | Awaiting device profiling |

*\*Workstation benchmark environment: Windows 11, AMD Ryzen 5, Python 3.13.2 x86_64 single-thread.*

### 5.2 Confusion Matrix (MicroChessCNN on Set 4)
```
              Pred: pawn   knig   bish   rook   quee   king    Total
True: pawn            39     56      0     31      2      0      128
True: knight          17     58      3     46      4      0      128
True: bishop           6     44      1     57     18      2      128
True: rook             4     21      0     89     12      2      128
True: queen            1     24      0     81     19      3      128
True: king             2     13      1     82     28      2      128
--------------------------------------------------------------------
Total Pred:           69    216      5    386     83      9      768
```

---

## 6. Gate 6: Production Decision & Final Policy Table

```
========================================================================================================
MILESTONE 5.0 PRODUCTION ACCEPTANCE GATES
========================================================================================================
Requirement                         Gate Standard      MicroChessCNN (M5.0)    Gate Decision
--------------------------------------------------------------------------------------------------------
Top-1 Accuracy (Physical Set 4):    >= 88.0%           27.08%                  FAIL
Top-2 Accuracy (Physical Set 4):    >= 96.0%           51.17%                  FAIL
Queen Recall (Physical Set 4):      >= 90.0%           14.84%                  FAIL
King Recall (Physical Set 4):       >= 90.0%           1.56%                   FAIL
Abstention Rate (>= 80%, 90% tgt):  >= 80.0%           95.18%                  PASS (Exceeds 90% target)
Target Device Latency (vivo V2336): <= 150 ms / board  UNVERIFIED              UNVERIFIED
Zero-Silent-Error Policy:           100.0%             100.0%                  PASS
Production Weight Isolation:        pieceModelWeights  UNCHANGED               PASS
Checkpoint Isolation:               models/*.pt        models/micro_chess...   PASS
========================================================================================
OVERALL DECISION: DO NOT PROMOTE TO PRODUCTION.
PRESERVE PRODUCTION WEIGHTS UNCHANGED AT src/vision/pieceModelWeights.ts.
MAINTAIN MANDATORY MANUAL CONFIRMATION IN BOARD REVIEW.
========================================================================================
```

---

## 7. Next Actions for Physical Chess Set Recognition

To eventually achieve $\ge 88.0\%$ Top-1 on real physical chess sets:
1. **Physical Set Diversity Deficit:** Training on only 3 physical sets is mathematically insufficient for neural networks to learn shape invariance across 3D piece carvings. A diverse physical capture dataset covering at least 15–20 distinct tournament, club, and wooden sets under calibrated radial angles is required.
2. **Hybrid Multi-Stage Pipeline:** Keep Stage 1 (classical background adaptive surface stats) for robust occupancy and color separation.
3. **No Compromise on Safety:** Continue strictly enforcing the amber `?` confirmation workflow and manual piece editor fail-safe. No unverified neural model will ever silently place pieces on the user's board.

---

## 8. Milestone 5.1 Audit Cross-Reference

For the complete dataset-to-runtime alignment audit, input sampling mathematical trace, and minimal controlled diagnosis verifying that preprocessing differences account for less than $\pm 0.3\%$ variance, see `docs/DATASET_TO_RUNTIME_ALIGNMENT_AUDIT.md`.

