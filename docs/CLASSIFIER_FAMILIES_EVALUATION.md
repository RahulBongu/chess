# Classifier-Family Evaluation & Honest Performance Gates

**Milestone:** 4.0 — Phase 4.3F  
**Date:** 2026-11-01  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Production Model:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, UNCHANGED)  
**Experimental Linear Candidate:** `experimentalPieceClassifierModel` (`src/vision/experimentalPieceModelWeights.ts`, version `1.1.0-real-adapted-candidate-v1`)  
**Evaluator Script:** `scripts/evaluate_classifier_families.py`  
**Dataset:** Zenodo Record 6656212 (Campos & Berlanga, Universidad Carlos III de Madrid, CC BY 4.0)  

---

## 1. Executive Summary & Objective

In Phase 4.3F, we investigated whether stronger or nonlinear classifier families can overcome the physical chess piece generalization barrier. 

We conducted a controlled experimental comparison across **five classifier families** over the exact 288-bin HOG feature extractor:
1. **Linear Logistic Regression (Control):** L2 penalty, $C \in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]$.
2. **Linear with Feature Standardization:** `StandardScaler` (fit strictly on training folds) + Logistic Regression.
3. **Linear Ridge Classifier:** Squared-error loss with L2 regularization.
4. **Compact Nonlinear Neural Multilayer Perceptron (MLP):** Single and double hidden layers with ReLU activation (32 to 64 units).
5. **Nonlinear Support Vector Machines:** Radial Basis Function (RBF) kernel ($\gamma=\text{scale}$) and Polynomial kernel (degree=2).

### Key Takeaways
1. **Generalization Ceiling of HOG Features:** No candidate family reaches the production acceptance standard ($\ge 88.0\%$ Top-1 on unseen physical chess sets). The best model achieves **46.88% Top-1** (RBF SVM) and **45.18% Top-1** (MLP), compared to **40.62% Top-1** for the linear control.
2. **Extreme RBF Resource Inflation (+1,300× Size, +700× Latency):** RBF SVM stores **4,044 support vectors (87.76% of training data)**, inflating model memory from **6.8 KB to 9.1 MB** and increasing per-patch inference latency from $1.5\ \mu\text{s}$ to $1,085\ \mu\text{s}$ for only a +1.3 pp gain in cross-validation.
3. **MLP Overfitting:** A 64-unit neural MLP achieves 34.81% in Leave-One-Set-Out cross-validation (slightly lower than the linear control's 35.33%), demonstrating that increased parameter capacity does not solve out-of-domain physical piece variations.
4. **Decision:** In adherence to the project's **Zero-Silent-Error policy**, production weights in `src/vision/pieceModelWeights.ts` remain **100% UNCHANGED**.

---

## 2. Experimental Setup & Split Methodology

### Leave-One-Physical-Set-Out (LOSO) Cross-Validation
All model selection and hyperparameter exploration were performed strictly on physical Sets 1, 2, and 3 from Zenodo 6656212 ($N = 4,608$ samples with horizontal mirror augmentation):
* **Fold 1:** Train on Sets 2 + 3 ($N = 3,072$) $\to$ Evaluate on unaugmented Set 1 ($N = 768$).
* **Fold 2:** Train on Sets 1 + 3 ($N = 3,072$) $\to$ Evaluate on unaugmented Set 2 ($N = 768$).
* **Fold 3:** Train on Sets 1 + 2 ($N = 3,072$) $\to$ Evaluate on unaugmented Set 3 ($N = 768$).

**Data Leakage Guard:** Mirrored versions of source images are strictly grouped with their parent set and never split across train/test folds. Set 4 is held out completely during model selection.

---

## 3. Classifier-Family Cross-Validation Results (Sets 1–3)

*Workstation Benchmark Environment: Windows 11, AMD Ryzen 5, Python 3.13.2 x86_64, single-thread.*

| Model Family / Configuration | Parameters | Memory Size | Mean LOSO Top-1 | Mean LOSO Top-2 | Fold 1 (Set 1) | Fold 2 (Set 2) | Fold 3 (Set 3) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control: LogReg L2 ($C=0.20$)** | 1,734 | 6.8 KB | **35.33%** | **52.56%** | 38.5% | 32.9% | 34.5% | **1.53 $\mu$s / patch** |
| **Control: LogReg L2 ($C=1.00$)** | 1,734 | 6.8 KB | 34.64% | 53.39% | 38.3% | 32.7% | 32.9% | 1.58 $\mu$s / patch |
| **Linear: Scaled LogReg ($C=0.20$)** | 1,734 | 6.8 KB | 31.86% | 51.48% | 35.3% | 29.6% | 30.7% | 1.57 $\mu$s / patch |
| **Linear: Scaled LogReg ($C=1.00$)** | 1,734 | 6.8 KB | 30.21% | 50.04% | 34.4% | 26.8% | 29.4% | 1.49 $\mu$s / patch |
| **Linear: Ridge ($alpha=10.0$)** | 1,734 | 6.8 KB | 34.55% | 52.91% | 37.6% | 30.2% | 35.8% | 1.82 $\mu$s / patch |
| **MLP: (32,) ReLU early_stop** | 9,446 | 37.8 KB | 33.29% | 53.34% | 36.1% | 32.0% | 31.8% | 1.83 $\mu$s / patch |
| **MLP: (64,) ReLU early_stop** | 18,886 | 75.5 KB | 34.81% | 52.99% | 36.1% | 35.5% | 32.8% | 2.34 $\mu$s / patch |
| **RBF SVM: $C=1.0, \gamma=\text{scale}$** | 1,164,672 | 9,099 KB | **36.63%** | 52.30% | 38.8% | 38.7% | 32.4% | 1,085.04 $\mu$s / patch |
| **RBF SVM: $C=2.0, \gamma=\text{scale}$** | 1,164,672 | 9,099 KB | 36.55% | 52.86% | 38.4% | 37.1% | 34.1% | 1,122.49 $\mu$s / patch |
| **Poly SVM: degree=2, $C=1.0$** | 895,680 | 7,000 KB | 35.20% | 52.56% | 37.8% | 36.3% | 31.5% | 504.62 $\mu$s / patch |

---

## 4. Final Evaluation on Held-Out Physical Set 4 ($N = 768$)

Each model was fit on all training data (Sets 1–3, $N = 4,608$) and evaluated on Set 4:

| Model Candidate | Set 4 Top-1 | Set 4 Top-2 | Parameter Count | Binary Size | Inference Feasibility on Mobile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Linear Control ($C=0.20$)** | 40.62% (312/768) | 56.77% (436/768) | 1,734 | 6.8 KB | **Pure TypeScript forward pass (zero native deps)** |
| **2. Compact MLP (64, ReLU)** | 45.18% (347/768) | 63.80% (490/768) | 18,886 | 75.5 KB | Pure TypeScript forward pass (Matrix mult + ReLU) |
| **3. RBF SVM ($C=1.0$)** | 46.88% (360/768) | 62.76% (482/768) | 1,164,672 | 9.1 MB | Infeasible (Requires 4,044 kernel evals / patch) |

---

## 5. Risk-versus-Coverage (Abstention) Analysis

*Evaluated on Set 4 using the Linear Candidate Model:*

| Confidence Threshold | Margin Threshold | Coverage | Abstention Rate | Accuracy on Accepted | Error Rate on Accepted | Accepted Samples |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 0.00 | 100.00% | 0.00% | 40.62% | 59.38% | 768 / 768 |
| 0.30 | 0.05 | 72.01% | 27.99% | 46.65% | 53.35% | 553 / 768 |
| 0.40 | 0.10 | 42.45% | 57.55% | 56.44% | 43.56% | 326 / 768 |
| 0.50 | 0.15 | 21.88% | 78.12% | 68.45% | 31.55% | 168 / 768 |
| **0.60** | **0.20** | **10.42%** | **89.58%** | **72.50%** | **27.50%** | **80 / 768** |
| 0.70 | 0.25 | 3.12% | 96.88% | 87.50% | 12.50% | 24 / 768 |
| 0.80 | 0.30 | 1.56% | 98.44% | 100.00% | 0.00% | 12 / 768 |

### Abstention Findings
* Under the default production thresholds ($\text{conf} \ge 0.60, \text{margin} \ge 0.20$), the model abstains on **89.58% (688 / 768)** of samples, achieving **72.50% precision** on the 80 accepted predictions.
* The 80% minimum abstention gate is passed ($89.58\% \ge 80.0\%$), but the 90% aspirational target is **NOT MET** ($89.58\% < 90.0\%$).

---

## 6. Honest Performance Gate Summary

```
========================================================================================
PHASE 4.3F HONEST PERFORMANCE GATE EVALUATION
========================================================================================
Acceptance Criteria                Standard           Linear (Control)    Compact MLP (64)    RBF SVM (C=1)    Status
------------------------------------------------------------------------------------------------------------------------
Top-1 Accuracy (Physical Set 4):   >= 88.0%           40.62%              45.18%              46.88%           FAIL
Top-2 Accuracy (Physical Set 4):   >= 96.0%           56.77%              63.80%              62.76%           FAIL
Queen Recall (Physical Set 4):     >= 90.0%           5.47%               10.94%              12.50%           FAIL
King Recall (Physical Set 4):      >= 90.0%           21.09%              24.22%              25.78%           FAIL
Abstention Rate on Ambiguous:      >= 80.0% (90% tgt) 89.58%              88.28%              87.50%           PARTIAL (>=80%, <90%)
Target Device Latency (vivo V2336):<= 150 ms / board  UNVERIFIED          UNVERIFIED          UNVERIFIED       UNVERIFIED
Zero-Silent-Error Enforcement:     100.0%             100.0%              100.0%              100.0%           PASS
Production Weight Isolation:       pieceModelWeights  UNCHANGED           UNCHANGED           UNCHANGED        PASS
========================================================================================
OVERALL VERDICT: ALL CANDIDATE FAMILIES FAIL PRODUCTION ACCURACY GATES.
PRESERVE PRODUCTION WEIGHTS UNCHANGED. MAINTAIN MANDATORY MANUAL CONFIRMATION.
========================================================================================
```

### Architectural Analysis & Path Forward
Handcrafted 288-bin HOG descriptors measure edge orientation histograms, which are invariant to minor illumination changes but fundamentally sensitive to:
1. 3D perspective distortion (pieces on corner squares viewed at oblique angles cast asymmetric silhouettes).
2. Carving style variations (e.g. Knight heads carved facing forward vs sideways, Bishop mitre slit orientation).

Increasing classifier complexity from linear to RBF or MLP yields only modest gains (+4 to +6 pp) while remaining far below the 88% production threshold. True $\ge 90\%$ cross-set physical piece recognition will require an end-to-end convolutional or attention-based neural backbone (e.g., MobileNetV4 / EfficientNet-Lite) trained on multi-angle physical sets. 

Until such an offline mobile pipeline is designed and verified, AI Chess Navigator's **Zero-Silent-Error policy**, prominent experimental UI disclosure, and authoritative manual piece editor remain the correct and honest architectural guarantees.
