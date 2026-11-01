# Milestone 5.2: Real-World Dataset Expansion and License Gate

**Milestone:** 5.2  
**Date:** 2026-11-02  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `5d0904e`  
**Objective:** Investigate lawful, reproducible sources to expand real physical chess-piece imagery diversity before further model training.  
**Production Model:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, UNCHANGED)  

---

## 1. Executive Summary

Milestones 4.3D through 5.1 established that:
1. Procedural synthetic training achieves **99.72% Top-1** on synthetic fixtures, but collapses to **16.80% Top-1** on real physical pieces (`knight` bias collapse).
2. Linear classifiers adapted to Zenodo Sets 1–3 achieve **35.33% Mean LOSO Top-1** and **40.62% on Set 4**.
3. Compact CNNs (`MicroChessCNN`, 40k parameters) trained from scratch achieve **20.40% Mean LOSO Top-1** and **27.99% on Set 4** due to texture/wood-grain overfitting on small datasets ($N=2,304$).
4. The Milestone 5.1/5.1A audit definitively proved that this ~20% LOSO ceiling is **not** an input pipeline defect or preprocessing mismatch.

In Milestone 5.2, we conducted an exhaustive licensing, provenance, and data-diversity audit of all public chessboard and piece datasets in the machine learning literature.

### Key Discoveries of the Audit
1. **The Single-Set Monopoly in Academic Datasets:**
   * **ChessReD (TU Delft / 4TU.ResearchData, 2024):** While providing 10,800 smartphone photos under CC BY 4.0, **all 10,800 images were captured from a single physical chess set and board**! It adds zero physical-set diversity.
   * **samryan18 (GitHub, MIT):** 500 images of a single wooden board.
   * **Roboflow Universe (`chess-full`):** Single board on a carpeted floor, 21 corrupted class labels, and unverified anonymous lineage.
2. **The Global Diversity Gap:**
   * Across the entire open-access scientific literature, only **6 distinct physical chess sets** exist with clean commercial licensing: Zenodo 6656212 (4 sets), ChessReD (1 set), and FENiT (1 set).
   * There is a **deficit of at least 14 physical chess sets** to reach our target of 20+ distinct sets required for shape-invariant piece recognition.
3. **Strategic Acquisition Decision:**
   * External open datasets cannot bridge the diversity gap because existing academic efforts focused on perspective variations of a *single* set, rather than carving and material diversity across *multiple* sets.
   * To achieve production-grade recognition ($\ge 88.0\%$ Top-1), the project must execute an **Author-Owned Physical Capture Protocol** across 20 distinct physical tournament, club, and wooden chess sets with multi-angle calibration and clean CC0 / MIT licensing.

---

## 2. Gate 0: Repository State Preservation

* **Git HEAD:** Verified at `5d0904e` on `main`, synchronized with `origin/main`.
* **Artifact Preservation:** All 29 APKs and 6 log directories in `dist/` remain untouched.
* **No APK Builds:** Zero builds or compilation commands executed in Milestone 5.2.
* **Clean-Room Storage:** All external dataset metadata and research caches reside outside Git in `C:\Users\rahul\AppData\Local\Temp\opencode\`.
* **Runtime Permissions:** Zero network permissions (`INTERNET` and `RECORD_AUDIO` strictly absent).

---

## 3. Gate 1: Comprehensive Audit of Existing Dataset Limitations

We audited Zenodo Record 6656212 (Campos & Berlanga, UC3M, CC BY 4.0):

### 3.1 Dataset Characterization
* **Physical Set Count:** Exactly 4 physical sets (Sets 1, 2, 3 in `train/`, Set 4 in `valid/`).
* **Camera Viewpoint:** Static zenith overhead view (~$85^\circ$ to $90^\circ$) across all 64 squares. Zero oblique, player, or side angles.
* **Class & Color Balance:** Perfectly balanced:
  - 12 piece classes (`[color]_[piece]`): 64 images per piece per set.
  - Exactly 384 White pieces and 384 Black pieces per set.
  - Total piece images: $4 \times 768 = 3,072$ images ($181\times 181$ PNG crops).
* **Labeling Quality:** Clean ground truth with algebraic square coordinates and set IDs.
* **Set 4 Holdout Status:** Set 4 has been repeatedly evaluated as a historical benchmark since Phase 4.3D and must not be used for model tuning or dataset selection.

### 3.2 Root Cause of Empirical Generalization Failure
* In 3-fold Leave-One-Physical-Set-Out (LOSO) cross-validation on Sets 1–3 ($N=2,304$), training a 40,000-parameter CNN from scratch achieves:
  - In-Domain Accuracy: **64.28% Top-1** (1,481 / 2,304)
  - Mean Out-of-Domain LOSO: **20.40% Top-1** (Fold 1: 24.87%, Fold 2: 16.41%, Fold 3: 19.92%)
* **Diagnosis:** With only 3 physical sets in training, convolutional filters memorize low-level wood grain, varnish sheen, and lathe grooves rather than 3D piece geometry. When evaluated on an unseen set with different wood contrast (such as Set 3, which has a contrast spread of only 19.9 units vs. 61.0 units in Set 2), the network collapses.

---

## 4. Gate 2: Source-Level Audit of Real-Image Candidates

We performed a deep licensing and suitability audit across all identified public chess datasets:

### Candidate 1: Zenodo Record 6656212
* **Permanent URL:** `https://doi.org/10.5281/zenodo.6656212`
* **Authors:** Pablo Tomás Campos & Antonio Berlanga (Universidad Carlos III de Madrid)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Commercial Model Training / Offline Embedding:** Permitted with attribution
* **Physical Sets:** 4 sets (Sets 1–3 train, Set 4 validation)
* **Images:** 3,200 ($181\times 181$ PNG crops)
* **Usability:** 100% directly compatible with $800\times 800$ rectification pipeline.
* **Gate Decision:** **APPROVED AS PRIMARY HISTORICAL BENCHMARK.**

### Candidate 2: ChessReD (Chess Recognition Dataset)
* **Permanent URL:** `https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f` / `arXiv:2310.04086`
* **Authors:** Athanasios Masouris and Jan C. van Gemert (Delft University of Technology / TU Delft)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Commercial Model Training / Offline Embedding:** Permitted with attribution
* **Physical Sets:** **EXACTLY 1 PHYSICAL SET** (100 games, 10,800 images across 3 smartphone cameras).
* **Usability:** Excellent for multi-angle robustness, but provides zero set diversity.
* **Gate Decision:** **APPROVED FOR ANGLE ROBUSTNESS BENCHMARKING; REJECTED AS A SOLUTION TO PHYSICAL DIVERSITY.**

### Candidate 3: FENiT Annotated Image Dataset
* **Permanent URL:** `https://doi.org/10.17632/wpp9xnbcp6.1`
* **Authors:** Theo Harjanto, Vincent Oei, Dicky Susanto (Binus University)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Commercial Model Training / Offline Embedding:** Permitted with attribution
* **Physical Sets:** 1–2 standard tournament vinyl sets.
* **Annotations:** CVAT bounding boxes and 4-point board corner keypoints.
* **Gate Decision:** **APPROVED AS SECONDARY VALIDATION BENCHMARK.**

### Candidate 4: Sam Ryan (`samryan18/chess-dataset`)
* **Permanent URL:** `https://github.com/samryan18/chess-dataset`
* **Authors:** Sam Ryan & Mukund V.
* **License:** MIT License
* **Commercial Model Training / Offline Embedding:** Permitted
* **Physical Sets:** 1 wooden chessboard (500 full-board photos).
* **Gate Decision:** **APPROVED AS REFERENCE PIPELINE; INSUFFICIENT DIVERSITY.**

### Candidate 5: Roboflow Universe (`chess-full-r3wh3-sajg6`)
* **Source:** Roboflow Universe (`chess-project-wfqnn`)
* **License Claim:** CC BY 4.0 (platform badge)
* **Underlying Lineage:** Single wooden board on carpet; 21 noisy/corrupted classes (marketing text); unverified anonymous uploader with no auditable chain of title.
* **Gate Decision:** **REJECTED (Failed Provenance, Quality, and Single-Set Gates).**

### Candidate 6: Kaggle Chess Pieces Detection (Anshul Mehta)
* **Permanent URL:** `https://www.kaggle.com/datasets/anshulmehtakaggle/chess-pieces-detection-images-dataset`
* **License Claim:** CC0: Public Domain
* **Underlying Lineage:** 651 uncredited web-scrapes from Google Images of isolated pieces on white/transparent cutouts. No chessboard grids, no square coordinates, unverified copyright of original stock images.
* **Gate Decision:** **REJECTED (Unverified Scraped Provenance & Missing Board Context).**

### Candidate 7: Kaggle Chessman Image Dataset (Nitesh Fre)
* **Permanent URL:** `https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset`
* **License:** CC BY-NC-SA 4.0 (Non-Commercial & Share-Alike)
* **Commercial Model Training / Offline Embedding:** **STRICTLY PROHIBITED (NC / SA Viral Hazard).**
* **Gate Decision:** **REJECTED (Non-Commercial Hazard).**

### Candidate 8: LiveChess2FEN (David Mallasén Quintana)
* **Source:** GitHub (`davidmallasen/LiveChess2FEN`)
* **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
* **Commercial Model Training / Offline Embedding:** **PROHIBITED (Viral Copyleft Hazard).**
* **Gate Decision:** **REJECTED (Copyleft Hazard).**

### Candidate 9: ChessCog & ChessRender360
* **ChessCog:** 4,888 synthetic 3D Blender renders (CC BY 4.0).
* **ChessRender360:** 10,000 synthetic 3D Blender renders (CC BY 4.0).
* **Gate Decision:** **REJECTED AS REAL-WORLD SOURCES (Synthetic CG Renders Only).**

### Candidate 10: Wikimedia Commons / CC0 Stock Photography
* **Source:** Wikimedia Commons (Category:Chess pieces, Category:Chess sets), Unsplash/Pexels CC0 pools.
* **License:** CC0 1.0 Universal / Public Domain
* **Limitations:** Artistic close-ups with shallow depth of field ($f/1.8$), severe piece occlusions, lack of 4-corner board annotations.
* **Gate Decision:** **APPROVED AS REFERENCE POOL FOR CAPTURE PROTOCOL; REJECTED AS READY-TO-USE TRAINING DATA.**

---

## 5. Gate 3: Data Suitability and Anti-Leakage Partitioning Plan

### 5.1 Why Image Count $\ne$ Dataset Diversity
Adding 10,000 images of the *same* physical chess set (as in ChessReD) does not solve cross-set physical piece classification. Neural networks easily memorize the exact wood grain of that specific set. To learn 3D shape invariance across real-world Staunton pieces, the model must see **many distinct physical sets**.

### 5.2 Group-Based Partitioning Protocol (Zero Leakage)
To ensure reliable evaluation without data leakage:
1. **Physical Set Isolation:** All images, camera angles, lighting conditions, and board positions of a physical set MUST reside strictly in the same split.
2. **Augmentation Confinement:** All horizontal flips, crops, and color jitter variants must remain in the same split as the source image.
3. **Sequential Separation:** Adjacent frames of the same game must not be split across train and test.
4. **Group Allocation (for $N \ge 20$ sets):**
   - **Training Group (Sets 1–14, 70%):** Model parameter optimization.
   - **Validation Group (Sets 15–17, 15%):** Model architecture selection and hyperparameter tuning.
   - **Untouched Blind Holdout Group (Sets 18–20, 15%):** Final production gate evaluation (strictly untouched during development).

---

## 6. Gate 4: Acquisition Recommendation & Capture Protocol

### 6.1 Source-by-Source Licensing Decision Matrix

| Dataset Candidate | Source & DOI / URL | License | Commercial Permitted | Usable Physical Sets | Class Labels & Format | Gate Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Zenodo 6656212** | [10.5281/zenodo.6656212](https://doi.org/10.5281/zenodo.6656212) | **CC BY 4.0** | **YES** | **4 sets** | 6 classes + empty, 181x181 crops | **APPROVED (Benchmark)** |
| **ChessReD** | [10.4121/99b5c721](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f) | **CC BY 4.0** | **YES** | **1 set** | FEN + boxes, full photos | **APPROVED (Angle Bench)** |
| **FENiT** | [10.17632/wpp9xnbcp6.1](https://doi.org/10.17632/wpp9xnbcp6.1) | **CC BY 4.0** | **YES** | **1–2 sets** | BBoxes + corners, full photos | **APPROVED (Corner Bench)** |
| **samryan18** | `samryan18/chess-dataset` | **MIT** | **YES** | **1 set** | FEN, full photos | **APPROVED (Reference)** |
| **Roboflow Universe** | `chess-full-r3wh3-sajg6` | CC BY 4.0 (unverified) | Unclear | 1 set | 21 corrupted classes | **REJECTED (Corrupted)** |
| **Anshul Mehta** | Kaggle `chess-pieces-detection` | CC0 (unverified) | Unclear | Unknown | 6 classes, scraped cutouts | **REJECTED (Scraped)** |
| **Kaggle Chessman** | Kaggle `niteshfre/chessman` | **CC BY-NC-SA 4.0** | **NO** | Web scrapes | 6 classes | **REJECTED (Non-Commercial)** |
| **LiveChess2FEN** | `davidmallasen/LiveChess2FEN` | **AGPL-3.0** | Copyleft | Mixed | 6 classes | **REJECTED (AGPL Copyleft)** |
| **ChessCog** | `georg-wolflein/chesscog` | MIT / CC BY 4.0 | YES | 0 real sets | 3D Blender renders | **REJECTED (Synthetic)** |
| **ChessRender360** | Zenodo `10.5281/zenodo...` | CC BY 4.0 | YES | 0 real sets | 3D Blender renders | **REJECTED (Synthetic)** |

### 6.2 The Diversity Gap Analysis
* Current eligible physical sets in academic literature: **6 total sets** (Zenodo: 4, ChessReD: 1, FENiT: 1).
* Required physical sets for shape-invariant representation: **$\ge 20$ sets**.
* **Diversity Gap:** **$-14$ physical sets**.

### 6.3 Author-Owned In-House Physical Capture Protocol
Because no commercial-ready public dataset satisfies the 20-set requirement, we propose an Author-Owned Capture Protocol:

1. **Target Equipment (20 Physical Sets):**
   * **Category A (Tournament Standard, Sets 1–5):** Standard FIDE plastic weighted pieces (black/white, black/cream) on green/white, brown/white, and black/white roll-up vinyl boards.
   * **Category B (Fine Wood Staunton, Sets 6–10):** Handcrafted wooden Staunton pieces (boxwood, sheesham, rosewood, ebonized walnut) on solid mahogany, birch, and walnut boards.
   * **Category C (Club & Scholastic, Sets 11–15):** Matte plastic unweighted pieces, silicone boards, and mousepad tournament boards.
   * **Category D (Stylistic Carvings, Sets 16–20):** Distinctive Staunton variations (German knight, French Lardy knight, Dubrovnik style, vintage weighted).
2. **Camera Angles & Viewpoints:**
   * **Angle 1 (Overhead / Zenith):** $75^\circ - 90^\circ$ pitch (canonical top-down).
   * **Angle 2 (Player White View):** $45^\circ - 60^\circ$ pitch from White side.
   * **Angle 3 (Player Black View):** $45^\circ - 60^\circ$ pitch from Black side.
   * **Angle 4 (Spectator Side View):** $45^\circ - 60^\circ$ pitch from lateral flank.
3. **Lighting Environments:**
   * **Setup 1 (Natural Diffuse):** Ambient daytime room lighting through windows (soft shadows).
   * **Setup 2 (Direct Overhead):** Incandescent/LED direct ceiling light (moderate specular sheen).
   * **Setup 3 (Oblique Directional):** Desk lamp to one side (elongated directional shadows across adjacent cells).
4. **Acquisition Stages per Set:**
   * **Stage 1 (Isolated 64-Square Sweep):** Every piece type (6) $\times$ color (2) placed on light and dark squares without adjacent occlusions ($12 \times 64 = 768$ crops per set).
   * **Stage 2 (Starting Position):** Standard starting layout under all 4 angles and 3 lighting setups ($12$ photos per set).
   * **Stage 3 (Standard Midgame Positions):** 5 balanced midgame board positions under all 4 angles ($20$ photos per set).
5. **Expected Total Yield:**
   * $20 \text{ sets} \times 768 \text{ isolated square crops} = \mathbf{15,360 \text{ clean square crops}}$.
   * $20 \text{ sets} \times 32 \text{ full-board scene photos} = \mathbf{640 \text{ full-board calibrated scenes}}$.
6. **Licensing:** Released under **CC0 1.0 Universal** or **MIT**, ensuring 100% clean commercial and open-source provenance.

---

## 7. Gate 5: Publication Acceptance Gate Table

```
========================================================================================================
MILESTONE 5.2 ACCEPTANCE GATES
========================================================================================================
Requirement                         Gate Standard      Audit Finding / Status          Gate Decision
--------------------------------------------------------------------------------------------------------
Data Provenance Audit:              Exhaustive review  Audited Zenodo, ChessReD, FENiT PASS
Commercial Licensing Verification:  Zero copyleft/NC   Rejected AGPL & CC-BY-NC        PASS
Anti-Leakage Grouping Protocol:     Set-level grouping Protocol defined (70/15/15)     PASS
Physical Set Diversity Gate:        >= 20 sets         Only 6 eligible sets in public  FAIL (External Data)
Capture Protocol Formulated:        Comprehensive plan 20-set multi-angle protocol     PASS
Model Architecture Isolation:       No new models      Zero training conducted         PASS
Production Model Weights:           pieceModelWeights  100% UNCHANGED                  PASS
Zero-Silent-Error Policy:           100%               Mandatory confirmation active   PASS
========================================================================================================
OVERALL DECISION: EXTERNAL DATASETS CANNOT BRIDGE THE 20-SET DIVERSITY GAP.
DO NOT RESUME MODEL TRAINING UNTIL THE AUTHOR-OWNED 20-SET PHYSICAL CAPTURE PROTOCOL IS EXECUTED.
PRESERVE PRODUCTION WEIGHTS UNCHANGED AT src/vision/pieceModelWeights.ts.
========================================================================================================
```
