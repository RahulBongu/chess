# Milestone 5.2J — External Benchmark Integrity and Ground-Truth Audit

**Milestone:** 5.2J  
**Date:** 2026-11-02  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `a0def87` (verified synchronized with `origin/main`)  
**Production Model Weights:** `src/vision/pieceModelWeights.ts` (version `1.0.0-synthetic-staunton-v1`, strictly **UNCHANGED**)  
**Preserved Release Artifacts:** All 35 files under `dist/` intact; `modules/live-camera/android/build/` untouched.  
**Physical Capture Status:** Set 01 physical capture remains explicitly **PENDING**; external datasets do not substitute for in-house studio capture.

---

## 1. Executive Summary & Audit Scope

Milestone 5.2J executes an in-depth evaluation-integrity and ground-truth audit of the external benchmark results produced in Milestone 5.2I. The objective is to verify that the benchmark accurately distinguishes image processing, board orientation, square assignment, and piece classification, and to eliminate any risk of presenting oracle-assisted scores as autonomous production accuracy.

### Key Audit Findings
1. **Oracle-Use Disclosure (Gate 1):**
   - The reported **71.82% FEN placement match** was derived using an **Oracle Orientation Search**. Specifically, the adapter tested all 4 cyclic shifts of the board corners and selected the shift maximizing matches against the filename FEN ground truth.
   - Ground truth was thus used to choose the orientation before computing the placement score.
   - When orientation is determined **without the FEN oracle** using canonical screen-space geometry (Mode 2), the mean placement match drops from **71.82%** to **52.62%** (-19.20 pp), with image-driven orientation agreeing with the oracle in only **71.17%** of scenes.
   - The 71.82% metric is formally classified as an **oracle-assisted YOLO-to-FEN annotation alignment score**, NOT autonomous full-board detection accuracy.
2. **Annotation Quality & Lineage Validation (Gate 2):**
   - FENiT annotations were generated using CVAT with a pre-trained base model (`yolov8l-pose.pt`), as documented in the Mendeley metadata. They are model-assisted annotations with varying fidelity, not 100% human-verified ground truth.
   - **Corner Geometry:** 435 of 436 board annotations form strictly convex quadrilaterals; exactly 1 image (`2b5-4kp2-p6p-5P2-PP1Np1p1-6P1-4K2P-8.jpg`) contains self-intersecting/bowtie corner annotations and was correctly rejected by geometric validation. 54 images lack board annotations entirely.
   - **Piece Count Mismatches:** 26 of 385 processed images (6.75%) have piece count discrepancies between the YOLO annotations and the filename FEN (missing or spurious bounding boxes).
   - **Piece Class Confusions:** 20 of 385 processed images (5.19%) have identical piece counts but misclassified piece types or colors in the YOLO annotations (e.g., all 6 white pawns labeled as black pawns in `2rqr1k1-pp2bp2-2b2n1p-3pN1p1-3P3B-1QN5-PP3PPP-R4RK1.jpg`).
   - **Exact Multiset Agreement:** 339 of 385 images (88.05%) show exact agreement in piece type and color multiset between YOLO labels and filename FEN.
3. **Metric Clarifications (Gate 3):**
   - **FEN Placement Match (71.82%):** Arithmetic mean across 385 processed boards of the ratio $(\text{matched pieces} / \text{annotated piece count})$. Global aggregate ratio is $3,962 / 5,292 = 74.87\%$. Board-level exact placement (100% match) was achieved on **75 of 385 boards (19.48%)**.
   - **Piece Classifier Accuracy (11.00%):** Evaluated on 5,302 occupied squares across 385 oracle-aligned boards. Top-1 accuracy is $583 / 5,302 = \mathbf{10.9959\% \approx 11.00\%}$. Abstentions occurred on 1,703 squares (32.12%); coverage-filtered accuracy on accepted predictions is $583 / 3,599 = \mathbf{16.20\%}$.
   - **Zenodo Holdout Status:** The 16.80% Top-1 result on Zenodo Set 4 is explicitly identified as a historical benchmark on a previously inspected holdout from Milestone 4.0 (Phase 4.3D).
4. **Physical Set Identification (Gate 4):**
   - FENiT metadata does not catalogue or track physical chess sets across scenes. The distinct physical-set count is formally marked **UNVERIFIED**.

---

## 2. Dataset Acquisition, Provenance & Checksums

All external dataset assets and extraction caches reside exclusively in `C:\dev\chess-capture-workspace\external\`:

```
C:\dev\chess-capture-workspace\external\
├── fenit\
│   ├── raw\                  # 491 images (.jpg/.jpeg), 490 .txt annotations, data.yaml, Train.txt
│   ├── metadata\             # dataset.json, files.json (Mendeley API)
│   ├── benchmark_output\     # rectified 800x800 boards and 24,640 square crops
│   ├── reports\              # external_benchmark_manifest.jsonl, annotation_quality_audit.json
│   └── README.md             # provenance, CC BY 4.0 license, and attribution
└── zenodo\
    ├── archive\              # chess_pieces.zip (153.6 MB)
    ├── metadata\             # record.json (Zenodo API)
    ├── extracted\            # 3,200 PNG images across train/ and valid/
    └── README.md             # provenance, CC BY 4.0 license, and attribution
```

### Dataset Inventory & Checksum Table
| Dataset | Source URL | Version | License | Archive / File Count | Size | Checksum Verification |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **FENiT** | [Mendeley wpp9xnbcp6/1](https://data.mendeley.com/datasets/wpp9xnbcp6/1) | v1 (2026) | CC BY 4.0 | 983 files | 3.41 GB | 100% of files SHA-256 verified against Mendeley API |
| **Zenodo** | [Zenodo 6656212](https://zenodo.org/records/6656212) | v1.0.0 (2022) | CC BY 4.0 | 3,200 PNGs in zip | 153.6 MB | MD5: `4583133a23383c05c982dd4ddd549068`<br>SHA-256: `7C590F9C4E347526B6F98A57AC24323DB3CB2D379E0BDF1674E115AF47478CA3` |
| **Sam Ryan** | [GitHub samryan18](https://github.com/samryan18/chess-dataset) | master | MIT (repo) | 500 images | ~2.25 GB | **EXCLUDED:** Lack of photographic rights scope; zero board corner keypoints or piece bounding boxes |

---

## 3. Evaluation Architecture & Oracle-Use Audit (Gate 1)

### 3.1 Trace of Ground-Truth Label Influence
In Milestone 5.2I, the filename FEN influenced the processing pipeline at several critical junctions:

```
Raw Image + Annotation (.txt)
         │
         ▼
[Corner Convexity Check] ──> (Rejects bowtie quads; purely geometric)
         │
         ▼
[Orientation Selection] ──> ⚠️ USES FILENAME FEN ORACLE:
                             Tests shifts 0, 1, 2, 3 against FEN piece occupancy;
                             Picks shift with highest match rate.
         │
         ▼
[Corner Ordering & Homography H] ──> Influenced by Oracle shift.
         │
         ▼
[Square Assignment (a1..h8)] ──> Aligned via Oracle homography H.
         │
         ▼
[Placement Match Calculation] ──> Evaluates matched pieces under Oracle shift.
         │
         ▼
[Piece Classifier Evaluation] ──> Crops sliced from Oracle-rectified board.
```

### 3.2 Formal Evaluation Modes
To eliminate confounding oracle assistance with autonomous capability, the benchmark adapter implements distinct evaluation modes:

1. **Mode 1: Ground-Truth-Assisted Crop Evaluation (Oracle Orientation Mode)**
   - *Procedure:* Uses annotated board corners and tests all 4 cyclic shifts against the filename FEN to select the optimal rotation.
   - *Purpose:* Evaluates piece classification in isolation on well-aligned crops, bypassing orientation ambiguity.
   - *Status:* Oracle-assisted; not an autonomous vision capability claim.
2. **Mode 2: Image-Driven Geometry/Orientation Evaluation (Canonical Screen-Space Mode)**
   - *Procedure:* Uses annotated board corners but orders them strictly by image-plane screen geometry: top-left (min $x+y$), top-right (max $x-y$), bottom-right (max $x+y$), bottom-left (min $x-y$). Never reads the filename FEN.
   - *Purpose:* Evaluates how well standard geometric corner ordering aligns the board without oracle knowledge.
   - *Status:* Autonomous geometry; oracle-free.
3. **Mode 3: End-to-End Evaluation (Full Autonomous Pipeline)**
   - *Procedure:* Raw image is fed into the runtime contour-based board detector (`detectBoard.ts`), followed by parity assessment, square slicing, and classification.
   - *Findings:* Classical contour detection achieves **0% detection** on unprocessed FENiT images due to tile grout lines, reflections, and perspective foreshortening. The mobile app relies on manual corner calibration and preview guides for uncalibrated raw scenes.

### 3.3 Comparative Performance: Mode 1 vs Mode 2 across 385 FENiT Boards

| Metric | Mode 1 (Oracle Orientation) | Mode 2 (Image-Driven Screen Space) | Delta |
| :--- | :---: | :---: | :---: |
| **Mean Placement Match Score** | **71.82%** | **52.62%** | -19.20 pp |
| **Exact 100% Placement Matches** | **75 / 385 (19.48%)** | **48 / 385 (12.47%)** | -7.01 pp |
| **Agreement with Oracle Orientation** | 385 / 385 (100.0%) | 274 / 385 (71.17%) | -28.83 pp |
| **FEN Oracle Access** | Yes (tested all 4 rotations) | **No (pure image geometry)** | Independent |

*Conclusion:* In 28.83% of images (111 boards), the camera was positioned at a lateral flank or from the opposing player's perspective. Without the FEN oracle, screen-space ordering misaligns the board axes by 90°, 180°, or 270°.

---

## 4. Annotation Quality & Ground-Truth Integrity Audit (Gate 2)

An exhaustive audit of all 491 FENiT images and annotations was conducted via `scripts/audit_external_annotations.py`.

### 4.1 Annotation Lineage & Model-Generated Artifacts
The Mendeley Data publication notes that raw source images were labeled using CVAT with a pre-trained `yolov8l-pose.pt` model to infer preliminary bounding box boundaries and 4-point board keypoint geometries.

The audit confirmed that while the dataset provides a valuable real-image benchmark, the annotations contain model-generated errors and omissions:

```
Total Downloaded Images: 491
├── Eligible Full-Board Scenes with FEN: 393 (80.04%)
│   ├── With Valid Board Annotations: 386 (78.62%)
│   │   ├── Strictly Convex Quadrilaterals: 385 (78.41%) -> PROCESSED
│   │   └── Non-Convex / Bowtie Quadrilaterals: 1 (0.20%) -> FAILED (2b5-4kp2-p6p-5P2-PP1Np1p1-6P1-4K2P-8.jpg)
│   └── Missing Board Annotations: 7 (1.43%) -> SKIPPED
└── Ineligible Candidate Images: 98 (19.96%)
    ├── Timestamp-Named (No FEN ground truth): 86 (17.52%) -> SKIPPED
    └── Single-Piece Reference Photos: 6 (1.22%) -> SKIPPED
    └── Unparseable Placement Ground Truth: 6 (1.22%) -> SKIPPED
```

### 4.2 Piece Annotation vs Filename FEN Consistency (385 Processed Boards)
Auditing the YOLO `.txt` annotations against the filename FEN revealed:

| Audit Category | Count | Percentage | Description / Impact |
| :--- | :---: | :---: | :--- |
| **Exact Piece Count Agreement** | **359** | **93.25%** | YOLO annotation count exactly matches FEN piece count. |
| **Piece Count Mismatches** | **26** | **6.75%** | YOLO model missed pieces or generated spurious detections (e.g. 18 annotated vs 20 in FEN). |
| **Exact Class Multiset Agreement** | **339** | **88.05%** | Exact agreement across all 12 piece class frequencies. |
| **Class Misclassifications** | **20** | **5.19%** | Correct total piece count, but YOLO model confused piece types or colors (e.g. White Pawns labeled as Black Pawns). |
| **Severe Alignment Discrepancies (<50%)** | **61** | **15.84%** | Steep perspective elevated piece bounding box centers, projecting them outside their contact squares. |

### 4.3 Verified Geometric Failure Detail
* **File:** `2b5-4kp2-p6p-5P2-PP1Np1p1-6P1-4K2P-8.jpg`
* **Annotated Board Keypoints:**
  `kpt0=(0.9727, 0.5412), kpt1=(0.0011, 0.3714), kpt2=(0.4233, 0.8800), kpt3=(0.5510, 0.2559)`
* **Geometric Fault:** The polygon traverses from right to left, then bottom, then top, crossing over itself (bowtie quadrilateral). The adapter's strict convexity check detected this defect and rejected the image (`non_convex_board_corners`), preventing corrupt homography generation.

---

## 5. Metric Clarifications & Detailed Reproductions (Gate 3)

### 5.1 FEN Placement Match Score (71.82%)
* **Exact Definition:** The arithmetic mean of per-board annotation alignment scores under oracle-selected 4-way orientation.
* **Formula for Board $i$:**
  $$\text{Score}_i = \frac{\text{Count of YOLO boxes whose mapped center lands on matching FEN square}}{\text{Total YOLO piece boxes in board } i}$$
* **Overall Score:**
  $$\text{Mean Score} = \frac{1}{385} \sum_{i=1}^{385} \text{Score}_i = \mathbf{71.8230\% \approx 71.82\%}$$
* **Global Ratio:**
  $$\frac{\sum \text{Matched Pieces}}{\sum \text{Annotated Pieces}} = \frac{3,962}{5,292} = \mathbf{74.8677\% \approx 74.87\%}$$
* **Board-Level Exact Placement (All pieces land on correct squares):**
  $$\text{Exact Boards} = 75 / 385 = \mathbf{19.48\%}$$

### 5.2 Piece Classifier Performance (11.00%)
* **Scope:** Deployed synthetic Staunton HOG model (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, linear 288-bin classifier with temperature scaling).
* **Sample Size:** 5,302 occupied square crops derived from 385 oracle-aligned boards.
* **Top-1 Accuracy:**
  $$\frac{583 \text{ correct}}{5,302 \text{ occupied squares}} = \mathbf{10.9959\% \approx 11.00\%}$$
* **Abstention Accounting:**
  - Production thresholds: Confidence $< 0.60$ or Top-1/Top-2 Margin $< 0.20$.
  - Abstained predictions: 1,703 / 5,302 (**32.12%**).
  - Accepted predictions: 3,599 / 5,302 (**67.88%**).
  - Coverage-filtered accuracy: $583 / 3,599 = \mathbf{16.20\%}$.

### 5.3 Zenodo Record 6656212 Re-Verification & Historical Status
The evaluation on Zenodo Set 4 was re-verified and is explicitly disclosed as a historical evaluation on a previously inspected holdout:
* **Set 4 (`valid/` — 768 pieces):** Top-1 = **16.80%** (129/768), Top-2 = **33.85%** (260/768), Abstention Rate = **16.54%** (127/768).
* **Sets 1–3 (`train/` — 2,304 pieces):** Top-1 = **16.32%** (376/2,304), Top-2 = **31.64%** (729/2,304), Abstention Rate = **36.28%** (836/2,304).

---

## 6. External-Set Limitations & Physical Identity Status (Gate 4)

### 6.1 Unverified Distinct Physical-Set Count
- Neither the Mendeley Data metadata, the publication abstract, `data.yaml`, nor the YOLO annotation files track distinct physical chess sets across scenes.
- While the authors state that chessboards were captured under varied lighting and angles, it is impossible to verify whether the 385 boards represent 1 physical chess set, 3 sets, or 10 sets.
- **Formal Status:** The distinct physical-set count of FENiT is marked **UNVERIFIED**.

### 6.2 Partition Isolation Invariant
- FENiT images and crops remain strictly excluded from in-house training, validation, and holdout partitions (`SET01`–`SET20`).
- No fictitious physical IDs (`SET01_W_P01`) or physical tag IDs (`S01-W-P01`) were created. All records retain their original dataset source IDs and observation-only tags.

---

## 7. Verification and Publication (Gate 5)

### 7.1 Test Suites and Typecheck
1. **Python Unit & Adapter Tests:**
   ```bash
   npm run test:python
   # py -m unittest tests.test_capture_pipeline -v
   ```
   *Result:* 12/12 tests pass in 3.95 s (includes new tests verifying Mode 1 vs Mode 2 disaggregation).
2. **JavaScript / TypeScript Test Suite:**
   ```bash
   npm test
   ```
   *Result:* 186/186 tests pass across 13 test suites in 12.2 s.
3. **TypeScript Typecheck:**
   ```bash
   npm run typecheck
   # tsc --noEmit
   ```
   *Result:* 0 errors.

### 7.2 Working Tree Audit
* All 35 `dist/` files preserved bit-for-bit.
* `modules/live-camera/android/build/` untouched.
* `src/vision/pieceModelWeights.ts` unchanged.
* Zero external dataset files, raw images, or crops staged in Git.

---

## 8. Gate Table

| Gate | Requirement | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Gate 0** | Preservation | **PASS** | 35 `dist/` files intact; `modules/live-camera/android/build/` untouched; production weights unchanged; no APK built. |
| **Gate 1** | Orientation & Label Use Audit | **PASS** | Traced FEN oracle influence; implemented and benchmarked Mode 1 (Oracle: 71.82%) vs Mode 2 (Image-driven: 52.62%); disclosed that Mode 1 is not autonomous. |
| **Gate 2** | Annotation Quality Validation | **PASS** | Audited all 491 images; identified 1 non-convex corner failure; verified 26 count mismatches and 20 class misclassifications; documented CVAT `yolov8l-pose.pt` origin. |
| **Gate 3** | Metric Clarifications | **PASS** | Mathematically defined 71.82% placement match and 11.00% piece classifier accuracy (with abstention accounting); clarified Zenodo Set 4 as historical inspected holdout. |
| **Gate 4** | External-Set Limitations | **PASS** | FENiT distinct physical-set count formally marked UNVERIFIED; partition isolation from in-house `SET01`–`SET20` verified; zero fake physical IDs emitted. |
| **Gate 5** | Verification & Publication | **PASS** | 186/186 Node tests pass; 12/12 Python tests pass; `tsc --noEmit` clean; documentation updated; zero dataset artifacts in Git. |

---

## 9. Next Recommended Action

With the evaluation integrity of external benchmarks fully audited, documented, and disaggregated from oracle assistance:
1. Review git status and stage code/documentation changes.
2. Commit and push Milestone 5.2J to `origin/main` under `AGENTS.md`.
3. Proceed to live physical capture preparation for Set 01 when physical hardware and studio setup become available.
