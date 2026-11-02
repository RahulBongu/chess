# Milestone 5.2I — External Real-Image Dataset Benchmark and Vision Pipeline Smoke Test

**Milestone:** 5.2I  
**Date:** 2026-11-02  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `7885e3d` (verified synchronized with `origin/main`)  
**Production Model Weights:** `src/vision/pieceModelWeights.ts` (version `1.0.0-synthetic-staunton-v1`, strictly **UNCHANGED**)  
**Preserved Release Artifacts:** All 35 files under `dist/` intact; `modules/live-camera/android/build/` untouched.  
**Physical Capture Status:** Set 01 physical capture remains explicitly **PENDING**; public datasets are an external benchmark, not a substitute for the 20-set in-house capture campaign.

---

## 1. Executive Summary

Milestone 5.2I advances vision-pipeline development without requiring physical chessboard capture by acquiring eligible public real-image chess datasets outside the repository, implementing an external benchmark adapter, executing an end-to-end smoke test on real source photographs, and reporting all performance metrics and limitations honestly.

Key results:
1. **Datasets Acquired Outside Git (`C:\dev\chess-capture-workspace\external\`):**
   - **FENiT (Primary Candidate):** Mendeley Data DOI `10.17632/wpp9xnbcp6.1` (v1, CC BY 4.0, Harjanto et al., Binus University). Downloaded and SHA-256 verified all 983 files (3,408,014,513 bytes), including 491 full-board real photographs (5712×4284 and 4000×3000), 490 YOLO pose annotations, `data.yaml`, and `Train.txt`.
   - **Zenodo 6656212 (Piece Benchmark):** DOI `10.5281/zenodo.6656212` (v1.0.0, CC BY 4.0, Campos & Berlanga, UC3M). Downloaded and verified archive `chess_pieces.zip` (153,649,683 bytes; MD5 `4583133a23383c05c982dd4ddd549068`, SHA-256 `7C590F9C4E347526B6F98A57AC24323DB3CB2D379E0BDF1674E115AF47478CA3`), extracted to 3,200 PNG images across 4 physical chess sets.
   - **Sam Ryan (Secondary Candidate Evaluated & Excluded):** 500 images (~2.25 GB). Repository declares MIT software license, but photographs are credited to two GitHub users (`@mukund-v`, `@samryan18`) without express photographic copyright assignment or CC license. Crucially, the dataset provides no board corner keypoints or piece bounding boxes; using it would require fabricating corner ground truth, violating Gate 2 rules. Documented reasoned exclusion.
2. **External Benchmark Adapter Implemented (`scripts/external_benchmark_adapter.py`):**
   - Enforces an **Observation-Only schema**. External records never emit `SETxx` physical piece IDs, physical tag IDs (`Sxx-W-Pxx`), or physical-set identities.
   - External data and derived crops are kept completely outside Git and strictly isolated from the in-house `SET01`–`SET20` training, validation, and holdout partitions.
   - Detects and rejects non-convex/bowtie board corners; resolves physical viewing perspective (`white_perspective`, `black_perspective`, `flank_left`, `flank_right`) via cyclic homography matching against filename FEN ground truth; rectifies boards to canonical 800×800 RGB/BGR; and extracts 64 100×100 square crops per board.
3. **Smoke Test Execution:**
   - **Representative Subset (8 images):** 5 full boards processed, 3 cleanly skipped (2 single-piece photos, 1 timestamp image), 0 geometric failures, 320 square crops emitted.
   - **Full FENiT Ingestion (491 images):** 385 full-board scenes successfully processed, 105 skipped with explicit reasons (86 timestamp-named without FEN, 7 missing board corner line, 6 unparseable FEN, 6 single-piece photos), 1 geometric failure (`2b5-4kp2-p6p-5P2-PP1Np1p1-6P1-4K2P-8.jpg` has crossed/bowtie corner annotations). Total 24,640 derived square crops extracted outside Git.
   - **Deployed HOG Classifier Smoke Test:** 11.00% accuracy on FENiT perspective piece crops (583/5,302), confirming severe domain shift on perspective images.
   - **Zenodo Piece Classifier Re-Verification:** 16.80% Top-1 on held-out Set 4 (129/768), 16.32% on Sets 1–3 (376/2,304).
   - **Synthetic Regression:** 186/186 TypeScript tests pass; 11/11 Python pipeline tests pass; `tsc --noEmit` clean.
4. **Honest Limitations:**
   - The 88% real-piece accuracy target is **NOT MET** by the synthetic model on real images (11.00% on FENiT, 16.80% on Zenodo).
   - Android frame latency remains **UNVERIFIED** on physical devices.
   - Set 01 physical capture remains **PENDING**; zero physical Set 01 photographs exist.

---

## 2. Dataset Acquisition, Provenance & License Verification (Gate 1)

All datasets were downloaded and stored exclusively in `C:\dev\chess-capture-workspace\external\`, keeping Git completely free of raw images, archives, crops, and external manifests.

### 2.1 Primary Candidate: FENiT (Mendeley Data)
* **Title:** *Annotated Image Dataset for Fenit: Automatic Chessboard Digitization and FEN Generation*
* **DOI:** [10.17632/wpp9xnbcp6.1](https://doi.org/10.17632/wpp9xnbcp6.1)
* **Authors:** Theo Harjanto, Vincent Oei, Dicky Susanto (Binus University, Jakarta)
* **Publication Date:** 14 April 2026 (Version 1)
* **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)
* **Attribution Notice:**
  > Harjanto, T., Oei, V., & Susanto, D. (2026). *Annotated Image Dataset for Fenit: Automatic Chessboard Digitization and FEN Generation* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/wpp9xnbcp6.1
* **Acquisition Method:** Downloaded all 983 files using concurrent SHA-256 verifying script (`download_mendeley.py`).
* **Archive Inventory:**
  - Total files: 983
  - Images: 491 files (340 `.jpg`, 151 `.jpeg`), dimensions 5712×4284 (high resolution) and 4000×3000 / 4032×3024 / 4096×3072
  - Annotations: 490 YOLO pose `.txt` files + `data.yaml` + `Train.txt` (490 lines)
  - Total downloaded size: 3,408,014,513 bytes
  - Hash verification: 100% of files matched the official Mendeley API SHA-256 hashes bit-for-bit.

### 2.2 Piece Benchmark: Zenodo Record 6656212
* **Title:** *Chess piece dataset for image classification*
* **DOI:** [10.5281/zenodo.6656212](https://doi.org/10.5281/zenodo.6656212)
* **Authors:** Pablo Tomas Campos, Antonio Berlanga (Supervisor), Universidad Carlos III de Madrid (UC3M)
* **Publication Date:** 22 June 2022 (Version v1.0.0)
* **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)
* **Attribution Notice:**
  > Campos, P. T., & Berlanga, A. (2022). *Chess piece dataset for image classification* (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6656212
* **Archive Integrity:**
  - Preserved archive: `C:\dev\chess-capture-workspace\external\zenodo\archive\chess_pieces.zip`
  - Size: 153,649,683 bytes
  - MD5: `4583133a23383c05c982dd4ddd549068` (verified against Zenodo record)
  - SHA-256: `7C590F9C4E347526B6F98A57AC24323DB3CB2D379E0BDF1674E115AF47478CA3`
* **Extracted Contents (`C:\dev\chess-capture-workspace\external\zenodo\extracted\`):**
  - Total images: 3,200 PNG images ($181\times 181$)
  - `train/`: 2,368 images across Sets 1, 2, 3 (192 per piece class + 64 empty)
  - `valid/`: 832 images from Set 4 (64 per piece class + 64 empty)

### 2.3 Secondary Candidate: Sam Ryan (`samryan18/chess-dataset`) — Reasoned Exclusion
* **Source:** `https://github.com/samryan18/chess-dataset`
* **Volume:** 500 images (~2.25 GB)
* **License Analysis:** The repository contains an MIT `LICENSE.md`, which is a standard software copyright license covering code. However, the README states:
  > *Photos by @mukund-v and @samryan18.*
  The repository does not contain a photographer release, model release, or Creative Commons deed granting commercial reuse of the photographic images themselves.
* **Technical Blocker:** The dataset contains only full-board photographs and warped heuristic outputs. It provides **zero board corner keypoints** and **zero piece bounding box annotations**.
* **Gate 2 Decision:** **EXCLUDED**. Utilizing Sam Ryan would require inventing or heuristically guessing board corners and piece positions, directly violating Gate 2 ("Do not fabricate unsupported board corners, boxes, or ground truth"). FENiT already provides verified CC BY 4.0 full-board photographs with explicit 4-point corner keypoints and multi-class piece bounding boxes.

---

## 3. External Benchmark Adapter Architecture (Gate 2)

### 3.1 Partition and Identity Separation Invariants
The external adapter enforces strict isolation between external benchmark data and the planned in-house capture campaign:
1. **No Fabricated Physical IDs:** External records never assign `SETxx` identifiers, canonical piece instance IDs (`SET01_W_P01`), or physical tag IDs (`S01-W-P01`).
2. **Observation-Only Schema:** Crops are tagged with `crop_type: "external_observation"`, `is_external: true`, square coordinate (`a1`..`h8`), and FEN-derived piece class/color.
3. **Partition Isolation:** Outputs are stored under `C:\dev\chess-capture-workspace\external\fenit\benchmark_output\`, never in `crops/train/SET01` or any partition reserved for in-house sets `SET01`–`SET20`.
4. **Validation Policy:** The strict in-house validator (`validate_capture_manifest.py`) is reserved for physical capture manifests requiring 1-to-1 physical piece tracking. External data uses its own schema validator.

### 3.2 Annotation & Geometry Verification Pipeline
The adapter executes five sequential stages for each candidate image:
1. **Decode & Dimension Validation:** Verifies image load, 3-channel depth, and minimum resolution ($\ge 500\times 500$).
2. **Annotation Parsing & Ineligibility Classification:**
   - Extracts Class 0 (Board) and Classes 1–12 (Pieces).
   - Classifies timestamp-only names (`^\d{8}_\d{6}$`) as `timestamp_named_no_ground_truth_fen`.
   - Classifies single-piece stems (`B.jpg`, `K.jpg`) as `isolated_piece_photo_no_board_grid`.
   - Classifies images without Class 0 as `missing_board_corner_annotations`.
3. **Convexity & Geometric Integrity Check:**
   - Evaluates consecutive edge cross-products around the 4 board keypoints.
   - Detects and rejects self-intersecting / bowtie quadrilaterals (`non_convex_board_corners`).
4. **Physical Orientation Resolution:**
   - Real photographs are captured from varying angles (White perspective, Black perspective, or side flanks).
   - In screen coordinates, the 4 YOLO board keypoints are consistently oriented clockwise.
   - The adapter tests all 4 cyclic permutations against ground-truth FEN piece occupancy, mapping piece box centers $(c_x, c_y)$ through the homography $H$ onto the 8×8 grid. The permutation maximizing piece class matches determines the physical viewing angle (`white_perspective`, `black_perspective`, `flank_left`, `flank_right`).
5. **Rectification & Crop Slicing:**
   - Warps source image with $H$ to canonical 800×800 RGB/BGR.
   - Slices into 64 $100\times 100$ square crops (`a1` through `h8`).
   - Evaluates the deployed HOG classifier (`classifyPiece.ts` / `pieceModelWeights.ts`) on each occupied square.

---

## 4. Smoke Test Execution and Results (Gate 3)

### 4.1 Representative Subset Smoke Test (8 Images)
Command:
```bash
py scripts/external_benchmark_adapter.py \
  --subset "1k1r3r_7p_p1p1Rp2_1pbp3b_5P1P_6PB_PPP5_RK6.jpeg" \
           "2b5-4kp2-p6p-2Np1P2-PP1Nn1p1-6P1-7P-4K3.jpg" \
           "2b5-5p2-p4k1p-5P2-PP1Np1p1-6P1-4K2P-8.jpg" \
           "8_8_8_8_8_8_8_8 (2).jpg" \
           "8_8_8_8_8_8_8_8(board).jpeg" \
           "20260219_222546.jpg" \
           "B.jpg" \
           "K.jpg"
```

Results:
* **Total Evaluated:** 8
* **Successfully Processed:** 5 full boards (320 crops)
* **Skipped / Ineligible:** 3
  - `isolated_piece_photo_no_board_grid`: 2 (`B.jpg`, `K.jpg`)
  - `timestamp_named_no_ground_truth_fen`: 1 (`20260219_222546.jpg`)
* **Geometric Failures:** 0
* **Execution Time:** 3.92 s

### 4.2 Full FENiT Smoke Test (491 Images)
Command:
```bash
py scripts/external_benchmark_adapter.py
```

Results Summary:
* **Total Images Evaluated:** 491
* **Successfully Processed:** 385 full-board scenes
* **Skipped / Ineligible:** 105
  - `timestamp_named_no_ground_truth_fen`: 86
  - `missing_board_corner_annotations`: 7
  - `unparseable_placement_ground_truth`: 6
  - `isolated_piece_photo_no_board_grid`: 6
* **Geometric Failures:** 1 (`2b5-4kp2-p6p-5P2-PP1Np1p1-6P1-4K2P-8.jpg` had crossed/bowtie corner annotations)
* **Total Derived Square Crops:** 24,640 (64 per board $\times$ 385 boards)
* **Average FEN Placement Match:** 71.82%
* **Resolved Orientations:** `flank_left`: 271, `flank_right`: 84, `white_perspective`: 27, `black_perspective`: 3
* **Deployed HOG Classifier Performance on FENiT Crops:**
  - Total Occupied Squares Evaluated: 5,302
  - Correct Predictions: 583
  - **Accuracy:** **11.00%** (583 / 5,302)
* **Run Time:** 286.91 s
* **Manifest Output:** `C:\dev\chess-capture-workspace\external\fenit\reports\external_benchmark_manifest.jsonl`

### 4.3 Zenodo 6656212 Piece Benchmark Re-Verification
Command:
```bash
py scripts/evaluate_real_benchmark.py --data-dir "C:\dev\chess-capture-workspace\external\zenodo\extracted"
```

Results Summary:
* **Held-Out Set 4 (`valid/` — 768 pieces):**
  - Top-1 Accuracy: **16.80%** (129/768)
  - Top-2 Accuracy: **33.85%** (260/768)
  - Abstention Rate: **16.54%** (127/768)
  - Workstation Extraction Latency: 12.39 ms / patch
  - Workstation Classifier Latency: 0.11 ms / patch
* **Sets 1–3 (`train/` — 2,304 pieces):**
  - Top-1 Accuracy: **16.32%** (376/2,304)
  - Top-2 Accuracy: **31.64%** (729/2,304)
  - Abstention Rate: **36.28%** (836/2,304)

---

## 5. Disaggregated Performance & Honest Limitations (Gate 4)

To prevent false claims or confusing digital regression with physical capability, results are strictly disaggregated across three separate testing regimes:

| Regime | Test Dataset | Sample Size | Top-1 Accuracy | Status / Limitations |
| :--- | :--- | :---: | :---: | :--- |
| **1. Real Full-Board Photos** | FENiT (Mendeley wpp9xnbcp6 v1) | 385 boards / 5,302 pieces | **11.00%** | Severe perspective angle, piece height distortion, edge-blur on tall pieces. |
| **2. Real Physical Pieces (Overhead)** | Zenodo 6656212 (Campos & Berlanga) | 768 pieces (Set 4 valid) | **16.80%** | Domain gap from synthetic silhouettes; knight bias dominates; diffuse lighting. |
| **3. Synthetic / Digital Regression** | Unit test suites & procedural shapes | 360 procedural / 186 unit tests | **99.72%** (359/360) / 100% tests pass | Evaluates software correctness, invariants, and mathematical repeatability only. |

### Explicit Declarations
1. **Set 01 Physical Capture Status: PENDING.**
   No physical photographs have been taken for Set 01. Public external datasets (FENiT, Zenodo) serve as validation smoke tests for the pipeline and benchmark tools; they do **not** replace the planned capture of 20 independently identified physical sets with underside QR/tag verification.
2. **Production Quality Target (88% Real-Piece Accuracy): NOT MET.**
   The deployed model (`1.0.0-synthetic-staunton-v1`) achieves 11.00% on FENiT and 16.80% on Zenodo. Model weights remain unchanged.
3. **Android Device Latency Target: UNVERIFIED.**
   Single-thread workstation Python timing is ~12.5 ms/patch, but on-device Android latency requires native frame profiling on physical hardware.

---

## 6. Verification and Publication (Gate 5)

### 6.1 Test Suites and Typecheck
1. **Python Capture Pipeline & External Adapter Tests:**
   ```bash
   npm run test:python
   # py -m unittest tests.test_capture_pipeline -v
   ```
   *Result:* 11/11 tests pass in 3.79 s.
2. **JavaScript / TypeScript Test Suite:**
   ```bash
   npm test
   ```
   *Result:* 186/186 tests pass across 13 test suites in 12.8 s.
3. **TypeScript Typecheck:**
   ```bash
   npm run typecheck
   # tsc --noEmit
   ```
   *Result:* 0 errors.

### 6.2 Git Cleanliness Check
* Tracked files modified/added:
  - `scripts/external_benchmark_adapter.py`
  - `tests/test_capture_pipeline.py`
  - `docs/EXTERNAL_DATASET_BENCHMARK.md`
  - `docs/PHYSICAL_CAPTURE_PROTOCOL.md`
* Working tree audit:
  - All 35 `dist/` files preserved bit-for-bit.
  - `modules/live-camera/android/build/` untouched.
  - No raw images, downloaded archives, extracted PNGs, or external manifests staged in Git.
  - Pre-existing untracked `modules/live-camera/android/build/` remains untracked.

---

## 7. Gate Table

| Gate | Requirement | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Gate 0** | Preservation | **PASS** | 35 `dist/` files intact; `modules/live-camera/android/build/` untouched; production weights unchanged; no APK built. |
| **Gate 1** | Candidate Dataset Acquisition | **PASS** | FENiT downloaded and SHA-256 verified (3.4 GB, 983 files); Zenodo verified (153.6 MB, 3,200 PNGs); Sam Ryan evaluated and excluded with reasoned analysis. |
| **Gate 2** | External Benchmark Adapter | **PASS** | `scripts/external_benchmark_adapter.py` built with observation-only schema, convex quadrilateral checking, orientation resolution, canonical 800×800 rectification, and no fake physical IDs. |
| **Gate 3** | Source Image Smoke Testing | **PASS** | Tested 8-image subset (5 processed, 3 skipped) and 491-image full dataset (385 processed, 105 skipped with explicit reasons, 1 geometric failure identified). |
| **Gate 4** | Honest Limitation Reporting | **PASS** | Disaggregated results across real full-board, physical individual-piece, and synthetic regression. Set 01 declared PENDING; 88% accuracy and Android latency declared not met / unverified. |
| **Gate 5** | Verification & Publication | **PASS** | 186/186 Node tests pass; 11/11 Python tests pass; `tsc --noEmit` clean; no dataset artifacts staged; documentation updated. |

---

## 8. Next Recommended Action

With external real-image datasets acquired and the external benchmark adapter verified, the vision pipeline has a repeatable real-image smoke test framework. The next recommended action is:
1. Stage and commit Milestone 5.2I code and documentation changes.
2. Push commit to `origin/main` under `AGENTS.md` guidelines.
3. Proceed to live physical capture preparation for Set 01 when the physical studio setup (board, pieces, tags, lighting, tripod) becomes physically available.
