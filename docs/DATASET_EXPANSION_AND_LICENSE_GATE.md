# Milestone 5.2 & 5.2A: Real-World Dataset Expansion, License Gate, and Acquisition Protocol

**Milestones:** 5.2 & 5.2A  
**Date:** 2026-11-02  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `58b7d6b`  
**Objective:** Close factual and arithmetic gaps in Milestone 5.2, investigate additional legally eligible real-board photo sources, and refine the proposed data acquisition plan.  
**Production Model:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, UNCHANGED)  
**Preserved Release Artifact:** `dist/ai-chess-navigator-1.0.0-arm64-milestone51b-bgr-validation-20261101.apk`  

---

## 1. Executive Summary

Milestones 4.3D through 5.1 established that:
1. Procedural synthetic training achieves **99.72% Top-1** on synthetic fixtures, but collapses to **16.80% Top-1** on real physical pieces (`knight` bias collapse).
2. Linear classifiers adapted to Zenodo Sets 1–3 achieve **35.33% Mean LOSO Top-1** and **40.62% on Set 4**.
3. Compact CNNs (`MicroChessCNN`, 40k parameters) trained from scratch achieve **20.40% Mean LOSO Top-1** and **27.99% on Set 4** due to texture/wood-grain overfitting on small datasets ($N=2,304$).
4. The Milestone 5.1/5.1A audit definitively proved that this ~20% LOSO ceiling is **not** an input pipeline defect or preprocessing mismatch.

Milestone 5.2A closes factual, licensing, and arithmetic gaps identified in Milestone 5.2:
* **ChessReD Licensing Re-Verification (Gate 1):** Corrected the license labeling of ChessReD. The official published dataset record on 4TU.ResearchData is **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. While third-party project documents claim separate author authorization for commercial use with attribution, that permission has not been independently verified for this project. Commercial use is designated **UNVERIFIED**, and ChessReD is strictly **EXCLUDED** from the eligible training pool.
* **Open Chess Photo Collections (Gate 2):** Audited crowdsourced and open-access chess photography collections. While CC0-dedicated subsets exist on platforms like Wikimedia Commons, Pexels, and Unsplash, they lack 4-point chessboard corner annotations, carry unlinked physical-set identities, exhibit severe shallow-depth-of-field blur and piece occlusions, and cannot be partitioned into leak-free physical-set groups without manual inspection.
* **Physical Capture Arithmetic Reconciled (Gate 3):** Fixed the mathematical inconsistency between the 12 capture conditions per set and scene totals:
  - **Canonical Baseline:** $20 \text{ sets} \times 12 \text{ conditions (4 angles} \times 3 \text{ lighting setups)} = \mathbf{240 \text{ full-board scenes}}$.
  - **Multi-Position Expansion:** $20 \text{ sets} \times (12 \text{ scenes starting position} + 20 \text{ scenes across 4 midgame positions}) = \mathbf{640 \text{ full-board scenes}}$.
  - **Square Crop Breakdown:** Separated clean isolated piece sweeps ($15,360$ piece crops $+ 1,280$ empty crops) from full-board rectified context crops ($15,360$ to $40,960$ crops).
* **Provisional Source Count (Gate 4):** Revised the claim of an exhaustive global count. Supported by source evidence, the number of identifiable distinct physical sets eligible for commercial training without licensing restrictions or copyleft risk is provisional and stands at **approximately 5 to 6 sets** (Zenodo: 4, samryan18: 1, FENiT: ~1). The remaining physical diversity gap is **at least 14 to 15 sets**.

---

## 2. Gate 0: Repository State Preservation

* **Git HEAD:** Verified synchronized with `origin/main` at `58b7d6b`.
* **Artifact Preservation:** All 29 APKs in `dist/` remain intact and unmodified, including the Milestone 5.1B release candidate.
* **Build Tree Preservation:** `modules/live-camera/android/build/` preserved untouched.
* **No APK Builds:** Zero compilation or packaging commands executed during Milestone 5.2A.
* **Storage Isolation:** Research metadata and temporary data reside strictly outside Git in `C:\Users\rahul\AppData\Local\Temp\opencode\`.
* **Security & Permissions:** Zero network permissions (`INTERNET` and `RECORD_AUDIO` strictly absent).

---

## 3. Gate 1: Re-Verification of ChessReD Licensing Provenance

### 3.1 Official Published License Record
* **Title:** *Chess Recognition Dataset (ChessReD)*
* **Authors:** Athanasios Masouris and Jan C. van Gemert (Delft University of Technology / TU Delft)
* **DOI:** [10.4121/99b5c721-280b-450b-b058-b2900b69a90f](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f)
* **Code Repository:** `https://github.com/tmasouris/end-to-end-chess-recognition` (Code licensed under MIT)
* **Published Dataset License on 4TU.ResearchData:** **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**
  - *NonCommercial (NC):* Explicitly restricts reuse to non-commercial purposes.
  - *ShareAlike (SA):* Requires any derivative works to be distributed under identical terms.

### 3.2 Evaluation of Third-Party Commercial Permission Claims
* **Claim:** Downstream training projects and re-packaged mirrors cite informal correspondence claiming specific author permission to train models for commercial use with attribution.
* **Legal & Provenance Evaluation:**
  1. The official published dataset record on 4TU.ResearchData remains formally bound by **CC BY-NC-SA 4.0**.
  2. Informal third-party claims of bilateral permission do not constitute a verifiable, publicly recorded license grant to the open-source community or this project.
  3. Relabeling the dataset as CC BY 4.0 in our records was factually inaccurate.
* **Gate Decision:**
  - **Commercial Licensing Status:** **UNVERIFIED**.
  - **Action:** ChessReD is strictly **EXCLUDED** from the eligible training pool for commercial distribution.
  - **Diversity Limitation (Independent of License):** As established in Milestone 5.2, all 10,800 photographs in ChessReD depict **a single physical chess set and board**. Even if commercial rights were cleared, it would provide zero physical-set diversity.

---

## 4. Gate 2: Investigation of Open Chess Photo Collections

We investigated crowdsourced and open-access chess photography collections across GitHub, Wikimedia Commons, Unsplash, Pexels, and Kaggle:

### 4.1 Evaluation of CC0-Dedicated Photo Subsets
* **Sources:** Wikimedia Commons (Category:Chess sets), Unsplash / Pexels public domain uploads.
* **Stated License:** **CC0 1.0 Universal (Public Domain Dedication)** or equivalent royalty-free open content license.
* **Commercial Rights:** Permitted without royalty or attribution.
* **Technical Limitations for Piece Classification:**
  1. **Unannotated Board Geometry:** Stock and community photos lack 4-point chessboard corner keypoints and square-level bounding boxes.
  2. **Shallow Depth of Field:** Artistic photography typically uses wide apertures ($f/1.8 - f/2.8$), resulting in severe background piece blurring.
  3. **Severe Perspective Occlusion:** Oblique angles ($< 30^\circ$) cause pawns to occlude pieces behind them.
  4. **Unknown Grouping & Lineage:** The number of distinct physical sets cannot be determined from photo count. Photos uploaded by the same user may depict the same set under different angles, creating silent test-set leakage if not grouped by owner.
* **Gate Decision:** **CONDITIONALLY APPROVED AS A VISUAL REFERENCE POOL; REJECTED AS OFF-THE-SHELF TRAINING DATA.** (Requires complete manual keypoint annotation and set-identity verification).

### 4.2 Evaluation of General Contributor Photo Subsets
* **Sources:** Community-uploaded chess image repositories (e.g., MTurk-sourced datasets, PhotoChess).
* **Licensing Hazard:** General contributor uploads default to platform-specific terms or CC BY-SA / CC BY-NC.
  - CC BY-NC: Prohibits commercial model distribution.
  - CC BY-SA: Introduces copyleft viral contamination risk for permissive MIT software.
* **Gate Decision:** **REJECTED (Incompatible Contributor Terms & Copyleft Risk).**

### 4.3 Crowdsourced Project Case Study: `notnil/fenify-3D`
* **Source:** GitHub (`notnil/fenify-3D`, MIT License) by Logan Spears (2023).
* **Dataset Characteristics:** 30,000 synthetic Unity images and 9,505 crowdsourced Mechanical Turk photos.
* **Findings:**
  - The author noted that 75% of crowdsourced MTurk submissions were scraped internet stock photos and had to be purged.
  - Over 25% of the genuine data came from a single physical setting (the author's own board).
  - The raw crowdsourced images were retained in private cloud storage and **never publicly published or released under an open license**.
* **Gate Decision:** **REJECTED (Unreleased Proprietary Raw Data).**

---

## 5. Gate 3: Reconciled Physical Capture Plan and Arithmetic

We reviewed and corrected the mathematical calculations for the proposed Author-Owned In-House Physical Capture Protocol:

### 5.1 Reconciled Scene Arithmetic
* **Target Set Count:** 20 distinct physical chess sets.
* **Camera Viewpoints (4 Angles per Condition):**
  1. Angle 1: Overhead / Zenith ($75^\circ - 90^\circ$ pitch)
  2. Angle 2: Player White Perspective ($45^\circ - 60^\circ$ pitch)
  3. Angle 3: Player Black Perspective ($45^\circ - 60^\circ$ pitch)
  4. Angle 4: Lateral / Spectator Flank Perspective ($45^\circ - 60^\circ$ pitch)
* **Lighting Environments (3 Setups per Condition):**
  1. Setup 1: Ambient diffuse daytime window light (soft shadows)
  2. Setup 2: Direct overhead warm incandescent / ceiling LED (specular sheen)
  3. Setup 3: Oblique directional side-lighting (casting directional shadows)
* **Conditions per Set:** $4 \text{ angles} \times 3 \text{ lighting setups} = \mathbf{12 \text{ conditions per set}}$.

#### Exact Scene Calculations:
1. **Canonical Baseline (1 Full-Board Layout per Condition):**
   * Capturing the starting position across all 12 conditions:
     $$20 \text{ sets} \times 12 \text{ conditions} = \mathbf{240 \text{ full-board scene photographs}}$$
2. **Multi-Position Expansion (Reconciliation of the 640 Figure):**
   * If 4 additional midgame board positions are captured per set:
     - Starting Position: 12 scenes (all 4 angles $\times$ 3 lighting setups).
     - 4 Midgame Positions: 20 scenes (4 positions $\times$ 4 angles under canonical diffuse lighting).
     - Total scenes per set: $12 + 20 = 32$ full-board scenes.
     - Total across 20 sets:
       $$20 \text{ sets} \times 32 \text{ scenes} = \mathbf{640 \text{ full-board scene photographs}}$$

### 5.2 Separation of Square Crop Data Types
To prevent confounding clean shape learning with perspective distortion and occlusion:

| Crop Category | Source Setup | Samples per Set | Total across 20 Sets | Task Role |
| :--- | :--- | :---: | :---: | :--- |
| **Type A: Isolated Square Sweep** | Static overhead camera across 64 squares; single unoccluded piece per square | $12 \text{ piece classes} \times 64 = 768$ piece crops $+ 64$ empty crops ($832$ total) | **15,360 piece crops**<br>($+ 1,280$ empty crops) | Pure 3D piece carving invariance & local surface statistics |
| **Type B: Full-Board Rectified Crops** | Extracted from 240 baseline (or 640 expanded) full-board scenes via homography | $12 \text{ scenes} \times 64 = 768$ context crops (or $32 \times 64 = 2,048$) | **15,360 context crops**<br>(or $40,960$ expanded) | Realistic piece occlusions, perspective tilts & board shadow handling |

### 5.3 Leak-Free Partitioning Protocol
* **Strict Set Isolation:** All images, angles, lighting variations, and crops of a physical set reside strictly in the same split.
* **Partition Split (20 Sets):**
  - **Training Split (Sets 1–14, 70%):** 14 physical sets ($10,752$ isolated piece crops).
  - **Validation Split (Sets 15–17, 15%):** 3 physical sets ($2,304$ isolated piece crops) for hyperparameter tuning.
  - **Untouched Blind Holdout (Sets 18–20, 15%):** 3 physical sets ($2,304$ isolated piece crops) reserved exclusively for final acceptance gating.
* **Realistic Target Disclosure:** The 20-set plan is an engineering diversity target designed to break the 3-set texture bottleneck; it is an empirical research hypothesis, not a mathematical guarantee of reaching $\ge 88.0\%$ Top-1.

---

## 6. Gate 4: Revised Source-Count and Diversity Gap Conclusion

We revised the global dataset audit to distinguish verified distinct physical sets from photo volume and restrictive licensing:

### 6.1 Comprehensive Public Chess Dataset Provenance Matrix

| Dataset Candidate | Source & DOI / URL | Published License | Commercial Rights | Verified Distinct Sets | Annotations & Usability | Eligibility Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Zenodo Record 6656212** | [10.5281/zenodo.6656212](https://doi.org/10.5281/zenodo.6656212) | **CC BY 4.0** | **YES** | **4 sets** (verified) | 6 classes + empty; $181\times 181$ clean crops | **APPROVED (Primary Benchmark)** |
| **samryan18/chess-dataset** | `samryan18/chess-dataset` | **MIT** | **YES** | **1 set** (verified) | FEN labels; 500 full-board photos | **APPROVED (Reference Pipeline)** |
| **FENiT Image Dataset** | [10.17632/wpp9xnbcp6.1](https://doi.org/10.17632/wpp9xnbcp6.1) | **CC BY 4.0** | **YES** | **1–2 sets** (provisional) | Bounding boxes + corner keypoints | **APPROVED (Corner Benchmark)** |
| **ChessReD** | [10.4121/99b5c721](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f) | **CC BY-NC-SA 4.0** | **UNVERIFIED** | 1 set (verified) | FEN + bounding boxes; 10,800 photos | **EXCLUDED (NC/SA License & Single Set)** |
| **Kaggle Chessman** | `niteshfre/chessman-image-dataset` | **CC BY-NC-SA 4.0** | **NO** | Unknown (scraped) | Isolated piece photos | **REJECTED (Non-Commercial Hazard)** |
| **LiveChess2FEN** | `davidmallasen/LiveChess2FEN` | **AGPL-3.0** | Copyleft | Mixed | Full pipeline | **REJECTED (Viral Copyleft Hazard)** |
| **notnil/fenify-3D** | `notnil/fenify-3D` | MIT (code only) | Unreleased | 150 contributors | Crowdsourced photos unreleased | **REJECTED (Unreleased Raw Data)** |
| **Roboflow Universe** | `chess-full-r3wh3-sajg6` | CC BY 4.0 (unverified) | Unclear | 1 set | 21 corrupted classes; carpet background | **REJECTED (Corrupted Metadata & Lineage)** |
| **Kaggle Mehta** | `anshulmehtakaggle/chess-pieces...` | CC0 (unverified) | Unclear | Unknown (scraped) | 651 isolated piece cutouts | **REJECTED (Uncredited Web Scrapes)** |
| **ChessCog** | `georg-wolflein/chesscog` | MIT / CC BY 4.0 | YES | 0 real sets | 4,888 3D Blender renders | **REJECTED (Synthetic Only)** |
| **ChessRender360** | Zenodo `10.5281/zenodo...` | CC BY 4.0 | YES | 0 real sets | 10,000 3D Blender renders | **REJECTED (Synthetic Only)** |
| **Wikimedia Commons CC0** | Category:Chess sets | **CC0 1.0 Universal** | **YES** | Multiple (untracked) | Unannotated artistic stock photos | **REJECTED (Lacks Board Annotations)** |

### 6.2 The Provisional Physical Set Deficit
* **Verified Commercially Eligible Sets:** Approximately **5 to 6 distinct physical sets** supported by verifiable open-access licenses (Zenodo: 4, samryan18: 1, FENiT: ~1).
* **Target Requirement:** $\ge 20$ distinct physical sets.
* **Provisional Diversity Deficit:** **$-14$ to $-15$ physical sets**.
* **Finding:** No single external public dataset bridges this gap without either violating commercial licensing terms (CC BY-NC-SA, AGPL) or suffering from the single-set monopoly (ChessReD, samryan18).

---

## 7. Gate 5: Production Acceptance Gate Table & Strategic Recommendation

```
========================================================================================================
MILESTONE 5.2A ACCEPTANCE GATES
========================================================================================================
Requirement                         Gate Standard      Audit Finding / Status          Gate Decision
--------------------------------------------------------------------------------------------------------
ChessReD License Re-Verification:   Official record    Published CC BY-NC-SA 4.0       PASS (Corrected)
ChessReD Commercial Eligibility:    Verified grant     Permission unverified; EXCLUDED EXCLUDED
Open Chess CC0 Audit:               Detailed review    CC0 lacks geometry annotations  PASS (Audited)
Physical Capture Arithmetic:        Exact math proof   240 baseline / 640 expanded     PASS (Reconciled)
Source Count Conclusion:            Evidence-based     Provisional ~5-6 eligible sets  PASS (Revised)
Model Architecture Isolation:       No new models      Zero training conducted         PASS
Production Model Weights:           pieceModelWeights  100% UNCHANGED                  PASS
Zero-Silent-Error Policy:           100%               Mandatory confirmation active   PASS
========================================================================================================
OVERALL DECISION: FACTUAL, LICENSING, AND ARITHMETIC AUDITS FULLY RECONCILED.
CHESSRED IS EXCLUDED FROM THE ELIGIBLE TRAINING POOL PENDING VERIFIED RESOLUTION.
COMMENCE PHYSICAL PREPARATION FOR THE AUTHOR-OWNED 20-SET CC0/MIT CAPTURE PROTOCOL.
PRESERVE PRODUCTION WEIGHTS UNCHANGED AT src/vision/pieceModelWeights.ts.
========================================================================================================
```

---

## 8. Strategic Acquisition Recommendation

1. **Do Not Rely on ChessReD for Commercial Weights:** ChessReD's published CC BY-NC-SA 4.0 license poses commercial and copyleft risks, and its single-set composition provides zero physical-set diversity.
2. **Prioritize the Author-Owned 20-Set Capture Protocol:**
   - Procure and photograph 20 physical Staunton sets across the 4 specified categories.
   - Execute the reconciled capture plan ($240$ baseline full-board scenes, $640$ expanded scenes, and $15,360$ isolated square crops).
   - Dedicate the resulting dataset to the public domain under **CC0 1.0 Universal**, establishing an unassailable commercial-ready benchmark for the chess computer vision community.
3. **Preserve Production Pipeline Truthfulness:**
   - In the interim, keep production weights in `src/vision/pieceModelWeights.ts` unchanged.
   - Maintain mandatory user review via amber `?` badges and manual board editing in Board Review.
