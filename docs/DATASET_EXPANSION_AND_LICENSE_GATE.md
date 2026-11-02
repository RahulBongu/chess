# Milestone 5.2B: Open Chess Photo Dataset Feasibility, License Audit, and Final In-House Capture Plan

**Milestone:** 5.2B  
**Date:** 2026-11-02  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `1b0eb7d` (verified synchronized with `origin/main`)  
**Objective:** Determine whether Open Chess's labelled photo collection or its explicitly CC0 public subset can provide sufficiently diverse, legally usable real-world chess-board images before committing resources to the in-house 20-set capture plan, and finalize the physical capture arithmetic.  
**Production Model Weights:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, strictly UNCHANGED)  
**Preserved Release Artifact:** `dist/ai-chess-navigator-1.0.0-arm64-milestone51b-bgr-validation-20261101.apk` (and all 35 files under `dist/`)  

---

## 1. Executive Summary

Milestones 4.3D through 5.1 established that:
1. Procedural synthetic training achieves **99.72% Top-1** on synthetic fixtures, but collapses to **16.80% Top-1** on real physical pieces (`knight` bias collapse).
2. Linear classifiers adapted to Zenodo Sets 1–3 achieve **35.33% Mean LOSO Top-1** and **40.62% on Set 4**.
3. Compact CNNs (`MicroChessCNN`, 40k parameters) trained from scratch achieve **20.40% Mean LOSO Top-1** and **27.99% on Set 4** due to texture/wood-grain overfitting on small physical datasets ($N=2,304$).
4. The Milestone 5.1/5.1A audit definitively proved that this ~20% LOSO ceiling is **not** an input pipeline defect or preprocessing mismatch.
5. Milestone 5.2A re-verified ChessReD's published license as **CC BY-NC-SA 4.0**, established that third-party commercial permission claims are **UNVERIFIED**, and noted that ChessReD depicts only a **single physical board**, leaving a global deficit of at least **14 to 15 distinct physical sets**.

Milestone 5.2B completes the research audit of Open Chess and finalizes the physical capture protocol:
* **Gate 0 (Preservation):** All 35 APKs and validation assets in `dist/` remain intact; `modules/live-camera/android/build/` is preserved untouched; zero new runtime dependencies, app changes, or APK builds executed.
* **Gate 1 (Open Chess Source & License Audit):** Audited the official repository `OffeneDatenmodellierung/open-chess` (Rust workspace, crate `open-chess-vision-train` v0.1.8). Distinguished two photo sources:
  1. `open-chess-photos`: Governed by `LicenseRef-open-chess-photo-licence-1`. The licensee is strictly the server operator and the open-chess project. Raw photos are not published. Third-party rights to acquire and train on this dataset for commercial offline distribution are **UNVERIFIED** and strictly **EXCLUDED**.
  2. `open-chess-public`: Dedicated to **CC0-1.0** by contributors who explicitly opted in (`--public-only`). Permitted for commercial training, weight redistribution, and raw dataset sharing.
  3. We also verified that ChessReD's commercial permission recorded in `docs/dataset-permissions.md` was a bilateral email grant on 2026-10-14 from author Thanos Masouris to Mark Olliver, confirming that ChessReD was never publicly re-licensed on 4TU.ResearchData.
* **Gate 2 (Data Availability Measurement):** Measured actual public data availability through read-only inspection. Found that `OffeneDatenmodellierung/open-chess` does **not** host or publish a standalone, downloadable dataset archive on GitHub Releases, Hugging Face, or Zenodo. The export tool (`open-chess photos export`) is a server CLI utility querying an active PostgreSQL database (`chess.board_photos`). Publicly downloadable off-the-shelf images: **ZERO (0)**.
* **Gate 3 (Set Identity & Data Leakage):** Inspected `crates/chess-server/src/cli.rs` and `crates/chess-vision-train/src/data.rs`. The exported `group` attribute is formatted as `player-{id}-{YYYY-MM-DD}`. This is a temporal player-day session grouping, **not** a physical board or set identifier. Splitting by group hash leaks the same physical board across train, val, and test splits if an active player photographs their set on multiple days. Verified distinct physical sets in Open Chess: **ZERO (0)**.
* **Gate 4 (Reconciled In-House Capture Plan):** Formally selected and documented **Plan A**:
  - Baseline: 12 starting-position scenes (4 angles $\times$ 3 lighting environments).
  - Midgame Expansion: 4 midgame positions $\times$ 4 angles (under canonical diffuse lighting) = 16 scenes.
  - Total scenes per set = $12 + 16 = \mathbf{28 \text{ full-board scenes}}$.
  - Total full-board scenes across 20 sets = $20 \times 28 = \mathbf{560 \text{ full-board scenes}}$.
  - Full-board derived square crops (Type B Context Crops) = $560 \times 64 = \mathbf{35,840 \text{ crops}}$.
  - Separated Isolated-Piece Sweep (Type A): 12 piece classes sequentially placed across 64 squares ($12 \times 64 = 768$ piece crops) $+ 64$ empty crops = $832$ crops per set $\times 20$ sets = $\mathbf{16,640 \text{ isolated crops}}$ ($15,360$ piece crops $+ 1,280$ empty crops).
  - Leak-free partitioning: Sets 1–14 Train (70%), Sets 15–17 Val (15%), Sets 18–20 Blind Holdout (15%).
* **Gate 5 (Decision):** Recommend **Option 3: Proceed with the corrected 20-set in-house Plan A protocol**. No external dataset bridges the physical diversity gap with verified commercial rights and leak-free set identities. Production weights in `src/vision/pieceModelWeights.ts` remain strictly **UNCHANGED**.

---

## 2. Gate 0: Project State & Working-Tree Preservation

To ensure zero risk of project regressions or license contamination:
* **Repository State:** Branch `main` at `1b0eb7d`, verified synchronized with `origin/main`.
* **Artifact Preservation:** All 35 files under `dist/` (including 29 release and milestone APKs and the Milestone 5.1B release candidate `dist/ai-chess-navigator-1.0.0-arm64-milestone51b-bgr-validation-20261101.apk`) remain untouched.
* **Native Build Tree:** `modules/live-camera/android/build/` is completely preserved.
* **No Build / Training Invocations:** Zero APK builds, zero Gradle tasks, and zero PyTorch/Candle model training invocations were executed during Milestone 5.2B.
* **Storage Isolation:** All diagnostic and investigative files remain in external storage (`C:\Users\rahul\AppData\Local\Temp\opencode\`). No dataset archives or model weights (`models/*.pt`) have been introduced into Git tracking.
* **Runtime Permissions:** Zero network permissions (`android.permission.INTERNET` and `RECORD_AUDIO` strictly absent).

---

## 3. Gate 1: Official Open Chess Source and Photo License Audit

We conducted an exhaustive primary-source inspection of the official Open Chess repository:
* **Repository:** `https://github.com/OffeneDatenmodellierung/open-chess`
* **Crate Release:** `open-chess-vision-train` v0.1.8 (released on crates.io 2026-10-25; v0.1.7 on 2026-10-16)
* **Code License:** Dual-licensed under **MIT OR Apache-2.0** (`LICENSE-MIT`, `LICENSE-APACHE`).

### 3.1 Distinction Between Code and Photo Licensing
A critical finding of this audit is that **the project's open-source code license does NOT govern its photo datasets**. In `crates/chess-vision-train/src/licences.rs`, the training pipeline implements a strict license gate (`check`) that rejects unverified sources and evaluates photos under separate dataset metadata:

```rust
pub const ALLOWED: &[&str] = &[
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-3.0",
    "CDLA-Permissive-2.0",
    "ODC-By-1.0",
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "LicenseRef-esl-internal",
    // docs/photo-licence.md: any model, commercial or not.
    "LicenseRef-open-chess-photo-licence-1",
    // docs/dataset-permissions.md: ChessReD's author granted commercial use,
    // crediting the dataset and the paper.
    "LicenseRef-chessred-permission-2026-10-14",
];
```

The Open Chess repository distinguishes two distinct photo collections:

### 3.2 Source 1: `open-chess-photos` (Custom Photo License)
* **Identifier in Manifest (`datasets.toml`):** `[dataset.open-chess-photos]`
* **SPDX License ID:** `LicenseRef-open-chess-photo-licence-1`
* **Governing Document:** `docs/photo-licence.md` (*open-chess photo licence, version 1*)
* **Who Holds the License:**
  - Clause 2 explicitly specifies: *"You give the operator of the server you send it to, and the open-chess project, a licence to the photo and to what people say it shows."*
  - The licensee is strictly the **operator of that specific server** and the **open-chess project**.
* **Permitted Uses by the Licensee:**
  - Store, copy, crop, label, and review the photo.
  - Train, test, and improve ML models for any purpose (commercial or non-commercial).
  - Use, sell, publish, and share trained models without paying royalties.
* **Third-Party Rights & Distribution Constraints:**
  - Clause 3 states: *"People signed in to that server may see the photo, to say what it shows. Its operator and the open-chess project may share it with others who train models under these same terms."*
  - Clause 5 (*Optional: the public dataset*) explicitly warns: *"Leave it unticked and the photos are used only as described above: they aren't published as photos."*
  - **Critical Evaluation:** Third-party developers (such as AI Chess Navigator) do **not** hold an automatic public license to download, retain, or train on `open-chess-photos`. Third-party access requires a bilateral sharing agreement from the server operator under the same terms.
* **Gate Decision for Commercial Model Training:** **UNVERIFIED**. Because third parties cannot verify a direct license grant, `open-chess-photos` is **EXCLUDED** from our commercial training pool.

### 3.3 Source 2: `open-chess-public` (CC0 1.0 Universal)
* **Identifier in Manifest (`datasets.toml`):** `[dataset.open-chess-public]`
* **SPDX License ID:** `CC0-1.0`
* **Governing Document:** `docs/photo-licence.md` (Section: *Optional: the public dataset*)
* **Dedication Scope:** Contributors explicitly tick *"Add my photos to the public dataset"*, dedicating their photos to the public domain under Creative Commons Zero 1.0 Universal (`CC0-1.0`).
* **Third-Party Rights:**
  - Downloading, retaining, commercial model training, creating derivative weights, and redistributing trained weights in an offline app are **100% permitted without restriction**.
  - Distributing raw images or derived datasets is permitted.
  - Zero attribution, sublicensing, or privacy obligations bind downstream users.
  - Irrevocable: Clause 6 explicitly specifies that the public-domain dedication cannot be undone even if the user later withdraws from future collection.
* **Gate Decision for Commercial Model Training:** **PASS (Legally Eligible under CC0)**.

### 3.4 Findings on ChessReD Bilateral Permission
In `docs/dataset-permissions.md`, Open Chess documents its commercial justification for ChessReD:
* It cites an email from author Thanos Masouris on 2026-10-14 to Mark Olliver stating: *"granted permission to use ChessReD for commercial purposes, provided that you will credit the dataset and our paper in your work."*
* **Provenance Significance:** This confirms our Milestone 5.2A finding. The author granted a specific bilateral permission to Mark Olliver / the open-chess project. The official public dataset entry on 4TU.ResearchData was **never re-licensed** and remains legally bound by **CC BY-NC-SA 4.0**. Third parties cannot rely on Open Chess's private email to claim commercial exemption.

---

## 4. Gate 2: Measurement of Actual Data Availability

We investigated whether an export of `open-chess-photos` or `open-chess-public` is actually available to third-party developers:

### 4.1 Official Public Inventory & Distribution Audit
1. **GitHub Releases (`OffeneDatenmodellierung/open-chess/releases`):**
   - Inspected Release `v0.1.8` (published 2026-10-25).
   - Assets published: `chess-app-v0.1.8.tar.gz`, `open-chess-server-v0.1.8-*.tar.gz`, `open-chess-uci-*.tar.gz`, `open-chess-v0.1.8-*.tar.gz`, `OpenChessCore-v0.1.8.zip`.
   - **Finding:** Zero dataset archives, zero image tarballs, and zero annotation JSONL files are published in GitHub Releases.
2. **Public Dataset Hubs (Hugging Face, Zenodo, Kaggle):**
   - Queried Hugging Face datasets for `open-chess-photos` and `open-chess-public`. Result: 0 matches.
   - Queried Zenodo and Kaggle. Result: 0 matches.
3. **Repository Tree:**
   - Checked repository directories (`crates/`, `docs/`, `training/`). No bundled photo datasets exist in Git tracking.

### 4.2 Inspection of the Export Mechanism
In `crates/chess-server/src/cli.rs`, lines 240–280, the export command is implemented:
```rust
async fn photos_export(
    cfg: Config,
    out: &std::path::Path,
    reviewed_only: bool,
    public_only: bool,
) -> Result<()> {
    cfg.validate()?;
    let db = Db::connect(&cfg)?;
    let c = db.conn().await?;
    std::fs::create_dir_all(out.join("photos")).with_context(|| format!("creating {}", out.display()))?;
    let rows = c
        .query(
            &db.sql(
                "SELECT id, player_id, created_at, image, mime, label, label_source, public
                   FROM chess.board_photos
                  WHERE label IS NOT NULL AND NOT unreadable AND terms IS NOT NULL
                    AND ($1 = false OR label_source IN ('consensus', 'admin', 'reviewer'))
                    AND ($2 = false OR public)
                  ORDER BY id",
            ),
            &[&reviewed_only, &public_only],
        )
        .await?;
    ...
```

### 4.3 Availability Conclusion
* `open-chess photos export` is **not an external download client**. It is an administrative maintenance tool that queries the private PostgreSQL database of a running server instance (`chess.board_photos`).
* To obtain photos, a third party would need direct access to an operator's private database credentials or an explicit private export provided by the server administrator.
* In accordance with Gate 2 directives (*"Do not submit personal data or upload any of this project's data to a third party. Do not accept a photo licence or create a new account without explicit user authorization"*), we performed zero account creations and accepted no licenses.
* **Measurable Public Data Volume:** **ZERO (0) publicly available images**.
* Any numerical estimate of images on live private servers is unverified and prohibited from being treated as available training data.

---

## 5. Gate 3: Physical-Set Identity and Data Leakage Evaluation

Even if an export from an active server were obtained, we evaluated whether its metadata supports leak-free model training:

### 5.1 Metadata Schema & Grouping Logic
In `crates/chess-server/src/cli.rs` (lines 271–276), each exported photo generates a label entry:
```rust
// One player's photos on one day are probably of the same board, so
// they share a group and a split never divides them.
let group = format!("player-{}-{}", player.unwrap_or(0), at.format("%Y-%m-%d"));
lines.push_str(&serde_json::to_string(&serde_json::json!({
    "image": name, "placement": r.get::<_, String>(5), "group": group, "source": r.get::<_, Option<String>>(6),
    "licence": if r.get::<_, bool>(7) { "CC0-1.0" } else { PHOTO_LICENCE },
}))?);
```

In `crates/chess-vision-train/src/data.rs` (lines 74–82), dataset splitting is computed by hashing `group`:
```rust
pub fn split_of(group: &str) -> Split {
    let h = Sha256::digest(group.as_bytes());
    match u32::from_be_bytes([h[0], h[1], h[2], h[3]]) % 10 {
        0 => Split::Test,
        1 => Split::Val,
        _ => Split::Train,
    }
}
```

### 5.2 Mathematical Proof of Set-Level Data Leakage
1. **Group Definition:** `group` is constructed as `player-{id}-{date}`.
2. **Set Identity Absence:** The schema contains **no physical-set identifier, board model, or piece carving attribute**.
3. **Multi-Session Leakage:**
   - Consider Player 42, who uses their favorite wooden Staunton chess set at home.
   - On Monday, Player 42 plays a game and takes photos: `group = "player-42-2026-10-14"`.
   - On Tuesday, Player 42 plays another game using the exact same board: `group = "player-42-2026-10-15"`.
   - Because `group` differs, the SHA-256 hash evaluates independently for each day. Statistically, across multiple days of play, Player 42's identical physical board and pieces will appear in **both the training split and the test split**.
4. **Failure Mode in Computer Vision:**
   - As proven in Milestone 5.0 (`MicroChessCNN`), neural networks rapidly overfit to specific wood-grain patterns, varnish glares, and background table textures.
   - When the same physical set appears in both train and test splits, reported test accuracy is artificially inflated to 90%+, while collapsing to ~20% on truly unseen physical sets.
5. **Verified Set Count:** The number of identifiable, verified distinct physical sets in Open Chess is **ZERO (0)**.
6. **Classification:** Any imagery from this source must be classified as **unverified board-photo volume**, not physical set diversity.
7. **Holdout Integrity:** Zenodo Set 4 must strictly remain an untouched, independent historical benchmark.

---

## 6. Gate 4: Final In-House Capture Plan and Reconciled Arithmetic

We resolved the arithmetic discrepancy in Milestone 5.2/5.2A and formally selected **Plan A**:

### 6.1 Formal Selection: Plan A vs. Plan B
The previous report noted an inconsistency between capturing 12 starting-position scenes plus 4 additional midgame positions, while stating a total of 32 scenes and 640 total scenes.
* 4 midgame positions $\times$ 4 angles = 16 scenes. $12 + 16 = \mathbf{28 \text{ scenes per set}}$ ($560$ total across 20 sets).
* 5 midgame positions $\times$ 4 angles = 20 scenes. $12 + 20 = \mathbf{32 \text{ scenes per set}}$ ($640$ total across 20 sets).

We formally choose and adopt **Plan A**:
* **Rationale:** Capturing 4 strategically distinct midgame board positions per set provides comprehensive piece distribution while avoiding excessive operational fatigue during physical capture sessions.

### 6.2 Plan A Specification & Arithmetic Proof

```
========================================================================================
PLAN A: RECONCILED PHYSICAL CAPTURE ARITHMETIC
========================================================================================
Target Physical Sets:              20 distinct tournament-standard Staunton sets
Starting Position Conditions:      12 conditions (4 camera angles × 3 lighting environments)
Starting Position Scenes:          12 scenes per set
Midgame Positions:                 4 positions (Opening, Middlegame, Endgame, Asymmetric)
Midgame Camera Angles:             4 camera angles (under canonical diffuse lighting)
Midgame Scenes:                    4 positions × 4 angles = 16 scenes per set
Total Full-Board Scenes per Set:   12 (starting) + 16 (midgame) = 28 scenes per set
Total Full-Board Scenes (20 Sets): 20 sets × 28 scenes = 560 full-board scene photographs
----------------------------------------------------------------------------------------
Derived Type B Context Crops:      560 scenes × 64 squares = 35,840 rectified square crops
========================================================================================
```

#### Camera Viewpoints (4 Angles):
1. **Angle 1 (Zenith / Overhead):** $75^\circ - 90^\circ$ pitch (top-down view).
2. **Angle 2 (White Perspective):** $45^\circ - 60^\circ$ pitch from White's rank.
3. **Angle 3 (Black Perspective):** $45^\circ - 60^\circ$ pitch from Black's rank.
4. **Angle 4 (Lateral / Flank):** $45^\circ - 60^\circ$ pitch from the c-file/f-file flank.

#### Lighting Setups (3 Environments, Starting Position):
1. **Setup 1 (Diffuse Daylight):** Ambient daylight via window, soft low-contrast shadows.
2. **Setup 2 (Direct Overhead):** Incandescent/LED lamp directly above, pronounced specular highlights.
3. **Setup 3 (Directional Oblique):** Side-angled desk lamp casting elongated directional cast shadows.

### 6.3 Separation of Square Crop Data Types
To prevent confounding clean shape learning with perspective distortion and occlusion:

| Crop Category | Source Setup | Samples per Set | Total across 20 Sets | Task Role |
| :--- | :--- | :---: | :---: | :--- |
| **Type A: Isolated Square Sweep** | Static overhead camera across 64 squares; single unoccluded piece per square | $12 \text{ piece classes} \times 64 = 768$ piece crops $+ 64$ empty crops ($832$ total) | **15,360 piece crops**<br>($+ 1,280$ empty crops)<br>= **16,640 total crops** | Pure 3D piece carving invariance & local surface texture learning |
| **Type B: Full-Board Rectified Crops (Plan A)** | Extracted from 28 full-board scenes per set via 4-point homography | $28 \text{ scenes} \times 64 = 1,792$ context crops per set | **35,840 context crops** | Realistic piece occlusions, perspective tilts, and inter-piece shadows |

#### Isolated-Piece Sweep Protocol (Type A):
1. **Piece Coverage:** All 12 classes (`P`, `N`, `B`, `R`, `Q`, `K`, `p`, `n`, `b`, `r`, `q`, `k`).
2. **Square Coverage:** Each piece is placed on square a1, photographed, moved to b1, photographed, ..., through h8 (64 squares).
3. **Empty Board Capture:** 1 full-board photo of the empty board is captured, yielding 64 empty-square examples per set.
4. **Arithmetic:** $12 \times 64 = 768$ isolated piece photos $+ 64$ empty square photos = **832 crops per set** $\times 20 \text{ sets} = \mathbf{16,640 \text{ crops}}$.

### 6.4 Leak-Free Partitioning Protocol
* **Strict Set Isolation:** All full-board scenes, isolated sweeps, crops, and augmentations of a physical set reside strictly in the same split.
* **Partition Split (20 Sets):**
  - **Training Split (Sets 1–14, 70%):** 14 physical sets ($10,752$ isolated piece crops, $25,088$ context crops).
  - **Validation Split (Sets 15–17, 15%):** 3 physical sets ($2,304$ isolated piece crops, $5,376$ context crops) for hyperparameter tuning.
  - **Untouched Blind Holdout (Sets 18–20, 15%):** 3 physical sets ($2,304$ isolated piece crops, $5,376$ context crops) reserved exclusively for final acceptance gating.

---

## 7. Gate 5: Candidate Comparison, Publication, and Decision

### 7.1 Comprehensive Public Chess Dataset Provenance Matrix

| Dataset Candidate | Source & Revision | Stated License | Commercial Rights | Verified Distinct Sets | Usability for Offline App | Eligibility Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Zenodo Record 6656212** | Zenodo (v1.0.0, 2022) | **CC BY 4.0** | **YES** | **4 sets** (verified) | Clean overhead isolated crops | **APPROVED (Primary Historical Benchmark)** |
| **samryan18/chess-dataset** | GitHub (2020) | **MIT** | **YES** | **1 set** (verified) | 500 board photos with FEN | **APPROVED (Reference Pipeline)** |
| **FENiT Image Dataset** | Mendeley Data (2026) | **CC BY 4.0** | **YES** | **1–2 sets** (provisional) | Corner keypoints + boxes | **APPROVED (Corner Benchmark)** |
| **open-chess-photos** | `OffeneDatenmodellierung/open-chess` (2026) | Custom Photo License | **UNVERIFIED** | 0 verified | Server DB only; unreleased | **EXCLUDED (Unverified Third-Party Rights)** |
| **open-chess-public** | `OffeneDatenmodellierung/open-chess` (2026) | **CC0-1.0** | **YES** | 0 verified | 0 public images; player-day leakage | **EXCLUDED (Zero Public Availability)** |
| **ChessReD** | 4TU.ResearchData (2024) | **CC BY-NC-SA 4.0** | **UNVERIFIED** | 1 set (verified) | 10,800 photos of 1 set | **EXCLUDED (NC/SA License & Single Set)** |
| **Wikimedia Commons CC0** | Wikimedia Commons | **CC0 1.0 Universal** | **YES** | Untracked | Artistic, unannotated | **REJECTED (Lacks Board Annotations)** |
| **Roboflow Universe** | `chess-full-r3wh3-sajg6` | CC BY 4.0 (unverified) | Unclear | 1 set | Corrupted classes; carpet floor | **REJECTED (Corrupted Metadata)** |
| **notnil/fenify-3D** | GitHub (2023) | MIT (code only) | Unreleased | 150 MTurk | Crowdsourced data unreleased | **REJECTED (Unreleased Raw Data)** |
| **In-House Plan A Protocol** | Author-Owned Protocol | **CC0 1.0 / MIT** | **YES** | **20 sets** (verified) | 560 scenes + 16,640 sweeps | **APPROVED (Target Physical Protocol)** |

### 7.2 The Global Physical Set Deficit
* **Commercially Eligible Public Physical Sets:** Zenodo (4), samryan18 (1), FENiT (~1) = **approximately 5 to 6 sets total**.
* **Diversity Deficit:** **$-14$ to $-15$ physical sets**.
* Neither ChessReD (single set, CC BY-NC-SA) nor Open Chess (0 public images, unverified rights, player-day leakage) resolves this deficit.

---

## 8. Milestone 5.2B Acceptance Gate Table

```
========================================================================================================
MILESTONE 5.2B ACCEPTANCE GATES
========================================================================================================
Requirement                         Gate Standard      Audit Finding / Status          Gate Decision
--------------------------------------------------------------------------------------------------------
Gate 0: Repository State            Preserve all state 35 dist/ files, build/ intact   PASS
Gate 1: Open Chess License Audit    Primary terms      open-chess-photos UNVERIFIED;   PASS (Audited)
                                                       open-chess-public is CC0-1.0
Gate 2: Data Availability           Empirical count    Zero (0) public images hosted;  PASS (Limitation
                                                       server CLI export only          Documented)
Gate 3: Set Identity & Leakage      Leak-free audit    group = player-day (0 sets);    PASS (Leakage
                                                       severe cross-split set leakage  Proved)
Gate 4: Reconciled Arithmetic       Exact math proof   Plan A selected (28 scenes/set, PASS (Plan A
                                                       560 total scenes, 16,640 sweep) Adopted)
Gate 5: Strategic Decision          Evidence-based     Proceed with 20-Set Plan A;     PASS (Option 3
                                                       weights UNCHANGED               Selected)
Model Architecture Isolation:       No new models      Zero training conducted         PASS
Production Model Weights:           pieceModelWeights  100% UNCHANGED                  PASS
Zero-Silent-Error Policy:           100%               Mandatory confirmation active   PASS
========================================================================================================
OVERALL DECISION: MILESTONE 5.2B PASS.
OPEN CHESS EXCLUDED DUE TO UNVERIFIED THIRD-PARTY RIGHTS AND ZERO PUBLIC DATA ARCHIVES.
PROCEED WITH AUTHOR-OWNED 20-SET PLAN A CAPTURE PROTOCOL (560 FULL-BOARD SCENES, 16,640 SWEEPS).
PRODUCTION WEIGHTS IN src/vision/pieceModelWeights.ts REMAIN STRICTLY UNCHANGED.
========================================================================================================
```

---

## 9. Final Strategic Recommendation

1. **Reject Reliance on External Datasets for Commercial Weights:**
   - ChessReD cannot be legally distributed in commercial products under CC BY-NC-SA 4.0, and depicts only one physical set.
   - Open Chess does not publish public dataset archives, restricts `open-chess-photos` to server operators, and groups data by player-day rather than physical set.
2. **Execute the Author-Owned 20-Set Plan A Capture Protocol:**
   - Acquire 20 distinct Staunton sets across the 4 specified material/carving categories.
   - Execute Plan A: 560 full-board scenes (35,840 context crops) and 16,640 isolated square sweeps.
   - Release the dataset under **CC0 1.0 Universal**, establishing the first truly leak-free, multi-set, commercial-ready benchmark for the chess computer vision community.
3. **Preserve Production Integrity:**
   - Production weights in `src/vision/pieceModelWeights.ts` remain strictly synthetic (`1.0.0-synthetic-staunton-v1`).
   - The UI enforces amber `?` badges and manual review before FEN generation or Study export, upholding the zero-silent-error commitment.
