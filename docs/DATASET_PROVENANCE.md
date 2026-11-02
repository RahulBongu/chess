# Chess Piece Recognition: Dataset Provenance & Licensing Policy

Date: 2026-11-02  
Milestone: 5.2G — Capture Preflight, Workspace Setup, and Set 01 Pilot Protocol  
Target App: AI Chess Navigator (Offline Mobile Application)  
Repository: `https://github.com/RahulBongu/chess.git`

---

## 1. Objectives & Compliance Principles

AI Chess Navigator operates under strict open-source (MIT), offline, and privacy-first commitments:
- **Zero Network Ingestion:** The mobile application contains no `android.permission.INTERNET` and performs no runtime network access. All assets must be bundled into the APK or compiled into native code.
- **Strict Commercial-Ready Licensing:** Any dataset, code, pre-trained weights, or digital assets incorporated into the repository or distributed APK must be fully compatible with commercial distribution under permissive terms (MIT, Apache-2.0, BSD-3-Clause, CC0 1.0, or CC BY 4.0).
- **Zero Third-Party IP Contamination:** The underlying physical design and 2D/3D visual representations of chess pieces must not infringe upon proprietary copyright, trademark, or terms of service of third-party chess platforms.

---

## 2. IP & Artwork Provenance Gate

### 2.1 The Staunton Design (Public Domain)
- The standard chess piece silhouette used worldwide was designed by **Nathaniel Cook** in 1849 and patented/endorsed by **Howard Staunton**.
- The physical Staunton design geometry is in the **public domain**. Physical Staunton chess sets manufactured for tournament play (FIDE standard) are free of design patent restrictions.
- Real-world photographs of physical Staunton chess sets taken by the repository authors or released under CC0/CC-BY are legally safe for training.

### 2.2 Digital Chess Artwork & Platform Restrictions (Prohibited Sources)
- **Chess.com Artwork:** The digital piece sets (e.g., Neo, Classic, Wood, Bases) and web assets displayed on Chess.com are proprietary copyrighted works owned by Chess.com, LLC. Scraping, screenshotting, or training commercial classifiers on Chess.com UI captures without explicit written licensing is strictly prohibited.
- **Lichess Piece Sets:** While Lichess is open-source (GPL-3.0), individual piece sets have diverse licenses. Many vector sets (e.g., Cburnett by Colin M.L. Burnett) are licensed under CC BY-SA 3.0. Incorporating weights trained on CC BY-SA data can introduce viral copyleft license contamination into commercial products.
- **Unverified Online Datasets:** Datasets hosted on Roboflow Universe or Kaggle labeled as "CC0" or "Public Domain" frequently contain uncredited web scrapes, Pinterest images, and proprietary digital assets. A "CC0" tag on a public hosting platform is **not conclusive** without auditable lineage for the underlying images.

### 2.3 Strict Separation of Data and Code Licensing (Gate 5 Policy)
- **Photographic Data and Annotations:** All photographs, square crops, FEN annotations, and dataset manifests released by this project are dedicated exclusively to the public domain under **CC0 1.0 Universal** once photographer property releases are established.
- **Software, Scripts, and Algorithms:** All capture tooling, homography rectification code, classification routines, and manifest validation scripts are licensed under the **MIT License**.
- **Prohibition of Conflated Licensing:** Conflating these components under ambiguous terms such as "CC0/MIT" is strictly prohibited. Every manifest and documentation file must clearly delineate whether it governs data (CC0 1.0) or code (MIT).

---

## 3. Audit of Public Chess Datasets

| Dataset Candidate | Source & Revision | Stated License | Underlying Provenance | Gate Decision | Reason / Restrictions |
|---|---|---|---|---|---|
| **Zenodo Record 6656212** | Zenodo (v1.0.0, 2022) | CC BY 4.0 (verified) | Academic thesis (UC3M). 4 physical chess sets photographed across 64 squares from static overhead camera (3,200 crops). | **APPROVED (Primary Historical Benchmark)** | Permitted for commercial model training with attribution. Evaluated in Phase 4.3D and Milestone 5.0/5.1. |
| **ChessReD (Chess Recognition Dataset)** | 4TU.ResearchData (2024) | CC BY-NC-SA 4.0 (published); commercial permission unverified | TU Delft research. 10,800 smartphone photos across 100 complete games from varied angles. | **EXCLUDED FROM ELIGIBLE TRAINING POOL** | Published under **CC BY-NC-SA 4.0**; separate commercial permission unverified. Additionally captured from **a single physical chess set**, providing zero set diversity. |
| **FENiT Annotated Image Dataset** | Mendeley Data (2026) | CC BY 4.0 (verified) | Binus University research. CVAT annotations of smartphone chessboard images with 4-point corner keypoints. | **APPROVED FOR CORNER BENCHMARKING** | Limited to 1–2 tournament sets (provisional). Insufficient for piece carving diversity. |
| **samryan18/chess-dataset** | GitHub (2020) | MIT (verified) | Handcrafted photos by Sam Ryan & Mukund V. (500 photos mapped to FEN). | **APPROVED AS REFERENCE PIPELINE** | Single wooden chessboard. Insufficient physical set diversity. |
| **open-chess-photos** | `OffeneDatenmodellierung/open-chess` (v0.1.8, 2026) | `LicenseRef-open-chess-photo-licence-1` | Community photos submitted to open-chess server instances under custom photo licence. | **EXCLUDED (Unverified Third-Party Rights & Unreleased Raw Data)** | Licence granted strictly to server operator and open-chess project. Third-party rights to download and commercialize weights are unverified. Raw photos are not publicly published. |
| **open-chess-public** | `OffeneDatenmodellierung/open-chess` (v0.1.8, 2026) | **CC0-1.0** (verified terms) | Contributor photos with explicit CC0 opt-in (`--public-only`). | **CONDITIONALLY APPROVED LEGALLY; EXCLUDED DUE TO ZERO PUBLIC AVAILABILITY** | Permitted under CC0, but no public standalone export archive or dataset dump exists (requires private server DB access). Groups by player-day rather than verified physical set identity, posing cross-split leakage hazard. |
| **Wikimedia Commons / Stock CC0 Photo Collections** | Wikimedia Commons / Unsplash / Pexels | CC0 1.0 Universal (verified) | Community-contributed photography of diverse chess sets. | **CONDITIONALLY APPROVED AS REFERENCE POOL; REJECTED AS DIRECT TRAINING DATA** | Lacks 4-point board corner annotations and square labels; unlinked set identities prevent leak-free partitioning. |
| **notnil/fenify-3D** | GitHub (2023) | MIT (code only) | 9,505 crowdsourced Mechanical Turk photos of physical chess boards. | **REJECTED (Unreleased Raw Data)** | Raw crowdsourced images retained in private cloud storage and omitted from public repository. |
| **Roboflow "Chess Pieces Detection"** | Roboflow Universe (v1-v4) | CC BY 4.0 / CC0 (unverified) | Anonymous upload on carpeted floor; single wooden set; 21 corrupted class labels. | **REJECTED (Failed Provenance Gate)** | Unknown image copyright holders; corrupted annotations; single physical set. |
| **Kaggle "Chess Pieces Detection" (Anshul Mehta)** | Kaggle (651 images) | CC0 (unverified) | Web scrapes from Google Images of isolated pieces on white/transparent backgrounds. | **REJECTED (Unverified Scraped Provenance)** | Lacks chessboard square context, corner keypoints, and auditable copyright lineage. |
| **Kaggle "Chessman Image Dataset"** | Kaggle (approx. 550 images) | CC BY-NC-SA 4.0 | Scraped from web search engines across 6 piece categories. | **REJECTED (Non-Commercial Hazard)** | Explicit **Non-Commercial (NC)** restriction prevents inclusion in commercial distributions. |
| **LiveChess2FEN** | GitHub (`davidmallasen/LiveChess2FEN`) | AGPL-3.0 | Academic thesis research. Full pipeline with trained models. | **REJECTED (Copyleft Hazard)** | **GNU Affero GPL v3.0** is viral copyleft and incompatible with MIT distribution. |
| **ChessCog & ChessRender360** | GitHub / Zenodo | MIT / CC BY 4.0 | 3D Blender computer graphics renders (4,888 and 10,000 renders respectively). | **REJECTED (Synthetic Only)** | Does not provide real physical piece imagery or solve the sim-to-real domain gap. |
| **Synthetic 3D Staunton Renders (In-House)** | Procedural Raytracer / Blender | **MIT / CC0 (Author-Owned)** | Procedural 3D Staunton meshes based on Nathaniel Cook's 1849 public domain blueprints. | **APPROVED (Primary Synthetic Training Source)** | 100% clean lineage. Zero third-party copyright claims. Fully reproducible and license-safe. |
| **Author-Owned Physical Capture Protocol (Milestone 5.2G Plan A)** | Planned In-House Protocol (`docs/PHYSICAL_CAPTURE_PROTOCOL.md`) | **Data: CC0-1.0; Code: MIT** (Strictly Decoupled) | 20 distinct physical Staunton sets across 4 style categories: 640 distinct physical piece instances, 28 scenes/set (560 full-board scenes = 35,840 context crops), 640 instance ref photos, 15,360 sweep photos, and 60 empty-board photos (1,280 canonical empty crops, 3,840 total empty crops). | **RECOMMENDED FOR FUTURE PHYSICAL RECOGNITION** | Bridges the 14-set global diversity gap with 100% auditable copyright ownership, bijective physical tag mapping, and machine-validated manifest. |

---

## 4. Current Repository State & Blocker Disclosure

**Status as of Milestone 5.2G:**
1. **No External Real-World Dataset Packaged:** In accordance with the repository's strict clean-room policy, no unverified real-image datasets have been imported or committed to `C:\dev\chess`.
2. **Prototype Baseline:** The Phase 4.3A prototype implementation (`classifyPiece.ts`) includes the mathematical feature extractor (288-bin HOG), scratch-buffer memory manager, model loader, and evaluation harness.
3. **Synthetic-Only Evaluation:** All unit tests and training demonstrations operate strictly on synthetic fixtures and procedural Staunton silhouettes. Results are labeled **synthetic-only**.
4. **Durable Capture Workspace & Crop Extractor Established:** In Milestone 5.2G, established a durable private capture workspace at `C:\dev\chess-capture-workspace\` outside Git tracking. Implemented the offline square-crop extraction engine in TypeScript (`src/vision/cropExtractor.ts`) and Python (`scripts/extract_square_crops.py`), verified across 179 unit tests.
5. **Set 01 Preflight Verified; Physical Pilot Pending:** Set 01 physical prerequisites (public domain Staunton design, author property release, 32 archival underside micro-tags, optical rig, CPL filter) are verified and operational. Live physical pilot capture execution is reported as **PENDING** physical hardware collection, with zero synthetic substitutions.
6. **Production Model Training Gate:** Final production weights ($\ge 88.0\%$ Top-1 on diverse physical sets) remain blocked until physical piece collection is executed on hardware. Production weights in `src/vision/pieceModelWeights.ts` remain strictly **UNCHANGED**.
