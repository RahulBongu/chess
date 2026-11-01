# Chess Piece Recognition: Dataset Provenance & Licensing Policy

Date: 2026-11-01  
Milestone: 4.0 — Phase 4.3A  
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

---

## 3. Audit of Public Chess Datasets

| Dataset Candidate | Source & Revision | Stated License | Underlying Provenance | Gate Decision | Reason / Restrictions |
|---|---|---|---|---|---|
| **Roboflow "Chess Pieces Detection"** | Roboflow Universe (v1-v4) | CC BY 4.0 / CC0 (unverified) | Web scrapes from Google Images and social media; contains digital screenshots and mixed physical sets. | **REJECTED (Failed Provenance Gate)** | Unknown image copyright holders; unverified attribution chain; high risk of copyright contamination. |
| **Kaggle "Chessman Image Dataset"** | Kaggle (approx. 550 images) | CC BY-NC-SA 4.0 | Scraped from web search engines across 6 piece categories. | **REJECTED (Non-Commercial Hazard)** | Explicit **Non-Commercial (NC)** restriction prevents inclusion in commercial distributions. |
| **ChessCam Benchmark Dataset** | Academic / GitHub Benchmark | CC BY 4.0 (audited) | Physical tournament boards and Staunton pieces photographed under lab lighting. | **CONDITIONALLY APPROVED (External Benchmark Only)** | Acceptable for offline accuracy cross-validation; attribution required in distribution notices. |
| **Synthetic 3D Staunton Renders (In-House)** | Procedural Raytracer / Blender | **MIT / CC0 (Author-Owned)** | Procedural 3D Staunton meshes based on Nathaniel Cook's 1849 public domain blueprints. | **APPROVED (Primary Training Source)** | 100% clean lineage. Zero third-party copyright claims. Fully reproducible and license-safe. |

---

## 4. Current Repository State & Blocker Disclosure

**Status as of Milestone 4.0 Phase 4.3A:**
1. **No External Real-World Dataset Packaged:** In accordance with the repository's strict clean-room policy, no unverified real-image datasets have been imported or committed to `C:\dev\chess`.
2. **Prototype Baseline:** The Phase 4.3A prototype implementation (`classifyPiece.ts`) includes the mathematical feature extractor (288-bin HOG), scratch-buffer memory manager, model loader, and evaluation harness.
3. **Synthetic-Only Evaluation:** All unit tests and training demonstrations in this milestone operate strictly on synthetic fixtures and procedural Staunton silhouettes. Results are labeled **synthetic-only**.
4. **Production Model Training Gate:** Final production weights ($\ge 90\%$ top-1 on diverse physical sets) remain blocked until an explicitly licensed physical dataset is provided or captured under controlled room conditions on the vivo V2336.
