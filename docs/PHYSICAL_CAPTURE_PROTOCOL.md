# Physical Capture Protocol — Set 01 Preflight and Pilot Execution

**Milestone:** 5.2H  
**Date:** 2026-11-02  
**Target repository:** `https://github.com/RahulBongu/chess.git`  
**Starting commit:** `1ae5f50` (verified synchronized with `origin/main`)  
**Production model weights:** `src/vision/pieceModelWeights.ts` (version `1.0.0-synthetic-staunton-v1`, strictly **UNCHANGED**)  
**Preserved release artifacts:** all 35 files under `dist/`, including `dist/ai-chess-navigator-1.0.0-arm64-milestone51b-bgr-validation-20261101.apk`

---

## 1. Executive summary

Milestones 4.3D–5.2G established why an in-house physical dataset is required:

1. Synthetic-only models collapse on real pieces (16.80% Top-1 in the earlier benchmark).
2. Small physical datasets trained from scratch overfit wood grain and piece texture.
3. Public datasets either lack commercial redistribution rights, lack public archives, or contain only one physical board.
4. Canonical manifest IDs and physical underside tags were reconciled in Milestone 5.2F.

Milestone 5.2H audited the extraction code itself. The audit found that the 5.2G extractor emitted 64 crops for every source image, including single-piece sweeps and references; the Python CLI also silently resized non-canonical inputs and hard-coded same-type pieces to instance `01`. Those behaviours were corrected with three explicit modes. The genuine physical pilot remains **PENDING** because no physical photographs exist yet.

> A passing synthetic software test is not a physical pilot result. No photographs, crops, or evidence for Set 01 were fabricated in this milestone.

---

## 2. Capture taxonomy and dataset accounting

### 2.1 Twenty-set target

- 20 distinct physical chess sets;
- 640 distinct physical pieces (32 per set);
- 12 piece classes plus one empty class;
- 4 camera viewpoints: zenith, White perspective, Black perspective, lateral flank;
- 3 lighting environments: diffuse daylight, overhead specular, directional oblique;
- 5 board layouts: starting position plus four midgame configurations.

### 2.2 Source photographs vs derived crops

| Category | Source photographs | Crops per photo | Derived crops |
| :--- | ---: | ---: | ---: |
| Full-board scenes (28/set × 20) | 560 | 64 | 35,840 |
| Isolated sweep (768/set × 20) | 15,360 | 1 | 15,360 |
| Individual piece reference (32/set × 20) | 640 | 1 | 640 |
| Empty-board reference (3/set × 20) | 60 | 64 | 3,840 |
| **Total** | **16,620** | — | **55,680** |

The 560 full-board scenes comprise 12 starting-position scenes and 16 midgame scenes (4 positions × 4 angles) per set.

### 2.3 Canonical initial pool and exclusions

The initial training/evaluation pool is **52,480** crops:

- 35,840 full-board context crops;
- 15,360 isolated sweep target crops;
- 1,280 diffuse empty-board crops (64 per set).

Excluded from initial training:

- **640 piece-reference crops** — reserved for human QC and physical-instance cataloguing;
- **2,560 specular/oblique empty crops** — reserved for shadow stress tests and hard-negative mining.

`35,840 + 15,360 + 1,280 + 640 + 2,560 = 55,680`.

### 2.4 Split policy

| Sets | Split | Canonical crops |
| :--- | :--- | ---: |
| 01–14 | train | 36,736 |
| 15–17 | val | 7,872 |
| 18–20 | holdout | 7,872 |

Set 01 is in the training partition. All images and crops of one physical set stay in exactly one split. Sets 18–20 are never used for model selection.

---

## 3. Extraction semantics

Full details and commands are in [`docs/CROP_EXTRACTION_SEMANTICS.md`](CROP_EXTRACTION_SEMANTICS.md).

| Mode | Emits | Manifest crop type | Ground truth |
| :--- | ---: | :--- | :--- |
| `full_board_context` | 64 | `full_board_context` | Valid placement/FEN plus square-to-piece-instance map |
| `occupied_target_only` | 1 | `isolated_piece` | Placement with exactly one occupied square, target square, canonical piece ID |
| `empty_board_grid` | 64 | `empty_square` | Empty placement |

### 3.1 Rectification policy

- An 800×800 image is treated as already rectified.
- A non-800×800 image requires explicit `--corners`.
- The CLI never silently resizes an oblique raw photograph into an 800×800 "rectified" board.

### 3.2 Full-board identity requirement

A full-board scene includes multiple same-type pieces. The extractor therefore requires a JSON map from occupied square to canonical physical piece ID. It refuses to invent identities such as assigning every white pawn to `SET01_W_P01`.

The map is validated against the FEN:

- every occupied square must be present;
- type, colour and set prefix must agree;
- a physical piece ID cannot appear on two squares in one scene;
- `EMPTY` is allowed only on empty squares.

### 3.3 Hard failures

The extractor and manifest validator reject:

- malformed or incomplete placement/FEN;
- non-canonical input without corners;
- missing or contradictory piece maps;
- zero or multiple occupied squares in `occupied_target_only`;
- wrong target coordinate or mismatched piece identity;
- `extraction_mode` / `crop_type` disagreement;
- square labels that contradict the placement;
- source-image/scene-ID conflicts;
- duplicate derived crop paths;
- cross-split set, piece, or tag leakage.

---

## 4. Piece namespace and physical tags

1. **Canonical identifier** (`piece_instance_id`): `SET01_W_P01`, used in manifests.
2. **Physical underside tag** (`physical_tag_id`): `S01-W-P01`, affixed to the piece.

The mapping is a deterministic one-to-one bijection:

```text
SET{ss}_{C}_{T}{nn}  <->  S{ss}-{C}-{T}{nn}
EMPTY                <->  NONE
```

Format rules:

- `piece_instance_id`: `^SET[0-9]{2}_[WB]_[PNBRQK][0-9]{2}$`, or `EMPTY`;
- `physical_tag_id`: `^S[0-9]{2}-[WB]-[PNBRQK][0-9]{2}$`, or `NONE`.

A physical tag may never map to two different pieces, and no tag or piece may appear in more than one split.

---

## 5. Board layouts and FEN ground truth

| Position ID | Scenario | Placement | Complete FEN | Pieces |
| :--- | :--- | :--- | :--- | ---: |
| `POS_00_START` | Starting position | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR` | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` | 32 |
| `POS_01_OPENING` | Berlin-style opening | `r1bq1rk1/ppp1bppp/2np1n2/1B2p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1` | `r1bq1rk1/ppp1bppp/2np1n2/1B2p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7` | 32 |
| `POS_02_TACTICAL` | Tactical middlegame | `r3r1k1/ppqbbppp/2p2n2/8/3N4/2PB4/PP1Q1PPP/R3R1K1` | `r3r1k1/ppqbbppp/2p2n2/8/3N4/2PB4/PP1Q1PPP/R3R1K1 w - - 4 16` | 25 |
| `POS_03_ASYMMETRIC` | Opposite castling | `3r1rk1/pb1n1p1p/1p4p1/2p5/2P2PP1/P1N1B2P/1P6/2KR3R` | `3r1rk1/pb1n1p1p/1p4p1/2p5/2P2PP1/P1N1B2P/1P6/2KR3R w - - 0 20` | 22 |
| `POS_04_ENDGAME` | Rook endgame | `4r3/3R1p2/5kp1/p6p/P6P/5KP1/5P2/8` | `4r3/3R1p2/5kp1/p6p/P6P/5KP1/5P2/8 w - - 2 34` | 12 |

Position 1 is reconstructed from the legal sequence `1.e4 e5 2.Nf3 Nc6 3.Bb5 Nf6 4.d3 d6 5.Nc3 Be7 6.O-O O-O`. Positions 2–4 are constructed snapshots and are documented as such.

---

## 6. Set 01 pilot categories

1. Starting-position full board, zenith, diffuse light.
2. Starting-position full board, White perspective, diffuse light.
3. `POS_01_OPENING` full board.
4. `POS_02_TACTICAL` full board.
5. `POS_03_ASYMMETRIC` full board.
6. `POS_04_ENDGAME` full board.
7. Physical reference photograph of `SET01_W_Q01`.
8. Physical reference photograph of `SET01_W_K01`.
9. Position-invariance sweep photograph of `SET01_W_P01` on `e4`.
10. Empty-square reference for `d4` from an empty-board lighting scene.

Expected software outputs once real photographs exist:

- categories 1–6: 6 × 64 = 384 full-board context crops;
- categories 7–9: 3 × 1 = 3 isolated target crops;
- category 10: 64 empty-square crops (the `d4` crop is the cited example, but the empty-board mode emits the full grid).

These are expected counts, not captured results.

---

## 7. Durable private workspace

**Path:** `C:\dev\chess-capture-workspace\`

```text
C:\dev\chess-capture-workspace\
├── raw\
│   └── SET01\             original photographs, unmodified
├── crops\
│   ├── train\
│   │   └── SET01\         derived 100x100 crops
│   ├── val\
│   └── holdout\
├── manifests\             JSONL manifests and piece maps
├── rejected\              ambiguous, out-of-focus, or glare-clipped sources
└── reports\               validation reports and audit logs
```

An `800×800` rectified working copy may be created for extraction, but the original source photograph must be retained unchanged. No raw image, crop, cache, checkpoint, or local manifest is committed to Git.

---

## 8. Operational checklist

1. Confirm all 32 physical pieces and underside tags are present.
2. Confirm the camera, tripod, lighting, CPL filter, and property-release record.
3. Capture the source photograph; keep the original unchanged.
4. Rectify with documented corners or provide an already-rectified 800×800 image.
5. Create the square-to-piece-instance map for every full-board layout.
6. Run the appropriate extraction mode.
7. Validate the manifest with `--check-files`.
8. Inspect the actual images and crops; log rejected/ambiguous samples rather than deleting them.
9. Record measured results in this protocol and `docs/DATASET_PROVENANCE.md`.
10. Do not claim that one set proves cross-set generalisation or classifier accuracy.

---

## 9. Pilot status

**PENDING.** The durable workspace contains zero Set 01 source photographs, zero derived crops, and no physical manifest.

The operational blocker is live studio capture. The repository documents the required set, tags, optics, lighting and release, but the physical operator actions have not been performed in this milestone. No synthetic fixture is treated as physical evidence.

Once real photographs exist:

```bash
py scripts/validate_capture_manifest.py \
    C:/dev/chess-capture-workspace/manifests/SET01_manifest.jsonl \
    --check-files
```

The pilot is complete only when that command validates against actual files and the image-to-label associations have been manually inspected.

---

## 10. Milestone 5.2H acceptance gates

| Gate | Requirement | Result |
| :--- | :--- | :--- |
| 0 | Preserve repository, `dist/`, build tree, weights | **PASS** — 35 `dist/` files, `modules/live-camera/android/build/` untouched, no APK build, no training |
| 1 | Inspect actual extraction behaviour | **PASS** — traced 64-crops-for-everything bug, silent resize, and hard-coded `_01` identities |
| 2 | Add explicit extraction modes and validation | **PASS** — `full_board_context`, `occupied_target_only`, `empty_board_grid` in TypeScript and Python |
| 3 | Deterministic tests and total reconciliation | **PASS** — synthetic tests cover 64 / 1 / 64 outputs, 32-photo workflow, rejection paths, and 16,620 / 55,680 accounting |
| 4 | No fake pilot | **PASS** — Set 01 physical pilot remains PENDING; zero photographs claimed |
| 5 | Tests and publication | **PASS after commit** — `npx tsc --noEmit` reported 0 errors; Node suite passed 186/186; Python suite passed 7/7; manifest validation was exercised against generated files. `expo lint` is **UNVERIFIED**: the repository has no ESLint configuration or lint dependency, and automatic bootstrap failed, so no lint result is claimed. |

**Overall decision:** the extraction software and accounting are approved for physical capture. The Set 01 physical pilot is **not complete**.
