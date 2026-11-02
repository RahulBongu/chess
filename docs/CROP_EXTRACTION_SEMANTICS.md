# Crop Extraction Semantics and Dataset Accounting

**Milestone:** 5.2H  
**Date:** 2026-11-02  
**Starting commit:** `1ae5f50`  
**Scope:** Software audit of the offline crop extractor and capture-manifest validator. No physical photographs were captured in this milestone.

---

## 1. Why this audit exists

Milestone 5.2G introduced an offline extractor but did not distinguish full-board scenes from isolated sweep/reference photographs. Tracing the actual code showed that every invocation emitted **64 crops and 64 manifest records**, including for a single-piece photograph. The Python CLI also silently resized any non-800x800 image to 800x800 and hard-coded every same-type piece to instance `01`. Those behaviours would have produced mislabeled data and an incorrect source-photo/crop audit trail.

Milestone 5.2H replaces the implicit behaviour with three explicit, validated modes. The TypeScript engine and Python CLI use the same mode names, validation rules and output counts.

---

## 2. Canonical mode matrix

| Mode | Source photograph | Crops emitted | Manifest `crop_type` | Required ground truth |
| :--- | :--- | :---: | :--- | :--- |
| `full_board_context` | Full-board starting or midgame scene | **64** | `full_board_context` | Valid placement or complete FEN **and** a square-to-piece-instance map |
| `occupied_target_only` | Isolated sweep or individual piece reference | **1** | `isolated_piece` | Placement with exactly one occupied square, target square, target canonical piece ID |
| `empty_board_grid` | Empty-board lighting reference | **64** | `empty_square` | Empty placement (or omitted placement, which defaults to `8/8/8/8/8/8/8/8`) |

A source photograph is counted once. Derived crops are counted separately. The extractor reports both `sourcePhotoCount` / `source_photo_count` and `derivedCropCount` / `derived_crop_count`.

### 2.1 `full_board_context`

- Requires an already-rectified 800×800 image, or explicit `--corners` for a documented homography.
- Emits exactly 64 crops in white-bottom or black-bottom orientation.
- Each crop receives the FEN occupant at that square.
- Every occupied square must have an entry in the supplied piece map.
- Piece map values are canonical IDs such as `SET01_W_P01`; `EMPTY` is allowed only on empty squares.
- The map is rejected if a physical piece ID appears twice, if a map value contradicts the FEN type/colour, or if an occupied square is missing.
- The extractor refuses to invent physical identities from FEN types alone.

Example piece-map fragment:

```json
{
  "a1": "SET01_W_R01",
  "b1": "SET01_W_N01",
  "c1": "SET01_W_B01",
  "d1": "SET01_W_Q01",
  "e1": "SET01_W_K01",
  "a2": "SET01_W_P01",
  "b2": "SET01_W_P02"
}
```

Only occupied squares need to be listed. Empty squares default to `EMPTY` / `NONE`.

### 2.2 `occupied_target_only`

- The supplied placement must have exactly **one** occupied square.
- `--target-square` must equal that square.
- `--piece-instance-id` must match the FEN piece type, colour and set prefix.
- Exactly one crop is written for the target square.
- An ambiguous placement, wrong target coordinate, mismatched piece identity, or missing target identity is a hard failure.

### 2.3 `empty_board_grid`

- The placement must contain zero occupied squares.
- Exactly 64 crops are written, all labelled `EMPTY` with physical tag `NONE`.
- This supports the three empty-board lighting setups. The diffuse set contributes the canonical empty class; the specular and oblique sets remain reserved for stress testing.

---

## 3. Rejection rules

The extractor and manifest validator reject, rather than silently repair:

- malformed or incomplete FEN placement;
- non-800x800 input without explicit corners;
- silent resizing of oblique photographs;
- a full-board scene without a physical piece-instance map;
- duplicate physical piece IDs within one scene;
- a map value that contradicts the FEN occupant;
- a single-piece mode with zero or multiple occupied squares;
- a target square that does not match the occupied square;
- a target identity whose type, colour or set prefix disagrees with the placement;
- an `extraction_mode` that disagrees with `crop_type`;
- a square label that disagrees with the supplied placement/FEN;
- a source image mapped to multiple scene IDs, or a scene ID mapped to multiple source images;
- duplicate derived crop paths;
- cross-split set, piece, or physical-tag leakage.

`validate_capture_manifest.py --check-files` additionally requires every `original_image_path` and `derived_crop_path` to exist on disk.

---

## 4. Exact commands

### 4.1 Full-board scene

```bash
py scripts/extract_square_crops.py \
    --image C:/dev/chess-capture-workspace/raw/SET01/SET01_SCENE_START_ZENITH.jpg \
    --mode full_board_context \
    --board-fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    --piece-map C:/dev/chess-capture-workspace/manifests/SET01_start_piece_map.json \
    --out-dir C:/dev/chess-capture-workspace/crops/train/SET01 \
    --set-id SET01 \
    --scene-id SET01_SCENE_START_ZENITH \
    --camera-angle zenith \
    --lighting-setup diffuse_daylight \
    --manifest-out C:/dev/chess-capture-workspace/manifests/SET01_manifest.jsonl
```

Output: **1 source photograph → 64 crops → 64 manifest records.**

### 4.2 Isolated sweep or piece reference

```bash
py scripts/extract_square_crops.py \
    --image C:/dev/chess-capture-workspace/raw/SET01/SET01_SWEEP_W_P01_E4.jpg \
    --mode occupied_target_only \
    --placement "8/8/8/8/4P3/8/8/8" \
    --target-square e4 \
    --piece-instance-id SET01_W_P01 \
    --out-dir C:/dev/chess-capture-workspace/crops/train/SET01 \
    --set-id SET01 \
    --scene-id SET01_SWEEP_W_P01_E4 \
    --camera-angle zenith \
    --lighting-setup diffuse_daylight \
    --manifest-out C:/dev/chess-capture-workspace/manifests/SET01_manifest.jsonl
```

Output: **1 source photograph → 1 target crop → 1 manifest record.**

### 4.3 Empty-board reference

```bash
py scripts/extract_square_crops.py \
    --image C:/dev/chess-capture-workspace/raw/SET01/SET01_EMPTY_DIFFUSE.jpg \
    --mode empty_board_grid \
    --out-dir C:/dev/chess-capture-workspace/crops/train/SET01 \
    --set-id SET01 \
    --scene-id SET01_EMPTY_DIFFUSE \
    --camera-angle zenith \
    --lighting-setup diffuse_daylight \
    --manifest-out C:/dev/chess-capture-workspace/manifests/SET01_manifest.jsonl
```

Output: **1 source photograph → 64 empty-square crops → 64 manifest records.**

### 4.4 Validate a real capture manifest

```bash
py scripts/validate_capture_manifest.py \
    C:/dev/chess-capture-workspace/manifests/SET01_manifest.jsonl \
    --check-files
```

---

## 5. Dataset accounting after the fix

| Capture category | Source photographs | Crops per photo | Derived crops |
| :--- | ---: | ---: | ---: |
| Full-board scenes (28/set × 20 sets) | 560 | 64 | 35,840 |
| Isolated sweep (768/set × 20 sets) | 15,360 | 1 | 15,360 |
| Individual piece references (32/set × 20 sets) | 640 | 1 | 640 |
| Empty-board references (3/set × 20 sets) | 60 | 64 | 3,840 |
| **Total** | **16,620** | — | **55,680** |

The canonical initial training/evaluation pool remains **52,480** crops:

- full-board context: 35,840;
- isolated sweep: 15,360;
- diffuse empty-board set: 1,280 (64 per set).

Two categories remain excluded from initial training:

- **640 piece-reference crops** — reserved for human QC and physical-instance cataloguing;
- **2,560 specular/oblique empty crops** — reserved for shadow stress tests and hard-negative mining.

`35,840 + 15,360 + 1,280 = 52,480`; `52,480 + 640 + 2,560 = 55,680`.

Set assignment is unchanged: Sets 01–14 train, Sets 15–17 validation, Sets 18–20 blind holdout. Set 01 remains in the training partition.

---

## 6. Synthetic software verification

```bash
node --test --experimental-strip-types tests/cropExtractor.test.ts tests/manifestValidator.test.ts
py -m unittest tests.test_capture_pipeline -v
```

The Python suite creates temporary 800×800 synthetic fixtures, extracts crops, writes a temporary manifest, and validates it with `--check-files`. These are software tests only. They are **not** physical Set 01 evidence and do not constitute a completed pilot.

---

## 7. Physical pilot status

**PENDING.** As of this milestone:

- `C:\dev\chess-capture-workspace\raw\SET01\` contains **0 source photographs**;
- no derived Set 01 crops exist;
- no physical pilot manifest exists;
- no image-to-label association has been inspected.

The blocker is live studio capture: the physical set, 32 underside tags, camera, tripod, lighting, CPL filter and operator actions must be present. Running synthetic tests does not satisfy that requirement. The genuine Set 01 pilot remains pending until actual photographs exist and the manifest validates against those files.
