# Milestone 5.3D — Explicit Geometry Accounting, Strict Production Gates, and Calibration Robustness

**Milestone:** 5.3D  
**Date:** 2026-11-03  
**Target Repository:** `https://github.com/RahulBongu/chess.git`  
**Starting Commit:** `4325310` (verified synchronized with `origin/main`)  
**Production Model Weights:** `src/vision/pieceModelWeights.ts` (version `1.0.0-synthetic-staunton-v1`, strictly **UNCHANGED**)  
**Preserved Release Artifacts:** All 35 top-level `dist/` entries intact; `modules/live-camera/android/build/` untouched.  
**Physical Capture Status:** Set 01 physical capture remains explicitly **PENDING**; external benchmarks do not substitute for in-house studio capture.

---

## 1. Executive Summary

Milestone 5.3D replaces inferred benchmark accounting with explicit per-image records. Every image now carries four independent facts for each approach — `candidate_generated`, `geometry_valid`, `accepted`, and `rejection_reason` — plus a reproducible JSONL record and a SHA-256 digest. Aggregate counts and means are computed only from those records, and an internal assertion fails the benchmark if a summary disagrees with its records.

The milestone also corrects three claims that were not supported by the earlier implementation:

1. **Approach B strict production was never actually evaluated in 5.3C.** The earlier “0 / 385” claim came from an asserted threshold, not a measured checkerboard-pattern gate. 5.3D implements the production pattern evidence in Python with the same thresholds and multi-inset sampling as `src/vision/boardEvidence.ts`. The measured result is **4 / 385 accepted (1.04%)**, with **89.51%** conditional accepted IoU. The earlier 0/385 claim is withdrawn.
2. **Approach A’s earlier 0 / 385 accepted count was hard-coded.** 5.3C returned confidence `0.25` for every candidate, which could never clear the `0.32` production threshold. 5.3D evaluates the production pattern gate over a shortlist of candidates and measures **9 / 385 accepted (2.34%)**. This remains far below autonomous use.
3. **Approach D’s earlier 384 / 385 count depended on a different validity check and unclamped simulated taps.** 5.3D clamps simulated touches to the decoded-image boundary, applies the shared structural validator, and reports all three required views: initial baseline, paired before/after, and all-input penalized. The measured result is **385 / 385 structurally valid**, with paired IoU improving from **81.75% to 83.54%**. This is a synthetic simulation, not human-operator or device accuracy.

The relaxed Approach B result (**273 / 385 accepted, 55.68% conditional IoU, 39.48% all-input IoU**) is retained only as an experimental proposal-quality diagnostic. It is never presented as production performance.

---

## 2. Benchmark Method and Accounting Contract

### 2.1 Dataset and evaluation-only labels

- **Dataset:** FENiT chessboard photographs (Mendeley Data DOI `10.17632/wpp9xnbcp6.1`), CC BY 4.0.
- **Processed images:** 385 of 491; 105 skipped and 1 bowtie failure are inherited from the frozen external benchmark manifest.
- **View split:** 30 frontal (`white_perspective` / `black_perspective`) and 355 lateral.
- **Ground-truth use:** FENiT corner annotations are used **only** for IoU and corner-error evaluation. They are never detector inputs for Approaches A, B, or C. Approach D is explicitly a ground-truth-seeded simulation and is disclosed as such.

### 2.2 Per-image record schema

Each image produces one record per approach. The required fields are:

| Field | Meaning |
| :--- | :--- |
| `candidate_generated` | The approach emitted a quadrilateral proposal. |
| `geometry_valid` | The proposal passed production structural checks: finite corners, in-bounds, non-self-intersecting, strictly convex, at least 5% of image area, and at least 15% span in both axes. |
| `accepted` | The proposal passed the approach-specific acceptance gate. |
| `rejection_reason` | Machine-readable code for the first failing stage, or `null` when accepted. |
| `rejection_stage` | `candidate`, `geometry`, or `acceptance`. |
| `iou`, `corner_error` | Metrics for the proposal, with documented penalties. |
| `pattern` | Checkerboard evidence metrics for A and B, or `null` where pattern evidence is not part of the gate. |
| `pattern_skipped_reason` | Why pattern sampling was not attempted (for example, boundary score below the production pre-sampling minimum), or `null`. |
| `strict_production` | Approach B only: the separate strict production gate result. |

Missing or structurally invalid proposals contribute **IoU 0.0** and **corner error 1.0**. Accepted-by-threshold is not assumed from metric values.

### 2.3 Three metric views with explicit denominators

| View | Denominator | Definition |
| :--- | :---: | :--- |
| `accepted_quality` | `accepted_count` | Conditional mean over accepted records only. |
| `proposal_quality` | `geometry_valid_count` | Conditional mean over geometry-valid proposals, regardless of acceptance. |
| `all_input_penalized` | `total_images` (385) | Every input counts; rejected or invalid proposals are penalized with IoU 0.0 and error 1.0. |

These views are not interchangeable. In particular, the relaxed B conditional IoU of 55.68% and its all-input IoU of 39.48% answer different questions and share the same 273 accepted proposals.

### 2.4 Acceptance gates

| Approach | Gate |
| :--- | :--- |
| **A — Production baseline** | Structural checks + 8 × 8 checkerboard pattern evidence (`fit ≥ 0.55`, `alternation ≥ 0.80`, `contrast ≥ 8`, `score ≥ 0.45`, `contrast > noise`) + `boundary × pattern ≥ 0.32`. |
| **B — Relaxed** | Any structurally valid convex quadrilateral. Experimental only. |
| **B — Strict production** | Same structural checks and checkerboard pattern evidence as A, evaluated on the same B proposal. |
| **C — Hough experimental** | Its own confidence `≥ 0.32`; no checkerboard pattern evidence. This is not a production gate. |
| **D — Simulated assisted calibration** | Ground-truth corners + Gaussian touch noise + local Shi-Tomasi snapping; accepted when refined corners remain structurally valid. |

Approach A ranks up to the **20 largest** structurally eligible contour candidates by combined boundary × pattern confidence, matching production's candidate-ranking behavior while bounding benchmark runtime. If every shortlisted candidate fails geometry, the first failing candidate is recorded; a valid smaller candidate can still be selected when the largest contour extends outside the frame. Pattern sampling is skipped when the boundary score is below half the production threshold (`0.16`); those records carry `pattern_skipped_reason = boundary_score_below_production_minimum` and are rejected at the acceptance stage, not the geometry stage.

### 2.5 Reproducible records

The benchmark writes both the aggregate JSON and a JSONL records file:

```text
C:\dev\chess-capture-workspace\external\fenit\reports\board_geometry_benchmark.json
C:\dev\chess-capture-workspace\external\fenit\reports\board_geometry_benchmark_records.jsonl
```

For this run:

- Records: **1,540** (`385 × 4` approach records).
- SHA-256: `3483e80cd401e090040748235baca6bf9138cda48afdcd0dd9f2bbbf0426dd0a`.
- Runtime: **329.73 s**.
- The digest is computed over the exact UTF-8 bytes written (LF line endings).

Reproduction command:

```powershell
py scripts\benchmark_board_geometry.py
```

---

## 3. Verified Benchmark Results (385 Real FENiT Boards)

### 3.1 Candidate, geometry, and acceptance accounting

| Approach | Candidates | Geometry valid | Accepted | Accepted % |
| :--- | :---: | :---: | :---: | :---: |
| **A — Production baseline** | 385 / 385 | 140 / 385 | **9 / 385** | **2.3377%** |
| **B — Relaxed (NOT production)** | 273 / 385 | 273 / 385 | 273 / 385 | 70.9091% |
| **B — Strict production** | 273 / 385 | 273 / 385 | **4 / 385** | **1.0390%** |
| **C — Hough experimental** | 344 / 385 | 175 / 385 | 0 / 385 | 0.0000% |

### 3.2 Metric views

| Approach | Accepted IoU | Accepted error | Proposal IoU (geometry-valid) | Proposal error | All-input IoU | All-input error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A — Production** | **88.7616%** (N=9) | **2.1300%** | 30.9830% (N=140) | 21.5667% | **2.0749%** (N=385) | **97.7121%** |
| **B — Relaxed** | 55.6836% (N=273) | 15.6832% | 55.6836% (N=273) | 15.6832% | 39.4848% (N=385) | 40.2118% |
| **B — Strict production** | **89.5085%** (N=4) | **1.4586%** | 55.6836% (N=273) | 15.6832% | **0.9300%** (N=385) | **98.9762%** |
| **C — Hough** | N/A (N=0) | N/A | 43.7144% (N=175) | 19.6612% | 0.0000% (N=385) | 100.0000% |

The accepted IoU figures are conditional and must not be compared to all-input figures without their denominators. For example, A’s 88.76% describes only the nine images that passed the production gate; 376 images were rejected.

### 3.3 Approach B reconciliation and correction

The two historically reported B figures are both reproducible under explicit definitions:

- **55.6836%** — accepted/proposal IoU over the **273** images where B emitted a geometry-valid candidate.
- **39.4848%** — all-input IoU over all **385** images, with the 112 missing candidates penalized to IoU 0.0.

Neither is strict production performance. Strict production requires checkerboard pattern evidence, which the relaxed gate bypasses. The measured strict gate accepts only **4 / 385 (1.04%)** B proposals, with:

- conditional accepted IoU **89.5085%** and corner error **1.4586%**;
- all-input IoU **0.9300%** and corner error **98.9762%**;
- rejections: 267 `pattern_fit_below_threshold`, 112 `no_convex_hull_quadrilateral_found`, 2 `combined_confidence_below_production_threshold`.

The 5.3C statement that strict production was “0 / 385” is therefore **withdrawn**. The correct, record-backed statement is **4 / 385**, and the 5.3C evidence did not implement the gate it described.

### 3.4 Approach D — ground-truth-seeded assisted calibration simulation

Approach D uses FENiT ground-truth corners, adds Gaussian noise with standard deviation equal to 3% of the image diagonal (seed `42`), clamps taps to the decoded-image boundary, and refines each corner with Shi-Tomasi inside a 5%-of-diagonal search window.

| View | Denominator | Mean IoU | Mean corner error |
| :--- | :---: | :---: | :---: |
| Initial perturbed taps (all input) | 385 | **81.7496%** | **3.5511%** |
| Refined, all-input penalized | 385 | **83.5411%** | **3.0073%** |
| Refined, accepted only | 385 | 83.5411% | 3.0073% |
| Paired before/after | 385 | 81.7496% → 83.5411% | 3.5511% → 3.0073% |

Paired deltas: **+1.7914 IoU percentage points** and **−0.5438 corner-error percentage points**. Paired outcomes: **240 improved, 145 worsened, 0 unchanged**. There were **0 / 385** structural rejections after refinement.

Every D record is a synthetic simulation. It measures local snapping on perturbed ground-truth and is **not** a measured human-operator, live-camera, or device result. The earlier 384/385 figure is superseded by this record-backed recomputation; no human performance is inferred from it.

### 3.5 Rejection reasons

| Approach | Rejection reasons |
| :--- | :--- |
| **A — Production** | 245 `geometry_out_of_bounds`, 90 `pattern_fit_below_threshold`, 41 `boundary_score_below_production_minimum` |
| **B — Relaxed** | 112 `no_convex_hull_quadrilateral_found` |
| **B — Strict production** | 267 `pattern_fit_below_threshold`, 112 `no_convex_hull_quadrilateral_found`, 2 `combined_confidence_below_production_threshold` |
| **C — Hough** | 175 `confidence_below_production_threshold`, 169 `geometry_out_of_bounds`, 41 `cannot_cluster_orthogonal_vanishing_directions` |
| **D — Simulated** | None (0 / 385 structural rejections) |

### 3.6 Viewpoint split (all-input penalized IoU)

| Approach | Frontal (N=30) | Lateral (N=355) |
| :--- | :---: | :---: |
| **A — Production** | 3.0303% | 1.9942% |
| **B — Relaxed** | 48.3489% | 38.7357% |
| **B — Strict production** | 3.0468% | 0.7511% |
| **C — Hough** | 0.0000% | 0.0000% |

---

## 4. Convexity, Corner Ordering, and Coordinate Quality Gates

### 4.1 Explicit strict convexity

`src/calibration/geometry.ts` now exports `isConvexQuadrilateral` and `MIN_CONVEX_TURN_SINE = 1e-6`. Each consecutive turn is measured by the sine of the turn angle, not by an absolute cross product, so the tolerance is scale-independent in normalized `[0, 1]` coordinates. A vertex with a turn sine below `1e-6` (approximately `0.000057°`) is treated as collinear.

- Concave, nearly collinear, repeated-point, and self-intersecting selections remain invalid.
- Convexity itself is orientation-agnostic: clockwise and counter-clockwise convex quads both return `true`.
- Confirmation still requires clockwise winding (`signedArea > 0`), because rectification expects top-left, top-right, bottom-right, bottom-left order.
- Reordering a concave or degenerate selection cannot make it valid. Tests cover all 24 permutations for convex, trapezoid, diamond, and sheared quadrilaterals, and verify that concave, collinear, coincident, and tiny selections stay invalid under every permutation.

### 4.2 Deterministic cyclic order is not semantic chess orientation

`reorderCalibrationPoints` starts at the screen-space top-left (`min(x + y)` with deterministic tie-breaking) and proceeds clockwise. This is a geometric convention, not proof of which physical side is White. The camera-setup screen now states the screen-space order, and board review displays the mapping card:

- White at bottom: 1 → a8, 2 → h8, 3 → h1, 4 → a1.
- Black at bottom: 1 → h1, 2 → a1, 3 → a8, 4 → h8.

Cyclic ordering cannot resolve White/Black perspective. The view selector or a re-capture remains the explicit corrective action.

### 4.3 Shared coordinate helpers

`src/calibration/coordinates.ts` centralizes preview-touch, normalized, and pixel conversions. Conventions:

- A touch location is preview-local: `(0, 0)` is the top-left of the exact View that received the touch, with no safe-area, scroll, or screen offset.
- Normalized calibration coordinates are fractions of that preview: `x = 0` / `y = 0` are the top/left edges and `x = 1` / `y = 1` are the right/bottom edges.
- Decoded-image coordinates are continuous boundary coordinates: `(0, 0)` is the top-left boundary and `(W, H)` is the bottom-right boundary. `x = 1` maps to `W`, not `W − 1`. Pixel centers remain at integers `0..W−1`.
- Captures returned by expo-camera are already rotated to the device orientation; no additional EXIF rotation is applied.
- The captured-image review container uses the decoded image aspect ratio rounded to an integer height. `resizeMode="contain"` may introduce less than one displayed pixel of letterboxing.

The shared helper is used by both camera setup and board review. Tests cover landscape, portrait, 16:9, 9:16, 1:1, and FENiT `4284 × 5712` dimensions.

### 4.4 Measured aspect-ratio bound

Integer display-height rounding (`h = round(w · H / W)`) introduces at most half a display pixel of rounding. Across display widths `320, 360, 390, 420, 480` px and the tested sensor formats, the measured maximum relative aspect-ratio error is **0.2463%** (16:9 at 360 px), strictly below the documented **0.35%** bound. This supersedes the earlier, too-tight 0.25% claim.

Boundary touches `[0, 0]` through `[1, 1]` map exactly to `(0, 0)` and `(W, H)`.

### 4.5 Known live-camera limitation

Manual live-preview corners are not currently passed to rectification when no captured image exists. The live `CameraView` may also crop or letterbox differently from the captured image: Android defaults to `FILL`, while `ratio` switches to `FIT` per the Expo SDK 57 camera documentation. Physical on-device validation of the preview-to-capture mapping is therefore still required and is **not** claimed here.

---

## 5. Verification and Preservation

### 5.1 Test suites and typecheck

| Check | Command | Result |
| :--- | :--- | :--- |
| Python unit/benchmark tests | `npm run test:python` | **26 / 26 passed** in 3.83 s |
| Node/TypeScript tests | `npm test` | **205 / 205 passed** across 13 suites in 13.57 s |
| TypeScript typecheck | `npx tsc --noEmit` | **0 errors** |
| Geometry benchmark | `py scripts\benchmark_board_geometry.py` | **385 images**, 1,540 records, 329.73 s |

`npm run lint` cannot run in this checkout: there is no ESLint configuration, and Expo's automatic setup failed while installing `eslint`/`eslint-config-expo` because the pnpm virtual store was created with a different `virtual-store-dir-max-length`. No dependency or lockfile change was left behind. This is an environment limitation, not a lint-clean result.

New Python coverage includes:

- summary counts and means derived from records;
- documented IoU/error penalties for missing and invalid proposals;
- strict production vs relaxed B gate separation;
- checkerboard pattern acceptance and non-checkerboard rejection;
- concave and nearly collinear geometry rejection;
- Approach D initial/paired/all-input denominators;
- summary-vs-record consistency guards;
- SHA-256 equality with the exact written record bytes.

### 5.2 Working-tree audit

- All 35 top-level `dist/` entries remain byte-for-byte unchanged from HEAD; Git reports no `dist/` modifications.
- `modules/live-camera/android/build/` is untouched and remains the pre-existing untracked directory.
- `src/vision/pieceModelWeights.ts` is unchanged (`1.0.0-synthetic-staunton-v1`).
- No external images, crops, manifests, or benchmark outputs are staged in Git.
- No APK was built or installed; no training was run.

### 5.3 Publication

The milestone commit is created only after all checks above pass, and is pushed normally (no force-push, no history rewrite). The final commit hash is reported in the Git history and in the milestone handoff.

---

## 6. Gate Table

| Gate | Requirement | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **Gate 0** | Preservation constraints | **PASS** | 35 `dist/` entries unchanged; build directory untouched; weights unchanged; no APK/training. |
| **Gate 1** | Explicit candidate/geometry/acceptance accounting | **PASS** | Per-image records with all four required fields; summaries recomputed from records and guarded by assertions. |
| **Gate 2** | Approach B strict production evidence | **PASS** | Checkerboard pattern gate implemented with production thresholds; measured **4 / 385** accepted, not the unsupported 0/385 claim. |
| **Gate 3** | Approach D initial/paired/all-input views | **PASS** | Exact denominators 385 / 385 / 385; paired deltas and outcome counts recorded. |
| **Gate 4** | Convexity and permutation robustness | **PASS** | Scale-independent turn-sine validation; 24 permutations per geometry; concave/degenerate inputs remain invalid. |
| **Gate 5** | Coordinate helper and conventions | **PASS** | Shared helpers wired into both screens; boundary, aspect, round-trip, and payload tests pass. |
| **Gate 6** | Regression verification and publication | **PASS** | 26 Python tests, 205 Node tests, `tsc --noEmit` clean, benchmark reproduced, records hash verified. |

---

## 7. Limitations and Next Action

- **Set 01 physical capture remains PENDING.** There are still zero physical photographs/crops; live studio hardware, lighting, tags, and tripod setup are required.
- **Real human calibration remains unverified.** Approach D is a ground-truth-seeded simulation only.
- **Android latency remains unverified.** No benchmark result in this milestone supports an 88% accuracy or 150 ms latency claim.
- **Autonomous detection is not production.** A’s 9/385 accepted and B strict’s 4/385 accepted are insufficient; assisted manual calibration remains the production workflow.
- **Live-preview coordinate mapping needs device validation.** The captured-image path is covered by tests, but preview crop/letterbox behavior and the no-captured-image path are not.
- **FENiT physical-set count and annotation quality remain unverified/model-assisted.** External results must not tune production weights.

**Next action:** when physical hardware and studio conditions are available, prepare and capture Set 01, then run the on-device calibration and latency validation. Until then, do not claim physical-device accuracy or latency.
