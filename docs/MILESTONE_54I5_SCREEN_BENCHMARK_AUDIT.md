# Milestone 5.4I.5 — Screen benchmark audit (Part A/C)

Status: HOLD (device acceptance still blocked; production model unchanged).
Date: 2026-11-06. Screen domain only; physical 3D-board results kept separate.

## Commits & production

- `main = origin/main = 92305a1` (85.4I.4). This audit adds one tracked
  benchmark script + this report. No push/reset/clean/discard, and no model swap.
- Production asset
  `modules/live-camera/android/src/main/assets/rgb_square_classifier_v1.onnx`
  = `models/rgb_square_classifier_v1_balanced_strong_universal150.onnx`
  (235,025 bytes, SHA-256 `4a78b7ad6145f2849f6d7ef18be6c1c4b124f909f984131f860820f90ddfffab`).
  `universal150` is the only candidate trained with the digital crops
  (`domain_train_samples: fenit 20736, zenodo 2368, digital 2600`).

## What the old benchmark got wrong

`dist/benchmark_i3.py` (`dist/milestone_54i3_domain_benchmark.json`) reported
"32/64" for `test_e4e5` with every model. That is a benchmark artefact:

1. `orange_corners()` took axis-aligned extremes of all orange pixels
   (`y>400`). On `dist/screen_nchw_clear.png` there are only 8,368 orange
   pixels, largest orange contour area 246 — no board overlay. The extremes
   warp a tiny non-board blob, so every square crop is blur and every model
   predicts `empty` everywhere (`empty 32/32`, `occ 33`, identical blur/noblur).
2. `exact` counted per-square correct / 64, not full-board exact-match
   (which would be 0/1 boards). On the buggy warp this hides the failure,
   because 32 empty squares are accidentally right.
3. Digital "sanity" used 2 crops/class (26 total) and full-square
   `resize(64,64)`, while training resizes to 72x72 then centre-crops 64x64
   (`4:68` in `train_rgb_square_model.evaluate()`). That 12% scale shift
   makes the check non-comparable.
4. Warp (perspective 640 + 80px squares + resize 64) differs from the device
   path: `ChessSquareClassifier` bilinear-samples directly at
   `SQUARE_INSET=(1-64/72)/2` with per-pixel homography.

## Fixed benchmark

Tracked script: `scripts/benchmark_screen_i5.py` (copied to the ignored
`dist/benchmark_i3.py` for continuity). It:

- Finds the board by largest orange-mask quad (`approxPolyDP` 4 points),
  then Canny outer quad (square aspect only), then explicit MANUAL fallback.
- Replicates the device path (`sample_device_batch`) and the old warp path
  (`warp_batch`) and reports both.
- Reports `per_square_exact_n`, `per_square_exact_rate`, and true
  `board_exact` separately; plus occupancy, occupied/empty exact, colour,
  mean confidence, uncertain (`conf<0.55 | margin<0.08`), per-class recall.
- Uses all digital crops: 2,600 (200/class, 72→centre-64).

Corners found:

| Board | Method | Corners (TL,TR,BR,BL) | Overlay |
|---|---|---|---|
| `green_start` (`screen_assist_prompt.png`) | `orange_contour` (area 779,291) | `[104,730],[979,689],[970,1613],[117,1598]` | `dist/audit_i5/corners_i5_screen_assist_prompt_orange_contour.png` |
| `green_e4e5` (`screen_nchw_clear.png`) | `manual_reused_from_start_orange` (no large orange quad; Canny largest 4-pt is phone UI 33,53–1046,318, aspect 3.36) | same as above (verified orange quad, same propped camera ~7 min apart; ±~15 px) | `dist/audit_i5/corners_i5_screen_nchw_clear_manual_reused_from_start_orange.png` |
| `brown_start` (`i4_auto_13s.png`) | `orange_contour` (area 770,648) | `[131,682],[1011,709],[996,1570],[130,1574]` | `dist/audit_i5/corners_i5_i4_auto_13s_orange_contour.png` |

Machine output: ignored `dist/milestone_54i5_screen_benchmark.json`.

## Results (per-square exact / 64; `board_exact=false` everywhere)

| Model | green_start dev/warp | green_e4e5 dev/warp | brown_start dev/warp | uncertain (dev) | digital top1 (n=2600) |
|---|---|---|---|---|---|
| universal150 (prod) | 23/23 | 31/28 | 51/49 | 43/39/16 | 1.000 |
| balanced150fix | 33/25 | 31/31 | 41/46 | 43/34/12 | 0.189 |
| zenodo_long | 33/31 | 34/35 | 40/40 | 19/20/3 | 0.150 |
| wide150 | 36/37 | 36/37 | 41/40 | 11/13/10 | 0.129 |

Occupancy + colour (device on brown: universal `occ 64, colour 32/32,
meanConf 0.724`; others `occ 64` but `colour 16–19/32`; on green: wide
`occ 64, colour 30–32/32` while universal `occ 50–55`).

Occupied-piece exact is low everywhere (universal brown 19/32 device,
others 8–14/32; green 0–5/32). Predictions collapse bishops/knights/queens/
kings to pawns, and kings/queens to other pieces; on green, colours also
flip (e.g. brown universal row0: `N→P,B→P,Q→N`).

### Start-gate replica (occupancy + colour per `verifyStartingPosition`)

Recomputed from summed `occupiedProbability` / `white/blackProbability`
(centre occ>0.30 uncertain; home occ/colour <0.70 uncertain):

| Board | universal | balanced | zenodo | wide |
|---|---|---|---|---|
| green_start: startUncert / centreOcc / whiteHome / blackHome / wrongCol | 31/10/12/12/4 | 26/2/16/9/7 | 15/0/16/11/5 | 4/0/16/14/2 |
| green_e4e5 (not a start) | 42/5/8/8/10 | 23/2/13/10/8 | 21/2/12/8/7 | 10/2/15/14/1 |
| brown_start | 1/0/16/16/0 | 17/0/1/15/16 | 16/0/3/16/13 | 17/0/4/15/13 |

No candidate passes all three boards. Universal is the only one that
satisfies brown start occupancy + colour (16/16 both colours, 1 uncertain).
The Zenodo-only candidates invert white home pieces to black on brown
(white 1–4/16) despite looking "confident" (zenodo brown uncertain 3 with
meanConf 0.907 — confidently wrong, which would also trip the
confident-contradiction block). Wide is closest on green start
(4 uncertain, 2 wrong-colour) but still fails and fails brown colour.

Digital top1 for non-universal models is training leakage in reverse:
only universal trained on the 2,600 digital crops, so 1.000 is train
accuracy, not screen generalisation — and screen occupied-exact remains
1–5/32 on green despite that. No preprocessing claim is made here; earlier
audit found Gaussian/median/down-up/clamp/tone/bilateral within 49–51/64
and not a reopening of the pipeline (NCHW, `(x/127.5-1)`, 72→64 inset were
verified correct).

## Verdict for Part C (model comparison)

- Keep `universal150` in production. No swap is justified by 3 screen frames.
- Do not raise uncertainty allowances. The start prior still blocks at
  >3 prior-resolved uncertain squares, and confident contradictions (≥0.8,
  margin ≥0.4) still block — widening that to force a pass would hide the
  colour inversions above.
- HOLD remains until automatic start + one-ply e2–e4 pass on capture-driven
  evidence (Part E). Screen-only results must stay labelled separate from
  physical-board results.

## Blockers & next move (Parts B/E)

`adb devices` is empty (no vivo attached). No physical 3D board is
available, and the previous phone was fixed sideways/moved mid-session —
so no genuine vivo-crop benchmark with session/position splits can be built
from existing files. Do not fabricate splits.

Needed capture protocol (once device is back):

1. Fixed wooden/propped rig, white-bottom, dark-room + fixed laptop
   brightness; record session ID, board corners, orientation, device APK hash.
2. Sessions × positions: start, e4, e5, e4+e5, knight development, capture —
   each as a burst (≥10 frames, 1 s apart), split by session (not randomly
   by crop) for held-out evaluation.
3. Vary one factor at a time (distance/angle/brightness) and keep a
   verified manifest so corner error is measurable.
4. Evaluate all four models on identical crops with per-class /
   occupancy / colour / calibration metrics; only swap on held-out win,
   in a separate model commit.

Until then: `npm test` 397/397 pass, `npx tsc --noEmit` clean;
`npx expo lint` blocked by pre-existing pnpm virtual-store mismatch
(`ERR_PNPM_VIRTUAL_STORE_DIR_MAX_LENGTH_DIFF`), unrelated to this change.

## Files

- `scripts/benchmark_screen_i5.py` — fixed benchmark (tracked).
- `docs/MILESTONE_54I5_SCREEN_BENCHMARK_AUDIT.md` — this report (tracked).
- Ignored artefacts: `dist/milestone_54i5_screen_benchmark.json`,
  `dist/audit_i5/corners_i5_*.png`.
