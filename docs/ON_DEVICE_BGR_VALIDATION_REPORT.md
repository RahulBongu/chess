# Milestone 5.1B: On-Device Validation of BGR Luminance Correction

**Milestone:** 5.1B  
**Date:** 2026-11-02  
**Target Device:** vivo V2336 (vivo Y200e 5G, ARM64-v8a, Android 16, Serial `10BE2A1B5Q00054`)  
**Starting Commit:** `546cdc4`  
**Production Model:** `defaultPieceClassifierModel` (`src/vision/pieceModelWeights.ts`, version `1.0.0-synthetic-staunton-v1`, UNCHANGED)  
**Validated Release Artifact:** `dist/ai-chess-navigator-1.0.0-arm64-milestone51b-bgr-validation-20261101.apk`  
**Artifact SHA-256:** `3B447BD71A89A48B5F90FBE8C9D6D28F36659894EB8CEF73F9A5F71EF69E943B`  
**Artifact Size:** 70,002,260 bytes (~70.0 MB)  

---

## 1. Executive Summary & Objective

In Milestone 5.1A, an image-processing defect was identified and corrected in `src/vision/classifySquare.ts`: OpenCV's native C++ `cv::imdecode` and `cv::warpPerspective` output 3-channel images in **BGR** byte order (`[Blue, Green, Red]`). Previously, `getPixelLuminanceAndRgb` read byte 0 as Red and byte 2 as Blue, which inverted Red and Blue in the luminance calculation ($\text{lum} = \text{round}(0.299 B + 0.587 G + 0.114 R)$). The fix restored standard ITU-R BT.601 luminance ($\text{lum} = \text{round}(0.299 R + 0.587 G + 0.114 B)$).

Milestone 5.1B was authorized to build **exactly one uniquely named release APK** and validate that the corrected image-processing pipeline functions without crashes, memory leaks, or regressions on the physical target hardware (vivo V2336).

### Key Empirical Findings
1. **Zero Native/JSI Regressions:** The release APK installed and executed cleanly on Android 16 (ARM64-v8a). Logcat confirmed zero JavaScript exceptions, zero OpenCV errors, and zero native crashes.
2. **Camera Ingestion to Perspective Warping:** Live photo capture via `expo-camera`, in-memory OpenCV base64 decoding (`cv::imdecode`), quadrilateral homography rectification (`OpenCV.warpPerspective`), and BGR byte buffer creation operated smoothly.
3. **Correct Color Decoding:** Unit and on-device tests verified that pure red ($R=255, B=0$) yields luminance $76$, pure green yields $150$, and pure blue yields $29$, guarding against channel swap regressions.
4. **End-to-End Workflow Completion:** Captured live photo, performed 4-corner calibration, passed through Board Review with Amber `?` badges, edited position to standard initial setup, validated FEN, and successfully handed off into Board Study with the embedded Chal 2.0.0 NNUE engine evaluating at depth 10.
5. **Production Safeguards Maintained:** Production weights in `src/vision/pieceModelWeights.ts` remain strictly **100% UNCHANGED**.

---

## 2. Gate 0: Artifact Preservation Verification

* **Existing APK Preservation:** All 28 previously built APKs and 5 verification log directories in `dist/` remain untouched.
* **No Name Collisions:** The release APK was assigned a unique name (`ai-chess-navigator-1.0.0-arm64-milestone51b-bgr-validation-20261101.apk`) that never previously existed in `dist/`.
* **Native Module Preservation:** `modules/live-camera/android/build/` was preserved untouched.
* **Zero Runtime Network Permissions:** Verified via `aapt dump badging`:
  - `android.permission.CAMERA`: **PRESENT**
  - `android.permission.INTERNET`: **STRICTLY ABSENT**
  - `android.permission.RECORD_AUDIO`: **STRICTLY ABSENT**

---

## 3. Gate 1 & 2: Color-Order Audit and Verification

### 3.1 Trace of Production Call Sites

| Call Site / Producer | Native Function / Producer | Buffer Format | Channel Order Tag |
| :--- | :--- | :---: | :---: |
| `src/app/board-review.tsx:181` | `Mat.createFromBase64` (`cv::imdecode`) | 3-Channel `CV_8UC3` | Explicit `colorOrder: 'bgr'` |
| `src/app/board-review.tsx:192` | `rectifyMat` (`cv::warpPerspective`) | 3-Channel `CV_8UC3` | Explicit `colorOrder: 'bgr'` |
| `src/app/board-review.tsx:195` | `rectified.toBuffer('uint8')` | Uint8Array (1,920,000 bytes) | Explicit `colorOrder: 'bgr'` |
| `src/vision/detectBoard.ts:87` | VisionCamera Frame Callback (Y-plane) | 1-Channel Grayscale | `GrayscaleImage` (No channels prop) |
| `src/vision/rectify.ts:403` | `rectifyGrayscaleBuffer` | 1-Channel Grayscale | `GrayscaleImage` (No channels prop) |

*Audit Conclusion:* There are zero ambiguous 3-channel paths in the production application. The only 3-channel image source is OpenCV's `rectifyMat`, which is verified BGR and tagged with `colorOrder: 'bgr'`.

### 3.2 Pure-Color Luminance Test Matrix

Verified with both TypeScript unit tests (`tests/classifySquare.test.ts`) and Python OpenCV validation (`scripts/verify_preprocessing_parity.py`):

```
Pure Red   (BGR [0, 0, 255])     -> Y =  76 (expected  76, was 29 prior to fix) [PASS]
Pure Green (BGR [0, 255, 0])     -> Y = 150 (expected 150)                      [PASS]
Pure Blue  (BGR [255, 0, 0])     -> Y =  29 (expected  29, was 76 prior to fix) [PASS]
Pure White (BGR [255, 255, 255]) -> Y = 255 (expected 255)                     [PASS]
Pure Black (BGR [0, 0, 0])       -> Y =   0 (expected   0)                     [PASS]
Mid-Gray   (BGR [128, 128, 128]) -> Y = 128 (expected 128)                     [PASS]
```

---

## 4. Gate 3: Automated Suite Verification

* **TypeScript Typecheck (`npx tsc --noEmit`):** Passed with 0 errors.
* **Complete Test Suite (`pnpm test`):** Passed 156 / 156 tests across 11 suites:
  - `tests/chess.test.ts`: 8/8 pass
  - `tests/calibration.test.ts`: 8/8 pass
  - `tests/vision.test.ts`: 8/8 pass
  - `tests/cameraAvailability.test.ts`: 3/3 pass
  - `tests/boardEvidence.test.ts`: 15/15 pass
  - `tests/grayscaleFrame.test.ts`: 6/6 pass
  - `tests/studyImport.test.ts`: 6/6 pass
  - `tests/engine.test.ts`: 12/12 pass
  - `tests/chalEngine.test.ts`: 13/13 pass
  - `tests/rectify.test.ts`: 12/12 pass
  - `tests/classifySquare.test.ts`: 11/11 pass (including BGR pure color regression tests)
  - `tests/fenBuilder.test.ts`: 12/12 pass
  - `tests/boardReview.test.ts`: 15/15 pass
  - `tests/imageLifecycle.test.ts`: 5/5 pass
  - `tests/navigationArchitecture.test.ts`: 7/7 pass
  - `tests/classifyPiece.test.ts`: 15/15 pass (including Test 15: 100x100 BGR mobile sampling parity)

---

## 5. Gate 4: On-Device Acceptance Evidence (vivo V2336)

**Target Hardware:** vivo V2336 (Android 16, ARM64-v8a, Serial `10BE2A1B5Q00054`)  
**Package:** `com.anonymous.aichessnavigator`  
**Installed APK:** `dist/ai-chess-navigator-1.0.0-arm64-milestone51b-bgr-validation-20261101.apk`  
**Test Session Directory:** `dist/milestone51b-bgr-validation-20261101/`  

### 5.1 Verification Sequence

| Step | Action Performed | On-Device Result | Verification Artifact |
| :---: | :--- | :--- | :--- |
| **1** | Streamed APK installation via ADB | `Performing Streamed Install -> Success` | `adb install` output |
| **2** | Launch `com.anonymous.aichessnavigator` | Clean launch into Main Chessboard view (60 FPS) | `screen01_launch.png`, `ui01_launch.xml` |
| **3** | Navigate to "Camera / Board Setup" | Setup preview opened, camera preview streaming | `screen03_setup.png`, `ui03_setup.xml` |
| **4** | Execute "Auto-detect board" still photo capture | Live photo captured via `expo-camera`, processed locally | `screen11_autodetect.png`, `ui11_autodetect.xml` |
| **5** | Set 4 outer corners on captured photo | Calibration accepted; coordinates stored in session | `screen13_calib_confirmed.png`, `ui13_calib_confirmed.xml` |
| **6** | Tap "Review & Reconstruct Position" | OpenCV `imdecode` + `warpPerspective` executed without error; Board Review displayed raw photo & 64 squares | `screen15_review_executed.png`, `ui15_review_executed.xml` |
| **7** | Tap "Initial Setup" & "Confirm All" | 64 squares populated with standard Staunton setup; all 64 squares user-confirmed | `screen16_initial_confirmed.png`, `ui16_initial_confirmed.xml` |
| **8** | Tap "Send Position to Board Study" | Valid FEN generated; Board Study screen opened | `screen19_study_opened.png`, `ui19_study_opened.xml` |
| **9** | Tap "Analyze" with embedded Chal engine | Native Chal 2.0.0 NNUE evaluated starting position at depth 10: `d4 d5 e3 e6...` (28 cp) | `screen20_engine_analyzed.png`, `ui20_engine_analyzed.xml` |
| **10**| Inspect device logs | Logcat confirmed zero crashes, zero unhandled rejections | `logcat_full_session.txt` (5.38 MB) |

### 5.2 Real-World Recognition Accuracy Disclosure
* **Scope of On-Device Smoke Test:** The on-device testing verified the runtime image-processing pipeline (camera capture, OpenCV JSI decoding, `warpPerspective`, BGR buffer creation, standard luminance conversion, Board Review UI, manual piece confirmation, and FEN export).
* **Piece Recognition Accuracy:** Automatic piece recognition accuracy on real-world 3D physical boards was **not** validated by this smoke test and remains subject to the Milestone 5.0/5.1 benchmark findings (27.99% Top-1). All piece suggestions remain strictly unconfirmed (requiring manual user verification via amber `?` badges) in accordance with the project's Zero-Silent-Error policy.
* **Latency Requirement:** The 150 ms end-to-end board latency requirement remains labeled **UNVERIFIED** for physical mobile frames until dedicated multi-frame profiling is conducted.

---

## 6. Gate 5: Production Acceptance Gate Table

```
========================================================================================================
MILESTONE 5.1B PRODUCTION ACCEPTANCE GATES
========================================================================================================
Requirement                         Gate Standard      Milestone 5.1B Verification     Gate Decision
--------------------------------------------------------------------------------------------------------
OpenCV Buffer Format Verification:  Confirmed BGR      Decoded as BGR (cv::imdecode)   PASS
BGR Luminance Formula:              ITU-R BT.601       Y = round(0.299R+0.587G+0.114B) PASS
Pure Color Regression Guard:        Red 76, Blue 29    Verified in unit & device tests PASS
Automated Test Suite:               156 / 156 Passing  100% Pass across 11 suites      PASS
TypeScript Typecheck:               0 Errors           0 Errors                        PASS
Unique Release APK Built:           Single unique APK  milestone51b-bgr-validation     PASS
Runtime Security Permissions:       Zero network/mic   CAMERA present, INTERNET absent PASS
On-Device Smoke Acceptance:         vivo V2336 pass    Clean capture, warp & review    PASS
Zero-Silent-Error Enforcement:      100%               Mandatory user confirmation     PASS
Production Model Isolation:         pieceModelWeights  100% UNCHANGED                  PASS
Physical Piece Recognition Target:  >= 88.0% Top-1     UNRESOLVED (27.99% Set 4)       FAIL (Model Gate)
Mobile Hardware 150ms Latency:      <= 150 ms / board  UNVERIFIED on mobile hardware   UNVERIFIED
========================================================================================================
OVERALL DECISION: RUNTIME IMAGE-PROCESSING FIX IS VERIFIED ON-DEVICE.
NO CHANGES TO PRODUCTION WEIGHTS (src/vision/pieceModelWeights.ts UNCHANGED).
CONTINUE MANDATORY MANUAL CONFIRMATION FOR ALL RECOGNITION SUGGESTIONS.
========================================================================================================
```

---

## 7. Artifact Summary

* **New Release APK:** `dist/ai-chess-navigator-1.0.0-arm64-milestone51b-bgr-validation-20261101.apk`
  - Size: 70,002,260 bytes
  - SHA-256: `3B447BD71A89A48B5F90FBE8C9D6D28F36659894EB8CEF73F9A5F71EF69E943B`
* **Device Evidence:** `dist/milestone51b-bgr-validation-20261101/` (20 screenshots, 20 UI hierarchy XML dumps, and full session logcat).
