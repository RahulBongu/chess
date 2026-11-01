# Chal native runtime verification

Date: 2026-11-01

Purpose: execute the release `libchal-engine.so` JNI entry points on physical
Android hardware without Expo or JavaScript, using an isolated `app_process`
host.

## Tested artifact

- Device: vivo V2336, Android 16 (SDK 36), arm64-v8a.
- Library: `modules/chal-engine/android/build/intermediates/stripped_native_libs/release/stripReleaseDebugSymbols/out/lib/arm64-v8a/libchal-engine.so`
- Library SHA-256: `8d7c3c754c0b406bb3adad87bb38f8c125a132ac163b35f686496438f6836760`
- The same bytes are packaged at `jni/arm64-v8a/libchal-engine.so` in
  `chal-engine-release.aar`.
- Harness dex SHA-256: `5742adb064347780bbad1056a559caa8ea0680bb1aaecde87233aefdd0a3e24b`
- Full log: `dist/chal-runtime-verify-20261101/results.txt`
- Result: 69 checks, 0 failures.

## Synchronization design

- `chal.c` uses `atomic_int time_over_flag` (C11 `<stdatomic.h>`).
  - `chal_request_stop()`: `atomic_store_explicit(..., 1, memory_order_release)`
  - `chal_is_stopped()`: `atomic_load_explicit(..., memory_order_acquire)`
  - `chal_reset_stop()`: `atomic_store_explicit(..., 0, memory_order_release)`
- `search()` and `qsearch()` check `chal_is_stopped()` at entry; recursive
  callers re-check after returning. The iterative root checks after each PVS
  pass. This is the flag the native search itself reads.
- `chal_bridge.cpp` keeps one active search via `s_engine_mutex` and
  `std::try_to_lock`; overlapping requests return a JSON "busy" error.
- Search generations are published under `s_stop_mutex`. The same mutex
  protects the `s_cancelled_search_id >= search_id` check and
  `chal_reset_stop()`. Therefore a cancellation either:
  - wins before the reset and the new search refuses to start, or
  - arrives after the reset and sets the engine flag, which the search reads.
  It cannot be erased by the reset.
- `nativeTeardown()` requests cancellation first, then blocks on
  `s_engine_mutex` before calling `tt_clear()`/`free(tt)`. Memory is not freed
  while a search is active.
- FEN input is validated (six fields, 8x8 board, exactly one king per side,
  side/castling/en-passant/clock fields) before `parse_fen()` is called, because
  the upstream parser has no bounds checks.

## Reproduction

Build the release native library offline:

```powershell
cd C:\dev\chess\android
$env:JAVA_HOME='C:\Users\rahul\.gradle\jdks\eclipse_adoptium-17-amd64-windows.2'
$env:CHESS_OFFLINE_BUILD='1'
.\gradlew.bat ':chal-engine:assembleRelease' '--console=plain' '-PreactNativeArchitectures=arm64-v8a' '-Pandroid.enableResourceOptimizations=false'
```

Compile the JNI host:

```powershell
$v='C:\dev\chess\modules\chal-engine\verification'
$env:JAVA_HOME='C:\Users\rahul\.gradle\jdks\eclipse_adoptium-17-amd64-windows.2'
$androidJar='C:\Users\rahul\AppData\Local\Android\Sdk\platforms\android-36\android.jar'
& "$env:JAVA_HOME\bin\javac.exe" -source 11 -target 11 -classpath $androidJar -d "$v\out\classes" (Get-ChildItem "$v\src" -Recurse -Filter '*.java' | ForEach-Object FullName)
& 'C:\Users\rahul\AppData\Local\Android\Sdk\build-tools\36.0.0\d8.bat' --release --min-api 24 --lib $androidJar --output "$v\out\dex" (Get-ChildItem "$v\out\classes" -Recurse -Filter '*.class' | ForEach-Object FullName)
```

Run on the device:

```powershell
$adb='C:\Users\rahul\AppData\Local\Android\Sdk\platform-tools\adb.exe'
$dev='10BE2A1B5Q00054'
$lib='C:\dev\chess\modules\chal-engine\android\build\intermediates\stripped_native_libs\release\stripReleaseDebugSymbols\out\lib\arm64-v8a'
& $adb -s $dev shell "mkdir -p /data/local/tmp/chal-verify-20261101"
& $adb -s $dev push "$lib\libc++_shared.so" /data/local/tmp/chal-verify-20261101/
& $adb -s $dev push "$lib\libchal-engine.so" /data/local/tmp/chal-verify-20261101/
& $adb -s $dev push "$v\out\dex\classes.dex" /data/local/tmp/chal-verify-20261101/
& $adb -s $dev shell "chmod 644 /data/local/tmp/chal-verify-20261101/*; cd /data/local/tmp/chal-verify-20261101 && CLASSPATH=classes.dex app_process /data/local/tmp/chal-verify-20261101 expo.modules.chalengine.ChalRuntimeHarness"
```

## Scope limits

- This exercises the JNI bridge directly. It does not exercise the Expo/Kotlin
  `ChalEngineModule` wrapper or `System.loadLibrary()` from an installed app
  APK; that remains a Stage 2 packaging check.
- The move check is structural (origin piece, target occupancy, movement
  geometry, sliding path, promotion rank/suffix). Chal itself only emits moves
  from its legal move generator.
- RSS figures include the whole `app_process`/ART process, not the engine
  alone.
