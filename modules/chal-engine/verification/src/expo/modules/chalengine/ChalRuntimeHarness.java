package expo.modules.chalengine;

import android.os.Build;
import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Test-only runtime harness for the production libchal-engine.so.
 *
 * Executed on-device with app_process. It calls the exact JNI entry points
 * exported by the native library; it does not use Expo or JavaScript.
 */
public final class ChalRuntimeHarness {
  private static final String TAG = "ChalRuntimeTest";

  private static final String START =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  private static final String TACTICAL =
      "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1";
  private static final String MATE =
      "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1";
  private static final String STALEMATE =
      "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";
  private static final String PROMOTION =
      "7k/P7/6K1/8/8/8/8/8 w - - 0 1";

  private static final String[] INVALID_FENS = {
      "not a fen",
      "8/8/8/8/8/8/8/8 w - - 0 1",
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0",
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0",
      "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra"
  };

  private static final ChalEngineModule ENGINE = new ChalEngineModule();
  private static PrintStream fileOut;
  private static int checks = 0;
  private static int failures = 0;

  public static void main(String[] args) {
    try {
      fileOut = new PrintStream(
          new FileOutputStream(System.getProperty(
              "chal.out", "/data/local/tmp/chal-verify-20261101/results.txt")),
          true, "UTF-8");
    } catch (Exception e) {
      fileOut = null;
    }

    startWatchdog();

    out("DEVICE|model=" + Build.MODEL
        + "|release=" + Build.VERSION.RELEASE
        + "|sdk=" + Build.VERSION.SDK_INT
        + "|abis=" + Arrays.toString(Build.SUPPORTED_ABIS)
        + "|arch=" + System.getProperty("os.arch"));
    out("MEMORY|start|" + memoryLine());

    ExecutorService workers = Executors.newCachedThreadPool();
    try {
      runSuite(workers);
    } catch (Throwable t) {
      failures++;
      out("EXCEPTION|" + t.getClass().getName() + "|" + t.getMessage());
      StackTraceElement[] trace = t.getStackTrace();
      for (int i = 0; i < trace.length && i < 12; i++) {
        out("TRACE|" + trace[i]);
      }
    } finally {
      try {
        ENGINE.nativeTeardown();
      } catch (Throwable ignored) {
      }
      workers.shutdownNow();
      out("MEMORY|end|" + memoryLine());
      out("SUMMARY|checks=" + checks + "|failures=" + failures
          + "|verdict=" + (failures == 0 ? "PASS" : "FAIL"));
      System.out.flush();
      if (fileOut != null) {
        fileOut.flush();
      }
      System.exit(failures == 0 ? 0 : 1);
    }
  }

  private static void runSuite(ExecutorService workers) throws Exception {
    boolean init1 = ENGINE.nativeInit();
    boolean init2 = ENGINE.nativeInit();
    check("init", init1 && init2, "first=" + init1 + " second=" + init2);

    JSONObject startResult = runPosition(
        "startpos", START, 8, 2000, false, null, null, false);
    if (startResult != null) {
      check("startpos:positive_nodes",
          startResult.optLong("nodes", 0) > 0,
          "nodes=" + startResult.optLong("nodes", 0));
    }

    JSONObject tacticalResult = runPosition(
        "tactical", TACTICAL, 8, 2000, false, null, null, false);
    if (tacticalResult != null) {
      check("tactical:depth", tacticalResult.optInt("depth", 0) >= 4,
          "depth=" + tacticalResult.optInt("depth", 0));
    }

    JSONObject mateResult = runPosition(
        "checkmate", MATE, 8, 1000, true, null, Integer.valueOf(0), false);
    if (mateResult != null) {
      check("checkmate:losing_score",
          mateResult.optInt("score", 1) < 0,
          "score=" + mateResult.optInt("score", 1));
    }

    JSONObject stalemateResult = runPosition(
        "stalemate", STALEMATE, 8, 1000, true, null, null, true);
    if (stalemateResult != null) {
      check("stalemate:zero_score",
          stalemateResult.optInt("score", 1) == 0,
          "score=" + stalemateResult.optInt("score", 1));
    }

    JSONObject promotionResult = runPosition(
        "promotion", PROMOTION, 10, 2000, false, "a7a8[qrbn]", null, false);
    if (promotionResult != null) {
      check("promotion:winning_score",
          promotionResult.optInt("score", 0) > 0,
          "score=" + promotionResult.optInt("score", 0)
              + " mate=" + promotionResult.opt("mate"));
    }

    // ---- invalid FEN rejection -------------------------------------------
    for (String fen : INVALID_FENS) {
      String raw = ENGINE.nativeSearch(fen, 4, 200);
      check("invalid_fen|" + abbreviate(fen),
          raw != null && raw.contains("\"error\""), "raw=" + raw);
    }
    String afterInvalid = ENGINE.nativeSearch(START, 6, 1000);
    check("invalid_fen:engine_still_usable",
        afterInvalid != null && !afterInvalid.contains("\"error\""),
        abbreviate(afterInvalid));

    // ---- overlapping search rejection ------------------------------------
    boolean overlapRejected = false;
    boolean overlapStopped = false;
    for (int attempt = 1; attempt <= 3 && !overlapRejected; attempt++) {
      Future<String> longSearch = workers.submit(
          () -> ENGINE.nativeSearch(START, 64, 30000));
      boolean active = waitForActive(longSearch, 500);
      long overlapStart = System.nanoTime();
      String overlap = ENGINE.nativeSearch(START, 4, 200);
      long overlapMs = (System.nanoTime() - overlapStart) / 1_000_000L;
      overlapRejected = active && !longSearch.isDone()
          && overlap != null && overlap.contains("busy");
      out("OVERLAP|attempt=" + attempt + "|active=" + active
          + "|elapsed_ms=" + overlapMs + "|raw=" + overlap);
      ENGINE.nativeStop();
      String longResult = longSearch.get(40, TimeUnit.SECONDS);
      boolean thisAttemptStopped =
          longResult != null && !longResult.contains("\"error\"");
      if (overlapRejected) {
        overlapStopped = thisAttemptStopped;
      }
      if (!overlapRejected) {
        // The probe ran before the long search acquired the engine lock;
        // the long search has been stopped and joined, so retry cleanly.
        Thread.sleep(100);
      }
    }
    check("overlap:rejected", overlapRejected, "busy response observed");
    check("overlap:stopped", overlapStopped, "long search stopped after probe");
    check("overlap:restart",
        !ENGINE.nativeSearch(START, 6, 1000).contains("\"error\""), "restart");

    // ---- cancellation trials ---------------------------------------------
    cancellationTrials(workers, 10);

    // ---- teardown while a search is active -------------------------------
    for (int i = 1; i <= 3; i++) {
      Future<String> search = workers.submit(
          () -> ENGINE.nativeSearch(START, 64, 30000));
      boolean searching = waitForActive(search, 500);
      long t0 = System.nanoTime();
      Future<Boolean> teardown = workers.submit(() -> ENGINE.nativeTeardown());
      Boolean teardownOk = teardown.get(40, TimeUnit.SECONDS);
      String raw = search.get(10, TimeUnit.SECONDS);
      long us = (System.nanoTime() - t0) / 1_000L;
      check("teardown_cycle_" + i + ":was_searching",
          searching && us < 5_000_000L,
          "active_before=" + searching + " teardown_us=" + us
              + " teardown_ms=" + String.format(Locale.US, "%.3f", us / 1000.0)
              + " teardown_ok=" + teardownOk);
      check("teardown_cycle_" + i + ":returned", raw != null,
          abbreviate(raw));
      check("teardown_cycle_" + i + ":not_initialized",
          ENGINE.nativeSearch(START, 4, 200).contains("not initialized"),
          "post-teardown search");
      check("teardown_cycle_" + i + ":reinit", ENGINE.nativeInit(),
          "nativeInit");
      String after = ENGINE.nativeSearch(START, 6, 1000);
      check("teardown_cycle_" + i + ":usable_after_reinit",
          after != null && !after.contains("\"error\""), abbreviate(after));
    }

    // ---- repeated init / search / teardown cycles ------------------------
    for (int i = 1; i <= 10; i++) {
      boolean init = ENGINE.nativeInit();
      String raw = ENGINE.nativeSearch(START, 6, 1000);
      boolean searchOk = raw != null && !raw.contains("\"error\"");
      boolean teardown = ENGINE.nativeTeardown();
      check("repeat_cycle_" + i, init && searchOk && teardown,
          "init=" + init + " search_ok=" + searchOk
              + " teardown=" + teardown + " raw=" + abbreviate(raw));
      if (i < 10) {
        // Re-init for the next cycle; the final teardown stays in effect.
        check("repeat_cycle_" + i + ":reinit", ENGINE.nativeInit(), "nativeInit");
      }
    }

    // Leave the engine initialized for the stale-result check.
    boolean reinit = ENGINE.nativeInit();
    check("final_reinit", reinit, "nativeInit");

    // ---- stale result isolation ------------------------------------------
    // PROMOTION produced a7a8q/a7a8r; a cancelled START search must never
    // return that move (it is illegal in the starting position).
    Future<String> staleSearch = workers.submit(
        () -> ENGINE.nativeSearch(START, 64, 30000));
    boolean staleActive = waitForActive(staleSearch, 500);
    long staleStopStart = System.nanoTime();
    ENGINE.nativeStop();
    String staleRaw = staleSearch.get(40, TimeUnit.SECONDS);
    long staleStopUs = (System.nanoTime() - staleStopStart) / 1_000L;
    String staleMove = "";
    try {
      staleMove = new JSONObject(staleRaw).optString("bestMove", "");
    } catch (Exception ignored) {
    }
    boolean staleLegal = staleMove.isEmpty() || plausible(START, staleMove);
    check("stale:cancelled_result_legal_for_current_position",
        staleActive && staleStopUs < 5_000_000L && staleLegal
            && !staleMove.startsWith("a7a8"),
        "move=" + staleMove + " stop_us=" + staleStopUs
            + " stop_ms=" + String.format(Locale.US, "%.3f", staleStopUs / 1000.0)
            + " raw=" + abbreviate(staleRaw));

    String freshRaw = ENGINE.nativeSearch(START, 8, 2000);
    check("stale:fresh_search_after_cancel",
        freshRaw != null && !freshRaw.contains("\"error\""), abbreviate(freshRaw));
  }

  private static void cancellationTrials(ExecutorService workers, int trials)
      throws Exception {
    long[] latenciesUs = new long[trials];
    long[] stopCallMicros = new long[trials];

    for (int i = 0; i < trials; i++) {
      Future<String> search = workers.submit(
          () -> ENGINE.nativeSearch(START, 64, 30000));
      boolean active = waitForActive(search, 400);
      long t0 = System.nanoTime();
      boolean stopOk = ENGINE.nativeStop();
      long t1 = System.nanoTime();
      if (!active) {
        check("cancel_" + (i + 1) + ":was_active", false, "search never became active");
      }
      String raw;
      try {
        raw = search.get(40, TimeUnit.SECONDS);
      } catch (TimeoutException te) {
        out("FATAL|cancel_" + (i + 1) + "|search did not return within 40s");
        Runtime.getRuntime().halt(96);
        return;
      }
      long t2 = System.nanoTime();
      latenciesUs[i] = (t2 - t0) / 1_000L;
      stopCallMicros[i] = (t1 - t0) / 1_000L;

      boolean resultOk = raw != null && !raw.contains("\"error\"");
      String move = "";
      try {
        move = new JSONObject(raw).optString("bestMove", "");
      } catch (Exception ignored) {
      }
      boolean moveOk = move.isEmpty() || plausible(START, move);

      // Immediately start a fresh analysis after cancellation.
      long restartStart = System.nanoTime();
      String restart = ENGINE.nativeSearch(START, 8, 2000);
      long restartMs = (System.nanoTime() - restartStart) / 1_000_000L;
      boolean restartOk = restart != null && !restart.contains("\"error\"");

      out("CANCEL|trial=" + (i + 1)
          + "|active_at_stop=" + active
          + "|latency_us=" + latenciesUs[i]
          + "|latency_ms=" + String.format(Locale.US, "%.3f", latenciesUs[i] / 1000.0)
          + "|stop_call_us=" + stopCallMicros[i]
          + "|stop_ok=" + stopOk
          + "|result_ok=" + resultOk
          + "|move=" + move
          + "|move_legal=" + moveOk
          + "|restart_ok=" + restartOk
          + "|restart_ms=" + restartMs
          + "|raw=" + abbreviate(raw));

      boolean stoppedQuickly = latenciesUs[i] < 5_000_000L;
      check("cancel_" + (i + 1),
          active && stopOk && resultOk && moveOk && restartOk && stoppedQuickly,
          "latency_us=" + latenciesUs[i]
              + " stopped_quickly=" + stoppedQuickly);
    }

    long[] sorted = latenciesUs.clone();
    Arrays.sort(sorted);
    long median = sorted[sorted.length / 2];
    long max = sorted[sorted.length - 1];
    long min = sorted[0];
    out("CANCEL_SUMMARY|trials=" + trials
        + "|latencies_us=" + Arrays.toString(latenciesUs)
        + "|min_us=" + min
        + "|median_us=" + median
        + "|max_us=" + max
        + "|min_ms=" + String.format(Locale.US, "%.3f", min / 1000.0)
        + "|median_ms=" + String.format(Locale.US, "%.3f", median / 1000.0)
        + "|max_ms=" + String.format(Locale.US, "%.3f", max / 1000.0));
  }

  /**
   * Gives the worker thread time to enter nativeSearch() and verifies the
   * future is still running. The subsequent stop latency itself proves the
   * search was active: a stop delivered before the search started would let
   * the 30-second search run to its time limit instead of returning in
   * milliseconds.
   */
  private static boolean waitForActive(Future<String> search, long settleMs)
      throws Exception {
    Thread.sleep(settleMs);
    return !search.isDone();
  }

  private static JSONObject runPosition(
      String label, String fen, int depth, long timeMs,
      boolean expectNoMove, String moveRegex, Integer expectedMate,
      boolean requireMateNull) {
    long t0 = System.nanoTime();
    String raw = ENGINE.nativeSearch(fen, depth, timeMs);
    long elapsed = (System.nanoTime() - t0) / 1_000_000L;
    out("POSITION|" + label + "|elapsed_ms=" + elapsed + "|raw=" + raw);

    JSONObject json = null;
    try {
      json = new JSONObject(raw);
    } catch (Exception e) {
      check("position:" + label, false, "unparseable JSON: " + raw);
      return null;
    }

    boolean ok = !json.has("error");
    String move = json.optString("bestMove", "");
    if (ok && expectNoMove) {
      ok = move.isEmpty();
    } else if (ok) {
      ok = !move.isEmpty() && plausible(fen, move);
    }
    if (ok && moveRegex != null) {
      ok = move.matches(moveRegex);
    }
    if (ok && expectedMate != null) {
      ok = !json.isNull("mate") && json.optInt("mate", -999) == expectedMate;
    }
    if (ok && requireMateNull) {
      ok = json.isNull("mate");
    }
    if (ok) {
      ok = json.optInt("depth", 0) > 0;
    }

    check("position:" + label, ok,
        "move=" + move + " depth=" + json.optInt("depth", 0)
            + " score=" + json.optInt("score", 0)
            + " mate=" + json.opt("mate") + " raw=" + abbreviate(raw));
    return json;
  }

  /**
   * Structural legality check for the returned UCI move: origin piece belongs
   * to the side to move, target is not occupied by that side, piece geometry is
   * respected, sliders have a clear path, and pawn promotion rank/suffix agree.
   * This is not a full check-legality engine; Chal itself only stores moves
   * produced by its legal move generator.
   */
  private static boolean plausible(String fen, String uci) {
    if (uci == null || !uci.matches("[a-h][1-8][a-h][1-8][qrbn]?")) {
      return false;
    }
    String[] fields = fen.split(" ");
    if (fields.length != 6) {
      return false;
    }
    boolean white = fields[1].equals("w");
    char[][] board = parseBoard(fields[0]);

    int fromFile = uci.charAt(0) - 'a';
    int fromRank = uci.charAt(1) - '1';
    int toFile = uci.charAt(2) - 'a';
    int toRank = uci.charAt(3) - '1';
    if (fromFile == toFile && fromRank == toRank) {
      return false;
    }

    char piece = board[7 - fromRank][fromFile];
    char target = board[7 - toRank][toFile];
    if (piece == '.') {
      return false;
    }
    if (Character.isUpperCase(piece) != white) {
      return false;
    }
    if (target != '.' && Character.isUpperCase(target) == white) {
      return false;
    }

    char type = Character.toLowerCase(piece);
    boolean promotion = uci.length() == 5;
    boolean lastRank = white ? toRank == 7 : toRank == 0;
    if (promotion && (type != 'p' || !lastRank)) {
      return false;
    }
    if (type == 'p' && lastRank != promotion) {
      return false;
    }

    int dr = toRank - fromRank;
    int df = toFile - fromFile;
    int forward = white ? 1 : -1;

    switch (type) {
      case 'p':
        if (df == 0) {
          if (target != '.') {
            return false;
          }
          if (dr == forward) {
            return true;
          }
          if (dr == 2 * forward && fromRank == (white ? 1 : 6)) {
            return board[7 - (fromRank + forward)][fromFile] == '.';
          }
          return false;
        }
        if (Math.abs(df) != 1 || dr != forward) {
          return false;
        }
        if (target != '.') {
          return true;
        }
        return fields[3].equals("" + (char) ('a' + toFile) + (char) ('1' + toRank));
      case 'n':
        return (Math.abs(df) == 1 && Math.abs(dr) == 2)
            || (Math.abs(df) == 2 && Math.abs(dr) == 1);
      case 'b':
        return Math.abs(df) == Math.abs(dr) && df != 0
            && pathClear(board, fromRank, fromFile, toRank, toFile);
      case 'r':
        return (df == 0) != (dr == 0)
            && pathClear(board, fromRank, fromFile, toRank, toFile);
      case 'q':
        return ((Math.abs(df) == Math.abs(dr) && df != 0)
            || ((df == 0) != (dr == 0)))
            && pathClear(board, fromRank, fromFile, toRank, toFile);
      case 'k':
        return Math.max(Math.abs(df), Math.abs(dr)) == 1;
      default:
        return false;
    }
  }

  private static boolean pathClear(
      char[][] board, int fromRank, int fromFile, int toRank, int toFile) {
    int dr = Integer.signum(toRank - fromRank);
    int df = Integer.signum(toFile - fromFile);
    int rank = fromRank + dr;
    int file = fromFile + df;
    while (rank != toRank || file != toFile) {
      if (board[7 - rank][file] != '.') {
        return false;
      }
      rank += dr;
      file += df;
    }
    return true;
  }

  private static char[][] parseBoard(String boardField) {
    char[][] board = new char[8][8];
    String[] ranks = boardField.split("/");
    for (int r = 0; r < 8; r++) {
      int file = 0;
      for (char c : ranks[r].toCharArray()) {
        if (c >= '1' && c <= '8') {
          for (int i = 0; i < c - '0'; i++) {
            board[r][file++] = '.';
          }
        } else {
          board[r][file++] = c;
        }
      }
    }
    return board;
  }

  private static void check(String name, boolean ok, String detail) {
    checks++;
    if (!ok) {
      failures++;
    }
    out("CHECK|" + name + "|" + (ok ? "PASS" : "FAIL") + "|" + detail);
  }

  private static synchronized void out(String message) {
    String line = "CHALTEST|" + message;
    System.out.println(line);
    Log.i(TAG, line);
    if (fileOut != null) {
      fileOut.println(line);
    }
  }

  private static String memoryLine() {
    try {
      BufferedReader reader = new BufferedReader(
          new FileReader("/proc/self/status"));
      String line;
      StringBuilder sb = new StringBuilder();
      while ((line = reader.readLine()) != null) {
        if (line.startsWith("VmRSS:") || line.startsWith("VmHWM:")) {
          if (sb.length() > 0) {
            sb.append('|');
          }
          sb.append(line.replaceAll("\\s+", " ").trim());
        }
      }
      reader.close();
      return sb.toString();
    } catch (Exception e) {
      return "unavailable:" + e.getMessage();
    }
  }

  private static String abbreviate(String value) {
    if (value == null) {
      return "null";
    }
    return value.length() <= 400 ? value : value.substring(0, 400) + "...";
  }

  private static void startWatchdog() {
    Thread watchdog = new Thread(() -> {
      try {
        Thread.sleep(240_000L);
      } catch (InterruptedException ignored) {
      }
      out("WATCHDOG|timeout_after_240s");
      Runtime.getRuntime().halt(95);
    }, "chal-watchdog");
    watchdog.setDaemon(true);
    watchdog.start();
  }

  private ChalRuntimeHarness() {
  }
}
