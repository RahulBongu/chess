#include "chal_bridge.h"

#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "ChalBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {
  void init_attacks(void);
  void init_zobrist(void);
  int nnue_init(const char *path);
  int tt_allocate(size_t mb);
  void tt_clear(void);

  void chal_request_stop(void);
  int chal_is_stopped(void);
  void chal_reset_stop(void);

  typedef struct {
    uint64_t pieces[2][6], occ[3], hash;
    int side, castling, ep, halfmove, fullmove;
    uint8_t mailbox[64];
  } Position;

  void parse_fen(Position *pos, const char *fen);
  int search_root(Position *pos, int max_depth, int64_t hard_limit_ms, int64_t soft_limit_ms);

  extern uint64_t nodes_searched;
  typedef struct { uint64_t key; int16_t score; uint16_t move; uint8_t depth, bound, age, padding; } TTEntry;
  extern TTEntry *tt;
  extern size_t tt_mask;
  extern int history[2][64][64][2][2];
  extern int lmr_table[64][64];
}

static void bridge_init_lmr(void) {
  for (int d = 1; d < 64; d++) {
    for (int m = 1; m < 64; m++) {
      double r = log((double)d) * log((double)m) / 2.1350 + 0.2319;
      lmr_table[d][m] = r < 0.0 ? 0 : (int)r;
    }
  }
  lmr_table[0][0] = lmr_table[0][1] = lmr_table[1][0] = lmr_table[1][1] = 0;
}

struct SearchResultState {
  std::string bestMove;
  int score = 0;
  bool isMate = false;
  int mateIn = 0;
  int depth = 0;
  uint64_t nodes = 0;
  std::vector<std::string> pv;
  bool foundBestMove = false;

  void reset() {
    bestMove.clear();
    score = 0;
    isMate = false;
    mateIn = 0;
    depth = 0;
    nodes = 0;
    pv.clear();
    foundBestMove = false;
  }
};

static std::mutex s_engine_mutex;
static std::mutex s_buffer_mutex;
static std::mutex s_stop_mutex;
static std::atomic<bool> s_stop_requested{false};
static std::atomic<uint64_t> s_active_search_id{0};
static std::atomic<uint64_t> s_cancelled_search_id{0};
static bool s_initialized = false;
static std::string s_line_buffer;
static SearchResultState s_current_result;

static bool parse_uint_field(const std::string& field, unsigned long long* value) {
  if (field.empty()) return false;
  unsigned long long parsed = 0;
  for (char c : field) {
    if (c < '0' || c > '9') return false;
    parsed = parsed * 10 + (unsigned long long)(c - '0');
    if (parsed > 1000000000ULL) return false;
  }
  *value = parsed;
  return true;
}

// The upstream parse_fen() performs no bounds checking: an over-full rank can
// write past the 64-square mailbox. Validate the complete six-field FEN before
// it is allowed anywhere near the engine's mutable state.
static bool is_valid_fen(const std::string& fen, std::string& reason) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (true) {
    size_t pos = fen.find(' ', start);
    if (pos == std::string::npos) {
      fields.push_back(fen.substr(start));
      break;
    }
    fields.push_back(fen.substr(start, pos - start));
    start = pos + 1;
  }
  if (fields.size() != 6) {
    reason = "FEN must contain exactly 6 space-separated fields";
    return false;
  }
  for (const std::string& field : fields) {
    if (field.empty()) {
      reason = "FEN fields must not be empty";
      return false;
    }
  }

  int file = 0;
  int slashes = 0;
  int whiteKings = 0;
  int blackKings = 0;
  for (char c : fields[0]) {
    if (c == '/') {
      if (file != 8) {
        reason = "each FEN rank must describe exactly 8 squares";
        return false;
      }
      slashes++;
      file = 0;
    } else if (c >= '1' && c <= '8') {
      file += c - '0';
      if (file > 8) {
        reason = "FEN rank contains too many squares";
        return false;
      }
    } else if (c != '\0' && strchr("pnbrqkPNBRQK", c) != nullptr) {
      if (file >= 8) {
        reason = "FEN rank contains too many squares";
        return false;
      }
      if (c == 'K') whiteKings++;
      if (c == 'k') blackKings++;
      file++;
    } else {
      reason = "FEN board contains an invalid piece character";
      return false;
    }
  }
  if (slashes != 7 || file != 8) {
    reason = "FEN board must contain 8 ranks of 8 squares";
    return false;
  }
  if (whiteKings != 1 || blackKings != 1) {
    reason = "FEN must contain exactly one king per side";
    return false;
  }

  if (fields[1] != "w" && fields[1] != "b") {
    reason = "FEN side to move must be w or b";
    return false;
  }

  if (fields[2] != "-") {
    bool seen[4] = {false, false, false, false};
    for (char c : fields[2]) {
      const char* p = (c != '\0') ? strchr("KQkq", c) : nullptr;
      if (p == nullptr) {
        reason = "FEN castling field is invalid";
        return false;
      }
      int idx = (int)(p - "KQkq");
      if (seen[idx]) {
        reason = "FEN castling field contains duplicates";
        return false;
      }
      seen[idx] = true;
    }
  }

  if (fields[3] != "-") {
    if (fields[3].size() != 2) {
      reason = "FEN en passant field is invalid";
      return false;
    }
    char epFile = fields[3][0];
    char epRank = fields[3][1];
    if (epFile < 'a' || epFile > 'h') {
      reason = "FEN en passant file is invalid";
      return false;
    }
    char expectedRank = (fields[1] == "w") ? '6' : '3';
    if (epRank != expectedRank) {
      reason = "FEN en passant rank is invalid";
      return false;
    }
  }

  unsigned long long halfmove = 0;
  unsigned long long fullmove = 0;
  if (!parse_uint_field(fields[4], &halfmove)) {
    reason = "FEN halfmove clock is invalid";
    return false;
  }
  if (!parse_uint_field(fields[5], &fullmove) || fullmove == 0) {
    reason = "FEN fullmove number is invalid";
    return false;
  }
  (void)halfmove;

  return true;
}

static void parse_info_line(const std::string& line, SearchResultState& res) {
  std::istringstream iss(line);
  std::string token;
  while (iss >> token) {
    if (token == "depth") {
      iss >> res.depth;
    } else if (token == "score") {
      std::string type;
      if (iss >> type) {
        if (type == "cp") {
          iss >> res.score;
          res.isMate = false;
        } else if (type == "mate") {
          iss >> res.mateIn;
          res.isMate = true;
          res.score = res.mateIn > 0 ? (100000 - res.mateIn) : (-100000 - res.mateIn);
        }
      }
    } else if (token == "nodes") {
      iss >> res.nodes;
    } else if (token == "pv") {
      res.pv.clear();
      std::string moveToken;
      while (iss >> moveToken) {
        res.pv.push_back(moveToken);
      }
      break;
    }
  }
}

static void parse_bestmove_line(const std::string& line, SearchResultState& res) {
  std::istringstream iss(line);
  std::string token;
  if (iss >> token && token == "bestmove") {
    if (iss >> res.bestMove) {
      res.foundBestMove = true;
    }
  }
}

extern "C" int chal_bridge_printf(const char* format, ...) {
  char buf[1024];
  va_list args;
  va_start(args, format);
  int len = vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);

  if (len > 0) {
    std::lock_guard<std::mutex> lock(s_buffer_mutex);
    s_line_buffer.append(buf, len);
    size_t newline_pos;
    while ((newline_pos = s_line_buffer.find('\n')) != std::string::npos) {
      std::string line = s_line_buffer.substr(0, newline_pos);
      s_line_buffer.erase(0, newline_pos + 1);
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.rfind("info ", 0) == 0) {
        parse_info_line(line, s_current_result);
      } else if (line.rfind("bestmove ", 0) == 0) {
        parse_bestmove_line(line, s_current_result);
      }
    }
  }

  if (s_stop_requested.load(std::memory_order_relaxed)) {
    chal_request_stop();
  }

  return len;
}

extern "C" int chal_bridge_fflush(FILE* /*stream*/) {
  return 0;
}

static std::string format_json_result(const SearchResultState& res) {
  std::ostringstream json;
  json << "{\"bestMove\":\"" << res.bestMove << "\""
       << ",\"score\":" << res.score
       << ",\"depth\":" << res.depth
       << ",\"nodes\":" << res.nodes
       << ",\"mate\":" << (res.isMate ? std::to_string(res.mateIn) : "null")
       << ",\"pv\":[";
  for (size_t i = 0; i < res.pv.size(); i++) {
    if (i > 0) json << ",";
    json << "\"" << res.pv[i] << "\"";
  }
  json << "]}";
  return json.str();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_expo_modules_chalengine_ChalEngineModule_nativeInit(JNIEnv* /*env*/, jobject /*thiz*/) {
  std::unique_lock<std::mutex> lock(s_engine_mutex);
  if (!s_initialized) {
    init_attacks();
    init_zobrist();
    bridge_init_lmr();
    int net_ok = nnue_init(nullptr);
    if (!net_ok) {
      LOGE("Failed to initialize embedded Chal NNUE network");
      return JNI_FALSE;
    }
    int tt_ok = tt_allocate(16);
    if (!tt_ok) {
      LOGE("Failed to allocate 16 MB transposition table");
      return JNI_FALSE;
    }
    memset(history, 0, sizeof(history));
    s_initialized = true;
    LOGI("Chal engine initialized successfully with embedded NNUE and 16MB TT");
  }
  return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_expo_modules_chalengine_ChalEngineModule_nativeSearch(
    JNIEnv* env,
    jobject /*thiz*/,
    jstring jFen,
    jint maxDepth,
    jlong maxTimeMs) {
  uint64_t search_id = 0;
  {
    // Publish the generation under the same mutex used by stop/teardown so a
    // cancellation can never be interleaved with, and erased by, the reset.
    std::lock_guard<std::mutex> stopLock(s_stop_mutex);
    search_id = ++s_active_search_id;
  }

  // Reject concurrent search attempts safely
  std::unique_lock<std::mutex> lock(s_engine_mutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    return env->NewStringUTF("{\"error\":\"Engine is currently busy with another search\"}");
  }

  if (!s_initialized) {
    return env->NewStringUTF("{\"error\":\"Engine is not initialized\"}");
  }

  if (jFen == nullptr) {
    return env->NewStringUTF("{\"error\":\"Invalid FEN string parameter\"}");
  }
  const char* fenChars = env->GetStringUTFChars(jFen, nullptr);
  if (!fenChars) {
    return env->NewStringUTF("{\"error\":\"Invalid FEN string parameter\"}");
  }
  std::string fen(fenChars);
  env->ReleaseStringUTFChars(jFen, fenChars);

  std::string fenError;
  if (!is_valid_fen(fen, fenError)) {
    LOGE("Rejected invalid FEN: %s", fenError.c_str());
    return env->NewStringUTF("{\"error\":\"Invalid FEN\"}");
  }

  // This block is mutually exclusive with nativeStop()/nativeTeardown():
  // a stop that targets this search can never be erased by the reset below,
  // and a stop arriving afterwards sets the engine flag after the reset.
  {
    std::lock_guard<std::mutex> stopLock(s_stop_mutex);
    if (s_cancelled_search_id.load(std::memory_order_acquire) >= search_id) {
      return env->NewStringUTF("{\"error\":\"Search was cancelled before starting\"}");
    }
    s_stop_requested.store(false, std::memory_order_release);
    chal_reset_stop();
  }

  {
    std::lock_guard<std::mutex> bufLock(s_buffer_mutex);
    s_line_buffer.clear();
    s_current_result.reset();
  }

  Position pos;
  parse_fen(&pos, fen.c_str());

  int64_t hardLimit = maxTimeMs > 0 ? (int64_t)maxTimeMs : 0;
  int64_t softLimit = maxTimeMs > 0 ? (int64_t)maxTimeMs : 0;
  int depth = maxDepth > 0 ? (int)maxDepth : 64;

  search_root(&pos, depth, hardLimit, softLimit);

  // Flush remaining line buffer if needed
  {
    std::lock_guard<std::mutex> bufLock(s_buffer_mutex);
    if (!s_line_buffer.empty()) {
      if (s_line_buffer.rfind("info ", 0) == 0) {
        parse_info_line(s_line_buffer, s_current_result);
      } else if (s_line_buffer.rfind("bestmove ", 0) == 0) {
        parse_bestmove_line(s_line_buffer, s_current_result);
      }
      s_line_buffer.clear();
    }
  }

  std::string resultJson = format_json_result(s_current_result);
  return env->NewStringUTF(resultJson.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_expo_modules_chalengine_ChalEngineModule_nativeStop(JNIEnv* /*env*/, jobject /*thiz*/) {
  std::lock_guard<std::mutex> stopLock(s_stop_mutex);
  s_cancelled_search_id.store(s_active_search_id.load(std::memory_order_relaxed), std::memory_order_release);
  s_stop_requested.store(true, std::memory_order_release);
  chal_request_stop();
  return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_expo_modules_chalengine_ChalEngineModule_nativeTeardown(JNIEnv* /*env*/, jobject /*thiz*/) {
  {
    std::lock_guard<std::mutex> stopLock(s_stop_mutex);
    // Cancel the currently active search generation. Do not use active + 1:
    // that would reject the first search after the next nativeInit().
    s_cancelled_search_id.store(s_active_search_id.load(std::memory_order_relaxed), std::memory_order_release);
    s_stop_requested.store(true, std::memory_order_release);
    chal_request_stop();
  }

  // Block until any active search has returned before freeing the TT.
  std::unique_lock<std::mutex> lock(s_engine_mutex);
  if (s_initialized) {
    tt_clear();
    if (tt) {
      free(tt);
      tt = nullptr;
      tt_mask = 0;
    }
    s_initialized = false;
    LOGI("Chal engine torn down and resources released");
  }
  return JNI_TRUE;
}
