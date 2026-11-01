package expo.modules.chalengine

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import org.json.JSONArray
import org.json.JSONObject

class ChalEngineModule : Module() {
  companion object {
    init {
      System.loadLibrary("chal-engine")
    }
  }

  private external fun nativeInit(): Boolean
  private external fun nativeSearch(fen: String, maxDepth: Int, maxTimeMs: Long): String
  private external fun nativeStop(): Boolean
  private external fun nativeTeardown(): Boolean

  override fun definition() = ModuleDefinition {
    Name("ChalEngine")

    AsyncFunction("init") {
      nativeInit()
    }

    AsyncFunction("search") { fen: String, maxDepth: Int, maxTimeMs: Long ->
      val rawJson = nativeSearch(fen, maxDepth, maxTimeMs)
      val json = JSONObject(rawJson)

      if (json.has("error")) {
        mapOf(
          "error" to json.getString("error")
        )
      } else {
        val pvArray = json.optJSONArray("pv") ?: JSONArray()
        val pvList = mutableListOf<String>()
        for (i in 0 until pvArray.length()) {
          pvList.add(pvArray.getString(i))
        }

        mapOf(
          "bestMove" to json.optString("bestMove", ""),
          "score" to json.optInt("score", 0),
          "depth" to json.optInt("depth", 0),
          "nodes" to json.optLong("nodes", 0L),
          "mate" to if (json.isNull("mate")) null else json.optInt("mate"),
          "pv" to pvList
        )
      }
    }

    AsyncFunction("stop") {
      nativeStop()
    }

    AsyncFunction("teardown") {
      nativeTeardown()
    }

    OnDestroy {
      nativeStop()
      nativeTeardown()
    }
  }
}
