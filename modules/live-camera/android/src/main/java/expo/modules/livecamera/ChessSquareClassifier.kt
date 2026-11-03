package expo.modules.livecamera

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Local 13-class RGB square classifier.
 *
 * The model is trained clean-room on CC BY 4.0 FENiT and Zenodo 6656212 data.
 * Inference runs entirely on-device through ONNX Runtime; no image buffer or
 * frame leaves the process.
 */
class ChessSquareClassifier(private val context: Context) {
  companion object {
    const val MODEL_ASSET = "rgb_square_classifier_v1.onnx"
    const val INPUT_SIZE = 64
    const val CLASS_COUNT = 13
    const val SQUARE_COUNT = 64
    /**
     * Training resizes each square to 72x72, then samples the central 64x64
     * window. Sampling the whole square here would shift and shrink every
     * piece relative to the pixels the model was trained on.
     */
    const val TRAINING_CROP_RATIO = INPUT_SIZE / 72.0
    const val SQUARE_INSET = (1.0 - TRAINING_CROP_RATIO) / 2.0

    val CLASSES = arrayOf(
      "empty",
      "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
      "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"
    )
  }

  data class ClassificationResult(
    val scores: FloatArray,
    val cropPrepMs: Long,
    val onnxMs: Long,
  )

  private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
  private val sessionDelegate = lazy {
    val options = OrtSession.SessionOptions()
    options.setIntraOpNumThreads(2)
    val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
    environment.createSession(bytes, options)
  }
  private val session: OrtSession by sessionDelegate
  private val input = FloatArray(SQUARE_COUNT * 3 * INPUT_SIZE * INPUT_SIZE)
  private var unavailable = false

  /** Releases native ONNX memory when the hosting view is destroyed. */
  fun close() {
    if (sessionDelegate.isInitialized()) {
      runCatching { session.close() }
    }
  }

  fun warmUp() {
    // Loading the model is the expensive part. Run one dummy batch so the
    // first real frame does not absorb session initialisation.
    java.util.Arrays.fill(input, 0f)
    runInput()
  }

  /**
   * Returns 64 * 13 probabilities in board-view row-major order. Row 0 is the
   * top edge of the physical board as seen by the camera. The caller maps
   * row/column to algebraic squares using the user's orientation.
   */
  fun classify(
    rgba: ByteArray,
    width: Int,
    height: Int,
    normalizedCorners: DoubleArray,
  ): ClassificationResult? {
    if (normalizedCorners.size != 8 || width < 8 || height < 8) {
      return null
    }
    val prepStart = System.currentTimeMillis()
    val corners = arrayOf(
      Point(normalizedCorners[0] * width, normalizedCorners[1] * height),
      Point(normalizedCorners[2] * width, normalizedCorners[3] * height),
      Point(normalizedCorners[4] * width, normalizedCorners[5] * height),
      Point(normalizedCorners[6] * width, normalizedCorners[7] * height),
    )
    val homography = squareToQuadrilateral(corners) ?: return null

    var outIndex = 0
    for (row in 0 until 8) {
      for (column in 0 until 8) {
        for (y in 0 until INPUT_SIZE) {
          val v = (
            row + SQUARE_INSET + (y + 0.5) / INPUT_SIZE * TRAINING_CROP_RATIO
          ) / 8.0
          for (x in 0 until INPUT_SIZE) {
            val u = (
              column + SQUARE_INSET + (x + 0.5) / INPUT_SIZE * TRAINING_CROP_RATIO
            ) / 8.0
            val denominator = homography.g * u + homography.h * v + 1.0
            if (denominator == 0.0) {
              input[outIndex] = 0f
              input[outIndex + 1] = 0f
              input[outIndex + 2] = 0f
              outIndex += 3
              continue
            }
            val px = (homography.a * u + homography.b * v + homography.c) / denominator
            val py = (homography.d * u + homography.e * v + homography.f) / denominator
            sampleRgb(rgba, width, height, px, py, outIndex)
            outIndex += 3
          }
        }
      }
    }
    val cropPrepMs = System.currentTimeMillis() - prepStart
    val onnxStart = System.currentTimeMillis()
    val scores = runInput() ?: return null
    val onnxMs = System.currentTimeMillis() - onnxStart
    return ClassificationResult(scores, cropPrepMs, onnxMs)
  }

  private fun runInput(): FloatArray? {
    if (unavailable) {
      return null
    }
    val shape = longArrayOf(SQUARE_COUNT.toLong(), 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
    return try {
      OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), shape).use { tensor ->
        session.run(mapOf("input" to tensor)).use { result ->
          @Suppress("UNCHECKED_CAST")
          val scores = (result[0] as OnnxTensor).value as Array<FloatArray>
          val flat = FloatArray(SQUARE_COUNT * CLASS_COUNT)
          var index = 0
          for (square in 0 until SQUARE_COUNT) {
            val row = scores[square]
            for (classIndex in 0 until CLASS_COUNT) {
              flat[index++] = row[classIndex]
            }
          }
          flat
        }
      }
    } catch (error: Exception) {
      // The asset is immutable at runtime, so a load/shape failure is permanent.
      // Stop retrying every frame, but keep camera frames flowing without scores.
      unavailable = true
      android.util.Log.w("LiveCameraDiag", "RGB square classifier unavailable", error)
      null
    }
  }

  private fun sampleRgb(
    rgba: ByteArray,
    width: Int,
    height: Int,
    x: Double,
    y: Double,
    outIndex: Int,
  ) {
    val clampedX = max(0.0, min((width - 1).toDouble(), x))
    val clampedY = max(0.0, min((height - 1).toDouble(), y))
    val x0 = floor(clampedX).toInt()
    val y0 = floor(clampedY).toInt()
    val x1 = min(width - 1, x0 + 1)
    val y1 = min(height - 1, y0 + 1)
    val fx = clampedX - x0
    val fy = clampedY - y0

    val topLeft = (y0 * width + x0) * 4
    val topRight = (y0 * width + x1) * 4
    val bottomLeft = (y1 * width + x0) * 4
    val bottomRight = (y1 * width + x1) * 4
    for (channel in 0 until 3) {
      val top = (rgba[topLeft + channel].toInt() and 0xFF) * (1 - fx) +
        (rgba[topRight + channel].toInt() and 0xFF) * fx
      val bottom = (rgba[bottomLeft + channel].toInt() and 0xFF) * (1 - fx) +
        (rgba[bottomRight + channel].toInt() and 0xFF) * fx
      input[outIndex + channel] = (((top * (1 - fy) + bottom * fy) / 127.5) - 1.0).toFloat()
    }
  }

  private data class Point(val x: Double, val y: Double)

  private data class Homography(
    val a: Double,
    val b: Double,
    val c: Double,
    val d: Double,
    val e: Double,
    val f: Double,
    val g: Double,
    val h: Double,
  )

  /** Unit-square to quad mapping, matching src/vision/boardEvidence.ts. */
  private fun squareToQuadrilateral(corners: Array<Point>): Homography? {
    val topLeft = corners[0]
    val topRight = corners[1]
    val bottomRight = corners[2]
    val bottomLeft = corners[3]
    val twiceArea = kotlin.math.abs(
      topLeft.x * topRight.y - topRight.x * topLeft.y +
        topRight.x * bottomRight.y - bottomRight.x * topRight.y +
        bottomRight.x * bottomLeft.y - bottomLeft.x * bottomRight.y +
        bottomLeft.x * topLeft.y - topLeft.x * bottomLeft.y
    )
    if (twiceArea < 1e-9) return null

    val dx1 = topRight.x - bottomRight.x
    val dx2 = bottomLeft.x - bottomRight.x
    val dy1 = topRight.y - bottomRight.y
    val dy2 = bottomLeft.y - bottomRight.y
    val sx = topLeft.x - topRight.x + bottomRight.x - bottomLeft.x
    val sy = topLeft.y - topRight.y + bottomRight.y - bottomLeft.y

    if (kotlin.math.abs(sx) < 1e-9 && kotlin.math.abs(sy) < 1e-9) {
      return Homography(
        a = topRight.x - topLeft.x,
        b = bottomLeft.x - topLeft.x,
        c = topLeft.x,
        d = topRight.y - topLeft.y,
        e = bottomLeft.y - topLeft.y,
        f = topLeft.y,
        g = 0.0,
        h = 0.0,
      )
    }
    val denominator = dx1 * dy2 - dx2 * dy1
    if (kotlin.math.abs(denominator) < 1e-9) return null
    val g = (sx * dy2 - dx2 * sy) / denominator
    val h = (dx1 * sy - sx * dy1) / denominator
    return Homography(
      a = topRight.x - topLeft.x + g * topRight.x,
      b = bottomLeft.x - topLeft.x + h * bottomLeft.x,
      c = topLeft.x,
      d = topRight.y - topLeft.y + g * topRight.y,
      e = bottomLeft.y - topLeft.y + h * bottomLeft.y,
      f = topLeft.y,
      g = g,
      h = h,
    )
  }
}
