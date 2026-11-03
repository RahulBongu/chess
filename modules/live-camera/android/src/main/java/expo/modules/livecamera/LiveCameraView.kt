package expo.modules.livecamera

import android.content.Context
import android.util.Base64
import android.util.Size
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.Observer
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class LiveCameraView(
  context: Context,
  appContext: AppContext
) : ExpoView(context, appContext) {
  companion object {
    /**
     * RGB analysis runs at 1280x960 so a board that fills the frame yields
     * roughly 100 px squares for the classifier. The JS pipeline receives a
     * 4x-downsampled grayscale frame (320x240), keeping bridge traffic small.
     */
    private const val GRAY_DOWNSAMPLE = 4
    private const val ANALYSIS_WIDTH = 1280
    private const val ANALYSIS_HEIGHT = 960
  }

  private val previewView = PreviewView(context).apply {
    layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
    scaleType = PreviewView.ScaleType.FILL_CENTER
    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
  }

  private val onFrame by EventDispatcher()
  private val onPreviewReady by EventDispatcher()
  private val onAnalysisReady by EventDispatcher()
  private val onTrackingFrame by EventDispatcher()
  private val onCameraBound by EventDispatcher()
  private val onCameraError by EventDispatcher()

  private val cameraExecutor = Executors.newSingleThreadExecutor()
  private val trackingExecutor = Executors.newSingleThreadExecutor()
  private val analysisBusy = AtomicBoolean(false)
  private val trackingBusy = AtomicBoolean(false)
  private val analysisDroppedCount = java.util.concurrent.atomic.AtomicInteger(0)
  private val analysisAnalyzedCount = java.util.concurrent.atomic.AtomicInteger(0)
  private val classificationFailureCount = java.util.concurrent.atomic.AtomicInteger(0)
  private var rgbaFrameBuffer = ByteArray(0)
  private var grayFrameBuffer = ByteArray(0)
  private val trackingDroppedCount = java.util.concurrent.atomic.AtomicInteger(0)
  private val trackingAnalyzedCount = java.util.concurrent.atomic.AtomicInteger(0)
  private val cameraReadyEmitted = AtomicBoolean(false)
  private val previewReadyEmitted = AtomicBoolean(false)
  private val trackingReadyEmitted = AtomicBoolean(false)

  private var cameraProvider: ProcessCameraProvider? = null
  private var camera: Camera? = null
  private var imageAnalysis: ImageAnalysis? = null
  private var trackingAnalysis: ImageAnalysis? = null
  private var previewStreamObserver: Observer<PreviewView.StreamState>? = null
  private val squareClassifier by lazy { ChessSquareClassifier(context) }

  @Volatile
  private var classificationEnabled = true

  @Volatile
  private var classificationCorners: DoubleArray? = null

  @Volatile
  private var classificationRequestId = 0

  @Volatile
  private var lastFrameTimeMs = 0L
  @Volatile
  private var lastTrackingTimeMs = 0L

  @Volatile
  private var pendingStart = false

  private val frameIntervalMs = 250L
  // trackingIntervalMs / trackingEnabled / trackingTargetWidth / trackingTargetHeight
  // are public props (see LiveCameraModule) with setters below.

  init {
    previewView.setOnHierarchyChangeListener(object : ViewGroup.OnHierarchyChangeListener {
      override fun onChildViewRemoved(parent: View?, child: View?) = Unit

      override fun onChildViewAdded(parent: View?, child: View?) {
        val host = parent ?: return
        val width = previewView.width
        val height = previewView.height
        if (width <= 0 || height <= 0) {
          return
        }
        host.measure(
          View.MeasureSpec.makeMeasureSpec(width, View.MeasureSpec.EXACTLY),
          View.MeasureSpec.makeMeasureSpec(height, View.MeasureSpec.EXACTLY)
        )
        host.layout(0, 0, host.measuredWidth, host.measuredHeight)
      }
    })
    addView(previewView)
  }

  override fun onAttachedToWindow() {
    super.onAttachedToWindow()
    if (active && width > 0 && height > 0) {
      post { startCamera() }
    }
  }

  override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
    super.onLayout(changed, left, top, right, bottom)
    if (active && width > 0 && height > 0 && cameraProvider == null) {
      post { startCamera() }
    }
  }

  var analysisEnabled: Boolean = true

  var active: Boolean = false
    set(value) {
      field = value
      if (value) {
        pendingStart = true
        if (isAttachedToWindow && width > 0 && height > 0) {
          post { startCamera() }
        }
      } else {
        pendingStart = false
        cameraReadyEmitted.set(false)
        previewReadyEmitted.set(false)
        stopCamera()
      }
    }

  fun releaseFrame() {
    analysisBusy.set(false)
  }

  fun releaseTrackingFrame() {
    trackingBusy.set(false)
  }

  fun setClassificationCorners(corners: List<Double>, requestId: Int) {
    if (corners.size != 8 || corners.any { !it.isFinite() }) {
      classificationCorners = null
      return
    }
    classificationCorners = corners.toDoubleArray()
    classificationRequestId = requestId
  }

  fun clearClassificationCorners() {
    classificationCorners = null
  }

  fun setClassificationEnabled(enabled: Boolean) {
    classificationEnabled = enabled
    if (!enabled) {
      classificationCorners = null
    }
  }

  fun stopCamera() {
    cameraProvider?.unbindAll()
    cameraProvider = null
    camera = null
    imageAnalysis = null
    trackingAnalysis = null
    analysisBusy.set(false)
    trackingBusy.set(false)
    classificationCorners = null
    analysisDroppedCount.set(0)
    analysisAnalyzedCount.set(0)
    trackingDroppedCount.set(0)
    trackingAnalyzedCount.set(0)
    cameraReadyEmitted.set(false)
    previewReadyEmitted.set(false)
    trackingReadyEmitted.set(false)
    previewStreamObserver?.let { previewView.previewStreamState.removeObserver(it) }
    previewStreamObserver = null
  }

  /** Final cleanup when the native view is destroyed; do not reuse afterwards. */
  fun releaseResources() {
    stopCamera()
    cameraExecutor.execute {
      runCatching { squareClassifier.close() }
    }
    cameraExecutor.shutdown()
    trackingExecutor.shutdown()
  }

  var trackingEnabled: Boolean = true
    set(value) {
      field = value
      if (!value) {
        trackingAnalysis = null
      }
    }

  var trackingTargetWidth: Int = 160
    set(value) {
      field = value.coerceAtLeast(80)
    }

  var trackingTargetHeight: Int = 120
    set(value) {
      field = value.coerceAtLeast(60)
    }

  var trackingIntervalMs: Long = 80L
    set(value) {
      field = value.coerceAtLeast(33L) // ~30 fps max
    }

  private fun startCamera() {
    if (!active || cameraProvider != null || !isAttachedToWindow) {
      return
    }
    post {
      if (!active || cameraProvider != null || !isAttachedToWindow) {
        return@post
      }
      bindCamera()
    }
  }

  private fun bindCamera() {
    if (cameraProvider != null) {
      return
    }

    // Load the bundled ONNX session off the main thread so the first analysed
    // frame does not absorb model initialisation.
    cameraExecutor.execute {
      runCatching { squareClassifier.warmUp() }
    }

    val activity = appContext.currentActivity ?: appContext.throwingActivity
    val lifecycleOwner = activity as? LifecycleOwner
    if (lifecycleOwner == null) {
      android.os.Handler(android.os.Looper.getMainLooper()).post {
        onCameraError(mapOf("message" to "No Android lifecycle owner is available."))
      }
      return
    }

    val providerFuture = ProcessCameraProvider.getInstance(context)
    providerFuture.addListener({
      val provider = providerFuture.get()
      cameraProvider = provider

      previewStreamObserver?.let { previewView.previewStreamState.removeObserver(it) }
      previewStreamObserver = null

      val preview = Preview.Builder().build().apply {
        setSurfaceProvider(previewView.surfaceProvider)
      }

      val previewStreamObserver = Observer<PreviewView.StreamState> { state ->
        if (!active) {
          return@Observer
        }
        if (state == PreviewView.StreamState.STREAMING && previewReadyEmitted.compareAndSet(false, true)) {
          android.os.Handler(android.os.Looper.getMainLooper()).post {
            onPreviewReady(mapOf("width" to previewView.width, "height" to previewView.height))
          }
        }
      }
      this.previewStreamObserver = previewStreamObserver
      previewView.previewStreamState.observe(lifecycleOwner, previewStreamObserver)

      val analysis = if (analysisEnabled) {
        ImageAnalysis.Builder()
          // Detection works from a half-resolution grayscale frame, but the
          // square classifier needs near-native detail to resolve 13 classes.
          .setTargetResolution(Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
          .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
          .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
          .build()
      } else {
        null
      }

      analysis?.setAnalyzer(cameraExecutor) { imageProxy ->
        try {
          val now = System.currentTimeMillis()
          if (analysisBusy.get() || now - lastFrameTimeMs < frameIntervalMs) {
            analysisDroppedCount.incrementAndGet()
            return@setAnalyzer
          }
          lastFrameTimeMs = now
          analysisAnalyzedCount.incrementAndGet()

          val plane = imageProxy.planes.firstOrNull() ?: return@setAnalyzer

          val sourceWidth = imageProxy.width
          val sourceHeight = imageProxy.height
          val rowStride = plane.rowStride
          val pixelStride = plane.pixelStride

          // Respect the camera's crop rectangle; the plane buffer may contain a
          // larger (possibly padded/strided) frame than the valid analysis region.
          val cropRect = imageProxy.cropRect
          val cropLeft = if (cropRect.width() > 0) cropRect.left else 0
          val cropTop = if (cropRect.height() > 0) cropRect.top else 0
          val cropWidth = if (cropRect.width() > 0) cropRect.width() else sourceWidth
          val cropHeight = if (cropRect.height() > 0) cropRect.height() else sourceHeight

          // Downsample to a maximum dimension of 640 px while preserving aspect ratio.
          val maxDimension = 640
          val scale = minOf(
            maxDimension.toFloat() / cropWidth.toFloat(),
            maxDimension.toFloat() / cropHeight.toFloat()
          )
          val outputWidth = maxOf(1, (cropWidth.toFloat() * scale).toInt())
          val outputHeight = maxOf(1, (cropHeight.toFloat() * scale).toInt())
          // Detection/tracking use a 4x-downsampled grayscale frame, keeping
          // bridge bandwidth near the previous 320x240 class while RGB crops
          // stay at full analysis resolution for the classifier.
          val grayWidth = (outputWidth + GRAY_DOWNSAMPLE - 1) / GRAY_DOWNSAMPLE
          val grayHeight = (outputHeight + GRAY_DOWNSAMPLE - 1) / GRAY_DOWNSAMPLE

          val rgbaSize = outputWidth * outputHeight * 4
          val rgbaBytes = if (rgbaFrameBuffer.size == rgbaSize) {
            rgbaFrameBuffer
          } else {
            ByteArray(rgbaSize).also { rgbaFrameBuffer = it }
          }
          val graySize = grayWidth * grayHeight
          val bytes = if (grayFrameBuffer.size == graySize) {
            grayFrameBuffer
          } else {
            ByteArray(graySize).also { grayFrameBuffer = it }
          }

          val buffer = plane.buffer
          buffer.rewind()
          val bufferLimit = buffer.limit()
          for (outY in 0 until outputHeight) {
            val srcY = (cropTop + (outY * cropHeight) / outputHeight).coerceIn(0, sourceHeight - 1)
            val rowBase = srcY * rowStride
            for (outX in 0 until outputWidth) {
              val srcX = (cropLeft + (outX * cropWidth) / outputWidth).coerceIn(0, sourceWidth - 1)
              val position = rowBase + srcX * pixelStride
              val outputIndex = outY * outputWidth + outX
              val grayIndex = if (
                outX % GRAY_DOWNSAMPLE == 0 && outY % GRAY_DOWNSAMPLE == 0
              ) {
                (outY / GRAY_DOWNSAMPLE) * grayWidth + (outX / GRAY_DOWNSAMPLE)
              } else {
                -1
              }
              if (position in 0 until bufferLimit - 2) {
                val r = buffer.get(position).toInt() and 0xFF
                val g = buffer.get(position + 1).toInt() and 0xFF
                val b = buffer.get(position + 2).toInt() and 0xFF
                val a = if (position + 3 < bufferLimit) {
                  buffer.get(position + 3).toInt() and 0xFF
                } else {
                  0xFF
                }
                if (grayIndex >= 0) {
                  bytes[grayIndex] = ((77 * r + 150 * g + 29 * b) shr 8).toByte()
                }
                val rgbaIndex = outputIndex * 4
                rgbaBytes[rgbaIndex] = r.toByte()
                rgbaBytes[rgbaIndex + 1] = g.toByte()
                rgbaBytes[rgbaIndex + 2] = b.toByte()
                rgbaBytes[rgbaIndex + 3] = a.toByte()
              } else {
                if (grayIndex >= 0) {
                  bytes[grayIndex] = 0
                }
                val rgbaIndex = outputIndex * 4
                rgbaBytes[rgbaIndex] = 0
                rgbaBytes[rgbaIndex + 1] = 0
                rgbaBytes[rgbaIndex + 2] = 0
                rgbaBytes[rgbaIndex + 3] = -1
              }
            }
          }

          // Classify 64 perspective-correct RGB squares against the most recent
          // validated corners. This runs on the native camera executor and
          // sends only probabilities across the bridge.
          var inferenceMs = 0L
          var inferenceScores: FloatArray? = null
          var inferenceRequestId = 0
          val corners = classificationCorners
          if (classificationEnabled && corners != null && corners.size == 8) {
            inferenceRequestId = classificationRequestId
            val inferenceStart = System.currentTimeMillis()
            try {
              inferenceScores = squareClassifier.classify(
                rgbaBytes,
                outputWidth,
                outputHeight,
                corners,
              )
            } catch (error: Exception) {
              // A missing or incompatible model must not drop the camera frame:
              // detection/tracking still work, recognition simply reports no
              // square evidence until the model loads successfully.
              if (classificationFailureCount.getAndIncrement() % 50 == 0) {
                android.util.Log.w(
                  "LiveCameraDiag",
                  "square classification failed; emitting frame without scores",
                  error,
                )
              }
            }
            inferenceMs = System.currentTimeMillis() - inferenceStart
          }

          analysisBusy.set(true)
          if (cameraReadyEmitted.compareAndSet(false, true)) {
            android.os.Handler(android.os.Looper.getMainLooper()).post {
              onAnalysisReady(mapOf("width" to grayWidth, "height" to grayHeight))
            }
          }
          val base64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
          val rotationDegrees = imageProxy.imageInfo.rotationDegrees
          val analyzed = analysisAnalyzedCount.get()
          if (analyzed <= 3 || analyzed % 25 == 0) {
            android.util.Log.i(
              "LiveCameraDiag",
              "detection analysed=$analyzed skipped=${analysisDroppedCount.get()} ${grayWidth}x${grayHeight}"
            )
          }
          val payload = mutableMapOf<String, Any>(
            "base64" to base64,
            "width" to grayWidth,
            "height" to grayHeight,
            "rotationDegrees" to rotationDegrees,
            "stream" to "detection",
            "analyzedFrames" to analyzed,
            "droppedFrames" to analysisDroppedCount.get()
          )
          val scores = inferenceScores
          if (scores != null && corners != null) {
            payload["squareScores"] = scores.toList()
            payload["classificationCorners"] = corners.toList()
            payload["inferenceMs"] = inferenceMs
            payload["classificationRequestId"] = inferenceRequestId
          }

          android.os.Handler(android.os.Looper.getMainLooper()).post {
            onFrame(payload)
          }
        } catch (_: Exception) {
          // Keep the camera alive; a later frame can be analyzed.
        } finally {
          imageProxy.close()
        }
      }

      imageAnalysis = analysis

      // High-rate lightweight tracking analysis (lower resolution, higher fps)
      val trackingAnalysis = if (trackingEnabled) {
        ImageAnalysis.Builder()
          .setTargetResolution(Size(trackingTargetWidth, trackingTargetHeight))
          .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
          .build()
      } else {
        null
      }

      trackingAnalysis?.setAnalyzer(trackingExecutor) { imageProxy ->
        try {
          val now = System.currentTimeMillis()
          if (trackingBusy.get() || now - lastTrackingTimeMs < trackingIntervalMs) {
            trackingDroppedCount.incrementAndGet()
            return@setAnalyzer
          }
          lastTrackingTimeMs = now
          trackingAnalyzedCount.incrementAndGet()

          val plane = imageProxy.planes.firstOrNull() ?: return@setAnalyzer

          val sourceWidth = imageProxy.width
          val sourceHeight = imageProxy.height
          val rowStride = plane.rowStride
          val pixelStride = plane.pixelStride

          val cropRect = imageProxy.cropRect
          val cropLeft = if (cropRect.width() > 0) cropRect.left else 0
          val cropTop = if (cropRect.height() > 0) cropRect.top else 0
          val cropWidth = if (cropRect.width() > 0) cropRect.width() else sourceWidth
          val cropHeight = if (cropRect.height() > 0) cropRect.height() else sourceHeight

          // Downsample to tracking target resolution (e.g., 160x120)
          val scale = minOf(
            trackingTargetWidth.toFloat() / cropWidth.toFloat(),
            trackingTargetHeight.toFloat() / cropHeight.toFloat()
          )
          val outputWidth = maxOf(1, (cropWidth.toFloat() * scale).toInt())
          val outputHeight = maxOf(1, (cropHeight.toFloat() * scale).toInt())

          val bytes = ByteArray(outputWidth * outputHeight)

          val buffer = plane.buffer
          buffer.rewind()
          val bufferLimit = buffer.limit()
          for (outY in 0 until outputHeight) {
            val srcY = (cropTop + (outY * cropHeight) / outputHeight).coerceIn(0, sourceHeight - 1)
            val rowBase = srcY * rowStride
            for (outX in 0 until outputWidth) {
              val srcX = (cropLeft + (outX * cropWidth) / outputWidth).coerceIn(0, sourceWidth - 1)
              val position = rowBase + srcX * pixelStride
              bytes[outY * outputWidth + outX] =
                if (position in 0 until bufferLimit) buffer.get(position) else 0
            }
          }

          trackingBusy.set(true)
          if (trackingReadyEmitted.compareAndSet(false, true)) {
            android.os.Handler(android.os.Looper.getMainLooper()).post {
              onAnalysisReady(mapOf("width" to outputWidth, "height" to outputHeight, "stream" to "tracking"))
            }
          }
          val base64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
          val rotationDegrees = imageProxy.imageInfo.rotationDegrees
          val analyzed = trackingAnalyzedCount.get()
          if (analyzed <= 3 || analyzed % 50 == 0) {
            android.util.Log.i(
              "LiveCameraDiag",
              "tracking analysed=$analyzed skipped=${trackingDroppedCount.get()} ${outputWidth}x${outputHeight}"
            )
          }
          val payload = mapOf(
            "base64" to base64,
            "width" to outputWidth,
            "height" to outputHeight,
            "rotationDegrees" to rotationDegrees,
            "stream" to "tracking",
            "analyzedFrames" to analyzed,
            "droppedFrames" to trackingDroppedCount.get()
          )

          android.os.Handler(android.os.Looper.getMainLooper()).post {
            onTrackingFrame(payload)
          }
        } catch (_: Exception) {
        } finally {
          imageProxy.close()
        }
      }

      this.trackingAnalysis = trackingAnalysis

      try {
        val useCases = mutableListOf<UseCase>(preview)
        if (analysis != null) useCases.add(analysis)
        if (trackingAnalysis != null) useCases.add(trackingAnalysis)
        camera = provider.bindToLifecycle(
          lifecycleOwner,
          CameraSelector.DEFAULT_BACK_CAMERA,
          *useCases.toTypedArray()
        )
        android.os.Handler(android.os.Looper.getMainLooper()).post {
          onCameraBound(mapOf("analysisEnabled" to analysisEnabled, "trackingEnabled" to trackingEnabled))
        }
      } catch (error: Exception) {
        stopCamera()
        android.os.Handler(android.os.Looper.getMainLooper()).post {
          onCameraError(mapOf("message" to (error.message ?: "CameraX binding failed.")))
        }
      }
    }, ContextCompat.getMainExecutor(context))
  }
}
