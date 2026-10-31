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
  private val previewView = PreviewView(context).apply {
    layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
    scaleType = PreviewView.ScaleType.FILL_CENTER
    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
  }

  private val onFrame by EventDispatcher()
  private val onPreviewReady by EventDispatcher()
  private val onAnalysisReady by EventDispatcher()
  private val onCameraBound by EventDispatcher()
  private val onCameraError by EventDispatcher()

  private val cameraExecutor = Executors.newSingleThreadExecutor()
  private val analysisBusy = AtomicBoolean(false)
  private val cameraReadyEmitted = AtomicBoolean(false)
  private val previewReadyEmitted = AtomicBoolean(false)

  private var cameraProvider: ProcessCameraProvider? = null
  private var camera: Camera? = null
  private var imageAnalysis: ImageAnalysis? = null
  private var previewStreamObserver: Observer<PreviewView.StreamState>? = null

  @Volatile
  private var lastFrameTimeMs = 0L

  @Volatile
  private var pendingStart = false

  private val frameIntervalMs = 250L

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

  fun stopCamera() {
    cameraProvider?.unbindAll()
    cameraProvider = null
    camera = null
    imageAnalysis = null
    analysisBusy.set(false)
    cameraReadyEmitted.set(false)
    previewReadyEmitted.set(false)
    previewStreamObserver?.let { previewView.previewStreamState.removeObserver(it) }
    previewStreamObserver = null
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
          .setTargetResolution(Size(320, 240))
          .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
          .build()
      } else {
        null
      }

      analysis?.setAnalyzer(cameraExecutor) { imageProxy ->
        try {
          val now = System.currentTimeMillis()
          if (analysisBusy.get() || now - lastFrameTimeMs < frameIntervalMs) {
            return@setAnalyzer
          }
          lastFrameTimeMs = now

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

          analysisBusy.set(true)
          if (cameraReadyEmitted.compareAndSet(false, true)) {
            android.os.Handler(android.os.Looper.getMainLooper()).post {
              onAnalysisReady(mapOf("width" to outputWidth, "height" to outputHeight))
            }
          }
          val base64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
          val payload = mapOf(
            "base64" to base64,
            "width" to outputWidth,
            "height" to outputHeight
          )

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

      try {
        provider.unbindAll()
        camera = if (analysis != null) {
          provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            analysis
          )
        } else {
          provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview
          )
        }
        android.os.Handler(android.os.Looper.getMainLooper()).post {
          onCameraBound(mapOf("analysisEnabled" to analysisEnabled))
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