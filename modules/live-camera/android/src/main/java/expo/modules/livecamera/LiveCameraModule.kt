package expo.modules.livecamera

import android.view.View
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class LiveCameraModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("LiveCamera")

    View<LiveCameraView>(LiveCameraView::class) {
      Events("onFrame", "onPreviewReady", "onAnalysisReady", "onTrackingFrame", "onCameraBound", "onCameraError")

      Prop("active") { view: LiveCameraView, active: Boolean ->
        view.active = active
      }

      Prop("analysisEnabled") { view: LiveCameraView, enabled: Boolean ->
        view.analysisEnabled = enabled
      }

      Prop("trackingEnabled") { view: LiveCameraView, enabled: Boolean ->
        view.trackingEnabled = enabled
      }

      Prop("trackingTargetWidth") { view: LiveCameraView, width: Int ->
        view.trackingTargetWidth = width
      }

      Prop("trackingTargetHeight") { view: LiveCameraView, height: Int ->
        view.trackingTargetHeight = height
      }

      Prop("trackingIntervalMs") { view: LiveCameraView, ms: Long ->
        view.trackingIntervalMs = ms
      }

      OnViewDestroys { view: LiveCameraView ->
        view.releaseResources()
      }

      AsyncFunction("releaseFrame") { view: LiveCameraView ->
        view.releaseFrame()
      }

      AsyncFunction("releaseTrackingFrame") { view: LiveCameraView ->
        view.releaseTrackingFrame()
      }

      AsyncFunction("setClassificationCorners") { view: LiveCameraView, corners: List<Double>, requestId: Int ->
        view.setClassificationCorners(corners, requestId)
      }

      AsyncFunction("clearClassificationCorners") { view: LiveCameraView ->
        view.clearClassificationCorners()
      }

      AsyncFunction("setClassificationEnabled") { view: LiveCameraView, enabled: Boolean ->
        view.setClassificationEnabled(enabled)
      }
    }
  }
}