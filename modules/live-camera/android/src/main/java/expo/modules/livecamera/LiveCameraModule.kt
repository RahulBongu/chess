package expo.modules.livecamera

import android.view.View
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class LiveCameraModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("LiveCamera")

    View<LiveCameraView>(LiveCameraView::class) {
      Events("onFrame", "onPreviewReady", "onAnalysisReady", "onCameraBound", "onCameraError")

      Prop("active") { view: LiveCameraView, active: Boolean ->
        view.active = active
      }

      Prop("analysisEnabled") { view: LiveCameraView, enabled: Boolean ->
        view.analysisEnabled = enabled
      }

      OnViewDestroys { view: LiveCameraView ->
        view.stopCamera()
      }

      AsyncFunction("releaseFrame") { view: LiveCameraView ->
        view.releaseFrame()
      }
    }
  }
}