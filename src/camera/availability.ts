/**
 * `CameraView.isAvailableAsync()` is documented as web-only. On native platforms
 * the method rejects with an `UnavailabilityError`, which means "not implemented
 * here", not "this device has no camera". Treating that rejection as a missing
 * camera hides the camera UI even on devices with a working back camera.
 *
 * Native availability is reported by `CameraView` itself through
 * `onCameraReady` and `onMountError`, so native platforms return `null` here and
 * must not be mapped to an "unavailable" state.
 */
export type CameraAvailabilityProbe = () => Promise<boolean>;

export async function resolveCameraAvailability(
  platform: string,
  isAvailableAsync: CameraAvailabilityProbe,
): Promise<boolean | null> {
  if (platform !== 'web') {
    return null;
  }

  try {
    return await isAvailableAsync();
  } catch {
    return false;
  }
}
