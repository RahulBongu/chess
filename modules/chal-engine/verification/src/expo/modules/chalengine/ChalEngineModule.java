package expo.modules.chalengine;

/**
 * Test-only JNI host.
 *
 * This class deliberately has the same fully-qualified name as the production
 * Kotlin module so that the JNI symbols exported by libchal-engine.so
 * (Java_expo_modules_chalengine_ChalEngineModule_*) resolve unchanged.
 *
 * It is not part of the Expo app and is only executed through app_process on
 * the test device. The native library under test is the stripped release
 * binary produced by :chal-engine:assembleRelease.
 */
public final class ChalEngineModule {
  static {
    String libDir = System.getProperty("chal.libdir", "/data/local/tmp/chal-verify-20261101");
    System.load(libDir + "/libc++_shared.so");
    System.load(libDir + "/libchal-engine.so");
  }

  public native boolean nativeInit();
  public native String nativeSearch(String fen, int maxDepth, long maxTimeMs);
  public native boolean nativeStop();
  public native boolean nativeTeardown();
}
