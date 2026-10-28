// Extends the static app.json configuration. The CMake override is applied only
// for local Windows builds: CMake 3.31.6 (bundled Ninja 1.12.1) fixes the
// long-path "build.ninja still dirty" failure. Linux-based (EAS) builds keep the
// Expo default and are not forced onto a locally installed CMake.
const CMAKE_VERSION = '3.31.6';

function withAndroidCmakeVersion(plugins) {
  if (process.platform !== 'win32') {
    return plugins;
  }

  return plugins.map((plugin) => {
    const name = Array.isArray(plugin) ? plugin[0] : plugin;
    if (name !== 'expo-build-properties') {
      return plugin;
    }

    const options = Array.isArray(plugin) ? plugin[1] ?? {} : {};
    return [
      'expo-build-properties',
      {
        ...options,
        android: {
          ...(options.android ?? {}),
          cmakeVersion: CMAKE_VERSION,
        },
      },
    ];
  });
}

module.exports = ({ config }) => ({
  ...config,
  plugins: withAndroidCmakeVersion(config.plugins ?? []),
});
