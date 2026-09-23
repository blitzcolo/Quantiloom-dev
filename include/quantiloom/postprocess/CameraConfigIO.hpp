#pragma once

#include "core/Config.hpp"
#include "core/Platform.hpp"
#include "core/Types.hpp"
#include "postprocess/CameraPipeline.hpp"

namespace quantiloom {

/// The single reading of the versioned camera sections, shared by CLI and Studio.
/// A document without sensor.version is migrated from the legacy [sensor] keys.
/// A document with `sensor.preset` starts from that camera preset (see
/// CameraPresets.hpp) and every explicit versioned key overrides the preset
/// value it names; absent keys keep the preset's value. `sensor.version` may
/// be omitted with a preset but an explicit wrong version is still an error.
/// Serialization never writes the preset key: the config is expanded, so
/// preset documents round-trip losslessly.
[[nodiscard]] QL_API Result<camera::CameraConfig, String> ParseCameraConfig(
    const Config& config, SpectralMode mode, const String& baseDir = {});

/// Write the camera-owned TOML sections, including optional camera motion.
/// The caller writes the static [camera] table and the rest of the scene.
[[nodiscard]] QL_API String CameraConfigToToml(const camera::CameraConfig& config);

/// Validate an authored camera trajectory before an editor commits it.
namespace camera {
[[nodiscard]] QL_API Result<void, String> ValidateCameraMotion(
    const CameraMotionConfig& motion);

/// Resolve the camera pose at global scene time (linear keys, held endpoints).
[[nodiscard]] QL_API Result<CameraPoseKey, String> CameraPoseAt(
    const CameraMotionConfig& motion, f64 timeSeconds);

/// Map a requested scene time to the most recent physical device acquisition
/// on the frame-period grid anchored at the sequence's first export time.
[[nodiscard]] QL_API Result<u64, String> CameraAcquisitionIndexAt(
    f64 firstTimeSeconds, f64 framePeriodSeconds, f64 sceneTimeSeconds);
[[nodiscard]] QL_API Result<f64, String> CameraAcquisitionTimeAt(
    f64 firstTimeSeconds, f64 framePeriodSeconds, u64 acquisitionIndex);
} // namespace camera

} // namespace quantiloom
