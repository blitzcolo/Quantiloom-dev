#pragma once

#include "core/Config.hpp"
#include "core/Platform.hpp"
#include "core/Types.hpp"
#include "postprocess/CameraPipeline.hpp"

namespace quantiloom {

/// The single reading of the versioned camera sections, shared by CLI and Studio.
/// A document without sensor.version is migrated from the legacy [sensor] keys.
[[nodiscard]] QL_API Result<camera::CameraConfig, String> ParseCameraConfig(
    const Config& config, SpectralMode mode, const String& baseDir = {});

/// Write the camera-owned TOML sections, including optional camera motion.
/// The caller writes the static [camera] table and the rest of the scene.
[[nodiscard]] QL_API String CameraConfigToToml(const camera::CameraConfig& config);

} // namespace quantiloom
