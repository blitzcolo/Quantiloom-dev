#pragma once

#include "postprocess/CameraPipeline.hpp"
#include "postprocess/SensorModel.hpp"

namespace quantiloom {

/// Reuse the legacy TOML migration for a host that still calls SetGPUSensorParams.
/// Width/height and vertical FOV are the host camera at conversion time.
[[nodiscard]] Result<camera::CameraConfig, String> CameraConfigFromSensorParams(
    const SensorParams& sensor, SpectralMode mode, u32 physicalWidth,
    u32 physicalHeight, f64 verticalFovDegrees, f64 wavelengthNm);

} // namespace quantiloom
