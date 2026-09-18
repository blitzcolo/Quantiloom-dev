#pragma once

#include "postprocess/CameraPipeline.hpp"

#include <functional>

namespace quantiloom::camera {

// Called at each requested scene time and wavelength. The returned image is
// one channel of scene L_lambda in W/m^2/sr/nm at the configured sensor size.
// The callback must not have already applied lens/filter/QE.
using SpectralFrameSampler =
    std::function<Result<Image, String>(f64 timeSeconds, f64 wavelengthNm)>;

class CpuCameraPipeline {
public:
    explicit CpuCameraPipeline(CameraConfig config);

    // An acquisition advances state only after every spectral sample, readout
    // and requested product succeeds. firstRowMidpointSeconds is the clock
    // shared with geometry and thermal scene motion.
    [[nodiscard]] Result<CameraOutput, String>
    Capture(CaptureState& state, f64 firstRowMidpointSeconds,
            const SpectralFrameSampler& sampler) const;

    // Deterministic measured-input path for cross-backend operator tests.
    // Each pixel contains one CFA measurement or one value per MultiChannel
    // response, in e-/s for photons or W for thermal detectors.
    [[nodiscard]] Result<CameraOutput, String>
    CaptureMeasured(CaptureState& state, f64 firstRowMidpointSeconds,
                    const Image& measuredRate) const;

    // Explicit approximate entry for legacy/fast linear RGB radiance images.
    // It uses the configured absolute scale and RGB-to-device matrix, then
    // enters the same detector/ADC readout as spectral Capture.
    [[nodiscard]] Result<CameraOutput, String>
    CaptureFastRgb(CaptureState& state, f64 firstRowMidpointSeconds,
                   const Image& linearRgb) const;

private:
    [[nodiscard]] Result<CameraOutput, String>
    Readout(CaptureState& state, f64 firstRowMidpointSeconds,
            std::vector<f64> expected, bool fastRgbApproximation,
            bool allowSignedMonteCarloResidual) const;
    CameraConfig m_config;
};

} // namespace quantiloom::camera
