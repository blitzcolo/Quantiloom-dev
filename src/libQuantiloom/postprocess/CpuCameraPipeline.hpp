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

    // The full measurement path against a read-only state: internally copies
    // the state and runs Capture, but never commits the copy back. Identical
    // inputs give bit-identical outputs because every noise stream is keyed
    // on acquisitionIndex, so a restored history replays exactly.
    [[nodiscard]] Result<CameraOutput, String>
    CaptureReprocess(const CaptureState& state, f64 firstRowMidpointSeconds,
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
    // The config one acquisition actually runs with: the auto controller's
    // feedback (state.next*) layered over the authored config. Only committed
    // acquisitions consume feedback; reprocess replays the authored config.
    [[nodiscard]] CameraConfig EffectiveConfig(const CaptureState& state,
                                               bool commitState) const;
    [[nodiscard]] Result<CameraOutput, String>
    CaptureImpl(CaptureState& state, f64 firstRowMidpointSeconds,
                const SpectralFrameSampler& sampler, bool commitState) const;
    [[nodiscard]] Result<CameraOutput, String>
    Readout(CaptureState& state, f64 firstRowMidpointSeconds,
            std::vector<f64> expected, bool fastRgbApproximation,
            bool allowSignedMonteCarloResidual, bool commitState,
            const CameraConfig& config) const;
    CameraConfig m_config;
    // Warns once: auto_* flags on a thermal detector are ignored (AE has no
    // physical meaning on a power-responding detector).
    mutable bool m_autoThermalNoted = false;
};

// Acquisition-history operations shared by every host (offline renderer,
// sequence job, tests). A checkpoint is a value snapshot; restoring it bumps
// historyEpoch so a replayed stretch is distinguishable from the original.
// The advance callback runs exactly one closed acquisition against the state
// (products may be suppressed); on success it must have moved
// state.acquisitionIndex forward by one and state.frameTimeSeconds to the
// time it was given.
using CameraAdvanceFn =
    std::function<Result<void, String>(CaptureState& state, f64 timeSeconds)>;

[[nodiscard]] Result<CaptureCheckpoint, String>
CheckpointCamera(const CaptureState& state);

[[nodiscard]] Result<void, String>
RestoreCamera(CaptureState& state, const CaptureCheckpoint& checkpoint);

[[nodiscard]] Result<void, String>
AdvanceCameraState(CaptureState& state, f64 timeSeconds,
                   const CameraAdvanceFn& advance);

// Advance the state through round(seconds/framePeriodSeconds) product-free
// acquisitions placed on the frame grid immediately before the state's
// current time. Times that would fall before the clock origin clamp onto it;
// a clamped step measures an elapsed of zero, so it advances the acquisition
// counter and the RNG streams but leaves the thermal state untouched.
[[nodiscard]] Result<void, String>
WarmUpCamera(CaptureState& state, f64 seconds, f64 framePeriodSeconds,
             const CameraAdvanceFn& advance);

} // namespace quantiloom::camera
