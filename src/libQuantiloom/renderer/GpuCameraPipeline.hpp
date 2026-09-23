#pragma once

#include "core/Types.hpp"
#include "postprocess/CameraPipeline.hpp"
#include "postprocess/CameraPhysics.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <memory>

namespace quantiloom {
class GpuImage;
class VulkanContext;
}

namespace quantiloom::rendercore {

/// Maximum time strata the measurement image is allocated for (M4-1). The
/// runtime stratum count is min(gpu_time_positions, spp) and travels in the
/// push constants; the layers exist regardless so the image view is stable.
inline constexpr u32 kCameraTimeStrataMax = 8;

/// Per-time-stratum camera frame, uploaded each measurement for the dynamic
/// compositor. Reproduces raygen.rgen's ray construction exactly (five
/// float4 rows: origin.xyz, forward.xyz, right.xyz, up.xyz, then
/// fovScale/aspect/timeSeconds), so reprojection inverts tracing.
struct DynamicLayerCamera {
    std::array<f32, 4> origin{};
    std::array<f32, 4> forward{};
    std::array<f32, 4> right{};
    std::array<f32, 4> up{};
    std::array<f32, 4> params{};  // fovScale, aspect, timeSeconds, unused
};

/// Inputs for the M4-1 dynamic-exposure pass: the time-stratified measurement
/// layers plus the camera frame and timing of each stratum. With
/// `strataCount <= 1` the pass degenerates to a copy of layer 0.
struct DynamicExposureInput {
    const GpuImage* strataRate = nullptr;    // 2D array RGBA32F, >= strataCount layers
    const GpuImage* strataDepth = nullptr;   // 2D array R32F, same layering
    const DynamicLayerCamera* layerCameras = nullptr;  // strataCount entries
    u32 strataCount = 1;
    f64 firstRowMidSeconds = 0.0;            // t0: midpoint of row 0's exposure
    f64 exposureSeconds = 0.0;               // E: per-row integration window
    f64 rowDelaySeconds = 0.0;               // rolling-shutter per-row offset (0 = global)
};

/// GPU approximation of the shared detector/ADC chain. The caller supplies
/// a device-weighted rate image in GENERAL layout: photon e-/s or thermal W.
/// Mono uses R; Bayer and MultiChannel carry latent R/G/B rates across every
/// pixel, with alpha recording the Bayer pixel's actual response identity.
/// PSF acts on those fields before Bayer readout selects one noisy channel.
/// One call is one explicit acquisition; rendering spp or presenting again
/// never calls this method.
class GpuCameraPipeline {
public:
    struct Images {
        const GpuImage* rawDn = nullptr;
        const GpuImage* corrected = nullptr;
        const GpuImage* display = nullptr;
        const GpuImage* expectedElectrons = nullptr;
        const GpuImage* preAdcElectrons = nullptr;
        /// Dynamic compositor output (the image PSF reads). For unstratified
        /// measurements it holds a copy of layer 0; for legacy callers that
        /// never supplied strata it is never written -- null there.
        const GpuImage* compositedRate = nullptr;
        /// Linear RGB out of the demosaic pass (debug/analysis product).
        const GpuImage* linearRgb = nullptr;
        /// Encoded-sRGB color-pass output feeding the visible display branch
        /// (debug/analysis product).
        const GpuImage* colorRgb = nullptr;
        /// The pre-AGC scalar image the host's CLAHE path consumes: the
        /// infrared branch writes the corrected scalar here, the visible
        /// branch the display luminance.
        const GpuImage* agcSource = nullptr;
    };

    /// Synchronously read back acquisition statistics (camera_stats). The
    /// histogram bins unsaturated pixels over [min,max]; mean/min/max are over
    /// the same unsaturated set. The per-channel sums and counts describe the
    /// unsaturated display-domain scalar per CFA/device channel, before white
    /// balance -- the AWB measurement. Blocks on the device: call between
    /// frames.
    struct IspStats {
        f32 minValue = 0.0f;
        f32 maxValue = 0.0f;
        f32 meanValue = 0.0f;
        u32 saturatedCount = 0;
        u32 unsaturatedCount = 0;
        std::array<u32, 256> histogram{};
        std::array<f32, 3> channelSums{};
        std::array<u32, 3> channelCounts{};
    };

    struct TimingQueries {
        VkQueryPool pool = VK_NULL_HANDLE;
        u32 firstQuery = 0;
    };
    // start, after dynamic compositor, after effective PSF, after
    // detector/ADC, after the full ISP (stats through display), after the HSV
    // pass (equal to stamp 4 when HSV is off), end after the final pass.
    static constexpr u32 kTimingQueryCount = 7;

    [[nodiscard]] static Result<std::unique_ptr<GpuCameraPipeline>, String>
    Create(VulkanContext& context, u32 width, u32 height);
    ~GpuCameraPipeline();

    GpuCameraPipeline(const GpuCameraPipeline&) = delete;
    GpuCameraPipeline& operator=(const GpuCameraPipeline&) = delete;

    [[nodiscard]] Result<void, String> Configure(const camera::CameraConfig& config);
    /// Re-upload the per-acquisition dynamic parameters (exposure, analog
    /// gain, white balance, HSV/tone/palette rows) from an effective capture
    /// config -- the authored config with the AE/AWB feedback applied. Called
    /// by the host on every new acquisition tick before RecordMeasurement;
    /// the ISP/statistics shaders read the values from the re-uploaded
    /// buffers, so a same-tick re-record keeps the previous tick's values by
    /// simply not calling this. Buffers are CPU_TO_GPU; the upload is a
    /// host-side memcpy, safe at record time.
    [[nodiscard]] Result<void, String> ApplyEffectiveConfig(
        const camera::CameraConfig& config);
    /// Re-run the display half of the ISP over the last acquisition without
    /// touching anything stateful: the fused demosaic/color/display pass ->
    /// HSV from the standing corrected image. No statistics passes (the
    /// Linear-AGC window keeps riding in the statistics buffer), no AE/AWB, no
    /// thermal-state noise, no acquisition advance -- the noise streams key on
    /// the last acquisition index, so a reprocess is bit-identical for
    /// unchanged parameters. `config` carries the display parameters to apply
    /// (white balance, tone, palette, HSV).
    [[nodiscard]] Result<void, String> RecordDisplayReprocess(
        VkCommandBuffer cmd, const camera::CameraConfig& config);
    /// Re-run detector/ADC/ISP from the standing PSF-blurred measurement.
    /// Does not trace, resample wavelengths, blur again or advance the
    /// detector's acquisition index or thermal history.
    [[nodiscard]] Result<void, String> RecordReadoutReprocess(
        VkCommandBuffer cmd, const camera::CameraConfig& config);
    /// `measuredRate` feeds the dynamic compositor (layer 0 when unstratified);
    /// `dynamic` may be null-equivalent (strataCount <= 1 or null strata
    /// images) for the plain single-layer path existing callers use.
    [[nodiscard]] Result<void, String> RecordMeasurement(
        VkCommandBuffer cmd, const GpuImage& measuredRate, u64 acquisitionIndex,
        f64 frameTimeSeconds, TimingQueries timing = {},
        const DynamicExposureInput& dynamic = {});
    /// Explicit approximate linear-RGB input. Maps RGB to latent device
    /// channels, then uses the same PSF, detector, ADC and product path.
    [[nodiscard]] Result<void, String> RecordFastRgbMeasurement(
        VkCommandBuffer cmd, const GpuImage& linearRgb, u64 acquisitionIndex,
        f64 frameTimeSeconds, TimingQueries timing = {});
    [[nodiscard]] Images GetOutputs() const;

    /// 4 x u32 atomics written by the dynamic compositor: slot 0 disoccluded
    /// pixels. Synchronized readback for DynamicExposureReport.
    [[nodiscard]] Result<std::array<u32, 4>, String> ReadDynamicCounters() const;

    /// Full-resolution ISP statistics (min/max/mean of the unsaturated
    /// display scalar, saturated count, 256-bin histogram). Synchronized
    /// readback; call between submitted frames, never between record and
    /// submit.
    [[nodiscard]] Result<IspStats, String> ReadIspStats() const;

    /// Snapshot of everything one acquisition mutates on the device: both
    /// thermal ping-pong states, the previous-frame AGC window riding in the
    /// statistics buffer, and the host-side advance scalars. The noise
    /// streams are keyed on acquisitionIndex, so replaying the same tick
    /// sequence from a checkpoint is bit-identical to the original run.
    /// `historyEpoch` is host bookkeeping; the pipeline leaves it at zero
    /// and the owning facade stamps it.
    struct GpuCheckpoint {
        u64 acquisitionIndex = 0;
        f64 frameTimeSeconds = 0.0;
        u64 historyEpoch = 0;
        bool hasCapture = false;
        bool firstCaptureWasFirst = false;
        f64 beforeFrameTimeSeconds = 0.0;
        f64 currentDeltaSeconds = 0.0;
        u32 stateBefore = 0;
        u32 stateCurrent = 1;
    };

    /// Copy both thermal states into the checkpoint images and capture the
    /// host advance scalars. Records transfer commands on `cmd`; the caller
    /// submits. `stateBefore` is captured as well: it feeds the next tick's
    /// delta semantics, so a restore must hand it back unchanged.
    [[nodiscard]] Result<GpuCheckpoint, String> RecordStateCheckpoint(
        VkCommandBuffer cmd);

    /// Copy the checkpoint images back into the thermal ping-pong states and
    /// restore the advance scalars exactly as RecordStateCheckpoint captured
    /// them. Configuration and resources are untouched; clearing the
    /// detector history after a scrub is ResetState's job, not this.
    [[nodiscard]] Result<void, String> RestoreStateCheckpoint(
        const GpuCheckpoint& checkpoint, VkCommandBuffer cmd);

    /// Clear thermal detector history after a timeline scrub/reconfiguration.
    void ResetState();

    /// Test/debug path: each RGBA32_UINT texel contains four consecutive
    /// CounterRandomU32 outputs for that pixel, without float conversion.
    [[nodiscard]] Result<void, String> RecordRandomVectors(
        VkCommandBuffer cmd, u32 seed, u64 acquisitionIndex,
        camera::NoiseClass noiseClass, u32 counter = 0);
    [[nodiscard]] const GpuImage* GetRandomVectors() const;

private:
    struct Impl;
    explicit GpuCameraPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

} // namespace quantiloom::rendercore
