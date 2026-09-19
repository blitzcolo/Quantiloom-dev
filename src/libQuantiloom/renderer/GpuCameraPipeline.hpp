#pragma once

#include "core/Types.hpp"
#include "postprocess/CameraPipeline.hpp"
#include "postprocess/CameraPhysics.hpp"

#include <vulkan/vulkan.h>

#include <memory>

namespace quantiloom {
class GpuImage;
class VulkanContext;
}

namespace quantiloom::rendercore {

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
    };

    struct TimingQueries {
        VkQueryPool pool = VK_NULL_HANDLE;
        u32 firstQuery = 0;
    };
    // start, after effective PSF, after detector/ADC, after basic display.
    static constexpr u32 kTimingQueryCount = 4;

    [[nodiscard]] static Result<std::unique_ptr<GpuCameraPipeline>, String>
    Create(VulkanContext& context, u32 width, u32 height);
    ~GpuCameraPipeline();

    GpuCameraPipeline(const GpuCameraPipeline&) = delete;
    GpuCameraPipeline& operator=(const GpuCameraPipeline&) = delete;

    [[nodiscard]] Result<void, String> Configure(const camera::CameraConfig& config);
    [[nodiscard]] Result<void, String> RecordMeasurement(
        VkCommandBuffer cmd, const GpuImage& measuredRate, u64 acquisitionIndex,
        f64 frameTimeSeconds, TimingQueries timing = {});
    /// Explicit approximate linear-RGB input. Maps RGB to latent device
    /// channels, then uses the same PSF, detector, ADC and product path.
    [[nodiscard]] Result<void, String> RecordFastRgbMeasurement(
        VkCommandBuffer cmd, const GpuImage& linearRgb, u64 acquisitionIndex,
        f64 frameTimeSeconds, TimingQueries timing = {});
    [[nodiscard]] Images GetOutputs() const;

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
