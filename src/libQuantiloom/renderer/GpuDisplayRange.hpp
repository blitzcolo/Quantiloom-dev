#pragma once

#include "core/Types.hpp"

#include <memory>

namespace quantiloom {
class GpuImage;
class VulkanContext;
}

namespace quantiloom::rendercore {

/// Percentile window computed from an RGBA32F image's finite RGB pixels.
struct DisplayRange {
    f32 min = 0.0f;
    f32 max = 1.0f;
};

/// GPU histogram used by the interactive display-range calculation.
///
/// The image scan and 65,536-bin histogram remain on the device. Compute()
/// first reads three scalar words so the CPU computes the original f32 bin
/// scale exactly, then reads the fixed histogram. The host also retains the
/// original double-precision percentile thresholds and fallback rules.
class GpuDisplayRange {
public:
    static constexpr u32 kHistogramBins = 65536;

    [[nodiscard]] static Result<std::unique_ptr<GpuDisplayRange>, String>
    Create(VulkanContext& context);
    ~GpuDisplayRange();

    GpuDisplayRange(const GpuDisplayRange&) = delete;
    GpuDisplayRange& operator=(const GpuDisplayRange&) = delete;

    [[nodiscard]] Result<DisplayRange, String> Compute(
        const GpuImage& image, u32 width, u32 height,
        f32 percentileLow, f32 percentileHigh);

private:
    struct Impl;
    explicit GpuDisplayRange(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

}  // namespace quantiloom::rendercore
