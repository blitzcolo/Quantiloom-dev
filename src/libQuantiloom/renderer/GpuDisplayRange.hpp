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
/// The image scan and the 65,536-bin histogram stay on the device, and the
/// percentile crossing bins are scanned on the device too: the histogram
/// pass's last-finishing workgroup writes the two crossing bins into the
/// scratch buffer. Compute() reads one batch of scalar words per pass --
/// the extents first, so the CPU computes the original f32 bin scale
/// exactly, then the two bins -- instead of reading the histogram back.
/// The host retains the original double-precision percentile thresholds,
/// the fallback rules, and the CPU path for images the device flags as
/// needing exact small-value arithmetic.
class GpuDisplayRange {
public:
    /// Scalar words in the scratch buffer: ordered min, ordered max, valid
    /// count, needs-CPU flag, histogram done-group counter, then the low and
    /// high percentile crossing bins. Must match display_range.comp.hlsl.
    static constexpr u32 kScratchWords = 8;
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
