/**
 * @file GpuClahe.hpp
 * @brief The CLAHE display-enhancement passes, extracted from ExternalRenderContext
 *
 * P0.3 step 4a: the three compute passes (histogram, CDF, apply) that tone-map
 * and palette the display image, moved out of ExternalRenderContext::Impl
 * verbatim. The class is Vulkan-typed on purpose; step 4b moves it onto the
 * gpu:: compute RHI. ExternalRenderContext keeps owning the displayImage --
 * what the passes write -- and decides when a run is recorded.
 */

#pragma once

#include "core/Types.hpp"
#include "renderer/DisplayControl.hpp"

#include <memory>

#include <vulkan/vulkan.h>

namespace quantiloom {
class GpuImage;
class VulkanContext;
}

namespace quantiloom::rendercore {

/// What one Record() call needs that the caller owns: the extent, the
/// viewport's display parameters, the cached input window and whether the
/// input was written by compute (camera enabled) or by the ray tracing pass.
struct ClaheRun {
    u32 width = 0;
    u32 height = 0;
    DisplayEnhancementParams params;
    f32 inputMin = 0.0f;
    f32 inputMax = 1.0f;
    bool inputWrittenByCompute = false;
};

/// The three CLAHE passes over a caller-owned input/output image pair.
class GpuClahe {
public:
    /// Creates the pipelines, tile buffers and the descriptor set. Err
    /// carries the warning text ExternalRenderContext used to log verbatim
    /// ("CLAHE: Could not load shader files, CLAHE disabled", ...).
    [[nodiscard]] static Result<std::unique_ptr<GpuClahe>, String> Create(VulkanContext& context);
    ~GpuClahe();

    GpuClahe(const GpuClahe&) = delete;
    GpuClahe& operator=(const GpuClahe&) = delete;

    /// The five-binding update: input image, output image, histogram, CDF and
    /// min/max buffers.
    void BindAll(const GpuImage& input, const GpuImage& output);

    /// Rewrites binding 1 (the output image) after the caller recreated it.
    void SetOutput(const GpuImage& output);

    /// Records the three passes into @p cmd. Binding 0 is rewritten only when
    /// the input view moved since the last call.
    void Record(VkCommandBuffer cmd, const GpuImage& input, const ClaheRun& run);

private:
    struct Impl;
    explicit GpuClahe(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

}  // namespace quantiloom::rendercore
