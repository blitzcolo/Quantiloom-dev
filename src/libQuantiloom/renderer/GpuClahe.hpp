/**
 * @file GpuClahe.hpp
 * @brief The CLAHE display-enhancement passes (internal compute RHI)
 *
 * The three compute passes (histogram, CDF, apply) that tone-map and palette
 * the display image. P0.3 step 4a moved them out of ExternalRenderContext;
 * step 4b put them on the gpu:: compute RHI. ExternalRenderContext keeps
 * owning the displayImage -- what the passes write -- and decides when a run
 * is recorded.
 */

#pragma once

#include "core/Types.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/GpuTypes.hpp"
#include "renderer/DisplayControl.hpp"

#include <memory>

namespace quantiloom::gpu {
class Device;
}

namespace quantiloom::rendercore {

/// What one Record() call needs that the caller owns: the extent, the
/// viewport's display parameters, the cached input window and which pass
/// wrote the input (the camera chain's compute dispatches, or the ray
/// tracing pass) -- the source scope of Record's entry barrier.
struct ClaheRun {
    u32 width = 0;
    u32 height = 0;
    DisplayEnhancementParams params;
    f32 inputMin = 0.0f;
    f32 inputMax = 1.0f;
    gpu::Access inputProducer = gpu::Access::RayTraceWrite;
};

/// The three CLAHE passes over a caller-owned input/output image pair.
class GpuClahe {
public:
    /// Creates the pipelines, tile buffers and the descriptor set. Err
    /// carries the warning text ExternalRenderContext used to log verbatim
    /// ("CLAHE: Could not load shader files, CLAHE disabled", ...).
    [[nodiscard]] static Result<std::unique_ptr<GpuClahe>, String> Create(gpu::Device& device);
    ~GpuClahe();

    GpuClahe(const GpuClahe&) = delete;
    GpuClahe& operator=(const GpuClahe&) = delete;

    /// The five-binding update: input image, output image, histogram, CDF and
    /// min/max buffers.
    void BindAll(gpu::TextureRef input, gpu::TextureRef output);

    /// Rewrites binding 1 (the output image) after the caller recreated it.
    void SetOutput(gpu::TextureRef output);

    /// Records the three passes. Binding 0 is rewritten only when the input
    /// view moved since the last call.
    void Record(gpu::CommandList& cmd, gpu::TextureRef input, const ClaheRun& run);

private:
    struct Impl;
    explicit GpuClahe(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

}  // namespace quantiloom::rendercore
