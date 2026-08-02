/**
 * @file RenderDevice.cpp
 * @brief The scene-independent half of an offline render, created once
 *
 * Every line here is a step OfflineRenderer::Create used to take for itself. It
 * still takes them when no device is handed to it -- see the borrow branches over
 * there -- so this file is not a second setup path, it is the same four steps
 * moved somewhere they can be reached more than once.
 *
 * @author blitzcolo
 */

#include "renderer/RenderDeviceImpl.hpp"

#include "core/Log.hpp"
#include "renderer/RayTracingPipeline.hpp"

namespace quantiloom {

RenderDevice::Impl::~Impl() {
    // Before the members go, because the cache was created against the context
    // and the context is one of them. Saved on the way out so the next process
    // starts warm -- once for the whole batch rather than once per render.
    if (pipelineCache != VK_NULL_HANDLE && context) {
        if (!init.pipelineCachePath.empty()) {
            RayTracingPipeline::SavePipelineCache(*context, pipelineCache,
                                                  init.pipelineCachePath);
        }
        RayTracingPipeline::DestroyPipelineCache(*context, pipelineCache);
    }
}

RenderDevice::RenderDevice() : m_impl(std::make_unique<Impl>()) {}
RenderDevice::~RenderDevice() = default;

Result<std::unique_ptr<RenderDevice>, String> RenderDevice::Create(
    const InitParams& params) {
    using CreateResult = Result<std::unique_ptr<RenderDevice>, String>;

    // Not make_unique: the constructor is private, and Create() being the only
    // way in is the point -- the same shape as OfflineRenderer.
    std::unique_ptr<RenderDevice> self(new RenderDevice());
    Impl& impl = *self->m_impl;
    impl.init = params;

    QL_LOG_INFO("Initializing shared Vulkan context...");
    impl.context = std::make_unique<VulkanContext>();
    if (!impl.context->IsRayTracingSupported()) {
        return CreateResult::Err("Ray tracing not supported on this device");
    }

    if (!params.pipelineCachePath.empty()) {
        impl.pipelineCache = RayTracingPipeline::LoadPipelineCache(
            *impl.context, params.pipelineCachePath);
    }

    // Generated or read from its own binary cache, and either way seconds of work
    // at the default resolution. Once per device, not once per scene.
    impl.brdfLut = rendercore::BrdfLut::Create(*impl.context);
    if (!impl.brdfLut.IsValid()) {
        return CreateResult::Err("Failed to create BRDF LUT sampler");
    }

    impl.cieCMF_LUTBuffer = rendercore::CreateCieColourMatchingBuffer(*impl.context);
    impl.fallbackEnvMap = rendercore::EnvironmentCubemap::Fallback(*impl.context);

    QL_LOG_INFO("  Shared render device ready");
    return CreateResult(std::move(self));
}

}  // namespace quantiloom
