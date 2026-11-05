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
#include "core/ResourceLocator.hpp"
#include "gpu/Device.hpp"
#include "gpu/DeviceFactory.hpp"
#include "renderer/RayTracingPipeline.hpp"

namespace quantiloom {

RenderDevice::Impl::~Impl() {
    // Before the members go, because the cache was created against the device
    // and the device is one of them. Saved on the way out so the next process
    // starts warm -- once for the whole batch rather than once per render.
    if (pipelineCache && gpuDevice) {
        if (!init.pipelineCachePath.empty()) {
            RayTracingPipeline::SavePipelineCache(pipelineCache.get(),
                                                  init.pipelineCachePath);
        }
        RayTracingPipeline::DestroyPipelineCache(pipelineCache);
    }
}

RenderDevice::RenderDevice() : m_impl(std::make_unique<Impl>()) {}
RenderDevice::~RenderDevice() = default;

Result<std::unique_ptr<RenderDevice>, String> RenderDevice::Create(
    const InitParams& params) {
    using CreateResult = Result<std::unique_ptr<RenderDevice>, String>;

    // Fail before touching the device when no shader set can be found, so the
    // error names the roots that were probed rather than a missing file.
    if (const String resourcesError = core::ResourceNotFoundMessage(); !resourcesError.empty()) {
        return CreateResult::Err(resourcesError);
    }

    // Not make_unique: the constructor is private, and Create() being the only
    // way in is the point -- the same shape as OfflineRenderer.
    std::unique_ptr<RenderDevice> self(new RenderDevice());
    Impl& impl = *self->m_impl;
    impl.init = params;

    QL_LOG_INFO("Initializing shared {} device...", gpu::BackendName(gpu::DefaultBackend()));
    // What the context's constructor threw is now the Err; the callers test the Result.
    auto created = gpu::CreateSystemDevice(gpu::DefaultBackend());
    if (!created) {
        return CreateResult::Err(std::move(created).error());
    }
    impl.gpuDevice = std::move(created).value();
    gpu::Device& gpuDevice = *impl.gpuDevice;
    if (!gpuDevice.Info().hardwareRayTracing) {
        return CreateResult::Err("Ray tracing not supported on this device");
    }

    if (!params.pipelineCachePath.empty()) {
        impl.pipelineCache = RayTracingPipeline::LoadPipelineCache(
            gpuDevice, params.pipelineCachePath);
    }

    // Generated or read from its own binary cache, and either way seconds of work
    // at the default resolution. Once per device, not once per scene.
    impl.brdfLut = rendercore::BrdfLut::Create(gpuDevice);
    if (!impl.brdfLut.IsValid()) {
        return CreateResult::Err("Failed to create BRDF LUT sampler");
    }

    impl.cieCMF_LUTBuffer = rendercore::CreateCieColourMatchingBuffer(gpuDevice);
    impl.rgbToSpectrumBuffer = rendercore::CreateRgbToSpectrumBuffer(gpuDevice);
    impl.fallbackEnvMap = rendercore::EnvironmentCubemap::Fallback(gpuDevice);

    QL_LOG_INFO("  Shared render device ready");
    return CreateResult(std::move(self));
}

}  // namespace quantiloom
