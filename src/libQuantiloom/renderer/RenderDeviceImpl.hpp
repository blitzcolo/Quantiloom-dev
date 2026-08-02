/**
 * @file RenderDeviceImpl.hpp
 * @brief RenderDevice::Impl, so that OfflineRenderer can borrow what it holds
 *
 * Internal. The public header hides all of this behind a pimpl; OfflineRenderer is
 * declared a friend there and includes this to reach the members. Nothing else
 * should include it.
 *
 * @author blitzcolo
 */

#pragma once

#include "renderer/RenderDevice.hpp"

#include "renderer/GpuBuffer.hpp"
#include "renderer/RenderCore.hpp"
#include "renderer/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <memory>

namespace quantiloom {

/**
 * @brief The shared half of an offline render: device, cache, and two LUTs
 *
 * DECLARATION ORDER IS DESTRUCTION ORDER, for the reason spelled out in
 * OfflineRenderer.cpp: the context is declared first so it dies last, after every
 * resource allocated from it.
 */
struct RenderDevice::Impl {
    RenderDevice::InitParams init;

    std::unique_ptr<VulkanContext> context;
    rendercore::BrdfLut brdfLut;
    std::unique_ptr<GpuBuffer> cieCMF_LUTBuffer;
    /// Bound when a scene names no environment map, which is most of them.
    rendercore::EnvironmentCubemap fallbackEnvMap;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;

    ~Impl();
};

}  // namespace quantiloom
