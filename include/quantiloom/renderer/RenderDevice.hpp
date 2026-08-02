/**
 * @file RenderDevice.hpp
 * @brief A GPU device and its scene-independent resources, shared across renders
 *
 * OfflineRenderer creates a device, a pipeline cache, a BRDF LUT and a colour
 * matching table for every scene it renders, and none of those four depend on the
 * scene. A host rendering one image pays for that once and does not care. A host
 * rendering a hundred pays for it a hundred times: device creation alone is ~250 ms
 * against ~6 ms of actual tracing for a small frame, and the pipeline cache is read
 * from disk and written back once per image.
 *
 * This is that fixed cost, hoisted out. Create one, hand it to every
 * OfflineRenderer::InitParams::sharedDevice, destroy it when the batch is done.
 *
 * @code
 * auto device = RenderDevice::Create();
 * if (!device) { QL_LOG_ERROR("{}", device.error()); return 1; }
 *
 * for (const auto& config : configs) {
 *     OfflineRenderer::InitParams init;
 *     init.sharedDevice = device.value().get();
 *     auto renderer = OfflineRenderer::Create(config, init);
 *     // ... render, write, destroy the renderer; the device stays
 * }
 * @endcode
 *
 * @note Not thread-safe, and deliberately so: the renderers sharing one of these
 *       submit to a single unsynchronised queue. Renders sharing a device run one
 *       after another.
 * @note Must outlive every OfflineRenderer created against it.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Platform.hpp"
#include "core/Types.hpp"

#include <memory>

namespace quantiloom {

/**
 * @class RenderDevice
 * @brief A Vulkan device plus the render resources that do not depend on a scene
 */
class QL_API RenderDevice {
public:
    struct InitParams {
        /**
         * @brief Read once here and written back once on destruction
         *
         * The reason this class exists in miniature: OfflineRenderer reads and
         * rewrites this file per render, so a batch of a hundred does a hundred
         * round trips through it for a cache that stops growing after the first
         * few. Empty disables both.
         */
        String pipelineCachePath = "pipeline_cache.bin";
    };

    /**
     * @brief Create the device and everything on it that a scene cannot change
     *
     * @return The device, or why one could not be made
     */
    static Result<std::unique_ptr<RenderDevice>, String> Create(
        const InitParams& params = {});

    ~RenderDevice();

    RenderDevice(const RenderDevice&) = delete;
    RenderDevice& operator=(const RenderDevice&) = delete;
    RenderDevice(RenderDevice&&) = delete;
    RenderDevice& operator=(RenderDevice&&) = delete;

private:
    RenderDevice();

    struct Impl;
    std::unique_ptr<Impl> m_impl;

    /// The only consumer, and the reason nothing above hands out a VkDevice: what
    /// is shared is a library internal, and exposing it would put Vulkan in the
    /// public API for one caller's benefit.
    friend class OfflineRenderer;
};

}  // namespace quantiloom
