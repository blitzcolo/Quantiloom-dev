#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "VulkanContext.hpp"
#include <vulkan/vulkan.h>

// ============================================================================
// PerformanceLogger - GPU timestamps around the ray-tracing dispatch
// ============================================================================
// A pool of timestamp query pairs with independent write and read cursors, so
// several dispatches can be recorded into one submit and resolved later.
//
// Two consumers, two resolve paths:
//
//   OfflineRenderer waits on a fence after each batch, so its results are
//   guaranteed ready -- it calls ResolveLastGpuMs(), which uses WAIT_BIT and
//   would deadlock on a query that was never recorded (hence the pending-frame
//   guard).
//
//   ExternalRenderContext only *records* commands; the host (QVulkanWindow)
//   submits them after RenderFrame returns, and the next RenderFrame can start
//   while earlier ones are still in flight. It calls TryResolvePending(),
//   which drains every pair the GPU has finished and never blocks -- a stall
//   here would serialize the CPU on the GPU and show up as lost frames.
//
// References:
// - Vulkan spec: VkQueryPool for timestamp queries
// - SRS §3.2: Performance cost recording requirement
// ============================================================================

namespace quantiloom {

class PerformanceLogger {
public:
    // ========================================================================
    // Configuration
    // ========================================================================

    struct Config {
        u32 queryPoolSize = 128;  // Timestamps (2 per frame); 64 frames in flight
    };

    // ========================================================================
    // Lifecycle
    // ========================================================================

    PerformanceLogger(VulkanContext& context, const Config& config);
    explicit PerformanceLogger(VulkanContext& context);  // Uses default Config
    ~PerformanceLogger();

    // Non-copyable, non-movable
    PerformanceLogger(const PerformanceLogger&) = delete;
    PerformanceLogger& operator=(const PerformanceLogger&) = delete;

    // ========================================================================
    // Frame timing
    // ========================================================================

    // Begin frame timing (records start timestamp)
    void BeginFrame(VkCommandBuffer cmd);

    // End frame timing (records end timestamp)
    void EndFrame(VkCommandBuffer cmd);

    // ========================================================================
    // Accessors
    // ========================================================================

    // Get last resolved frame GPU time in milliseconds
    [[nodiscard]] f32 GetLastFrameGpuMs() const { return m_lastFrameGpuMs; }

    // Query the current timestamp pair, store result, advance query index.
    // Blocks until the result is available (WAIT_BIT) -- only call after the
    // GPU work is known complete (fence wait / vkQueueWaitIdle).
    f32 ResolveLastGpuMs();

    // Resolve every recorded pair the GPU has already finished, oldest first,
    // without blocking. GetLastFrameGpuMs() afterwards returns the newest
    // resolved pair. Returns true if at least one pair was resolved.
    bool TryResolvePending();

private:
    // ========================================================================
    // Internal state
    // ========================================================================

    VulkanContext& m_context;
    Config m_config;

    // False when the device reports no timestamp support; every method is a
    // no-op then.
    bool m_timestampsSupported = false;

    // Vulkan query pool for timestamps.
    // Write cursor advances as frames are recorded (EndFrame);
    // read cursor advances as results are resolved.
    // Separate cursors allow multiple frames recorded per submit (batching).
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    u32 m_writeQueryIndex = 0;
    u32 m_readQueryIndex = 0;
    u32 m_pendingFrames = 0;

    // Timestamp frequency (nanoseconds per tick)
    f64 m_timestampPeriod = 1.0;

    // Last frame statistics
    f32 m_lastFrameGpuMs = 0.0f;

    // ========================================================================
    // Helpers
    // ========================================================================

    void CreateQueryPool();
    f32 QueryGpuTimeMs(u32 queryIndex);
};

} // namespace quantiloom
