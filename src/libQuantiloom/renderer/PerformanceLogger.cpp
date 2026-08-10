#include "PerformanceLogger.hpp"
#include "core/Log.hpp"
#include <stdexcept>

namespace quantiloom {

// ============================================================================
// Lifecycle
// ============================================================================

PerformanceLogger::PerformanceLogger(VulkanContext& context, const Config& config)
    : m_context(context)
    , m_config(config)
{
    // Get timestamp period (ns per tick)
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_context.GetPhysicalDevice(), &properties);
    m_timestampPeriod = static_cast<f64>(properties.limits.timestampPeriod);

    // Check if timestamps are supported
    if (properties.limits.timestampComputeAndGraphics == VK_FALSE) {
        QL_LOG_WARN("GPU timestamps not supported on this device; "
                    "frame timing will read 0");
        return;
    }

    CreateQueryPool();
    m_timestampsSupported = true;

    QL_LOG_INFO("GPU frame timing initialized ({:.3f} ns/tick, {} query slots)",
                m_timestampPeriod, m_config.queryPoolSize);
}

PerformanceLogger::PerformanceLogger(VulkanContext& context)
    : PerformanceLogger(context, Config{})
{
}

PerformanceLogger::~PerformanceLogger() {
    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_context.GetDevice(), m_queryPool, nullptr);
    }
}

// ============================================================================
// Frame Timing
// ============================================================================

void PerformanceLogger::BeginFrame(VkCommandBuffer cmd) {
    if (!m_timestampsSupported) {
        return;
    }

    // If the pool is about to lap the read cursor, drop the oldest unread
    // pair rather than reset a slot whose result was never collected.
    if (m_pendingFrames >= m_config.queryPoolSize / 2) {
        m_readQueryIndex = (m_readQueryIndex + 1) % (m_config.queryPoolSize / 2);
        m_pendingFrames--;
    }

    // Reset queries for this frame
    u32 startQuery = m_writeQueryIndex * 2;
    vkCmdResetQueryPool(cmd, m_queryPool, startQuery, 2);

    // Write start timestamp
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queryPool, startQuery);
}

void PerformanceLogger::EndFrame(VkCommandBuffer cmd) {
    if (!m_timestampsSupported) {
        return;
    }

    // Write end timestamp, then advance the write cursor so the next
    // recorded frame gets its own query pair (required for batched submits)
    u32 endQuery = m_writeQueryIndex * 2 + 1;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queryPool, endQuery);

    m_writeQueryIndex = (m_writeQueryIndex + 1) % (m_config.queryPoolSize / 2);
    m_pendingFrames++;
}

f32 PerformanceLogger::ResolveLastGpuMs() {
    if (!m_timestampsSupported) {
        return 0.0f;
    }

    // Never query a pair that was not recorded: QueryGpuTimeMs uses
    // VK_QUERY_RESULT_WAIT_BIT, which deadlocks on an unwritten query
    if (m_pendingFrames == 0) {
        return 0.0f;
    }

    m_lastFrameGpuMs = QueryGpuTimeMs(m_readQueryIndex);
    m_readQueryIndex = (m_readQueryIndex + 1) % (m_config.queryPoolSize / 2);
    m_pendingFrames--;
    return m_lastFrameGpuMs;
}

bool PerformanceLogger::TryResolvePending() {
    if (!m_timestampsSupported) {
        return false;
    }

    bool resolvedAny = false;
    while (m_pendingFrames > 0) {
        const u32 startQuery = m_readQueryIndex * 2;

        // [value, availability] per query. WITH_AVAILABILITY instead of WAIT:
        // the caller records commands the host has not even submitted yet, so
        // the oldest pair may legitimately be unfinished -- stop there.
        u64 results[4] = {0, 0, 0, 0};
        VkResult result = vkGetQueryPoolResults(
            m_context.GetDevice(),
            m_queryPool,
            startQuery,
            2,
            sizeof(results),
            results,
            2 * sizeof(u64),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
        );
        if (result != VK_SUCCESS && result != VK_NOT_READY) {
            QL_LOG_WARN("Failed to query GPU timestamps (result: {})",
                        static_cast<i32>(result));
            break;
        }
        if (results[1] == 0 || results[3] == 0) {
            break;  // oldest pair not complete; newer ones cannot be either
        }

        const u64 elapsedTicks = results[2] - results[0];
        m_lastFrameGpuMs = static_cast<f32>(
            static_cast<f64>(elapsedTicks) * m_timestampPeriod / 1e6);
        m_readQueryIndex = (m_readQueryIndex + 1) % (m_config.queryPoolSize / 2);
        m_pendingFrames--;
        resolvedAny = true;
    }
    return resolvedAny;
}

// ============================================================================
// Internal Helpers
// ============================================================================

void PerformanceLogger::CreateQueryPool() {
    VkQueryPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    poolInfo.queryCount = m_config.queryPoolSize;  // 2 timestamps per frame

    VkResult result = vkCreateQueryPool(m_context.GetDevice(), &poolInfo, nullptr, &m_queryPool);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create query pool for performance logging");
    }
}

f32 PerformanceLogger::QueryGpuTimeMs(u32 queryIndex) {
    VkDevice device = m_context.GetDevice();

    u32 startQuery = queryIndex * 2;

    u64 timestamps[2] = {0, 0};
    VkResult result = vkGetQueryPoolResults(
        device,
        m_queryPool,
        startQuery,
        2,  // Query 2 timestamps (start and end)
        sizeof(timestamps),
        timestamps,
        sizeof(u64),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    );

    if (result != VK_SUCCESS) {
        QL_LOG_WARN("Failed to query GPU timestamps (result: {})", static_cast<i32>(result));
        return 0.0f;
    }

    // Calculate elapsed time in milliseconds
    u64 elapsedTicks = timestamps[1] - timestamps[0];
    f64 elapsedNs = static_cast<f64>(elapsedTicks) * m_timestampPeriod;
    f32 elapsedMs = static_cast<f32>(elapsedNs / 1e6);

    return elapsedMs;
}

} // namespace quantiloom
