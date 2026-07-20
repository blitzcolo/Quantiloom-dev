#include "PerformanceLogger.hpp"
#include "core/Log.hpp"
#include <stdexcept>
#include <cstring>

namespace quantiloom {

// ============================================================================
// Lifecycle
// ============================================================================

PerformanceLogger::PerformanceLogger(VulkanContext& context, const Config& config)
    : m_context(context)
    , m_config(config)
{
    if (!m_config.enableLogging) {
        QL_LOG_INFO("Performance logging disabled");
        return;
    }

    QL_LOG_INFO("Initializing performance logger...");
    QL_LOG_INFO("  CSV output: {}", m_config.csvFilePath);

    // Get timestamp period (ns per tick)
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_context.GetPhysicalDevice(), &properties);
    m_timestampPeriod = static_cast<f64>(properties.limits.timestampPeriod);

    QL_LOG_INFO("  GPU timestamp period: {:.3f} ns/tick", m_timestampPeriod);

    // Check if timestamps are supported
    if (properties.limits.timestampComputeAndGraphics == VK_FALSE) {
        QL_LOG_WARN("GPU timestamps not supported on this device!");
        m_config.enableLogging = false;
        return;
    }

    // Create query pool
    CreateQueryPool();

    // Open CSV file
    m_csvFile.open(m_config.csvFilePath, std::ios::out | std::ios::trunc);
    if (!m_csvFile.is_open()) {
        QL_LOG_ERROR("Failed to open CSV file: {}", m_config.csvFilePath);
        m_config.enableLogging = false;
        return;
    }

    WriteCSVHeader();

    QL_LOG_INFO("  Performance logger initialized");
}

PerformanceLogger::PerformanceLogger(VulkanContext& context)
    : PerformanceLogger(context, Config{})
{
}

PerformanceLogger::~PerformanceLogger() {
    if (m_csvFile.is_open()) {
        m_csvFile.close();
    }

    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_context.GetDevice(), m_queryPool, nullptr);
    }

    QL_LOG_INFO("Performance logger destroyed");
}

// ============================================================================
// Frame Timing
// ============================================================================

void PerformanceLogger::BeginFrame(VkCommandBuffer cmd) {
    if (!m_config.enableLogging || m_queryPool == VK_NULL_HANDLE) {
        return;
    }

    // Reset queries for this frame
    u32 startQuery = m_writeQueryIndex * 2;
    vkCmdResetQueryPool(cmd, m_queryPool, startQuery, 2);

    // Write start timestamp
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queryPool, startQuery);
}

void PerformanceLogger::EndFrame(VkCommandBuffer cmd) {
    if (!m_config.enableLogging || m_queryPool == VK_NULL_HANDLE) {
        return;
    }

    // Write end timestamp, then advance the write cursor so the next
    // recorded frame gets its own query pair (required for batched submits)
    u32 endQuery = m_writeQueryIndex * 2 + 1;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queryPool, endQuery);

    m_writeQueryIndex = (m_writeQueryIndex + 1) % (m_config.queryPoolSize / 2);
    m_pendingFrames++;
}

void PerformanceLogger::LogFrame(u32 frameIndex, u32 width, u32 height, u32 spp,
                                  f32 wavelength_nm, const String& spectralMode) {
    if (!m_config.enableLogging || m_queryPool == VK_NULL_HANDLE) {
        return;
    }

    // Resolve GPU time for the oldest unread frame
    f32 gpuMs = ResolveLastGpuMs();

    // Calculate rays per second
    // Total rays = width * height * spp
    u64 totalRays = static_cast<u64>(width) * static_cast<u64>(height) * static_cast<u64>(spp);
    f64 seconds = static_cast<f64>(gpuMs) / 1000.0;
    m_lastFrameRaysPerSec = seconds > 0.0 ? static_cast<f64>(totalRays) / seconds : 0.0;

    // Write to CSV
    if (m_csvFile.is_open()) {
        m_csvFile << frameIndex << ","
                  << width << ","
                  << height << ","
                  << spp << ","
                  << wavelength_nm << ","
                  << gpuMs << ","
                  << static_cast<u64>(m_lastFrameRaysPerSec) << ","
                  << spectralMode << "\n";
    }

    // Log to console
    QL_LOG_INFO("Frame {}: {}x{} @ {} spp, {:.1f} nm, GPU: {:.2f} ms, {:.2f} Mrays/s",
                frameIndex, width, height, spp, wavelength_nm,
                gpuMs, m_lastFrameRaysPerSec / 1e6);

}

f32 PerformanceLogger::ResolveLastGpuMs() {
    if (!m_config.enableLogging || m_queryPool == VK_NULL_HANDLE) {
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

void PerformanceLogger::Flush() {
    if (m_csvFile.is_open()) {
        m_csvFile.flush();
    }
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

    QL_LOG_INFO("  Created query pool with {} timestamps", m_config.queryPoolSize);
}

void PerformanceLogger::WriteCSVHeader() {
    if (m_csvFile.is_open() && !m_csvHeaderWritten) {
        m_csvFile << "frame,width,height,spp,wavelength_nm,gpu_ms,rays_per_sec,spectral_mode\n";
        m_csvHeaderWritten = true;
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
