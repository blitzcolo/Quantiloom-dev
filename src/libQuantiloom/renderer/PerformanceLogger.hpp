#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "VulkanContext.hpp"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <fstream>

// ============================================================================
// PerformanceLogger - GPU timestamp queries and CSV logging
// ============================================================================
// Measures GPU rendering performance using Vulkan timestamp queries
// and logs results to CSV files for analysis
//
// Usage:
//   PerformanceLogger logger(context);
//   logger.BeginFrame(cmdBuffer);
//   // ... render commands ...
//   logger.EndFrame(cmdBuffer);
//   logger.LogFrame(frameIndex, width, height, spp, wavelength);
//
// Output CSV format:
//   frame,width,height,spp,wavelength_nm,gpu_ms,rays_per_sec,spectral_mode
//
// References:
// - Vulkan spec: VkQueryPool for timestamp queries
// - SRS §3.2: Performance cost recording requirement
// ============================================================================

namespace quantiloom {

// TODO: Consider if this should be public API or internal tool
// Currently exported for backward compatibility with existing main.cpp
class QL_API PerformanceLogger {
public:
    // ========================================================================
    // Configuration
    // ========================================================================

    struct Config {
        String csvFilePath = "performance_log.csv";
        bool enableLogging = true;
        u32 queryPoolSize = 128;  // Number of frames to buffer
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

    // Log frame statistics to CSV
    // Call after GPU execution completes (after vkQueueWaitIdle or fence)
    void LogFrame(u32 frameIndex, u32 width, u32 height, u32 spp,
                  f32 wavelength_nm, const String& spectralMode);

    // Flush CSV file (ensure all data is written)
    void Flush();

    // ========================================================================
    // Accessors
    // ========================================================================

    // Get last frame GPU time in milliseconds
    [[nodiscard]] f32 GetLastFrameGpuMs() const { return m_lastFrameGpuMs; }

    // Get last frame rays per second
    [[nodiscard]] f64 GetLastFrameRaysPerSec() const { return m_lastFrameRaysPerSec; }

    // Query the current timestamp pair, store result, advance query index.
    // Call after GPU work has completed (fence wait / vkQueueWaitIdle).
    // Unlike LogFrame(), does NOT write to the CSV.
    f32 ResolveLastGpuMs();

private:
    // ========================================================================
    // Internal state
    // ========================================================================

    VulkanContext& m_context;
    Config m_config;

    // Vulkan query pool for timestamps.
    // Write cursor advances as frames are recorded (EndFrame);
    // read cursor advances as results are resolved (ResolveLastGpuMs/LogFrame).
    // Separate cursors allow multiple frames recorded per submit (batching).
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    u32 m_writeQueryIndex = 0;
    u32 m_readQueryIndex = 0;
    u32 m_pendingFrames = 0;

    // Timestamp frequency (nanoseconds per tick)
    f64 m_timestampPeriod = 1.0;

    // CSV output
    std::ofstream m_csvFile;
    bool m_csvHeaderWritten = false;

    // Last frame statistics
    f32 m_lastFrameGpuMs = 0.0f;
    f64 m_lastFrameRaysPerSec = 0.0;

    // ========================================================================
    // Helpers
    // ========================================================================

    void CreateQueryPool();
    void WriteCSVHeader();
    f32 QueryGpuTimeMs(u32 queryIndex);
};

} // namespace quantiloom
