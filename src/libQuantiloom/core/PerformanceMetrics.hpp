/**
 * @file PerformanceMetrics.hpp
 * @brief GPU/CPU performance timing and throughput measurement system
 *
 * Provides PerformanceMetrics class for measuring rendering performance:
 * - GPU execution time via Vulkan timestamp queries
 * - CPU frame time via std::chrono high-resolution clock
 * - Derived metrics: seconds/frame, rays/second, throughput
 * - CSV export for performance analysis
 *
 * Complies with SRS §3.2 requirement: "seconds-per-frame" metric.
 *
 * Metrics collected:
 * - GPU time (ms): vkCmdWriteTimestamp around ray tracing dispatch
 * - CPU time (ms): std::chrono wall-clock time
 * - Total rays: width × height × spp
 * - Rays/second: totalRays / (gpuTime_ms / 1000.0)
 * - Mrays/second: rays/second / 1e6
 *
 * Usage example:
 * @code
 * PerformanceMetrics metrics(context);
 *
 * // Frame rendering loop
 * metrics.BeginFrame(cmd);
 * pipeline.TraceRays(cmd, width, height);
 * metrics.EndFrame(cmd);
 *
 * // Query results
 * auto stats = metrics.GetLastFrameStats();
 * QL_LOG_INFO("GPU: {:.2f} ms, Throughput: {:.1f} Mrays/s",
 *             stats.gpuTime_ms, stats.mraysPerSecond());
 *
 * // Export to CSV
 * metrics.ExportCSV("performance.csv");
 * @endcode
 *
 * @note BeginFrame/EndFrame must be called in same command buffer
 * @note Timestamp query requires VK_QUERY_TYPE_TIMESTAMP support
 * @note Results available after GPU execution completes (vkQueueWaitIdle)
 *
 * @author blitzcolo
 */

// ============================================================================
// Quantiloom - Performance Metrics System
// ============================================================================

#pragma once

#include "core/Types.hpp"
#include "core/Log.hpp"
#include "renderer/VulkanContext.hpp"

#include <vulkan/vulkan.h>
#include <vector>
#include <chrono>
#include <fstream>

namespace quantiloom {

// ============================================================================
// PerformanceMetrics Class
// ============================================================================

class PerformanceMetrics {
public:
    // ========================================================================
    // Frame Statistics
    // ========================================================================

    struct FrameStats {
        f64 gpuTime_ms = 0.0;      // GPU execution time (milliseconds)
        f64 cpuTime_ms = 0.0;      // CPU frame time (milliseconds)
        u32 spp = 1;               // Samples per pixel
        u32 width = 0;             // Render resolution width
        u32 height = 0;            // Render resolution height
        u64 totalRays = 0;         // Total rays traced

        // Derived metrics
        f64 secondsPerFrame() const { return gpuTime_ms / 1000.0; }
        f64 raysPerSecond() const {
            const f64 seconds = secondsPerFrame();
            return seconds > 0.0 ? totalRays / seconds : 0.0;
        }
        f64 mraysPerSecond() const { return raysPerSecond() / 1e6; }  // Million rays/sec
    };

    // ========================================================================
    // Public Interface
    // ========================================================================

    explicit PerformanceMetrics(VulkanContext& context);
    ~PerformanceMetrics();

    // Initialize query pool
    bool Initialize(u32 maxFramesInFlight = 2);

    // Record timestamps
    void BeginFrame(VkCommandBuffer cmd);
    void EndFrame(VkCommandBuffer cmd);

    // Retrieve results
    FrameStats GetLastFrameStats();

    // Set render parameters (for throughput calculation)
    void SetRenderParams(u32 width, u32 height, u32 spp);

    // Export to CSV for analysis
    void ExportToCSV(const String& filepath, const std::vector<FrameStats>& history);

    // Cleanup
    void Destroy();

private:
    VulkanContext& m_context;

    VkQueryPool m_timestampPool = VK_NULL_HANDLE;
    u32 m_maxFramesInFlight = 2;
    u32 m_currentFrame = 0;

    // Timestamp period (nanoseconds per tick)
    f64 m_timestampPeriod = 1.0;

    // Render parameters
    u32 m_width = 0;
    u32 m_height = 0;
    u32 m_spp = 1;

    // CPU timing
    std::chrono::high_resolution_clock::time_point m_cpuFrameStart;
    f64 m_lastCpuTime_ms = 0.0;

    // Cached stats
    FrameStats m_lastFrameStats;
};

} // namespace quantiloom
