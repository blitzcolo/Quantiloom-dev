#include "PerformanceLogger.hpp"
#include "core/Log.hpp"
#include "gpu/Device.hpp"
#include <stdexcept>
#include <utility>

namespace quantiloom {

// ============================================================================
// Lifecycle
// ============================================================================

PerformanceLogger::PerformanceLogger(gpu::Device& device, const Config& config)
    : m_config(config)
{
    // Get timestamp period (ns per tick)
    const gpu::DeviceInfo& info = device.Info();
    m_timestampPeriod = info.timestampPeriodNs;

    // Check if timestamps are supported
    if (!info.timestampsOnAllQueues) {
        QL_LOG_WARN("GPU timestamps not supported on this device; "
                    "frame timing will read 0");
        return;
    }

    CreateQueryPool(device);
    m_timestampsSupported = true;

    QL_LOG_INFO("GPU frame timing initialized ({:.3f} ns/tick, {} query slots)",
                m_timestampPeriod, m_config.queryPoolSize);
}

PerformanceLogger::PerformanceLogger(gpu::Device& device)
    : PerformanceLogger(device, Config{})
{
}

PerformanceLogger::~PerformanceLogger() {
    if (m_droppedFrames > 0) {
        QL_LOG_WARN("PerformanceLogger dropped {} GPU frame timing(s) the pool "
                    "could not hold ({} pairs); enlarge queryPoolSize if the gap "
                    "was not expected",
                    m_droppedFrames, m_config.queryPoolSize / 2);
    }
    m_queryPool.reset();
}

// ============================================================================
// Frame Timing
// ============================================================================

void PerformanceLogger::BeginFrame(gpu::CommandList& list) {
    if (!m_timestampsSupported) {
        return;
    }

    // If the pool is about to lap the read cursor, drop the oldest unread
    // pair rather than reset a slot whose result was never collected. The
    // drop is counted and reported: a silent gap reads as a speedup.
    if (m_pendingFrames >= m_config.queryPoolSize / 2) {
        if (m_droppedFrames == 0) {
            QL_LOG_WARN("GPU frame timing pool exhausted ({} pairs); oldest unread "
                        "timing dropped. TryResolvePending/ResolveLastGpuMs is not "
                        "keeping up with BeginFrame",
                        m_config.queryPoolSize / 2);
        }
        m_droppedFrames++;
        m_readQueryIndex = (m_readQueryIndex + 1) % (m_config.queryPoolSize / 2);
        m_pendingFrames--;
    }

    // Reset queries for this frame
    u32 startQuery = m_writeQueryIndex * 2;
    list.ResetTimestamps(m_queryPool->Ref(), startQuery, 2);

    // Write start timestamp
    list.WriteTimestamp(m_queryPool->Ref(), startQuery, gpu::Stage::TopOfPipe);
}

void PerformanceLogger::EndFrame(gpu::CommandList& list) {
    if (!m_timestampsSupported) {
        return;
    }

    // Write end timestamp, then advance the write cursor so the next
    // recorded frame gets its own query pair (required for batched submits)
    u32 endQuery = m_writeQueryIndex * 2 + 1;
    list.WriteTimestamp(m_queryPool->Ref(), endQuery, gpu::Stage::BottomOfPipe);

    m_writeQueryIndex = (m_writeQueryIndex + 1) % (m_config.queryPoolSize / 2);
    m_pendingFrames++;
}

f32 PerformanceLogger::ResolveLastGpuMs() {
    if (!m_timestampsSupported) {
        return 0.0f;
    }

    // Never query a pair that was not recorded: QueryGpuTimeMs resolves with
    // TimestampResolve::Wait, which deadlocks on an unwritten query
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
        const auto resolved = m_queryPool->Resolve(
            startQuery, 2, gpu::TimestampResolve::WithAvailability, results);
        if (!resolved) {
            QL_LOG_WARN("Failed to query GPU timestamps (result: {})",
                        resolved.error());
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

void PerformanceLogger::CreateQueryPool(gpu::Device& device) {
    auto pool = device.CreateTimestampPool(m_config.queryPoolSize);  // 2 timestamps per frame
    if (!pool) {
        throw std::runtime_error("Failed to create query pool for performance logging");
    }
    m_queryPool = std::move(pool).value();
}

f32 PerformanceLogger::QueryGpuTimeMs(u32 queryIndex) {
    u32 startQuery = queryIndex * 2;

    u64 timestamps[2] = {0, 0};
    const auto resolved = m_queryPool->Resolve(
        startQuery,
        2,  // Query 2 timestamps (start and end)
        gpu::TimestampResolve::Wait,
        timestamps);

    // A wait returns only once both values are written, so anything but
    // Ready is a failure, as any result but success was before.
    if (!resolved || *resolved != gpu::ResolveStatus::Ready) {
        QL_LOG_WARN("Failed to query GPU timestamps (result: {})",
                    resolved ? String("not ready") : resolved.error());
        return 0.0f;
    }

    // Calculate elapsed time in milliseconds
    u64 elapsedTicks = timestamps[1] - timestamps[0];
    f64 elapsedNs = static_cast<f64>(elapsedTicks) * m_timestampPeriod;
    f32 elapsedMs = static_cast<f32>(elapsedNs / 1e6);

    return elapsedMs;
}

} // namespace quantiloom
