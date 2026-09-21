/**
 * @file camera_cdf.comp.hlsl
 * @brief Merge the per-tile histograms and prefix-sum them into an Equalize CDF
 *
 * One group of 256 threads. Each thread owns one histogram bin and sums that
 * bin across every 16x16 tile's private histogram (written by camera_stats
 * phase 2 into the tile histogram block of binding 16), which replaces the
 * former whole-image InterlockedAdd storm with a coalesced, contention-free
 * read. The merged counts are written back to the persistent global
 * histogram (the window builder of the NEXT frame's stats reduce reads them)
 * and scanned into the CDF the display pass reads for the Equalize tone.
 *
 * The scan is an exact integer Hillis-Steele prefix sum over shared memory:
 * bit-identical to the former serial accumulation, including the normalized
 * last entry (float(unsat) * (1 / float(unsat)), exactly 1 whenever any
 * pixel binned, as on the CPU). The CPU twin is the Equalize branch of
 * AgcTone in postprocess/CameraIsp.cpp.
 */

#include "camera_common.hlsli"

[[vk::binding(14, 0)]] RWStructuredBuffer<uint> ispStats;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> cdfBuffer; // 256 entries
[[vk::binding(16, 0)]] RWStructuredBuffer<uint> tileStats;  // stats rows + per-tile histograms

#define ISP_HISTOGRAM_BINS 256
#define ISP_STAT_UNSAT 260

groupshared uint sHist[ISP_HISTOGRAM_BINS];

uint TilesX() { return (cameraPush.width + 15u) / 16u; }
uint TileCount() { return TilesX() * ((cameraPush.height + 15u) / 16u); }
uint TileHistBase() { return TileCount() * 12u; }

[numthreads(256, 1, 1)]
void main(uint localIndex : SV_GroupIndex) {
    const uint tileCount = TileCount();
    const uint base = TileHistBase() + localIndex;
    uint sum = 0;
    for (uint t = 0; t < tileCount; ++t)
        sum += tileStats[base + t * ISP_HISTOGRAM_BINS];
    sHist[localIndex] = sum;
    // This frame's global histogram: the NEXT frame's window builder walks
    // these slots, and the host's statistics readback reports them.
    ispStats[localIndex] = sum;
    GroupMemoryBarrierWithGroupSync();
    // Exact inclusive prefix sum (uint additions cannot round).
    for (uint offset = 1u; offset < ISP_HISTOGRAM_BINS; offset <<= 1u) {
        const uint add = localIndex >= offset ? sHist[localIndex - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sHist[localIndex] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    const uint unsat = ispStats[ISP_STAT_UNSAT];
    const float invCount = unsat > 0u ? 1.0 / float(unsat) : 0.0;
    cdfBuffer[localIndex] = float(sHist[localIndex]) * invCount;
}
