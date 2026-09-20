/**
 * @file camera_cdf.comp.hlsl
 * @brief Prefix-sum the camera statistics histogram into an Equalize CDF
 *
 * One group of 256 threads: serial prefix sum of the persistent 256-bin
 * histogram written by camera_stats phase 2, normalized by the unsaturated
 * pixel count, into the cdf buffer the display pass reads. The CPU twin is
 * the Equalize branch of AgcTone in postprocess/CameraIsp.cpp; like that
 * implementation the last CDF entry is exactly 1 whenever any pixel binned.
 */

#include "camera_common.hlsli"

[[vk::binding(14, 0)]] RWStructuredBuffer<uint> ispStats;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> cdfBuffer; // 256 entries

#define ISP_HISTOGRAM_BINS 256
#define ISP_STAT_UNSAT 260

groupshared float sCdf[ISP_HISTOGRAM_BINS];

[numthreads(256, 1, 1)]
void main(uint localIndex : SV_GroupIndex) {
    if (localIndex == 0) {
        const uint unsat = ispStats[ISP_STAT_UNSAT];
        const float invCount = unsat > 0u ? 1.0 / float(unsat) : 0.0;
        uint cumulative = 0;
        for (uint b = 0; b < ISP_HISTOGRAM_BINS; ++b) {
            cumulative += ispStats[b];
            sCdf[b] = float(cumulative) * invCount;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    cdfBuffer[localIndex] = sCdf[localIndex];
}
