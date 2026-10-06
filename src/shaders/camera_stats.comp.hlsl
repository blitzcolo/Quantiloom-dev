/**
 * @file camera_stats.comp.hlsl
 * @brief Full-resolution acquisition statistics for the camera display chain
 *
 * The statistics describe the display-domain scalar: thermal detectors report
 * the corrected DN (the infrared display's AGC input); photon detectors report
 * the well fraction max(corrected,0)/fullWell (the visible chain's input).
 * Three phases, selected by cameraPush.ispPhase, run back to back between
 * readout and the fused demosaic/color/display pass:
 *
 *   Phase 0 (per-16x16-tile groups): local min/max/sum/satCount plus the
 *     per-channel unsaturated sums and counts -> three tileStats rows
 *   Phase 1 (one group of 256): reduce tiles to global min/max/mean, build the
 *     NEXT Linear-AGC window from the PREVIOUS frame's histogram, preloaded
 *     into shared memory so the percentile walk is a shared-memory scan
 *     instead of hundreds of serial dependent global reads
 *   Phase 2 (per-16x16-tile groups): bin unsaturated pixels (value < 0.98*adcMax
 *     in DN terms) into a groupshared 256-bin histogram, then flush it to this
 *     tile's row of the tile histogram block (appended to the per-tile stats
 *     rows in binding 16). Per-tile private histograms replace the former
 *     whole-image global atomics: the merge is a coalesced read with no
 *     contention.
 *
 * The per-channel rows serve the AWB closed loop: the channel means are
 * the unsaturated display-domain scalar gathered per CFA/device channel
 * BEFORE white balance, exactly CameraIsp::ComputeAcquisitionStats on the
 * CPU. The channel sums (float bits) and counts ride in stat slots 263-268.
 *
 * camera_cdf.comp.hlsl then merges the per-tile histograms into the
 * persistent global histogram and prefix-sums it into a CDF for the Equalize
 * tone. Everything lives in the ispStats buffer, whose slots persist across
 * frames: the window and the histogram describe the previous acquisition, the
 * globals describe the current one (they are written before the display pass
 * consumes them in the same command buffer). The global histogram no longer
 * needs zeroing between frames: the merge pass overwrites all 256 bins.
 *
 * The CPU twin of the Equalize tone is AgcTone in postprocess/CameraIsp.cpp;
 * it bins every pixel over [min,max], so the GPU histogram (unsaturated
 * pixels only) matches it exactly for images without saturated pixels.
 *
 * Why not share with display_range.comp.hlsl: that shader percentiles BT.709
 * luminance of every finite pixel into 65,536 bins over the true absolute
 * range and scans them on the host with a truncating rank, all for the
 * current frame. This window instead covers unsaturated pixels only, in 256
 * bins, with an llround nearest-rank over the unsaturated count, and reads
 * the PREVIOUS frame's histogram -- the AGC window has to exist before this
 * frame's display pass runs, so a readback round-trip is not an option.
 * Different population, rank definition, resolution and timing: merging them
 * would have to reconcile all four.
 */

#include "camera_common.hlsli"

[[vk::binding(0, 0)]] RWTexture2D<float4> correctedImage;
[[vk::binding(9, 0)]] StructuredBuffer<float4> cameraConfig;
[[vk::binding(12, 0)]] StructuredBuffer<float4> ispConfig;
[[vk::binding(14, 0)]] RWStructuredBuffer<uint> ispStats;    // 272 entries

#define ISP_HISTOGRAM_BINS 256
#define ISP_STAT_MIN    256
#define ISP_STAT_MAX    257
#define ISP_STAT_MEAN   258
#define ISP_STAT_SAT    259
#define ISP_STAT_UNSAT  260
#define ISP_STAT_WIN_LO 261
#define ISP_STAT_WIN_HI 262
#define ISP_STAT_CH_SUM_R 263
#define ISP_STAT_CH_SUM_G 264
#define ISP_STAT_CH_SUM_B 265
#define ISP_STAT_CH_CNT_R 266
#define ISP_STAT_CH_CNT_G 267
#define ISP_STAT_CH_CNT_B 268

// CFA channel at a pixel; mirrors CfaChannelAt in the fused demosaic pass.
uint CfaChannelAt(uint cfa, uint x, uint y) {
    const uint px = x & 1u;
    const uint py = y & 1u;
    if (cfa == 1u) return !py ? (px ? 1u : 0u) : (px ? 2u : 1u);  // RGGB
    if (cfa == 2u) return !py ? (px ? 0u : 1u) : (px ? 1u : 2u);  // GRBG
    if (cfa == 3u) return !py ? (px ? 2u : 1u) : (px ? 1u : 0u);  // GBRG
    if (cfa == 4u) return !py ? (px ? 1u : 2u) : (px ? 0u : 1u);  // BGGR
    return 0u;                                                     // Mono
}

float ChannelValue(float4 c, uint channel) {
    if (cameraPush.detector == CAMERA_THERMAL) return c[channel];
    const float fullWell = max(ispConfig[6].z, 1e-20);
    return max(c[channel], 0.0) / fullWell;
}

float PixelValue(uint2 coord) {
    return ChannelValue(correctedImage[coord], 0u);
}

// One scalar per pixel (Mono/Bayer) or up to three parallel channels
// (MultiChannel), matching the element loop of the CPU stats.
uint StatsChannelCount() {
    return cameraPush.cfa == CAMERA_MULTI_CHANNEL
               ? min(cameraPush.channelCount, 3u) : 1u;
}

// Saturation lives in the RAW DN domain (dn >= 0.98*adcMax); the statistics
// scalar is the display-domain value, so convert the ceiling into it. The
// thermal corrected signal is absorbed power in W (dn = W * responsivity);
// the photon corrected signal is electrons (dn = e- / electronsPerDn, and
// the scalar is the well fraction e- / fullWell).
float SaturationThreshold() {
    const float adcMax = max(ispConfig[6].w, 1.0);
    if (cameraPush.detector == CAMERA_THERMAL)
        return 0.98 * adcMax / max(cameraConfig[3].y, 1e-20);
    const float fullWell = max(ispConfig[6].z, 1e-20);
    const float electronsPerDn = max(cameraConfig[2].y, 1e-20);
    return 0.98 * adcMax * electronsPerDn / fullWell;
}

uint TilesX() { return (cameraPush.width + 15u) / 16u; }
uint TileCount() { return TilesX() * ((cameraPush.height + 15u) / 16u); }

// First uint slot of the per-tile histogram block in binding 16: the partial
// statistics occupy 3 float4 (= 12 uint) rows per tile.
uint TileHistBase() { return TileCount() * 12u; }

// ---------------------------------------------------------------------------
// Phase 0: per-tile partial statistics
// ---------------------------------------------------------------------------
#ifdef CAMERA_STATS_PHASE_TILES

[[vk::binding(16, 0)]] RWStructuredBuffer<float4> tileStats;

// Three rows per tile: global min/max/sum/sat, per-channel unsaturated
// sums, per-channel unsaturated counts.
groupshared float4 sPartial[256];
groupshared float4 sChSum[256];
groupshared float4 sChCnt[256];

[numthreads(16, 16, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID,
          uint localIndex : SV_GroupIndex) {
    const uint2 tileMin = groupId.xy * 16u;
    const uint2 first = tileMin + localId.xy;
    const uint2 limit = min(tileMin + 16u,
                            uint2(cameraPush.width, cameraPush.height));
    float localMin = 3.402823e+38f;
    float localMax = -3.402823e+38f;
    float localSum = 0.0f;
    uint localSat = 0;
    float3 localChSum = 0.0f;
    float3 localChCnt = 0.0f;
    const float satThreshold = SaturationThreshold();
    const uint channels = StatsChannelCount();
    for (uint y = first.y; y < limit.y; y += 16u) {
        for (uint x = first.x; x < limit.x; x += 16u) {
            const uint2 coord = uint2(x, y);
            const float4 c = correctedImage[coord];
            const float v = ChannelValue(c, 0u);
            if (v >= satThreshold) {
                ++localSat;
                continue;
            }
            localMin = min(localMin, v);
            localMax = max(localMax, v);
            localSum += v;
            if (channels == 1u) {
                // Mono/Bayer: one scalar per pixel, attributed to its CFA
                // channel (Mono always lands on channel 0).
                const uint ch = CfaChannelAt(cameraPush.cfa, x, y);
                localChSum[ch] += v;
                localChCnt[ch] += 1.0f;
            } else {
                for (uint ch = 0u; ch < channels; ++ch) {
                    const float vc = ChannelValue(c, ch);
                    if (vc >= satThreshold) continue;
                    localChSum[ch] += vc;
                    localChCnt[ch] += 1.0f;
                }
            }
        }
    }
    sPartial[localIndex] = float4(localMin, localMax, localSum,
                                  float(localSat));
    sChSum[localIndex] = float4(localChSum, 0.0f);
    sChCnt[localIndex] = float4(localChCnt, 0.0f);
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            const float4 other = sPartial[localIndex + stride];
            const float sat = sPartial[localIndex].w + other.w;
            sPartial[localIndex] = float4(
                min(sPartial[localIndex].x, other.x),
                max(sPartial[localIndex].y, other.y),
                sPartial[localIndex].z + other.z, sat);
            sChSum[localIndex] += sChSum[localIndex + stride];
            sChCnt[localIndex] += sChCnt[localIndex + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (localIndex == 0) {
        const uint row = (groupId.y * TilesX() + groupId.x) * 3u;
        tileStats[row] = sPartial[0];
        tileStats[row + 1u] = sChSum[0];
        tileStats[row + 2u] = sChCnt[0];
    }
}

#endif // CAMERA_STATS_PHASE_TILES

// ---------------------------------------------------------------------------
// Phase 1: global reduce, next window from the previous histogram
// ---------------------------------------------------------------------------
#ifdef CAMERA_STATS_PHASE_REDUCE

[[vk::binding(16, 0)]] RWStructuredBuffer<float4> tileStats;

groupshared float4 sTiles[256];
groupshared float4 sChSum[256];
groupshared float4 sChCnt[256];
groupshared uint sPrevHist[ISP_HISTOGRAM_BINS];

// Nearest-rank percentile of the PREVIOUS histogram, reconstructed as a bin
// position over [prevMin, prevMax]. Serial; called by one thread over the
// shared-memory preload (the bins are a frame old and cold in every cache,
// so this must not be a chain of dependent global reads).
float PreviousPercentile(float prevMin, float prevMax, uint prevUnsat,
                         float percent) {
    const float rank = floor(clamp(percent, 0.0, 100.0) * 0.01 *
                                 float(prevUnsat - 1u) +
                             0.5); // llround, as in CameraIsp.cpp
    uint cumulative = 0;
    uint bin = ISP_HISTOGRAM_BINS - 1u;
    for (uint b = 0; b < ISP_HISTOGRAM_BINS; ++b) {
        cumulative += sPrevHist[b];
        // The rank is a 0-based index: its value is the (rank+1)-th sample.
        if (cumulative > uint(rank)) {
            bin = b;
            break;
        }
    }
    return prevMin + float(bin) * (prevMax - prevMin) /
                         float(ISP_HISTOGRAM_BINS - 1u);
}

[numthreads(256, 1, 1)]
void main(uint localIndex : SV_GroupIndex) {
    // Preload the previous frame's histogram in parallel: it is a full frame
    // old and therefore cold in every cache, and the percentile walk would
    // otherwise be hundreds of serial dependent DRAM reads.
    sPrevHist[localIndex] = ispStats[localIndex];
    const uint tileCount = TileCount();
    float gmin = 3.402823e+38f;
    float gmax = -3.402823e+38f;
    float gsum = 0.0f;
    uint gsat = 0;
    float3 chSum = 0.0f;
    float3 chCnt = 0.0f;
    for (uint i = localIndex; i < tileCount; i += 256u) {
        const float4 t = tileStats[i * 3u];
        gmin = min(gmin, t.x);
        gmax = max(gmax, t.y);
        gsum += t.z;
        gsat += uint(t.w);
        chSum += tileStats[i * 3u + 1u].xyz;
        chCnt += tileStats[i * 3u + 2u].xyz;
    }
    sTiles[localIndex] = float4(gmin, gmax, gsum, float(gsat));
    sChSum[localIndex] = float4(chSum, 0.0f);
    sChCnt[localIndex] = float4(chCnt, 0.0f);
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            const float4 other = sTiles[localIndex + stride];
            sTiles[localIndex] = float4(
                min(sTiles[localIndex].x, other.x),
                max(sTiles[localIndex].y, other.y),
                sTiles[localIndex].z + other.z,
                sTiles[localIndex].w + other.w);
            sChSum[localIndex] += sChSum[localIndex + stride];
            sChCnt[localIndex] += sChCnt[localIndex + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex == 0) {
        // The window describes the PREVIOUS frame: read its globals and
        // histogram before anything below overwrites them.
        const float prevMin = asfloat(ispStats[ISP_STAT_MIN]);
        const float prevMax = asfloat(ispStats[ISP_STAT_MAX]);
        const uint prevUnsat = ispStats[ISP_STAT_UNSAT];
        float winLo = 0.0f;
        float winHi = 0.0f;
        if (prevUnsat > 0u && prevMax > prevMin) {
            winLo = PreviousPercentile(prevMin, prevMax, prevUnsat,
                                       ispConfig[5].x);
            winHi = PreviousPercentile(prevMin, prevMax, prevUnsat,
                                       ispConfig[5].y);
        }
        const uint total = cameraPush.width * cameraPush.height;
        const uint sat = uint(sTiles[0].w);
        const uint unsat = total - sat;
        ispStats[ISP_STAT_MIN] = asuint(sTiles[0].x);
        ispStats[ISP_STAT_MAX] = asuint(sTiles[0].y);
        ispStats[ISP_STAT_MEAN] = asuint(unsat > 0u ? sTiles[0].z / float(unsat) : 0.0f);
        ispStats[ISP_STAT_SAT] = sat;
        ispStats[ISP_STAT_UNSAT] = unsat;
        ispStats[ISP_STAT_WIN_LO] = asuint(winLo);
        ispStats[ISP_STAT_WIN_HI] = asuint(winHi);
        // Per-channel unsaturated sums and counts for the AWB loop. The
        // counts ride as exact float bits (a sensor has fewer than 2^24
        // pixels); the host converts them back to integers.
        ispStats[ISP_STAT_CH_SUM_R] = asuint(sChSum[0].x);
        ispStats[ISP_STAT_CH_SUM_G] = asuint(sChSum[0].y);
        ispStats[ISP_STAT_CH_SUM_B] = asuint(sChSum[0].z);
        ispStats[ISP_STAT_CH_CNT_R] = asuint(sChCnt[0].x);
        ispStats[ISP_STAT_CH_CNT_G] = asuint(sChCnt[0].y);
        ispStats[ISP_STAT_CH_CNT_B] = asuint(sChCnt[0].z);
    }
    // No histogram zeroing here: camera_cdf merges the per-tile histograms
    // into all 256 global bins, overwriting whatever the previous frame left.
}

#endif // CAMERA_STATS_PHASE_REDUCE

// ---------------------------------------------------------------------------
// Phase 2: per-tile groupshared histogram, flushed to the tile histogram
// block for the camera_cdf merge
// ---------------------------------------------------------------------------
#ifdef CAMERA_STATS_PHASE_HISTOGRAM

[[vk::binding(16, 0)]] RWStructuredBuffer<uint> tileStats;

groupshared uint sHist[ISP_HISTOGRAM_BINS];

[numthreads(16, 16, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID,
          uint localIndex : SV_GroupIndex) {
    sHist[localIndex] = 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint2 coord = groupId.xy * 16u + localId.xy;
    if (coord.x < cameraPush.width && coord.y < cameraPush.height) {
        const float v = PixelValue(coord);
        if (v < SaturationThreshold()) {
            const float gmin = asfloat(ispStats[ISP_STAT_MIN]);
            const float gmax = asfloat(ispStats[ISP_STAT_MAX]);
            if (gmax > gmin) {
                const uint bin = min(uint(saturate((v - gmin) / (gmax - gmin)) *
                                              float(ISP_HISTOGRAM_BINS - 1u) +
                                          0.5f),
                                     ISP_HISTOGRAM_BINS - 1u);
                InterlockedAdd(sHist[bin], 1u);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Every thread flushes its own bin slot: 1 KB of coalesced writes per
    // tile and no global atomic contention anywhere.
    const uint tileIndex = groupId.y * TilesX() + groupId.x;
    tileStats[TileHistBase() + tileIndex * ISP_HISTOGRAM_BINS + localIndex] =
        sHist[localIndex];
}

#endif // CAMERA_STATS_PHASE_HISTOGRAM
