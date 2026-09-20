/**
 * @file camera_demosaic.comp.hlsl
 * @brief Preprocess + MHC demosaic into linear RGB
 *
 * Consumes the corrected device signal (black level and NUC were applied in
 * the readout, so calibration stays visible in the display) and produces the
 * linear per-pixel RGB the color pass consumes:
 *
 *   Bayer:         white balance per CFA site, defect-pixel repair (mean of
 *                  the non-defect four-neighbors, exactly PreprocessPlane in
 *                  postprocess/CameraIsp.cpp), then Malvar-He-Cutler 5x5.
 *   Mono:          the scalar plane replicated.
 *   MultiChannel:  up to three device channels mapped onto display R/G/B,
 *                  unused slots replicating the last plane.
 *   Thermal:       passthrough of the corrected scalar; the infrared display
 *                  branch reads the corrected product directly.
 *
 * The 5x5 convolution runs from a 20x20 groupshared tile (16x16 threads plus
 * a two-pixel halo) with replicated borders, mirroring Convolve5's clamped
 * indexing on the CPU.
 */

#include "camera_common.hlsli"
#include "camera_isp.hlsli"

[[vk::binding(0, 0)]] RWTexture2D<float4> correctedImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> linearRgbImage;
[[vk::binding(12, 0)]] StructuredBuffer<float4> ispConfig;
[[vk::binding(13, 0)]] StructuredBuffer<uint2> defectPixels;

static const uint ISP_FLAG_DEFECTS = 16u;

groupshared float gPlane[20u * 20u];

// Row-major 5x5 MHC kernels are flat float[25] in camera_isp.hlsli.
float Convolve5(uint cx, uint cy, uint kernel) {
    float sum = 0.0f;
    for (uint dy = 0; dy < 5u; ++dy) {
        const uint sy = cy + dy; // cy already includes the -2 offset
        for (uint dx = 0; dx < 5u; ++dx) {
            const uint tap = dy * 5u + dx;
            float weight;
            if (kernel == 0u) weight = kMhcGreenAtRedBlue[tap];
            else if (kernel == 1u) weight = kMhcColorAtGreenHorizontal[tap];
            else if (kernel == 2u) weight = kMhcColorAtGreenVertical[tap];
            else weight = kMhcColorAtOpposite[tap];
            sum += weight * gPlane[sy * 20u + cx + dx];
        }
    }
    return sum;
}

// CFA channel at a pixel; mirrors CfaChannelAt in CameraIsp.cpp. The CFA enum
// order matches CfaPattern in CameraPipeline.hpp (Mono, RGGB, GRBG, GBRG,
// BGGR, MultiChannel).
uint CfaChannelAt(uint cfa, uint x, uint y) {
    const bool px = (x & 1u) != 0u, py = (y & 1u) != 0u;
    if (cfa == 1u) return !py ? (px ? 1u : 0u) : (px ? 2u : 1u);  // RGGB
    if (cfa == 2u) return !py ? (px ? 0u : 1u) : (px ? 1u : 2u);  // GRBG
    if (cfa == 3u) return !py ? (px ? 2u : 1u) : (px ? 1u : 0u);  // GBRG
    if (cfa == 4u) return !py ? (px ? 1u : 2u) : (px ? 0u : 1u);  // BGGR
    return 0u;
}

float FullWell() { return max(ispConfig[6].z, 1e-20); }

float WhiteBalanceAt(uint channel) {
    const float3 wb = ispConfig[0].xyz;
    return channel == 0u ? wb.x : (channel == 1u ? wb.y : wb.z);
}

// Component of the corrected signal for one device channel.
float CorrectedAt(uint2 coord, uint channel) {
    const float4 c = correctedImage[coord];
    return channel == 0u ? c.r : (channel == 1u ? c.g : c.b);
}

bool IsDefect(uint x, uint y, uint count) {
    for (uint d = 0; d < count; ++d)
        if (defectPixels[d].x == x && defectPixels[d].y == y) return true;
    return false;
}

// Well-normalized, white-balanced sample of the CFA plane at global coords.
float SamplePlane(uint x, uint y) {
    const float c = CorrectedAt(uint2(x, y), 0u);
    const uint channel = CfaChannelAt(cameraPush.cfa, x, y);
    return max(c, 0.0) / FullWell() * WhiteBalanceAt(min(channel, 2u));
}

// Mean of the non-defect, in-range four-neighbors of a defect pixel, over the
// plane in shared memory. Serial per defect; defect lists are short.
void RepairSharedDefects(uint2 tileMin, uint count) {
    for (uint d = 0; d < count; ++d) {
        const int gx = (int)defectPixels[d].x;
        const int gy = (int)defectPixels[d].y;
        const int sx = gx - (int)tileMin.x + 2;
        const int sy = gy - (int)tileMin.y + 2;
        if (sx < 0 || sy < 0 || sx >= 20 || sy >= 20) continue;
        float sum = 0.0f;
        uint neighbors = 0;
        const int2 offsets[4] = {int2(1, 0), int2(-1, 0),
                                 int2(0, 1), int2(0, -1)};
        for (uint o = 0; o < 4u; ++o) {
            const int nx = gx + offsets[o].x;
            const int ny = gy + offsets[o].y;
            if (nx < 0 || ny < 0 || nx >= (int)cameraPush.width ||
                ny >= (int)cameraPush.height)
                continue;
            if (IsDefect(uint(nx), uint(ny), count)) continue;
            sum += gPlane[(sy + offsets[o].y) * 20 + (sx + offsets[o].x)];
            ++neighbors;
        }
        gPlane[sy * 20 + sx] = neighbors > 0u ? sum / float(neighbors) : 0.0f;
    }
}

// Direct (non-shared) value for the Mono/MultiChannel paths, where no
// convolution follows: white balance, then defect repair by the mean of the
// non-defect four-neighbors of the corrected signal.
float PreprocessedValue(uint2 coord, uint channel, uint count) {
    const float wb = WhiteBalanceAt(min(channel, 2u));
    if (count == 0u || !IsDefect(coord.x, coord.y, count)) {
        const float c = CorrectedAt(coord, min(channel, 2u));
        return max(c, 0.0) / FullWell() * wb;
    }
    float sum = 0.0f;
    uint neighbors = 0;
    const int2 offsets[4] = {int2(1, 0), int2(-1, 0),
                             int2(0, 1), int2(0, -1)};
    for (uint o = 0; o < 4u; ++o) {
        const int nx = (int)coord.x + offsets[o].x;
        const int ny = (int)coord.y + offsets[o].y;
        if (nx < 0 || ny < 0 || nx >= (int)cameraPush.width ||
            ny >= (int)cameraPush.height)
            continue;
        if (IsDefect(uint(nx), uint(ny), count)) continue;
        const float c = CorrectedAt(uint2(nx, ny), min(channel, 2u));
        sum += max(c, 0.0) / FullWell() * wb;
        ++neighbors;
    }
    return neighbors > 0u ? sum / float(neighbors) : 0.0f;
}

[numthreads(16, 16, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID,
          uint localIndex : SV_GroupIndex) {
    const uint2 coord = groupId.xy * 16u + localId.xy;
    const uint2 tileMin = groupId.xy * 16u;

    if (cameraPush.detector == CAMERA_THERMAL) {
        if (coord.x < cameraPush.width && coord.y < cameraPush.height)
            linearRgbImage[coord] = correctedImage[coord];
        return;
    }

    if (cameraPush.cfa == CAMERA_MONO) {
        if (coord.x >= cameraPush.width || coord.y >= cameraPush.height)
            return;
        const uint count = uint(ispConfig[3].y);
        const float v = PreprocessedValue(coord, 0u, count);
        linearRgbImage[coord] = float4(v, v, v, 1.0);
        return;
    }

    if (cameraPush.cfa == CAMERA_MULTI_CHANNEL) {
        if (coord.x >= cameraPush.width || coord.y >= cameraPush.height)
            return;
        const uint count = uint(ispConfig[3].y);
        const uint used = clamp(cameraPush.channelCount, 1u, 3u);
        // Unused display slots replicate the last used plane, as on the CPU.
        const float v0 = PreprocessedValue(coord, 0u, count);
        const float v1 = used > 1u ? PreprocessedValue(coord, 1u, count) : v0;
        const float v2 = used > 2u ? PreprocessedValue(coord, 2u, count) : v1;
        linearRgbImage[coord] = float4(v0, v1, v2, 1.0);
        return;
    }

    // Bayer: load the 20x20 white-balanced tile (replicated borders) into
    // shared memory, repair defect pixels there, then convolve.
    for (uint i = localIndex; i < 400u; i += 256u) {
        const uint sx = i % 20u;
        const uint sy = i / 20u;
        const uint gx = (uint)clamp((int)tileMin.x + (int)sx - 2, 0,
                                    (int)cameraPush.width - 1);
        const uint gy = (uint)clamp((int)tileMin.y + (int)sy - 2, 0,
                                    (int)cameraPush.height - 1);
        gPlane[i] = SamplePlane(gx, gy);
    }
    GroupMemoryBarrierWithGroupSync();
    if (uint(ispConfig[0].w) & ISP_FLAG_DEFECTS) {
        const uint defectCount = uint(ispConfig[3].y);
        if (localIndex == 0) RepairSharedDefects(tileMin, defectCount);
        GroupMemoryBarrierWithGroupSync();
    }

    if (coord.x >= cameraPush.width || coord.y >= cameraPush.height) return;

    const uint lx = localId.x + 2u;
    const uint ly = localId.y + 2u;
    const uint here = CfaChannelAt(cameraPush.cfa, coord.x, coord.y);
    const float own = gPlane[ly * 20u + lx];

    float r, g, b;
    if (here == 1u) {
        // Green site: both colors use the orientation of the neighbors.
        g = own;
        const uint side = coord.x + 1u < cameraPush.width
                              ? CfaChannelAt(cameraPush.cfa, coord.x + 1u,
                                             coord.y)
                              : 3u;
        r = Convolve5(lx - 2u, ly - 2u, side == 0u ? 1u : 2u);
        b = Convolve5(lx - 2u, ly - 2u, side == 2u ? 1u : 2u);
    } else {
        g = Convolve5(lx - 2u, ly - 2u, 0u);
        if (here == 0u) {
            r = own;
            b = Convolve5(lx - 2u, ly - 2u, 3u);
        } else {
            b = own;
            r = Convolve5(lx - 2u, ly - 2u, 3u);
        }
    }
    linearRgbImage[coord] = float4(r, g, b, 1.0);
}
