/**
 * @file camera_demosaic.comp.hlsl
 * @brief Fused preprocess + MHC demosaic + color + display product
 *
 * M5 fused the three per-pixel display passes (camera_demosaic, camera_color,
 * camera_display) into this single dispatch. The arithmetic is unchanged from
 * the three original passes -- every formula and operation order is kept, so
 * all products (linearRgb, colorRgb, display, agcSource) are bit-identical to
 * the former chain:
 *
 *   1. Preprocess + demosaic. Bayer: white balance per CFA site, defect-pixel
 *      repair (mean of the non-defect four-neighbors, exactly PreprocessPlane
 *      in postprocess/CameraIsp.cpp), then Malvar-He-Cutler 5x5. The color
 *      pass consumed the demosaic product through a 18x18 neighborhood (16x16
 *      threads plus a one-pixel halo), so demosaic runs here on that same
 *      18x18 region, which needs a 22x22 raw tile (a two-pixel convolution
 *      halo around the color pass's one-pixel halo) with replicated borders,
 *      mirroring Convolve5's clamped indexing on the CPU. Mono and
 *      MultiChannel have no cross-pixel demosaic: the preprocessed value is
 *      computed directly at the 18x18 clamped coordinates. Thermal is a
 *      passthrough the color stage reads from the corrected product.
 *   2. Color. The 3x3 color correction matrix, the optional 3x3 Gaussian
 *      (sigma 1) denoise blended by denoiseStrength, the optional unsharp
 *      mask with sharpenStrength, then tone (pow 1/gamma), the gamut
 *      operator and the piecewise sRGB encode -- RunVisibleIsp in
 *      postprocess/CameraIsp.cpp, f32 against its f64.
 *   3. Display product. Photon: the encoded sRGB triple passes through to
 *      the display image and its luminance feeds agcSource (the host CLAHE
 *      fallback). Thermal: the corrected scalar gets the AGC tone operator
 *      and display palette (Linear window from the stats reduce, Equalize
 *      from the merged histogram's CDF, persistent CLAHE passing the scalar
 *      through for the host pipeline), then sRGB encoding -- AgcTone on the
 *      CPU. agcSource always carries the pre-AGC scalar.
 *
 * The 22x22 raw plane costs 9.7 KB of shared memory together with the two
 * 18x18 float3 tiles; one dispatch replaces three full-resolution image
 * round-trips (demosaic->linearRgb->color->colorRgb->display).
 */

#include "camera_common.hlsli"
#include "camera_isp.hlsli"
#include "display_palettes.hlsli"

[[vk::binding(0, 0)]] RWTexture2D<float4> correctedImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> linearRgbImage;
[[vk::binding(2, 0)]] RWTexture2D<float4> colorRgbImage;
[[vk::binding(3, 0)]] RWTexture2D<float4> displayImage;
[[vk::binding(4, 0)]] RWTexture2D<float4> agcSourceImage;
[[vk::binding(12, 0)]] StructuredBuffer<float4> ispConfig;
[[vk::binding(13, 0)]] StructuredBuffer<uint2> defectPixels;
[[vk::binding(14, 0)]] RWStructuredBuffer<uint> ispStats;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> cdfBuffer;

static const uint ISP_FLAG_DENOISE = 1u;
static const uint ISP_FLAG_SHARPEN = 2u;
static const uint ISP_FLAG_CLIP = 4u;
static const uint ISP_FLAG_CLAHE_PERSISTENT = 8u;
static const uint ISP_FLAG_DEFECTS = 16u;

#define ISP_HISTOGRAM_BINS 256
#define ISP_STAT_MIN 256
#define ISP_STAT_MAX 257
#define ISP_STAT_WIN_LO 261
#define ISP_STAT_WIN_HI 262

// DisplayToneMode order matches DisplayControl.hpp.
static const uint TONE_LINEAR = 0u;
static const uint TONE_EQUALIZE = 1u;

// 22x22 white-balanced CFA tile (two-pixel halo around the 18x18 region the
// color stages consume) plus the post-CCM and post-denoise 18x18 tiles.
groupshared float gPlane[22u * 22u];
groupshared float3 gA[18u * 18u]; // post-CCM
groupshared float3 gB[18u * 18u]; // post-denoise

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
            sum += weight * gPlane[sy * 22u + cx + dx];
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
        const int sx = gx - (int)tileMin.x + 3;
        const int sy = gy - (int)tileMin.y + 3;
        if (sx < 0 || sy < 0 || sx >= 22 || sy >= 22) continue;
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
            sum += gPlane[(sy + offsets[o].y) * 22 + (sx + offsets[o].x)];
            ++neighbors;
        }
        gPlane[sy * 22 + sx] = neighbors > 0u ? sum / float(neighbors) : 0.0f;
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

// MHC demosaic at a global coordinate whose shared-tile index is (lx, ly);
// the arithmetic is exactly the former camera_demosaic pass.
float3 DemosaicBayer(uint2 coord, uint lx, uint ly) {
    const uint here = CfaChannelAt(cameraPush.cfa, coord.x, coord.y);
    const float own = gPlane[ly * 22u + lx];
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
    return float3(r, g, b);
}

// Row-major 3x3 CCM times a linear RGB triple (former camera_color pass).
float3 ApplyCcm(float3 v) {
    const float4 r0 = ispConfig[1];
    const float4 r1 = ispConfig[2];
    const float4 r2 = ispConfig[3];
    return float3(dot(float3(r0.x, r0.y, r0.z), v),
                  dot(float3(r0.w, r1.x, r1.y), v),
                  dot(float3(r1.z, r1.w, r2.x), v));
}

// 3x3 Gaussian (sigma 1), normalized, over a shared tile with clamped
// (replicated-border) indices.
float3 Gaussian3A(uint cx, uint cy) {
    float3 sum = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        const uint sy = (uint)clamp((int)cy + dy, 0, 17);
        for (int dx = -1; dx <= 1; ++dx) {
            const uint sx = (uint)clamp((int)cx + dx, 0, 17);
            const float w = (dx == 0 && dy == 0)   ? 4.0f
                            : (dx == 0 || dy == 0) ? 2.0f
                                                   : 1.0f;
            sum += w * gA[sy * 18u + sx];
        }
    }
    return sum / 16.0f;
}

float3 Gaussian3B(uint cx, uint cy) {
    float3 sum = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        const uint sy = (uint)clamp((int)cy + dy, 0, 17);
        for (int dx = -1; dx <= 1; ++dx) {
            const uint sx = (uint)clamp((int)cx + dx, 0, 17);
            const float w = (dx == 0 && dy == 0)   ? 4.0f
                            : (dx == 0 || dy == 0) ? 2.0f
                                                   : 1.0f;
            sum += w * gB[sy * 18u + sx];
        }
    }
    return sum / 16.0f;
}

// Clamped global coordinate of the 18x18 region entry i (one-pixel halo
// around the 16x16 core), matching the former color pass's replicated-border
// loads of the demosaic product.
uint2 RegionCoord(uint2 tileMin, uint i) {
    const uint sx = i % 18u;
    const uint sy = i / 18u;
    return uint2((uint)clamp((int)tileMin.x + (int)sx - 1, 0,
                             (int)cameraPush.width - 1),
                 (uint)clamp((int)tileMin.y + (int)sy - 1, 0,
                             (int)cameraPush.height - 1));
}

[numthreads(16, 16, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID,
          uint localIndex : SV_GroupIndex) {
    const uint2 tileMin = groupId.xy * 16u;
    const uint2 coord = tileMin + localId.xy;
    const bool inRange =
        coord.x < cameraPush.width && coord.y < cameraPush.height;
    const uint flags = uint(ispConfig[0].w);

    // -- Stage 1+2a: demosaic (or per-pixel preprocess) and CCM into gA. ----
    if (cameraPush.detector == CAMERA_THERMAL) {
        // Passthrough: the color stage reads the corrected scalar; the
        // linearRgb product is written in the output stage below.
        for (uint i = localIndex; i < 324u; i += 256u)
            gA[i] = ApplyCcm(correctedImage[RegionCoord(tileMin, i)].rgb);
    } else if (cameraPush.cfa == CAMERA_MONO) {
        const uint count = uint(ispConfig[3].y);
        for (uint i = localIndex; i < 324u; i += 256u) {
            const uint2 c = RegionCoord(tileMin, i);
            const float v = PreprocessedValue(c, 0u, count);
            gA[i] = ApplyCcm(float3(v, v, v));
            // linearRgb product write for this thread's own core pixel.
            if (inRange && c.x == coord.x && c.y == coord.y)
                linearRgbImage[coord] = float4(v, v, v, 1.0);
        }
    } else if (cameraPush.cfa == CAMERA_MULTI_CHANNEL) {
        const uint count = uint(ispConfig[3].y);
        const uint used = clamp(cameraPush.channelCount, 1u, 3u);
        for (uint i = localIndex; i < 324u; i += 256u) {
            const uint2 c = RegionCoord(tileMin, i);
            // Unused display slots replicate the last used plane, as on the
            // CPU.
            const float v0 = PreprocessedValue(c, 0u, count);
            const float v1 = used > 1u ? PreprocessedValue(c, 1u, count) : v0;
            const float v2 = used > 2u ? PreprocessedValue(c, 2u, count) : v1;
            gA[i] = ApplyCcm(float3(v0, v1, v2));
            if (inRange && c.x == coord.x && c.y == coord.y)
                linearRgbImage[coord] = float4(v0, v1, v2, 1.0);
        }
    } else {
        // Bayer: load the 22x22 white-balanced tile (replicated borders)
        // into shared memory, repair defect pixels there, then demosaic the
        // 18x18 region.
        for (uint i = localIndex; i < 484u; i += 256u) {
            const uint sx = i % 22u;
            const uint sy = i / 22u;
            const uint gx = (uint)clamp((int)tileMin.x + (int)sx - 3, 0,
                                        (int)cameraPush.width - 1);
            const uint gy = (uint)clamp((int)tileMin.y + (int)sy - 3, 0,
                                        (int)cameraPush.height - 1);
            gPlane[i] = SamplePlane(gx, gy);
        }
        GroupMemoryBarrierWithGroupSync();
        if (flags & ISP_FLAG_DEFECTS) {
            const uint defectCount = uint(ispConfig[3].y);
            if (localIndex == 0) RepairSharedDefects(tileMin, defectCount);
            GroupMemoryBarrierWithGroupSync();
        }
        for (uint i = localIndex; i < 324u; i += 256u) {
            const uint sx = i % 18u;
            const uint sy = i / 18u;
            // Out-of-range region entries replicate the demosaic value at
            // the clamped coordinate, as the color pass's clamped loads did.
            // The 22x22 tile starts three pixels above/left of the tile
            // origin, so a global coordinate maps to shared index
            // c - tileMin + 3.
            const uint2 c = RegionCoord(tileMin, i);
            const uint lx = c.x - tileMin.x + 3u;
            const uint ly = c.y - tileMin.y + 3u;
            const float3 rgb = DemosaicBayer(c, lx, ly);
            gA[i] = ApplyCcm(rgb);
            if (inRange && c.x == coord.x && c.y == coord.y)
                linearRgbImage[coord] = float4(rgb, 1.0);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // -- Stage 2b: optional denoise into gB (former camera_color pass). -----
    const float blend = saturate(ispConfig[4].y);
    const float amount = max(ispConfig[4].z, 0.0f);
    if ((flags & ISP_FLAG_DENOISE) && blend > 0.0f) {
        for (uint i = localIndex; i < 324u; i += 256u) {
            const uint sx = i % 18u;
            const uint sy = i / 18u;
            gB[i] = gA[i] + blend * (Gaussian3A(sx, sy) - gA[i]);
        }
    } else {
        for (uint i = localIndex; i < 324u; i += 256u) gB[i] = gA[i];
    }
    // Border semantics: the CPU clamps coordinates when READING the denoised
    // plane, so out-of-range sharpen taps see the denoised EDGE pixel, not a
    // denoise recomputed at the halo position. Replicate the tile's edge rows
    // and columns over its halo; the two passes keep corner writes ordered.
    if (localIndex < 18u) {
        gB[localIndex] = gB[18u + localIndex];
        gB[17u * 18u + localIndex] = gB[16u * 18u + localIndex];
    }
    GroupMemoryBarrierWithGroupSync();
    if (localIndex >= 18u && localIndex < 36u) {
        const uint j = localIndex - 18u;
        gB[j * 18u] = gB[j * 18u + 1u];
        gB[j * 18u + 17u] = gB[j * 18u + 16u];
    }
    GroupMemoryBarrierWithGroupSync();

    float3 value;
    if ((flags & ISP_FLAG_SHARPEN) && amount > 0.0f) {
        const uint sx = localId.x + 1u;
        const uint sy = localId.y + 1u;
        value = max(0.0f, gB[sy * 18u + sx] +
                              amount * (gB[sy * 18u + sx] -
                                        Gaussian3B(sx, sy)));
    } else {
        value = gB[(localId.y + 1u) * 18u + (localId.x + 1u)];
    }

    if (!inRange) return;

    // -- Stage 2c+3: tone, gamut, sRGB encode, display product. -------------
    // Tone needs a nonnegative base; the sharpen stage already clamps there.
    const float inverseGamma = 1.0 / max(ispConfig[4].x, 1e-6);
    float3 toned = pow(max(value, 0.0f), inverseGamma.xxx);
    if (flags & ISP_FLAG_CLIP) toned = saturate(toned);
    else toned = toned / (1.0f + toned);
    const float3 encoded = float3(CameraSrgbEncode(toned.r),
                                  CameraSrgbEncode(toned.g),
                                  CameraSrgbEncode(toned.b));
    colorRgbImage[coord] = float4(encoded, 1.0);

    if (cameraPush.detector != CAMERA_THERMAL) {
        // Photon display branch: the encoded sRGB triple passes through and
        // its luminance feeds the host's CLAHE fallback source.
        displayImage[coord] = float4(encoded, 1.0);
        const float luminance = dot(encoded, float3(0.2126f, 0.7152f, 0.0722f));
        agcSourceImage[coord] = float4(luminance, luminance, luminance, 1.0);
        return;
    }

    // Thermal display branch (former camera_display pass): the linearRgb
    // product is the corrected passthrough.
    linearRgbImage[coord] = correctedImage[coord];
    const float v = correctedImage[coord].r;
    agcSourceImage[coord] = float4(v, v, v, 1.0);

    if (flags & ISP_FLAG_CLAHE_PERSISTENT) {
        // The host's CLAHE pipeline consumes agcSource and overwrites the
        // on-screen image; the pipeline display product keeps the scalar.
        displayImage[coord] = float4(v, v, v, 1.0);
        return;
    }

    const uint tone = uint(ispConfig[5].z);
    float t;
    if (tone == TONE_LINEAR) {
        // The window came from the previous frame's histogram (stats phase
        // 1); a degenerate one (first frame, flat previous image) falls back
        // to this frame's range, and a flat current frame is the CPU chain's
        // flat 0.5.
        float lo = asfloat(ispStats[ISP_STAT_WIN_LO]);
        float hi = asfloat(ispStats[ISP_STAT_WIN_HI]);
        if (!(hi > lo)) {
            lo = asfloat(ispStats[ISP_STAT_MIN]);
            hi = asfloat(ispStats[ISP_STAT_MAX]);
        }
        t = (hi > lo) ? saturate((v - lo) / (hi - lo)) : 0.5f;
    } else {
        // Equalize, and the offline Clahe degradation (a global
        // equalization), from this frame's merged histogram CDF.
        const float lo = asfloat(ispStats[ISP_STAT_MIN]);
        const float hi = asfloat(ispStats[ISP_STAT_MAX]);
        if (!(hi > lo)) {
            t = 0.5f;
        } else {
            const uint bin = min(uint(saturate((v - lo) / (hi - lo)) *
                                          float(ISP_HISTOGRAM_BINS - 1u) +
                                      0.5f),
                                 ISP_HISTOGRAM_BINS - 1u);
            t = cdfBuffer[bin];
        }
    }

    const uint palette = uint(ispConfig[5].w);
    const float3 rgb = ApplyPalette(t, palette);
    displayImage[coord] = float4(CameraSrgbEncode(rgb.r),
                                 CameraSrgbEncode(rgb.g),
                                 CameraSrgbEncode(rgb.b), 1.0);
}
