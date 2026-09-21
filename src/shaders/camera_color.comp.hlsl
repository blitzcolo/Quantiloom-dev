/**
 * @file camera_color.comp.hlsl
 * @brief CCM, optional denoise/sharpen, tone and sRGB encode (legacy)
 *
 * NOTE (M5): this pass is superseded by the fused camera_demosaic pass, which
 * runs the identical arithmetic (demosaic -> CCM -> denoise/sharpen ->
 * tone -> sRGB) in one dispatch together with the display product. The file
 * stays in the build because the shader list is fixed, but GpuCameraPipeline
 * never dispatches it. The description below documents the original pass.
 *
 * The linear RGB produced by camera_demosaic passes through, in order:
 *   1. the 3x3 color correction matrix,
 *   2. an optional 3x3 Gaussian (sigma 1) denoise, blended by denoiseStrength,
 *   3. an optional unsharp mask with sharpenStrength,
 *   4. tone (pow 1/gamma), a gamut operator (hard clamp or x/(1+x)) and the
 *      piecewise sRGB encode.
 *
 * Stages 1-3 run in an 18x18 groupshared tile (16x16 threads plus a
 * one-pixel halo, replicated borders) so the two 3x3 neighborhoods are read
 * from shared memory; every formula mirrors RunVisibleIsp in
 * postprocess/CameraIsp.cpp, which computes in f64 -- the differences are
 * float rounding only.
 */

#include "camera_common.hlsli"

[[vk::binding(1, 0)]] RWTexture2D<float4> linearRgbImage;
[[vk::binding(2, 0)]] RWTexture2D<float4> colorRgbImage;
[[vk::binding(12, 0)]] StructuredBuffer<float4> ispConfig;

static const uint ISP_FLAG_DENOISE = 1u;
static const uint ISP_FLAG_SHARPEN = 2u;
static const uint ISP_FLAG_CLIP = 4u;

groupshared float3 gA[18u * 18u]; // post-CCM
groupshared float3 gB[18u * 18u]; // post-denoise

float3 LoadLinear(uint2 coord) {
    return linearRgbImage[coord].rgb;
}

// Row-major 3x3 CCM times a linear RGB triple.
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
float3 Gaussian3(uint cx, uint cy) {
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

[numthreads(16, 16, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID,
          uint localIndex : SV_GroupIndex) {
    const uint2 tileMin = groupId.xy * 16u;
    const uint2 coord = tileMin + localId.xy;

    // Halo load with replicated borders, then the CCM.
    for (uint i = localIndex; i < 324u; i += 256u) {
        const uint sx = i % 18u;
        const uint sy = i / 18u;
        const uint gx = (uint)clamp((int)tileMin.x + (int)sx - 1, 0,
                                    (int)cameraPush.width - 1);
        const uint gy = (uint)clamp((int)tileMin.y + (int)sy - 1, 0,
                                    (int)cameraPush.height - 1);
        gA[i] = ApplyCcm(LoadLinear(uint2(gx, gy)));
    }
    GroupMemoryBarrierWithGroupSync();

    const uint flags = uint(ispConfig[0].w);
    const float blend = saturate(ispConfig[4].y);
    const float amount = max(ispConfig[4].z, 0.0f);

    // Optional denoise into gB.
    if ((flags & ISP_FLAG_DENOISE) && blend > 0.0f) {
        for (uint i = localIndex; i < 324u; i += 256u) {
            const uint sx = i % 18u;
            const uint sy = i / 18u;
            gB[i] = gA[i] + blend * (Gaussian3(sx, sy) - gA[i]);
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

    if (coord.x >= cameraPush.width || coord.y >= cameraPush.height) return;

    // Tone needs a nonnegative base; the sharpen stage already clamps there.
    const float inverseGamma = 1.0 / max(ispConfig[4].x, 1e-6);
    float3 toned = pow(max(value, 0.0f), inverseGamma.xxx);
    if (flags & ISP_FLAG_CLIP) toned = saturate(toned);
    else toned = toned / (1.0f + toned);
    colorRgbImage[coord] = float4(CameraSrgbEncode(toned.r),
                                  CameraSrgbEncode(toned.g),
                                  CameraSrgbEncode(toned.b), 1.0);
}
