/**
 * @file camera_hsv.comp.hlsl
 * @brief Display HSV grading and empirical display effects
 *
 * The last display-chain pass: encoded-sRGB in (binding 3), encoded-sRGB out
 * in place through binding 4 (aliased to the same image; every thread touches
 * only its own texel), mirroring ApplyHsv in postprocess/CameraIsp.cpp:
 *
 *   RGB -> HSV -> cyclic hue offset, faded out below S=0.05 by a
 *   smoothstep(0.05, 0.2, S) weight (grey has no hue, so an IR-grey frame
 *   stays exactly grey) -> S scaled and clamped -> V raised to 1/valueGamma
 *   -> HSV -> RGB.
 *
 * The two empirical effects ride on independent counter streams keyed on the
 * acquisition index, so one acquisition always reproduces the same pattern
 * (a same-tick reprocess is bit-identical) and two acquisitions never share
 * one: noise folds acquisition*2, drift folds acquisition>>4 (slowly varying,
 * constant across 16 acquisitions, plus the per-acquisition noise above it).
 *
 * Parameter contract with the host (GpuCameraPipeline):
 *   ispConfig[0].w bit 5  empirical noise on
 *   ispConfig[0].w bit 6  temporal drift on
 *   ispConfig[7] = {hueOffsetDeg, saturationScale, valueGamma, noiseSigma}
 *   ispConfig[8].x       = driftSigma
 * The host only dispatches this pass when bit 7 of ispConfig[0].w (hsv
 * enabled) is set, i.e. at least one parameter is off its default, so an
 * untouched chain never reaches here and stays bit-identical. The sigmas are
 * the CPU HsvConfig values (CameraPipeline.hpp defaults 0.02/0.02), uploaded
 * per effective config by the host.
 *
 * The output aliases the input: the display pass has finished before this
 * dispatch and every thread reads and writes only its own texel, so the
 * in-place update is exact. RAW, corrected and apparent-temperature products
 * never pass through here: this stage is display-only by construction.
 */

#include "camera_common.hlsli"

[[vk::binding(3, 0)]] RWTexture2D<float4> displayImage;
[[vk::binding(4, 0)]] RWTexture2D<float4> hsvImage;
[[vk::binding(12, 0)]] StructuredBuffer<float4> ispConfig;

static const uint ISP_FLAG_EMPIRICAL_NOISE = 32u;  // 1 << 5
static const uint ISP_FLAG_TEMPORAL_DRIFT = 64u;   // 1 << 6

float3 RgbToHsv(float3 rgb) {
    const float3 c = saturate(rgb);
    const float maxc = max(c.r, max(c.g, c.b));
    const float minc = min(c.r, min(c.g, c.b));
    const float chroma = maxc - minc;
    float hue = 0.0f;
    if (chroma > 0.0f) {
        if (maxc == c.r) {
            hue = (c.g - c.b) / chroma / 6.0f;
        } else if (maxc == c.g) {
            hue = (c.b - c.r) / chroma / 6.0f + 2.0f / 6.0f;
        } else {
            hue = (c.r - c.g) / chroma / 6.0f + 4.0f / 6.0f;
        }
        hue -= floor(hue);
    }
    const float saturation = maxc > 0.0f ? chroma / maxc : 0.0f;
    return float3(hue, saturation, maxc);
}

float3 HsvToRgb(float3 hsv) {
    const float h = hsv.x - floor(hsv.x);
    const float s = saturate(hsv.y);
    const float v = saturate(hsv.z);
    const float sector = h * 6.0f;
    const uint i = uint(sector) % 6u;
    const float f = sector - floor(sector);
    const float p = v * (1.0f - s);
    const float q = v * (1.0f - s * f);
    const float t = v * (1.0f - s * (1.0f - f));
    switch (i) {
    case 0u: return float3(v, t, p);
    case 1u: return float3(q, v, p);
    case 2u: return float3(p, v, t);
    case 3u: return float3(p, q, v);
    case 4u: return float3(t, p, v);
    default: return float3(v, p, q);
    }
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint2 coord = id.xy;
    if (coord.x >= cameraPush.width || coord.y >= cameraPush.height) return;
    const uint pixel = coord.y * cameraPush.width + coord.x;

    const float4 row7 = ispConfig[7];
    const float hueOffsetDeg = row7.x;
    const float saturationScale = row7.y;
    const float valueGamma = max(row7.z, 1e-6f);
    const uint flags = uint(ispConfig[0].w);

    float3 hsv = RgbToHsv(displayImage[coord].rgb);
    if (hueOffsetDeg != 0.0f) {
        // Grey has no hue: fade the offset out on low saturation.
        const float weight = smoothstep(0.05f, 0.2f, hsv.y);
        if (weight > 0.0f) {
            const float shifted = hsv.x + hueOffsetDeg / 360.0f;
            hsv.x = hsv.x + weight * (shifted - floor(shifted) - hsv.x);
        }
    }
    hsv.y = saturate(hsv.y * saturationScale);
    hsv.z = pow(saturate(hsv.z), 1.0f / valueGamma);

    // Empirical effects, keyed on the acquisition index: noise on *2, drift
    // on >>4 (the per-pixel index is the stream's second key, as everywhere
    // in camera_common.hlsli).
    if (flags & ISP_FLAG_EMPIRICAL_NOISE) {
        const uint lo = cameraPush.acquisitionLo * 2u;
        const uint hi = cameraPush.acquisitionHi * 2u +
                        (cameraPush.acquisitionLo >> 31);
        hsv.z += CameraGaussian(pixel, NOISE_EMPIRICAL_NOISE, lo, hi) * row7.w;
    }
    if (flags & ISP_FLAG_TEMPORAL_DRIFT) {
        const uint lo = (cameraPush.acquisitionLo >> 4) |
                        (cameraPush.acquisitionHi << 28);
        const uint hi = cameraPush.acquisitionHi >> 4;
        hsv.z += CameraGaussian(pixel, NOISE_EMPIRICAL_DRIFT, lo, hi) *
                 ispConfig[8].x;
    }
    hsv.z = saturate(hsv.z);

    hsvImage[coord] = float4(HsvToRgb(hsv), 1.0f);
}
