/**
 * @file camera_display.comp.hlsl
 * @brief Final camera display product and the AGC source image (legacy)
 *
 * NOTE (M5): this pass is superseded by the fused camera_demosaic pass, which
 * produces the display and agcSource products in the same dispatch as the
 * demosaic/color chain, with the identical arithmetic. The file stays in the
 * build because the shader list is fixed, but GpuCameraPipeline never
 * dispatches it. The description below documents the original pass.
 *
 * Two branches, one per detector family:
 *
 *   Photon (visible): the color pass already produced the encoded-sRGB image;
 *     this pass passes it through to the display product and writes the
 *     luminance the host's CLAHE viewport path can fall back on.
 *   Thermal (infrared): the corrected scalar gets an AGC tone operator and a
 *     display palette, then sRGB encoding:
 *       Linear   stretch over the percentile window the stats reduce built
 *                from the PREVIOUS frame (falling back to this frame's
 *                min/max when the window is degenerate, e.g. the first
 *                acquisition), matching AgcTone in CameraIsp.cpp;
 *       Equalize 256-bin histogram equalization from this frame's statistics
 *                (camera_stats + camera_cdf ran earlier in this command
 *                buffer), matching the CPU chain for unsaturated images;
 *       Clahe    persistent mode: no tone or palette here at all -- this pass
 *                writes the scalar to both outputs and the host runs the
 *                viewport CLAHE pipeline over the agcSource image, using its
 *                output as the final display.
 *
 * agcSource always carries the pre-AGC scalar: the host's ExecuteCLAHE reads
 * it (infrared camera enabled) instead of the corrected product, and the
 * viewport override tone-maps exactly what the camera chain would have.
 */

#include "camera_common.hlsli"
#include "display_palettes.hlsli"

[[vk::binding(0, 0)]] RWTexture2D<float4> correctedImage;
[[vk::binding(2, 0)]] RWTexture2D<float4> colorRgbImage;
[[vk::binding(3, 0)]] RWTexture2D<float4> displayImage;
[[vk::binding(4, 0)]] RWTexture2D<float4> agcSourceImage;
[[vk::binding(12, 0)]] StructuredBuffer<float4> ispConfig;
[[vk::binding(14, 0)]] RWStructuredBuffer<uint> ispStats;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> cdfBuffer;

#define ISP_HISTOGRAM_BINS 256
#define ISP_STAT_MIN 256
#define ISP_STAT_MAX 257
#define ISP_STAT_WIN_LO 261
#define ISP_STAT_WIN_HI 262

// DisplayToneMode order matches DisplayControl.hpp.
static const uint TONE_LINEAR = 0u;
static const uint TONE_EQUALIZE = 1u;

static const uint ISP_FLAG_CLAHE_PERSISTENT = 8u;

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint2 coord = id.xy;
    if (coord.x >= cameraPush.width || coord.y >= cameraPush.height) return;

    if (cameraPush.detector != CAMERA_THERMAL) {
        const float3 c = colorRgbImage[coord].rgb;
        displayImage[coord] = float4(c, 1.0);
        const float luminance = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
        agcSourceImage[coord] = float4(luminance, luminance, luminance, 1.0);
        return;
    }

    const float v = correctedImage[coord].r;
    agcSourceImage[coord] = float4(v, v, v, 1.0);

    if (uint(ispConfig[0].w) & ISP_FLAG_CLAHE_PERSISTENT) {
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
    } else if (tone == TONE_EQUALIZE) {
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
    } else {
        // Clahe without the persistent wiring (offline callers): behave like
        // the CPU chain's degradation -- a global equalization.
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
