/**
 * @file display_palettes.hlsli
 * @brief Scalar-to-colour display ramps shared by CLAHE and the camera ISP
 *
 * Evenly spaced control points, linearly interpolated. A ramp rather than a
 * polynomial fit because the control points are the specification -- someone
 * checking a palette against a reference reads numbers, not coefficients.
 *
 * The CPU twins live in src/libQuantiloom/postprocess/CameraIsp.hpp (namespace
 * isp::kIronbowPoints and friends) and are compared point for point by the
 * test suite; keep the tables identical.
 */

#ifndef DISPLAY_PALETTES_HLSLI
#define DISPLAY_PALETTES_HLSLI

// Palettes. Grey is the identity and the only one that leaves a colour image
// alone; everything else replaces the colour with the scalar's own.
#define PALETTE_GREY          0
#define PALETTE_GREY_INVERTED 1
#define PALETTE_IRONBOW       2
#define PALETTE_RAINBOW       3
#define PALETTE_VIRIDIS       4

float3 IronbowPalette(float t) {
    // The classic thermal ramp: black through violet and red into white.
    static const float3 c[8] = {
        float3(0.000f, 0.000f, 0.000f),
        float3(0.110f, 0.020f, 0.260f),
        float3(0.300f, 0.030f, 0.430f),
        float3(0.510f, 0.060f, 0.430f),
        float3(0.730f, 0.170f, 0.310f),
        float3(0.900f, 0.350f, 0.130f),
        float3(0.990f, 0.640f, 0.010f),
        float3(1.000f, 1.000f, 0.850f)
    };
    float x = saturate(t) * 7.0f;
    uint i = min((uint)x, 6u);
    return lerp(c[i], c[i + 1], x - (float)i);
}

float3 RainbowPalette(float t) {
    // Blue through cyan and green to red. High apparent contrast and no
    // perceptual order to speak of, which is exactly why it is not the
    // default -- but it is what "false colour" means to most people.
    static const float3 c[6] = {
        float3(0.0f, 0.0f, 0.5f),
        float3(0.0f, 0.0f, 1.0f),
        float3(0.0f, 1.0f, 1.0f),
        float3(1.0f, 1.0f, 0.0f),
        float3(1.0f, 0.0f, 0.0f),
        float3(0.5f, 0.0f, 0.0f)
    };
    float x = saturate(t) * 5.0f;
    uint i = min((uint)x, 4u);
    return lerp(c[i], c[i + 1], x - (float)i);
}

float3 ViridisPalette(float t) {
    // Perceptually uniform and monotone in lightness, so a difference in
    // colour is a difference in value rather than an artefact of the ramp.
    // The one to reach for when the picture is going into a paper.
    static const float3 c[11] = {
        float3(0.267f, 0.005f, 0.329f),
        float3(0.283f, 0.141f, 0.458f),
        float3(0.254f, 0.265f, 0.530f),
        float3(0.207f, 0.372f, 0.553f),
        float3(0.164f, 0.471f, 0.558f),
        float3(0.128f, 0.567f, 0.551f),
        float3(0.135f, 0.659f, 0.518f),
        float3(0.267f, 0.749f, 0.441f),
        float3(0.478f, 0.821f, 0.318f),
        float3(0.741f, 0.873f, 0.150f),
        float3(0.993f, 0.906f, 0.144f)
    };
    float x = saturate(t) * 10.0f;
    uint i = min((uint)x, 9u);
    return lerp(c[i], c[i + 1], x - (float)i);
}

float3 ApplyPalette(float t, uint palette) {
    if (palette == PALETTE_IRONBOW) return IronbowPalette(t);
    if (palette == PALETTE_RAINBOW) return RainbowPalette(t);
    if (palette == PALETTE_VIRIDIS) return ViridisPalette(t);
    if (palette == PALETTE_GREY_INVERTED) return (1.0f - saturate(t)).xxx;
    return saturate(t).xxx;
}

#endif // DISPLAY_PALETTES_HLSLI
