// ============================================================================
// Quantiloom - Camera Dynamic Exposure Compositor (M4-1)
// ============================================================================
// Composites the time-stratified measurement layers (binding 28 of the ray
// tracing pipeline, written by raygen.rgen) into the single device-rate image
// the detector chain consumes. This is a GPU preview approximation of what
// the CPU reference integrates continuously:
//
//   - Each pixel's row integrates the scene over [t_row - E/2, t_row + E/2]
//     with t_row = t0 (+ y * rowDelay when the shutter rolls).
//   - The anchor (layer 0) depth reconstructs a world point, which is
//     reprojected into the two strata surrounding t_row. Each candidate must
//     pass a depth-consistency check against that stratum's own primary depth
//     (binding 31); a failed candidate means the stratum sees different
//     geometry there (disocclusion), and the pixel falls back toward the
//     anchor value. Fully failed pixels are counted for the report.
//   - Thermal state does not participate: nothing here re-runs the detector.
//
// With timeStratumCount <= 1 the pass is a pure copy of layer 0.
// ============================================================================

#include "camera_common.hlsli"

[[vk::binding(0, 0)]] RWTexture2DArray<float4> strataRate;
[[vk::binding(1, 0)]] RWTexture2DArray<float> strataDepth;
[[vk::binding(2, 0)]] RWTexture2D<float4> compositedRate;
// 4 x u32 atomics: slot 0 pixels where every reprojection candidate failed
// (disocclusion). Zeroed by the host before each measurement.
[[vk::binding(11, 0)]] RWByteAddressBuffer dynamicCounters;

// Per-stratum camera frames, uploaded by the host each measurement. Five
// float4 rows per stratum: origin.xyz, forward.xyz, right.xyz, up.xyz, then
// (fovScale, aspect, timeSeconds, unused). The frame reproduces raygen.rgen's
// ray construction exactly, so reprojection inverts tracing to the bit.
[[vk::binding(10, 0)]] StructuredBuffer<float4> layerCameras;

static const uint kLayerCameraStride = 5u;

float3 LayerOrigin(uint layer) {
    return layerCameras[layer * kLayerCameraStride + 0u].xyz;
}
float3 LayerForward(uint layer) {
    return layerCameras[layer * kLayerCameraStride + 1u].xyz;
}
float3 LayerRight(uint layer) {
    return layerCameras[layer * kLayerCameraStride + 2u].xyz;
}
float3 LayerUp(uint layer) {
    return layerCameras[layer * kLayerCameraStride + 3u].xyz;
}
float LayerFovScale(uint layer) {
    return layerCameras[layer * kLayerCameraStride + 4u].x;
}
float LayerAspect(uint layer) {
    return layerCameras[layer * kLayerCameraStride + 4u].y;
}
float LayerTime(uint layer) {
    return layerCameras[layer * kLayerCameraStride + 4u].z;
}

// Ray direction for a pixel centre under the given camera frame; identical
// math to raygen.rgen (without the subpixel jitter).
float3 PixelRayDirection(uint2 pixel, uint layer) {
    const float2 uv = (float2(pixel) + 0.5) /
                      float2(cameraPush.width, cameraPush.height);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    return normalize(LayerForward(layer) +
                     ndc.x * LayerRight(layer) * LayerFovScale(layer) *
                         LayerAspect(layer) +
                     ndc.y * LayerUp(layer) * LayerFovScale(layer));
}

// Projects a world point into the stratum's film plane. Returns the texture
// uv and, by reference, the expected Euclidean distance from the stratum
// camera (what the layer depth should read if the same surface is there).
float2 ProjectWorld(float3 world, uint layer, out float expectedDistance) {
    const float3 rel = world - LayerOrigin(layer);
    const float along = dot(rel, LayerForward(layer));
    expectedDistance = length(rel);
    const float x = dot(rel, LayerRight(layer));
    const float y = dot(rel, LayerUp(layer));
    const float2 ndc = float2(x / (along * LayerFovScale(layer) * LayerAspect(layer)),
                              y / (along * LayerFovScale(layer)));
    float2 uv = ndc * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

// Rates and validity for one candidate stratum: bilinear over the four
// texels, valid only where every texel's stored primary depth agrees with
// the reprojected world point within max(1e-3, 1% of its own depth).
bool SampleStratum(uint layer, float2 uv, float expectedDistance,
                   out float4 rate) {
    const float w = float(cameraPush.width);
    const float h = float(cameraPush.height);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ||
        expectedDistance <= 0.0)
        return false;
    const float2 texel = uv * float2(w, h) - 0.5;
    const int2 base = int2(floor(texel));
    const float2 frac = texel - float2(base);
    rate = float4(0.0, 0.0, 0.0, 0.0);
    float weightSum = 0.0;
    [loop]
    for (int dy = 0; dy <= 1; ++dy) {
        [loop]
        for (int dx = 0; dx <= 1; ++dx) {
            const int2 tap = base + int2(dx, dy);
            if (tap.x < 0 || tap.y < 0 || tap.x >= int(w) || tap.y >= int(h))
                continue;
            const float weight =
                (dx == 0 ? 1.0 - frac.x : frac.x) *
                (dy == 0 ? 1.0 - frac.y : frac.y);
            if (weight <= 0.0) continue;
            const float depth =
                strataDepth[uint3(uint2(tap), layer)];
            const float tolerance = max(1e-3, 0.01 * depth);
            if (depth < 0.0 || abs(depth - expectedDistance) > tolerance)
                return false;
            rate += weight * strataRate[uint3(uint2(tap), layer)];
            weightSum += weight;
        }
    }
    if (weightSum <= 0.0) return false;
    rate /= weightSum;
    return true;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID) {
    const uint2 pixel = dispatchID.xy;
    if (pixel.x >= cameraPush.width || pixel.y >= cameraPush.height) return;

    const uint strata = max(cameraPush.timeStratumCount, 1u);
    const float4 anchor = strataRate[uint3(pixel, 0u)];
    if (strata <= 1u) {
        compositedRate[pixel] = anchor;
        return;
    }

    const float rowTime = cameraPush.firstRowMidSeconds +
        (cameraPush.rollingShutter != 0u
             ? float(pixel.y) * cameraPush.rowDelaySeconds
             : 0.0f);

    // Reconstruct the world point the anchor depth implies. A miss carries a
    // far point along the pixel ray: exact for a rotating camera looking at
    // the sky, and sky parallax under translation is negligible at any
    // realistic exposure. Documented approximation.
    const float anchorDistance = strataDepth[uint3(pixel, 0u)];
    const float3 world = LayerOrigin(0u) +
        PixelRayDirection(pixel, 0u) * (anchorDistance >= 0.0
                                            ? anchorDistance
                                            : 1.0e6f);

    // The two strata whose times bracket the row's integration window,
    // clamped at the ends. Layer k integrates the window centred at
    // LayerTime(k); centres are E/T apart.
    const float windowStart = cameraPush.firstRowMidSeconds -
        0.5 * cameraPush.exposureSeconds;
    const float stratumSpacing = cameraPush.exposureSeconds / float(strata);
    float position = (rowTime - windowStart) / stratumSpacing - 0.5;
    int lower = int(floor(position));
    lower = clamp(lower, 0, int(strata) - 2);
    const float upperWeight = saturate(position - float(lower));

    float4 accumulated = float4(0.0, 0.0, 0.0, 0.0);
    float weightSum = 0.0;

    float expected = 0.0;
    float2 uv = ProjectWorld(world, uint(lower), expected);
    float4 lowerRate = float4(0.0, 0.0, 0.0, 0.0);
    if (SampleStratum(uint(lower), uv, expected, lowerRate)) {
        accumulated += (1.0 - upperWeight) * lowerRate;
        weightSum += 1.0 - upperWeight;
    }
    uv = ProjectWorld(world, uint(lower + 1), expected);
    float4 upperRate = float4(0.0, 0.0, 0.0, 0.0);
    if (SampleStratum(uint(lower + 1), uv, expected, upperRate)) {
        accumulated += upperWeight * upperRate;
        weightSum += upperWeight;
    }

    if (weightSum > 0.0) {
        // Renormalize when only one candidate passed: it is the honest value
        // at that time, better than a blend with an untrusted stratum.
        compositedRate[pixel] = accumulated / weightSum;
    } else {
        compositedRate[pixel] = anchor;
        dynamicCounters.InterlockedAdd(0, 1u);
    }
}
