/*
 * Two-pass display percentile histogram.
 *
 * Pass 1 filters pixels whose RGB channels are all finite and atomically
 * reduces the absolute BT.709 luminance range plus the valid-pixel count.
 * Pass 2 maps the same values into exactly 65,536 bins. The host scans that
 * fixed-size result with the pre-existing f64 percentile definition.
 *
 * camera_stats.comp.hlsl is the other percentile window in the tree and is
 * deliberately not this: it bins only unsaturated pixels of the camera's
 * display-domain scalar into 256 bins and walks a nearest-rank percentile on
 * the GPU, one frame late, because the AGC window must exist before the same
 * frame's display pass. This shader serves a readback path instead: every
 * finite pixel, the true absolute range, 65,536 bins, and an exact f64 scan
 * on the host -- where a serial walk costs a buffer read instead of hundreds
 * of dependent global loads inside a single thread.
 */

[[vk::binding(0, 0)]] RWTexture2D<float4> inputImage;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> rangeScratch;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> histogram;

struct RangePush {
    uint width;
    uint height;
    float absMin;
    float scale;
};
[[vk::push_constant]] RangePush params;

static const uint HISTOGRAM_BINS = 65536u;

uint FloatToOrdered(float value) {
    const uint bits = asuint(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// Keep this source expression identical to the CPU path. `precise` asks DXC
// not to reassociate or contract it across a histogram-bin boundary.
float Luminance(float3 rgb) {
    precise float lum = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
    return lum;
}

bool ValidRgb(float3 rgb) {
    return all(isfinite(rgb));
}

bool RequiresCpuRange(float3 rgb) {
    const uint3 magnitude = asuint(rgb) & 0x7fffffffu;
    const uint safeMagnitude = asuint(0x1p-98f);
    return any((magnitude != 0u) & (magnitude < safeMagnitude));
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    if (dispatchId.x >= params.width || dispatchId.y >= params.height)
        return;

    const float3 rgb = inputImage[dispatchId.xy].rgb;
    if (!ValidRgb(rgb)) return;

#ifdef DISPLAY_RANGE_PASS_EXTENTS
    uint ignored;
    // Vulkan devices may flush small arithmetic to zero. Values below 2^-98
    // can produce a subnormal BT.709 product or cancellation result, so tell
    // the host to use its exact original CPU path for this image. Test the
    // bits: a floating-point comparison may itself flush a denormal to zero.
    if (RequiresCpuRange(rgb)) {
        InterlockedOr(rangeScratch[3], 1u, ignored);
        return;
    }
    const float lum = Luminance(rgb);
    InterlockedMin(rangeScratch[0], FloatToOrdered(lum), ignored);
    InterlockedMax(rangeScratch[1], FloatToOrdered(lum), ignored);
    InterlockedAdd(rangeScratch[2], 1u, ignored);
#endif

#ifdef DISPLAY_RANGE_PASS_HISTOGRAM
    const float lum = Luminance(rgb);
    precise float offset = lum - params.absMin;
    precise float binFloat = offset * params.scale;
    const uint bin = min((uint)binFloat, HISTOGRAM_BINS - 1u);
    uint ignored;
    InterlockedAdd(histogram[bin], 1u, ignored);
#endif
}
