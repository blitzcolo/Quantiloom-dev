/*
 * Two-pass display percentile histogram.
 *
 * Pass 1 filters pixels whose RGB channels are all finite and atomically
 * reduces the absolute BT.709 luminance range plus the valid-pixel count.
 * Pass 2 maps the same values into exactly 65,536 bins, and the last
 * workgroup to finish its writes scans those bins on the device and writes
 * the two percentile crossing bins into rangeScratch. The host reads the
 * extents and the two bins -- a handful of words -- instead of the whole
 * 256 KiB histogram.
 *
 * The crossing-bin definition is bit-identical to the f64 host scan it
 * replaced: binLow is the first bin whose inclusive cumulative count reaches
 * targetLow, binHigh the first to reach targetHigh. Two-phase scan: each of
 * the 256 threads subtotals one contiguous 256-bin chunk, thread 0 walks the
 * subtotals to find which chunks hold the crossings, and the owning threads
 * re-walk their chunk to the exact bin.
 *
 * camera_stats.comp.hlsl is the other percentile window in the tree and is
 * deliberately not this: it bins only unsaturated pixels of the camera's
 * display-domain scalar into 256 bins and walks a nearest-rank percentile on
 * the GPU, one frame late, because the AGC window must exist before the same
 * frame's display pass. This shader serves a readback path instead: every
 * finite pixel and the true absolute range.
 */

[[vk::binding(0, 0)]] RWTexture2D<float4> inputImage;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> rangeScratch;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> histogram;

struct RangePush {
    uint width;
    uint height;
    float absMin;
    float scale;
    uint targetLow;
    uint targetHigh;
};
[[vk::push_constant]] RangePush params;

static const uint HISTOGRAM_BINS = 65536u;

// rangeScratch layout, one u32 per word (see GpuDisplayRange::kScratchWords).
static const uint SCRATCH_ORDERED_MIN = 0u;
static const uint SCRATCH_ORDERED_MAX = 1u;
static const uint SCRATCH_VALID_COUNT = 2u;
static const uint SCRATCH_NEEDS_CPU = 3u;
static const uint SCRATCH_DONE_GROUPS = 4u;
static const uint SCRATCH_BIN_LOW = 5u;
static const uint SCRATCH_BIN_HIGH = 6u;

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
void main(uint3 dispatchId : SV_DispatchThreadID, uint flatTid : SV_GroupIndex) {
#ifdef DISPLAY_RANGE_PASS_EXTENTS
    if (dispatchId.x >= params.width || dispatchId.y >= params.height)
        return;

    const float3 rgb = inputImage[dispatchId.xy].rgb;
    if (!ValidRgb(rgb)) return;

    uint ignored;
    // Vulkan devices may flush small arithmetic to zero. Values below 2^-98
    // can produce a subnormal BT.709 product or cancellation result, so tell
    // the host to use its exact original CPU path for this image. Test the
    // bits: a floating-point comparison may itself flush a denormal to zero.
    if (RequiresCpuRange(rgb)) {
        InterlockedOr(rangeScratch[SCRATCH_NEEDS_CPU], 1u, ignored);
        return;
    }
    const float lum = Luminance(rgb);
    InterlockedMin(rangeScratch[SCRATCH_ORDERED_MIN], FloatToOrdered(lum), ignored);
    InterlockedMax(rangeScratch[SCRATCH_ORDERED_MAX], FloatToOrdered(lum), ignored);
    InterlockedAdd(rangeScratch[SCRATCH_VALID_COUNT], 1u, ignored);
#endif

#ifdef DISPLAY_RANGE_PASS_HISTOGRAM
    // No early returns anywhere: every thread in every workgroup must reach
    // the completion barrier below.
    const bool inBounds = dispatchId.x < params.width && dispatchId.y < params.height;
    float lum = 0.0f;
    bool valid = false;
    if (inBounds) {
        const float3 rgb = inputImage[dispatchId.xy].rgb;
        if (ValidRgb(rgb)) {
            lum = Luminance(rgb);
            valid = true;
        }
    }
    if (valid) {
        precise float offset = lum - params.absMin;
        precise float binFloat = offset * params.scale;
        const uint bin = min((uint)binFloat, HISTOGRAM_BINS - 1u);
        uint ignored;
        InterlockedAdd(histogram[bin], 1u, ignored);
    }

    // The last workgroup to finish its histogram writes scans the bins.
    // Each workgroup fences its writes, then bumps the done-counter; the
    // workgroup that sees the final increment knows every other write is
    // complete and visible, and does the scan alone.
    AllMemoryBarrierWithGroupSync();
    groupshared uint isLast;
    const uint2 groupCount = uint2((params.width + 15u) / 16u, (params.height + 15u) / 16u);
    if (flatTid == 0) {
        uint done;
        InterlockedAdd(rangeScratch[SCRATCH_DONE_GROUPS], 1u, done);
        isLast = (done == groupCount.x * groupCount.y - 1u) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (isLast == 0u) return;
    AllMemoryBarrier();

    // 65,536 bins over 256 threads: thread t owns the contiguous chunk
    // [t*256, (t+1)*256). Subtotal the chunks, find the chunks holding the
    // two crossings, then walk those chunks to the exact bin.
    static const uint CHUNK_BINS = HISTOGRAM_BINS / 256u;
    groupshared uint partial[256];
    const uint chunkBase = flatTid * CHUNK_BINS;
    uint subtotal = 0;
    [loop] for (uint i = 0; i < CHUNK_BINS; ++i)
        subtotal += histogram[chunkBase + i];
    partial[flatTid] = subtotal;

    groupshared uint loThread, loBase, hiThread, hiBase;
    GroupMemoryBarrierWithGroupSync();
    if (flatTid == 0) {
        uint running = 0;
        uint loT = 255u;
        uint hiT = 255u;
        uint loB = 0;
        uint hiB = 0;
        bool haveLo = false;
        bool haveHi = false;
        [loop] for (uint t = 0; t < 256u; ++t) {
            const uint next = running + partial[t];
            if (!haveLo && next >= params.targetLow) {
                loT = t;
                loB = running;
                haveLo = true;
            }
            if (!haveHi && next >= params.targetHigh) {
                hiT = t;
                hiB = running;
                haveHi = true;
            }
            running = next;
        }
        loThread = loT;
        loBase = loB;
        hiThread = hiT;
        hiBase = hiB;
    }
    GroupMemoryBarrierWithGroupSync();

    if (flatTid == loThread) {
        const uint remaining = params.targetLow - loBase;
        uint local = 0;
        [loop] for (uint i = 0; i < CHUNK_BINS; ++i) {
            local += histogram[chunkBase + i];
            if (local >= remaining) {
                rangeScratch[SCRATCH_BIN_LOW] = chunkBase + i;
                break;
            }
        }
    }
    if (flatTid == hiThread) {
        const uint remaining = params.targetHigh - hiBase;
        uint local = 0;
        [loop] for (uint i = 0; i < CHUNK_BINS; ++i) {
            local += histogram[chunkBase + i];
            if (local >= remaining) {
                rangeScratch[SCRATCH_BIN_HIGH] = chunkBase + i;
                break;
            }
        }
    }
#endif
}
