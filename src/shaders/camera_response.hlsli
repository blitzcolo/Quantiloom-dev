#ifndef QUANTILOOM_CAMERA_RESPONSE_HLSLI
#define QUANTILOOM_CAMERA_RESPONSE_HLSLI

// Mirror of renderer/CameraResponseGpu.hpp. Binding 29 is one raw buffer;
// offsets in the channel records are byte offsets into that buffer.
static const uint CAMERA_RESPONSE_MAGIC = 0x514C4341u;
static const uint CAMERA_RESPONSE_VERSION = 1u;
static const uint CAMERA_HEADER_WORDS = 16u;
static const uint CAMERA_CHANNEL_WORDS = 12u;
static const uint CAMERA_MAX_CHANNELS = 3u;
static const uint CAMERA_DETECTOR_PHOTON = 0u;
static const uint CAMERA_CFA_MONO = 0u;
static const uint CAMERA_CFA_RGGB = 1u;
static const uint CAMERA_CFA_GRBG = 2u;
static const uint CAMERA_CFA_GBRG = 3u;
static const uint CAMERA_CFA_BGGR = 4u;
static const uint CAMERA_CFA_MULTI = 5u;

uint CameraHeaderWord(ByteAddressBuffer table, uint word) {
    return table.Load(word * 4u);
}
float CameraHeaderFloat(ByteAddressBuffer table, uint word) {
    return asfloat(CameraHeaderWord(table, word));
}
uint CameraChannelWord(ByteAddressBuffer table, uint channel, uint word) {
    return table.Load((CAMERA_HEADER_WORDS + channel * CAMERA_CHANNEL_WORDS + word) * 4u);
}
float CameraChannelFloat(ByteAddressBuffer table, uint channel, uint word) {
    return asfloat(CameraChannelWord(table, channel, word));
}
float CameraPointFloat(ByteAddressBuffer table, uint offsetBytes) {
    return asfloat(table.Load(offsetBytes));
}

bool CameraResponseValid(ByteAddressBuffer table) {
    const uint channels = CameraHeaderWord(table, 4u);
    return CameraHeaderWord(table, 0u) == CAMERA_RESPONSE_MAGIC &&
           CameraHeaderWord(table, 1u) == CAMERA_RESPONSE_VERSION &&
           channels > 0u && channels <= CAMERA_MAX_CHANNELS &&
           CameraHeaderWord(table, 5u) > 0u &&
           CameraHeaderWord(table, 7u) > 0u &&
           CameraHeaderWord(table, 11u) > 0u;
}

float CameraCurveAt(ByteAddressBuffer table, uint offsetBytes,
                    uint count, float lambdaNm) {
    if (count < 2u) return 0.0;
    const float first = CameraPointFloat(table, offsetBytes);
    const float last = CameraPointFloat(table, offsetBytes + 8u * (count - 1u));
    if (lambdaNm < first || lambdaNm > last) return 0.0;
    uint low = 0u, high = count - 1u;
    [loop]
    while (high - low > 1u) {
        const uint mid = (low + high) / 2u;
        if (lambdaNm < CameraPointFloat(table, offsetBytes + 8u * mid))
            high = mid;
        else
            low = mid;
    }
    const float x0 = CameraPointFloat(table, offsetBytes + 8u * low);
    const float x1 = CameraPointFloat(table, offsetBytes + 8u * high);
    const float y0 = CameraPointFloat(table, offsetBytes + 8u * low + 4u);
    const float y1 = CameraPointFloat(table, offsetBytes + 8u * high + 4u);
    return lerp(y0, y1, saturate((lambdaNm - x0) / max(x1 - x0, 1e-12)));
}

float CameraResponseAt(ByteAddressBuffer table, uint channel, float lambdaNm) {
    float response = CameraCurveAt(table,
        CameraChannelWord(table, channel, 0u),
        CameraChannelWord(table, channel, 1u), lambdaNm);
    const uint lensCount = CameraChannelWord(table, channel, 3u);
    if (lensCount > 0u)
        response *= CameraCurveAt(table, CameraChannelWord(table, channel, 2u),
                                  lensCount, lambdaNm);
    const uint filterCount = CameraChannelWord(table, channel, 5u);
    if (filterCount > 0u)
        response *= CameraCurveAt(table, CameraChannelWord(table, channel, 4u),
                                  filterCount, lambdaNm);
    return response;
}

float CameraProposalPdf(ByteAddressBuffer table, uint channel, float lambdaNm) {
    const float lo = CameraChannelFloat(table, channel, 8u);
    const float hi = CameraChannelFloat(table, channel, 9u);
    if (lambdaNm < lo || lambdaNm > hi) return 0.0;
    const uint bins = CameraChannelWord(table, channel, 7u);
    if (bins == 0u || hi <= lo) return 0.0;
    const float width = (hi - lo) / float(bins);
    const uint bin = min(uint((lambdaNm - lo) / width), bins - 1u);
    const uint offset = CameraChannelWord(table, channel, 6u) + 8u * bin;
    const float cdfHi = CameraPointFloat(table, offset + 4u);
    const float cdfLo = bin == 0u ? 0.0 : CameraPointFloat(table, offset - 4u);
    const float mass = max(cdfHi - cdfLo, 0.0);
    const float mix = CameraHeaderFloat(table, 10u);
    return (1.0 - mix) * mass / width + mix / (hi - lo);
}

float CameraSampleWavelength(ByteAddressBuffer table, uint channel, float u,
                             out float pdf) {
    const float lo = CameraChannelFloat(table, channel, 8u);
    const float hi = CameraChannelFloat(table, channel, 9u);
    const float mix = CameraHeaderFloat(table, 10u);
    const uint bins = CameraChannelWord(table, channel, 7u);
    u = clamp(u, 0.0, 0.99999994);
    float lambdaNm;
    if (u < mix) {
        lambdaNm = lo + (u / mix) * (hi - lo);
    } else {
        const float target = (u - mix) / (1.0 - mix);
        const uint offset = CameraChannelWord(table, channel, 6u);
        uint low = 0u, high = bins - 1u;
        [loop]
        while (low < high) {
            const uint mid = (low + high) / 2u;
            const float cdfHi = CameraPointFloat(table, offset + 8u * mid + 4u);
            if (target > cdfHi) low = mid + 1u;
            else high = mid;
        }
        const float previous = low == 0u ? 0.0 :
            CameraPointFloat(table, offset + 8u * (low - 1u) + 4u);
        const float current = CameraPointFloat(table, offset + 8u * low + 4u);
        const float mass = max(current - previous, 0.0);
        const float fraction = mass > 0.0 ?
            saturate((target - previous) / mass) : 0.0;
        lambdaNm = lo + (float(low) + fraction) * (hi - lo) / float(bins);
    }
    lambdaNm = min(lambdaNm, hi);
    pdf = CameraProposalPdf(table, channel, lambdaNm);
    return lambdaNm;
}

uint CameraCfaChannel(ByteAddressBuffer table, uint2 pixel) {
    const uint pattern = CameraHeaderWord(table, 3u);
    if (pattern == CAMERA_CFA_MONO || pattern == CAMERA_CFA_MULTI) return 0u;
    const bool oddX = (pixel.x & 1u) != 0u;
    const bool oddY = (pixel.y & 1u) != 0u;
    if (pattern == CAMERA_CFA_RGGB) return oddY ? (oddX ? 2u : 1u) : (oddX ? 1u : 0u);
    if (pattern == CAMERA_CFA_GRBG) return oddY ? (oddX ? 1u : 2u) : (oddX ? 0u : 1u);
    if (pattern == CAMERA_CFA_GBRG) return oddY ? (oddX ? 1u : 0u) : (oddX ? 2u : 1u);
    if (pattern == CAMERA_CFA_BGGR) return oddY ? (oddX ? 0u : 1u) : (oddX ? 1u : 2u);
    return 0u;
}

uint2 CameraPhysicalPixel(ByteAddressBuffer table, uint2 launchPixel,
                          uint2 launchExtent) {
    const uint width = CameraHeaderWord(table, 7u);
    const uint height = CameraHeaderWord(table, 11u);
    return min(uint2((float2(launchPixel) + 0.5) *
                      float2(width, height) / float2(launchExtent)),
               uint2(width - 1u, height - 1u));
}

uint CameraAtmosLambdaIndex(ByteAddressBuffer table, float lambdaNm,
                             uint bakedCount) {
    const uint count = min(CameraHeaderWord(table, 14u), bakedCount);
    if (count <= 1u) return 0u;
    const float lo = CameraHeaderFloat(table, 12u);
    const float step = CameraHeaderFloat(table, 13u);
    return uint(clamp(round((lambdaNm - lo) / max(step, 1e-9)),
                      0.0, float(count - 1u)));
}

#endif // QUANTILOOM_CAMERA_RESPONSE_HLSLI
