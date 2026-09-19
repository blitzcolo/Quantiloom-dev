#include "camera_common.hlsli"

[[vk::binding(0, 0)]] RWTexture2D<float4> inputRate;
[[vk::binding(1, 0)]] RWTexture2D<float4> outputRate;

float CameraSigma(uint channel) {
    if (cameraPush.direction == 0u)
        return channel == 0u ? cameraPush.sigmaR :
               channel == 1u ? cameraPush.sigmaG : cameraPush.sigmaB;
    return channel == 0u ? cameraPush.sigmaRy :
           channel == 1u ? cameraPush.sigmaGy : cameraPush.sigmaBy;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint2 coord = id.xy;
    if (coord.x >= cameraPush.width || coord.y >= cameraPush.height) return;
    const float4 centre = inputRate[coord];
    // Bayer input carries latent R/G/B rates at every pixel. Blur the three
    // optical fields before the actual CFA selects one in camera_readout.
    const uint validChannels = cameraPush.cfa == CAMERA_MONO
        ? 1u : min(cameraPush.channelCount, 3u);
    float3 sum = 0.0;
    float3 weightSum = 0.0;
    [loop]
    for (uint channel = 0u; channel < validChannels; ++channel) {
        const float sigma = CameraSigma(channel);
        if (sigma < 0.1) {
            sum[channel] = centre[channel];
            weightSum[channel] = 1.0;
            continue;
        }
        const int radius = min(12, int(ceil(3.0 * sigma)));
        [loop]
        for (int offset = -radius; offset <= radius; ++offset) {
            int2 sampleCoord = int2(coord);
            if (cameraPush.direction == 0u)
                sampleCoord.x = clamp(sampleCoord.x + offset, 0,
                                      int(cameraPush.width) - 1);
            else
                sampleCoord.y = clamp(sampleCoord.y + offset, 0,
                                      int(cameraPush.height) - 1);
            const float4 sample = inputRate[uint2(sampleCoord)];
            const float u = float(offset) / sigma;
            const float w = exp(-0.5 * u * u);
            sum[channel] += sample[channel] * w;
            weightSum[channel] += w;
        }
    }
    float3 result = 0.0;
    [unroll]
    for (uint channel = 0u; channel < 3u; ++channel)
        if (channel < validChannels)
            result[channel] = sum[channel] / max(weightSum[channel], 1e-20);
    outputRate[coord] = float4(result, centre.a);
}
