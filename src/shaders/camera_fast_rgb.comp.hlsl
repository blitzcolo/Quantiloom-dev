#include "camera_common.hlsli"

[[vk::binding(0, 0)]] RWTexture2D<float4> linearRgbImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> measuredRate;
[[vk::binding(9, 0)]] StructuredBuffer<float4> cameraConfig;

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint2 coord = id.xy;
    if (coord.x >= cameraPush.width || coord.y >= cameraPush.height) return;

    uint inputWidth, inputHeight;
    linearRgbImage.GetDimensions(inputWidth, inputHeight);
    const uint2 source = uint2(
        min(inputWidth - 1u,
            ((2u * coord.x + 1u) * inputWidth) / (2u * cameraPush.width)),
        min(inputHeight - 1u,
            ((2u * coord.y + 1u) * inputHeight) / (2u * cameraPush.height)));
    const float3 rgb = max(linearRgbImage[source].rgb, 0.0);
    float3 rate = max(float3(
        dot(cameraConfig[5].xyz, rgb),
        dot(cameraConfig[6].xyz, rgb),
        dot(cameraConfig[7].xyz, rgb)), 0.0);

    const uint2 physical = uint2(
        min(cameraPush.physicalWidth - 1u,
            ((2u * coord.x + 1u) * cameraPush.physicalWidth) /
                (2u * cameraPush.width)),
        min(cameraPush.physicalHeight - 1u,
            ((2u * coord.y + 1u) * cameraPush.physicalHeight) /
                (2u * cameraPush.height)));
    uint cfaChannel = 0u;
    if (cameraPush.cfa == 1u)
        cfaChannel = (physical.y & 1u) == 0u
            ? ((physical.x & 1u) == 0u ? 0u : 1u)
            : ((physical.x & 1u) == 0u ? 1u : 2u);
    else if (cameraPush.cfa == 2u)
        cfaChannel = (physical.y & 1u) == 0u
            ? ((physical.x & 1u) == 0u ? 2u : 1u)
            : ((physical.x & 1u) == 0u ? 1u : 0u);
    else if (cameraPush.cfa == 3u)
        cfaChannel = (physical.y & 1u) == 0u
            ? ((physical.x & 1u) == 0u ? 1u : 0u)
            : ((physical.x & 1u) == 0u ? 2u : 1u);
    else if (cameraPush.cfa == 4u)
        cfaChannel = (physical.y & 1u) == 0u
            ? ((physical.x & 1u) == 0u ? 1u : 2u)
            : ((physical.x & 1u) == 0u ? 0u : 1u);
    if ((cameraPush.flags & 512u) != 0u) {
        const float pitchM = cameraConfig[8].y;
        const float focalM = max(cameraConfig[8].x, 1e-20);
        const float dx = (float(physical.x) + 0.5 -
                          float(cameraPush.physicalWidth) * 0.5) * pitchM;
        const float dy = (float(physical.y) + 0.5 -
                          float(cameraPush.physicalHeight) * 0.5) * pitchM;
        const float cosine = rsqrt(1.0 + (dx * dx + dy * dy) /
                                   (focalM * focalM));
        const float cosine2 = cosine * cosine;
        rate *= cosine2 * cosine2;
    }
    measuredRate[coord] = float4(rate, cameraPush.cfa == CAMERA_MULTI_CHANNEL
        ? 0.0 : float(cfaChannel));
}
