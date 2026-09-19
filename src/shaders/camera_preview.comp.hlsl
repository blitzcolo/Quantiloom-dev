#include "camera_common.hlsli"

[[vk::binding(0, 0)]] RWTexture2D<float4> correctedImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> displayImage;
[[vk::binding(2, 0)]] RWTexture2D<float4> rawDnImage;
[[vk::binding(9, 0)]] StructuredBuffer<float4> cameraConfig;

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint2 coord = id.xy;
    if (coord.x >= cameraPush.width || coord.y >= cameraPush.height) return;
    const float4 corrected = correctedImage[coord];
    float3 linearValue;
    if (cameraPush.detector == CAMERA_PHOTON) {
        const float fullWell = max(cameraConfig[0].y, 1e-20);
        linearValue = cameraPush.cfa == CAMERA_MULTI_CHANNEL
            ? corrected.rgb / fullWell
            : corrected.rrr / fullWell;
    } else {
        // Basic M3 display only: monotonic DN preview. Persistent IR AGC,
        // palette and HSV are later ISP/display stages, never RAW operations.
        const float maxDn = exp2(float(cameraPush.adcBits)) - 1.0;
        linearValue = rawDnImage[coord].rrr / max(maxDn, 1.0);
    }
    const float3 encoded = float3(
        CameraSrgbEncode(linearValue.r),
        CameraSrgbEncode(linearValue.g),
        CameraSrgbEncode(linearValue.b));
    displayImage[coord] = float4(encoded, 1.0);
}
