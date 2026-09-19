#include "camera_common.hlsli"

[[vk::binding(2, 0)]] RWTexture2D<uint4> randomVectors;

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint2 coord = id.xy;
    if (coord.x >= cameraPush.width || coord.y >= cameraPush.height) return;
    const uint pixel = coord.y * cameraPush.width + coord.x;
    randomVectors[coord] = uint4(
        CameraCounterU32(cameraPush.seed, pixel, cameraPush.acquisitionLo,
                         cameraPush.acquisitionHi, cameraPush.rngNoiseClass,
                         cameraPush.rngCounter + 0u),
        CameraCounterU32(cameraPush.seed, pixel, cameraPush.acquisitionLo,
                         cameraPush.acquisitionHi, cameraPush.rngNoiseClass,
                         cameraPush.rngCounter + 1u),
        CameraCounterU32(cameraPush.seed, pixel, cameraPush.acquisitionLo,
                         cameraPush.acquisitionHi, cameraPush.rngNoiseClass,
                         cameraPush.rngCounter + 2u),
        CameraCounterU32(cameraPush.seed, pixel, cameraPush.acquisitionLo,
                         cameraPush.acquisitionHi, cameraPush.rngNoiseClass,
                         cameraPush.rngCounter + 3u));
}
