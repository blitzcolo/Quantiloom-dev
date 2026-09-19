#include "camera_common.hlsli"

[[vk::binding(0, 0)]] RWTexture2D<float4> measuredRate;
[[vk::binding(1, 0)]] RWTexture2D<float4> thermalBefore;
[[vk::binding(2, 0)]] RWTexture2D<float4> rawDnImage;
[[vk::binding(3, 0)]] RWTexture2D<float4> correctedImage;
[[vk::binding(4, 0)]] RWTexture2D<float4> expectedImage;
[[vk::binding(5, 0)]] RWTexture2D<float4> preAdcImage;
[[vk::binding(6, 0)]] RWTexture2D<float4> thermalAfter;
[[vk::binding(7, 0)]] StructuredBuffer<float> nucGain;
[[vk::binding(8, 0)]] StructuredBuffer<float> nucOffset;
[[vk::binding(9, 0)]] StructuredBuffer<float4> cameraConfig;

// row0: exposure_s, fullWell_e, darkCurrent_e_s, readNoise_e_rms
// row1: prnuSigma, dsnu_e_rms, dsnuReferenceExposure_s, biasDnRms
// row2: analogGain, electronsPerDn, blackLevelDn, nucResidualFraction
// row3: thermalTau_s, thermalResponsivityDnPerW, driftDnPerSecond, thermalNoise0
// row4: thermalNoise1, thermalNoise2, unused, unused

float CameraQuantize(float analogDn) {
    const float maxDn = exp2(float(cameraPush.adcBits)) - 1.0;
    return clamp(floor(analogDn + 0.5), 0.0, maxDn);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint2 coord = id.xy;
    if (coord.x >= cameraPush.width || coord.y >= cameraPush.height) return;

    const float4 rate = measuredRate[coord];
    const float4 prior = thermalBefore[coord];
    const bool multi = cameraPush.cfa == CAMERA_MULTI_CHANNEL;
    const uint channels = multi ? min(cameraPush.channelCount, 3u) : 1u;
    const uint cfaChannel = min(2u, uint(max(0.0, rate.a) + 0.5));
    const uint physicalPixel = CameraPhysicalPixel(coord);
    const bool noiseFree = (cameraPush.flags & CAMERA_NOISE_FREE) != 0u;
    const float4 p0 = cameraConfig[0];
    const float4 p1 = cameraConfig[1];
    const float4 p2 = cameraConfig[2];
    const float4 p3 = cameraConfig[3];
    const float4 p4 = cameraConfig[4];

    float4 raw = 0.0;
    float4 corrected = 0.0;
    float4 expected = 0.0;
    float4 preAdc = 0.0;
    float4 thermalState = prior;
    raw.a = rate.a;
    corrected.a = rate.a;
    expected.a = rate.a;
    preAdc.a = rate.a;
    thermalState.a = rate.a;

    [loop]
    for (uint channel = 0u; channel < channels; ++channel) {
        const uint noisePixel = physicalPixel * channels + channel;
        const uint deviceChannel = multi ? channel : cfaChannel;
        const uint responseChannel = cameraPush.cfa == CAMERA_MONO
            ? 0u : deviceChannel;
        const float rateOrPower = max(0.0, rate[responseChannel]);
        const float gainMap = (cameraPush.flags & CAMERA_NUC_GAIN_MAP) != 0u
            ? nucGain[noisePixel] : 1.0;
        const float offsetMap = (cameraPush.flags & CAMERA_NUC_OFFSET_MAP) != 0u
            ? nucOffset[noisePixel] : 0.0;
        float analogDn = 0.0;
        float correctedValue = 0.0;

        if (cameraPush.detector == CAMERA_PHOTON) {
            const float exposure = p0.x;
            const float baselineLight = rateOrPower * exposure;
            expected[channel] = baselineLight;
            const float prnu = !noiseFree &&
                (cameraPush.flags & CAMERA_FPN) != 0u
                ? p1.x * CameraGaussian(noisePixel, NOISE_FIXED_PRNU, 0u, 0u)
                : 0.0;
            const float expectedLight = max(0.0, baselineLight * (1.0 + prnu));
            const float dsnu = !noiseFree &&
                (cameraPush.flags & CAMERA_FPN) != 0u
                ? p1.y * (exposure / max(p1.z, 1e-20)) *
                  CameraGaussian(noisePixel, NOISE_FIXED_DSNU, 0u, 0u)
                : 0.0;
            const float darkMean = (cameraPush.flags & CAMERA_DARK_CURRENT) != 0u
                ? max(0.0, p0.z * exposure + dsnu) : 0.0;
            const float light = !noiseFree &&
                (cameraPush.flags & CAMERA_SHOT) != 0u
                ? CameraPoisson(expectedLight, noisePixel, NOISE_PHOTON_SHOT)
                : expectedLight;
            const float dark = !noiseFree &&
                (cameraPush.flags & CAMERA_DARK_SHOT) != 0u
                ? CameraPoisson(darkMean, noisePixel, NOISE_DARK_SHOT)
                : darkMean;
            float electrons = min(p0.y, light + dark);
            if (!noiseFree && (cameraPush.flags & CAMERA_READ) != 0u)
                electrons += p0.w * CameraGaussian(
                    noisePixel, NOISE_READ,
                    cameraPush.acquisitionLo, cameraPush.acquisitionHi);
            preAdc[channel] = electrons;
            const float fixedBias = !noiseFree &&
                (cameraPush.flags & CAMERA_FPN) != 0u
                ? p1.w * CameraGaussian(noisePixel, NOISE_BIAS, 0u, 0u)
                : 0.0;
            analogDn = p2.x * electrons / p2.y + p2.z + fixedBias;
            const float dn = CameraQuantize(analogDn);
            raw[channel] = dn;
            correctedValue = (dn - p2.z - fixedBias) * p2.y / p2.x;
            if ((cameraPush.flags & CAMERA_DARK_CURRENT) != 0u)
                correctedValue -= p0.z * exposure;
            if ((cameraPush.flags & CAMERA_NUC) != 0u) {
                correctedValue = correctedValue * gainMap + offsetMap;
                if (!noiseFree)
                    correctedValue += p2.w * (prnu * rateOrPower * exposure + dsnu);
            }
        } else {
            const float previous = cameraPush.firstThermal != 0u
                ? rateOrPower : max(0.0, prior[channel]);
            const float retain = p3.x <= 0.0
                ? 0.0 : exp(-max(0.0, cameraPush.deltaSeconds) / p3.x);
            const float state = previous + (1.0 - retain) * (rateOrPower - previous);
            thermalState[channel] = state;
            expected[channel] = state;
            float thermalNoise = deviceChannel == 0u ? p3.w :
                                 deviceChannel == 1u ? p4.x : p4.y;
            analogDn = state * p3.y + p2.z +
                       p3.z * cameraPush.frameTimeSeconds;
            if (!noiseFree)
                analogDn += thermalNoise * CameraGaussian(
                    noisePixel, NOISE_THERMAL_READ,
                    cameraPush.acquisitionLo, cameraPush.acquisitionHi);
            const float dn = CameraQuantize(analogDn);
            raw[channel] = dn;
            correctedValue = (dn - p2.z -
                p3.z * cameraPush.frameTimeSeconds) / p3.y;
            if ((cameraPush.flags & CAMERA_NUC_GAIN_MAP) != 0u)
                correctedValue *= gainMap;
            if ((cameraPush.flags & CAMERA_NUC_OFFSET_MAP) != 0u)
                correctedValue += offsetMap / p3.y;
            preAdc[channel] = state;
        }
        corrected[channel] = max(0.0, correctedValue);
    }
    rawDnImage[coord] = raw;
    correctedImage[coord] = corrected;
    expectedImage[coord] = expected;
    preAdcImage[coord] = preAdc;
    thermalAfter[coord] = thermalState;
}
