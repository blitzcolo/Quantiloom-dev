// Shared GPU camera readout constants and counter stream. Keep Mix32 and the
// five folds identical to postprocess/CameraPhysics.cpp::CounterRandomU32.
struct CameraPush {
    uint width;
    uint height;
    uint physicalWidth;
    uint physicalHeight;
    uint channelCount;
    uint cfa;
    uint detector;
    uint flags;
    uint seed;
    uint acquisitionLo;
    uint acquisitionHi;
    uint adcBits;
    float frameTimeSeconds;
    float deltaSeconds;
    uint firstThermal;
    uint direction;
    float sigmaR;
    float sigmaG;
    float sigmaB;
    float sigmaRy;
    float sigmaGy;
    float sigmaBy;
    uint rngNoiseClass;
    uint rngCounter;
    // Dynamic-exposure compositor (camera_dynamic.comp). The per-stratum
    // camera frames ride in binding 10; these scalars describe the exposure.
    uint timeStratumCount;      // T; 0/1 = plain copy of layer 0
    uint rollingShutter;        // 0/1: rows integrate at firstRowMid + y*rowDelay
    float exposureSeconds;      // E: integration window each row integrates over
    float rowDelaySeconds;      // rolling-shutter per-row offset
    float firstRowMidSeconds;   // t0: midpoint of row 0's exposure
    // Multi-phase camera passes (camera_stats) select their stage here; 0 for
    // every other pass. Must match CameraPush in GpuCameraPipeline.cpp.
    uint ispPhase;
};
[[vk::push_constant]] CameraPush cameraPush;

// bit 0 noiseFree, 1 shot, 2 dark current, 3 dark shot, 4 read,
// 5 fixed pattern, 6 NUC, 7 NUC gain map, 8 NUC offset map.
static const uint CAMERA_NOISE_FREE = 1u;
static const uint CAMERA_SHOT = 2u;
static const uint CAMERA_DARK_CURRENT = 4u;
static const uint CAMERA_DARK_SHOT = 8u;
static const uint CAMERA_READ = 16u;
static const uint CAMERA_FPN = 32u;
static const uint CAMERA_NUC = 64u;
static const uint CAMERA_NUC_GAIN_MAP = 128u;
static const uint CAMERA_NUC_OFFSET_MAP = 256u;

static const uint CAMERA_MONO = 0u;
static const uint CAMERA_MULTI_CHANNEL = 5u;
static const uint CAMERA_PHOTON = 0u;
static const uint CAMERA_THERMAL = 1u;

// NoiseClass public enum values. Must match the NoiseClass enum in
// postprocess/CameraPhysics.hpp bit for bit; new classes append at the end.
static const uint NOISE_PHOTON_SHOT = 0u;
static const uint NOISE_DARK_SHOT = 1u;
static const uint NOISE_READ = 2u;
static const uint NOISE_BIAS = 3u;
static const uint NOISE_FIXED_PRNU = 4u;
static const uint NOISE_FIXED_DSNU = 5u;
static const uint NOISE_THERMAL_READ = 6u;
static const uint NOISE_EMPIRICAL_NOISE = 9u;
static const uint NOISE_EMPIRICAL_DRIFT = 10u;

uint CameraMix32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint CameraCounterU32(uint seed, uint pixelIndex, uint acquisitionLo,
                      uint acquisitionHi, uint noiseClass, uint counter) {
    const bool fixedPattern = noiseClass == NOISE_FIXED_PRNU ||
                              noiseClass == NOISE_FIXED_DSNU;
    const uint lo = fixedPattern ? 0u : acquisitionLo;
    const uint hi = fixedPattern ? 0u : acquisitionHi;
    uint state = CameraMix32(seed ^ 0x9e3779b9u);
    state = CameraMix32(state ^ CameraMix32(pixelIndex + 0x85ebca6bu));
    state = CameraMix32(state ^ CameraMix32(lo + 0xc2b2ae35u));
    state = CameraMix32(state ^ CameraMix32(hi + 0x27d4eb2fu));
    state = CameraMix32(state ^ CameraMix32(noiseClass + 0x165667b1u));
    return CameraMix32(state ^ CameraMix32(counter + 0xd3a2646cu));
}

float CameraUniform(uint pixelIndex, uint noiseClass, uint counter,
                    uint acquisitionLo, uint acquisitionHi) {
    const uint bits = CameraCounterU32(cameraPush.seed, pixelIndex,
                                      acquisitionLo, acquisitionHi,
                                      noiseClass, counter);
    return (float(bits) + 0.5) * (1.0 / 4294967296.0);
}

float CameraGaussian(uint pixelIndex, uint noiseClass,
                     uint acquisitionLo, uint acquisitionHi) {
    const float u1 = max(CameraUniform(pixelIndex, noiseClass, 0u,
                                       acquisitionLo, acquisitionHi), 1e-12);
    const float u2 = CameraUniform(pixelIndex, noiseClass, 1u,
                                   acquisitionLo, acquisitionHi);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

float CameraPoisson(float mean, uint pixelIndex, uint noiseClass) {
    if (mean <= 0.0) return 0.0;
    // Exact inversion below 32 electrons. The high-count normal branch is
    // statistically validated against Poisson in the GPU test suite.
    if (mean < 32.0) {
        const float threshold = exp(-mean);
        float product = 1.0;
        uint draws = 0u;
        // At tiny positive means exp(-mean) rounds to 1 in f32; still draw
        // once so draws-1 cannot underflow to UINT_MAX.
        [loop]
        do {
            product *= CameraUniform(pixelIndex, noiseClass, draws,
                                     cameraPush.acquisitionLo,
                                     cameraPush.acquisitionHi);
            ++draws;
        } while (product > threshold && draws < 512u);
        return float(draws - 1u);
    }
    const float sample = mean + sqrt(mean) *
        CameraGaussian(pixelIndex, noiseClass,
                       cameraPush.acquisitionLo, cameraPush.acquisitionHi);
    return floor(max(sample, 0.0) + 0.5);
}

uint CameraPhysicalPixel(uint2 coord) {
    // Match raygen's pixel-centre mapping, including non-integer render
    // scales. Left-edge mapping sends 3 preview columns into 2 sensor
    // columns differently at x=1, so CFA and fixed pattern would disagree.
    const uint x = min(cameraPush.physicalWidth - 1u,
                       ((2u * coord.x + 1u) * cameraPush.physicalWidth) /
                           (2u * cameraPush.width));
    const uint y = min(cameraPush.physicalHeight - 1u,
                       ((2u * coord.y + 1u) * cameraPush.physicalHeight) /
                           (2u * cameraPush.height));
    return y * cameraPush.physicalWidth + x;
}

float CameraSrgbEncode(float linearValue) {
    const float x = saturate(linearValue);
    return x <= 0.0031308 ? 12.92 * x :
           1.055 * pow(x, 1.0 / 2.4) - 0.055;
}
