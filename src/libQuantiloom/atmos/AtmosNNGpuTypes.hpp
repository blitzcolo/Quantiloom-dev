#pragma once

#include <cstdint>

namespace quantiloom {

// ============================================================================
// AtmosNNHeaderGPU - mirrors AtmosNNHeader in src/shaders/atmosphere_nn.hlsli
// ============================================================================
// Bound at binding 17 (storage buffer). The bulk LUT data lives in a flat
// float buffer at binding 20:
//   tau:   [numLambda, countA]            tauOffset + iLambda*countA + ia
//   lpath: [numLambda, countA, countAz]   lpathOffset + (iLambda*countA+ia)*countAz + iaz
//   ldown: [numLambda]                    ldownOffset + iLambda   (thermal bands)
//   sky:   reserved (hasSky/skyOffset), for future <band>_sky networks
//
// Axis a: pathMode 0 (ground)  -> a = ln(range_km), uniform in log space
//         pathMode 1 (slant)   -> a = cos(view_zenith), uniform
// Axis az: relative azimuth between view and sun horizontal projections,
//          radians-free: stored in degrees [0, 180].

struct AtmosNNHeaderGPU {
    uint32_t enabled = 0;      // 0 = analytic-free passthrough (no atmosphere)
    uint32_t pathMode = 0;     // 0 = ground (range axis), 1 = slant (cos zenith)
    uint32_t numLambda = 0;    // Spectral samples (= render loop sample count)
    uint32_t countA = 0;       // Samples along axis a

    uint32_t countAz = 1;      // Samples along relative-azimuth axis (lpath)
    uint32_t hasLdown = 0;
    uint32_t hasSky = 0;       // Reserved: sky dome network not trained yet
    uint32_t thermalBand = 0;  // 1 for mwir/lwir

    float aStart = 0.0f;       // First a sample
    float aStep = 0.0f;        // Uniform step in a
    float azStart = 0.0f;      // Degrees
    float azStep = 0.0f;       // Degrees

    uint32_t tauOffset = 0;    // Float offsets into the binding-20 data blob
    uint32_t lpathOffset = 0;
    uint32_t ldownOffset = 0;
    uint32_t skyOffset = 0;

    float sunDirWorld[3] = {0.0f, 1.0f, 0.0f};  // Unit vector toward the sun
    float worldUnitsToMeters = 1.0f;
};

static_assert(sizeof(AtmosNNHeaderGPU) == 80,
              "AtmosNNHeaderGPU must stay in sync with atmosphere_nn.hlsli");

}  // namespace quantiloom
