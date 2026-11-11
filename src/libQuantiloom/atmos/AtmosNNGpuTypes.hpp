#pragma once

#include <cstdint>

// AtmosNNHeader, defined once for atmosphere_nn.hlsli and the host. By a path
// relative to this file, as renderer/RtBindings.hpp includes its manifest.
#include "../../shaders/ql_shared_scene.hlsli"

namespace quantiloom {

// ============================================================================
// AtmosNNHeaderGPU - the host's name for AtmosNNHeader (ql_shared_scene.hlsli)
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
// The defaults are the disabled header; sunDirWorld defaults to +Y.

using AtmosNNHeaderGPU = shader::AtmosNNHeader;

}  // namespace quantiloom
