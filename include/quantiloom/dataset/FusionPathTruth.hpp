#pragma once
#include "core/Types.hpp"

namespace quantiloom::dataset {
/// Little-endian GPU trace records. Header: 32 u32; rays: 8 words each;
/// vertices: 20 words each. A JSON artifact describes fields and admission.
struct FusionPathChunk {
    u32 width = 0, height = 0, spp = 0, rayStride = 1, storedRays = 0, slotsPerRay = 0;
    f64 wavelengthNm = 0, referenceTimeSeconds = 0;
    u64 acquisitionIndex = 0;
    u32 diagnosticFlags = 0;
    Vector<u8> bytes;
};
}
