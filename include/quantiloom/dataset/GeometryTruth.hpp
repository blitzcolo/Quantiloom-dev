#pragma once
#include "core/Image.hpp"
#include "io/ImageIO.hpp"
#include "dataset/ProductGeometry.hpp"

namespace quantiloom::dataset {
struct SurfaceQuery {
    u32 instanceId = 0;
    u32 validity = 0;
    glm::vec3 worldPosition{};
};
struct OpticalProbe {
    glm::vec2 nativePixel{};
    f64 wavelengthNm = 550;
    u32 branchMask = 0; // bit per interface depth: 1 reflection, 0 transmission
};
struct OpticalProbeResult {
    SurfaceQuery surface;
    u32 primitiveId = 0, flags = 0;
    f32 throughput = 0;
};
struct GeometryTruth {
    Image rayDistanceMeters;
    Image cameraDepthMeters;
    Image worldPosition;
    Image worldNormal;
    /// 0 miss, 1 opaque correspondence eligible, 2 lens invalid,
    /// 3 partial coverage, 4 transmissive surface (requires path truth).
    Image validity;
    UIntImage instanceId;
    ProductGeometry geometry;
    String instancesJson;
};
} // namespace quantiloom::dataset
