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
    /// Per-pixel surface temperature in K: the solved element temperature
    /// where a thermal solve ran and the instance took part, else the
    /// material's authored ir_temperature_k, else the scene default. -1 on a
    /// miss. This is the surface's own temperature, not the radiometric
    /// apparent temperature a sensor would report.
    Image surfaceTemperatureKelvin;
    /// Per-pixel material index + 1 into materialsJson; 0 marks a miss.
    UIntImage materialId;
    /// JSON array of one record per material: material_id, name, temperature
    /// provenance, and the band-averaged emissivity model and value used for
    /// the render band this truth was captured against.
    String materialsJson;
    ProductGeometry geometry;
    String instancesJson;
};
} // namespace quantiloom::dataset
