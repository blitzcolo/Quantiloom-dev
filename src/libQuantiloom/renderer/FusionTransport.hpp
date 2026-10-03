#pragma once
#include "scene/Scene.hpp"
#include "core/Config.hpp"
#include <vector>

namespace quantiloom::rendercore {
struct FusionTransportGpu {
    u32 mode=0,nodeId=0,materialId=0,flags=0;
    f32 absorptionPerMeter=0,sheetReflectance=0,sheetTransmittance=0,orientation=1;
};
static_assert(sizeof(FusionTransportGpu)==32);
/// Validates smooth thin sheets and watertight homogeneous solids; called before
/// BLAS build so solid back faces are present. No shared material layout changes.
Result<Vector<FusionTransportGpu>,String> ResolveFusionTransport(const Config& config,Scene& scene);
Result<Vector<u32>,String> InitialFusionMedia(const Scene& scene,
    const Vector<FusionTransportGpu>& records,const glm::vec3& origin);
}
