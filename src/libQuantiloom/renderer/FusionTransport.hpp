#pragma once
#include "scene/Scene.hpp"
#include "core/Config.hpp"
#include <vector>
// FusionTransportData, defined once for fusion_transport.hlsli and the host. By a
// path relative to this file, as RtBindings.hpp includes its manifest.
#include "../../shaders/ql_shared_fusion.hlsli"

namespace quantiloom::rendercore {
using FusionTransportGpu = shader::FusionTransportData;
/// Record 0 is the summary ResolveFusionTransport writes: mode 0 means the
/// config named fusion transport but no interface qualified, anything else
/// means rays must walk fusion media. Every host branch on it goes through
/// this.
inline bool FusionOpticsActive(const Vector<FusionTransportGpu>& transport) {
    return !transport.empty() && transport[0].mode != 0;
}
/// Validates smooth thin sheets and watertight homogeneous solids; called before
/// BLAS build so solid back faces are present. No shared material layout changes.
Result<Vector<FusionTransportGpu>,String> ResolveFusionTransport(const Config& config,Scene& scene);
Result<Vector<u32>,String> InitialFusionMedia(const Scene& scene,
    const Vector<FusionTransportGpu>& records,const glm::vec3& origin);
}
