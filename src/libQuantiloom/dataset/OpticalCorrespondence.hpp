#pragma once
#include "dataset/FusionPathTruth.hpp"
#include "dataset/ProductGeometry.hpp"
#include "renderer/OfflineRenderer.hpp"

namespace quantiloom::dataset {
struct OpticalEndpoint {
    String productId;
    u32 row=0,pixel=0,nodeId=0,primitiveId=0,branchMask=0;
    f64 wavelengthNm=0;
    u64 acquisitionIndex=0;
    glm::dvec3 position{};
    glm::dvec2 nativePixel{};
    bool throughOptics=false;
};
Vector<OpticalEndpoint> DecodeOpticalEndpoints(const FusionPathChunk& chunk,
    const String& productId,const ProductGeometry& geometry);
Result<String,String> MatchOpticalPaths(OfflineRenderer& target,
    const Vector<OpticalEndpoint>& source,const Vector<OpticalEndpoint>& targetSamples,
    const ProductGeometry& targetGeometry,u32 sourceLimit=64,
    const std::function<bool()>& cancelled={});
}
