#ifndef QUANTILOOM_SHADE_TYPES_HLSLI
#define QUANTILOOM_SHADE_TYPES_HLSLI

// What a hit leaves behind for the vertex that shades it. raygen cannot call
// ObjectToWorld3x4()/WorldToObject3x4(), so both 3x3s travel in the record:
// normals want the inverse transpose and tangents the forward matrix, and
// recomputing either would not give the same bits.
struct HitRecord {
    float  t;                  // RayTCurrent(); < 0 means the ray missed
    float2 bary;
    uint   instanceIndex;
    uint   primitiveIndex;
    uint   reserved;
    float3 o2w0, o2w1, o2w2;
    float3 w2o0, w2o1, w2o2;
};

float3x3 QlObjectToWorld(HitRecord hit) {
    float3x3 m;
    m[0] = hit.o2w0; m[1] = hit.o2w1; m[2] = hit.o2w2;
    return m;
}
float3x3 QlWorldToObject(HitRecord hit) {
    float3x3 m;
    m[0] = hit.w2o0; m[1] = hit.w2o1; m[2] = hit.w2o2;
    return m;
}

// Per-vertex state <-> ray payload. The two carry the same fields while a
// subtree can still be traced recursively; only their owners differ.
PathVertex QlFromRay(Payload p) {
    PathVertex v;
    v.radiance = p.radiance;
    v.isShadowed = p.isShadowed;
    v.depth = p.depth;
    v.rngState = p.rngState;
    v.heroLambda = p.heroLambda;
    v.primaryHitT = p.primaryHitT;
    v.bsdfPdf = p.bsdfPdf;
    v.primaryMaterialFlags = p.primaryMaterialFlags;
    v.fusionMediumCount = p.fusionMediumCount;
    v.fusionFlags = p.fusionFlags;
    v.fusionRoute = p.fusionRoute;
    v.fusionContributions = p.fusionContributions;
    v.fusionPathId = p.fusionPathId;
    v.fusionResidual = p.fusionResidual;
    v.fusionSegmentOffset = p.fusionSegmentOffset;
    v.fusionPreviousPosition = p.fusionPreviousPosition;
    v.fusionPreviousNormal = p.fusionPreviousNormal;
    v.fusionPreviousInstance = p.fusionPreviousInstance;
    v.fusionTerminalDepth = p.fusionTerminalDepth;
    v.fusionForced = p.fusionForced;
    v.fusionBranchMask = p.fusionBranchMask;
    [unroll] for (uint i = 0; i < kFusionMaxInitialMedia; ++i) v.fusionMedia[i] = p.fusionMedia[i];
    return v;
}
Payload QlToRay(PathVertex v) {
    Payload p;
    p.radiance = v.radiance;
    p.isShadowed = v.isShadowed;
    p.depth = v.depth;
    p.rngState = v.rngState;
    p.heroLambda = v.heroLambda;
    p.primaryHitT = v.primaryHitT;
    p.bsdfPdf = v.bsdfPdf;
    p.primaryMaterialFlags = v.primaryMaterialFlags;
    p.fusionMediumCount = v.fusionMediumCount;
    p.fusionFlags = v.fusionFlags;
    p.fusionRoute = v.fusionRoute;
    p.fusionContributions = v.fusionContributions;
    p.fusionPathId = v.fusionPathId;
    p.fusionResidual = v.fusionResidual;
    p.fusionSegmentOffset = v.fusionSegmentOffset;
    p.fusionPreviousPosition = v.fusionPreviousPosition;
    p.fusionPreviousNormal = v.fusionPreviousNormal;
    p.fusionPreviousInstance = v.fusionPreviousInstance;
    p.fusionTerminalDepth = v.fusionTerminalDepth;
    p.fusionForced = v.fusionForced;
    p.fusionBranchMask = v.fusionBranchMask;
    [unroll] for (uint i = 0; i < kFusionMaxInitialMedia; ++i) p.fusionMedia[i] = v.fusionMedia[i];
    return p;
}

#endif // QUANTILOOM_SHADE_TYPES_HLSLI
