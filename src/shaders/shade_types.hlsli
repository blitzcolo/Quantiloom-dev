#ifndef QUANTILOOM_SHADE_TYPES_HLSLI
#define QUANTILOOM_SHADE_TYPES_HLSLI

// HitRecord itself is declared in common.hlsli, ahead of Payload, which
// carries one back to a vertex that raygen shades.
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

// What a vertex asks the driver to trace next. The fields are exactly what the
// old child Payload was filled with before InheritFusionMedium ran over it, so a
// request carries no fusion state: the driver supplies the medium.
static const uint QL_CHILD_BOUNCE   = 0u;  // environment bounce, front lobe
static const uint QL_CHILD_BOUNCE_DT = 1u; // environment bounce, back lobe
static const uint QL_CHILD_HERO     = 2u;  // hero-split refraction or reflection
static const uint QL_CHILD_DISPERSE = 3u;  // one of the three RGB dispersion rays
static const uint QL_CHILD_REFRACT  = 4u;  // standard refraction or reflection

struct ChildRequest {
    RayDesc ray;
    uint    rayFlags;
    uint    kind;
    uint    depth;
    uint    rng;
    float   heroLambda;
    float   bsdfPdf;
};

// Per-vertex state <-> ray payload. The two carry the same fields while a
// subtree can still be traced recursively; only their owners differ. The
// payload's hit record and recordOnly have no vertex counterpart: QlToRay
// zeroes them, so a payload built from a vertex asks for full shading.
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
    Payload p = (Payload)0;
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
