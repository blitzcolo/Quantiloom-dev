#ifndef QUANTILOOM_QL_TRACE_HLSLI
#define QUANTILOOM_QL_TRACE_HLSLI

#include "shade_types.hlsli"

// The acceleration structure every ray in this pipeline is traced against.
// Kept here, and not in a shared resource header, so that a stage which never
// traces (the sky evaluator) does not acquire a binding it must then satisfy.
RaytracingAccelerationStructure scene : register(t1, space0);

// A visibility ray: true when nothing stands between the origin and TMax. Every
// shadow, sun-correction, light-sampling and medium-midpoint test in the shading
// library goes through here. Only isShadowed is read back (shadow_miss clears
// it), and the any-hit only looks at fusionPreviousInstance, which is the
// no-path value InheritFusionMedium gave these rays.
bool QlVisible(RayDesc ray) {
    Payload p = (Payload)0;
    p.isShadowed = 1;   // cleared by shadow_miss
    p.fusionPreviousInstance = kFusionNoPath;
    TraceRay(scene,
             RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 0, 0, 1, ray, p);
    return p.isShadowed == 0;
}

// The one site that traces a radiance ray for a vertex's child. The statics
// fusionActiveMedia* hold the vertex's entry medium here, as they did at every
// TraceRay of the old recursive body.
PathVertex QlTraceRadiance(ChildRequest req) {
    Payload child = (Payload)0;
    child.radiance    = float4(0.0, 0.0, 0.0, 0.0);
    child.isShadowed  = 0;
    child.depth       = req.depth;
    child.rngState    = req.rng;
    child.heroLambda  = req.heroLambda;
    child.primaryHitT = -1.0;
    child.bsdfPdf     = req.bsdfPdf;
    const FusionMediumContext context = SaveFusionMedium();
    InheritFusionMedium(child);
    TraceRay(scene, req.rayFlags, 0xFF, 0, 0, 0, req.ray, child);
    RestoreFusionMedium(context);
    return QlFromRay(child);
}

#endif // QUANTILOOM_QL_TRACE_HLSLI
