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
// no-path value an ordinary child carries (InheritFusionMedium).
bool QlVisible(RayDesc ray) {
    Payload p = (Payload)0;
    p.isShadowed = 1;   // cleared by shadow_miss
    p.fusionPreviousInstance = kFusionNoPath;
    TraceRay(scene,
             RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 0, 0, 1, ray, p);
    return p.isShadowed == 0;
}

// A child's vertex as its parent hands it over: the request's fields, and the
// medium the parent vertex is in.
PathVertex QlChildVertex(ChildRequest req, PathVertex parent) {
    PathVertex child = (PathVertex)0;
    child.radiance    = float4(0.0, 0.0, 0.0, 0.0);
    child.isShadowed  = 0;
    child.depth       = req.depth;
    child.rngState    = req.rng;
    child.heroLambda  = req.heroLambda;
    child.primaryHitT = -1.0;
    child.bsdfPdf     = req.bsdfPdf;
    InheritFusionMedium(child, parent);
    return child;
}

// Where a ray lands, and nothing more: closest-hit writes the hit record, miss
// writes hit.t = -1, and the caller shades the vertex. The payload carries in
// only what the any-hit stage reads: the vertex's fusionPrevious* fields, for
// the re-hit test at a fusion interface.
bool QlTraceRecord(RayDesc ray, uint rayFlags, PathVertex v, out HitRecord rec) {
    Payload p = (Payload)0;
    p.fusionPreviousInstance = v.fusionPreviousInstance;
    p.fusionPreviousPosition = v.fusionPreviousPosition;
    p.fusionPreviousNormal   = v.fusionPreviousNormal;
    TraceRay(scene, rayFlags, 0xFF, 0, 0, 0, ray, p);
    rec = p.hit;
    return rec.t >= 0.0;
}

#endif // QUANTILOOM_QL_TRACE_HLSLI
