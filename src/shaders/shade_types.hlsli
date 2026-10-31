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

// What a vertex asks the driver to trace next. For the kinds Advance issues,
// the fields are exactly what the old child Payload was filled with before
// InheritFusionMedium ran over it, so a request carries no fusion state: the
// driver builds the child from it and supplies the medium (QlChildVertex). The
// fusion interface's child is the exception: BeginFusionInterface builds that
// whole vertex, medium included, and its request only carries the ray.
static const uint QL_CHILD_BOUNCE   = 0u;  // environment bounce, front lobe
static const uint QL_CHILD_BOUNCE_DT = 1u; // environment bounce, back lobe
static const uint QL_CHILD_HERO     = 2u;  // hero-split refraction or reflection
static const uint QL_CHILD_DISPERSE = 3u;  // one of the three RGB dispersion rays
static const uint QL_CHILD_REFRACT  = 4u;  // standard refraction or reflection
static const uint QL_CHILD_FUSION   = 5u;  // the one child of a fusion interface vertex

struct ChildRequest {
    RayDesc ray;
    uint    rayFlags;
    uint    kind;
    uint    depth;
    uint    rng;
    float   heroLambda;
    float   bsdfPdf;
};

#endif // QUANTILOOM_SHADE_TYPES_HLSLI
