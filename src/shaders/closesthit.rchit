// ============================================================================
// Quantiloom - Closest Hit Shader
// ============================================================================
// Reports where a ray landed, and nothing else: raygen shades every vertex of
// the path itself (path_driver.hlsli). What raygen cannot query on its own --
// the barycentrics, the instance and primitive, the hit distance and both
// instance transforms -- comes back in the payload's HitRecord. This stage
// traces nothing, which is what lets the pipeline's recursion depth be 1.
// ============================================================================

#include "common.hlsli"
#include "hit_common.hlsli"  // HitAttributes, shared with the any-hit stage

[shader("closesthit")]
void main(inout Payload payload, in HitAttributes attribs) {
    payload.hit.t = RayTCurrent();
    payload.hit.bary = attribs.bary;
    payload.hit.instanceIndex = InstanceIndex();
    payload.hit.primitiveIndex = PrimitiveIndex();
    payload.hit.reserved = 0u;
    const float3x3 o2w = (float3x3)ObjectToWorld3x4();
    const float3x3 w2o = (float3x3)WorldToObject3x4();
    payload.hit.o2w0 = o2w[0]; payload.hit.o2w1 = o2w[1]; payload.hit.o2w2 = o2w[2];
    payload.hit.w2o0 = w2o[0]; payload.hit.w2o1 = w2o[1]; payload.hit.w2o2 = w2o[2];
}
