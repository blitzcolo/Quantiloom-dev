// ============================================================================
// Quantiloom - Closest Hit Shader
// ============================================================================
// The shading itself lives in shade_vertex.hlsli, callable from any stage; this
// stage only turns the traversal's intrinsics into a HitRecord and hands the
// ray's payload over as that vertex's state. A record-only ray belongs to a
// vertex that raygen shades (path_driver.hlsli): it gets the record and
// nothing else.
// ============================================================================

#include "shade_vertex.hlsli"

[shader("closesthit")]
void main(inout Payload payload, in HitAttributes attribs) {
    HitRecord rec;
    rec.t = RayTCurrent();
    rec.bary = attribs.bary;
    rec.instanceIndex = InstanceIndex();
    rec.primitiveIndex = PrimitiveIndex();
    rec.reserved = 0u;
    const float3x3 o2w = (float3x3)ObjectToWorld3x4();
    const float3x3 w2o = (float3x3)WorldToObject3x4();
    rec.o2w0 = o2w[0]; rec.o2w1 = o2w[1]; rec.o2w2 = o2w[2];
    rec.w2o0 = w2o[0]; rec.w2o1 = w2o[1]; rec.w2o2 = w2o[2];
    if (payload.recordOnly != 0u) {
        payload.hit = rec;
        return;
    }

    RayDesc ray;
    ray.Origin = WorldRayOrigin();
    ray.Direction = WorldRayDirection();
    ray.TMin = RayTMin();
    ray.TMax = rec.t;

    PathVertex v = QlFromRay(payload);
    ShadeVertex(rec, ray, v);
    payload = QlToRay(v);
}
