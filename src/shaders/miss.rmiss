// ============================================================================
// Quantiloom - Miss Shader
// ============================================================================
// The sky evaluation lives in shade_miss.hlsli; this stage only supplies the
// ray. In a miss shader RayTCurrent() is the ray's TMax, which is what the
// atmosphere's view-path length is measured from.
// ============================================================================

#include "shade_miss.hlsli"

[shader("miss")]
void main(inout Payload payload) {
    RayDesc ray;
    ray.Origin = WorldRayOrigin();
    ray.Direction = WorldRayDirection();
    ray.TMin = RayTMin();
    ray.TMax = RayTCurrent();

    PathVertex v = QlFromRay(payload);
    ShadeMiss(v, ray);
    payload = QlToRay(v);
}
