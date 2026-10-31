// ============================================================================
// Quantiloom - Miss Shader
// ============================================================================
// The sky evaluation lives in shade_miss.hlsli; this stage only supplies the
// ray. In a miss shader RayTCurrent() is the ray's TMax, which is what the
// atmosphere's view-path length is measured from. A record-only ray belongs to
// a vertex that raygen shades (path_driver.hlsli), which evaluates the sky
// itself; it is only told that the ray missed.
// ============================================================================

#include "shade_miss.hlsli"

[shader("miss")]
void main(inout Payload payload) {
    if (payload.recordOnly != 0u) {
        payload.hit.t = -1.0;
        return;
    }

    RayDesc ray;
    ray.Origin = WorldRayOrigin();
    ray.Direction = WorldRayDirection();
    ray.TMin = RayTMin();
    ray.TMax = RayTCurrent();

    PathVertex v = QlFromRay(payload);
    ShadeMiss(v, ray);
    payload = QlToRay(v);
}
