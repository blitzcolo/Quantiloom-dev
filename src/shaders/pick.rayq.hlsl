// ============================================================================
// Quantiloom - Viewport Pick (inline ray query)
// ============================================================================
// One thread, one ray: traces the primary camera ray through the pixel the
// host asked about and reports what it hit. Drives click-to-select in the
// GUI via ExternalRenderContext::Pick.
//
// The ray MUST match raygen.rgen's primary ray for the same pixel -- same
// NDC mapping, same Y flip, same basis expression -- or the pick lands on a
// different surface than the one under the cursor. Pixel center, no jitter:
// a pick is a question about the pixel, not a Monte Carlo sample.
//
// Compiled as cs_6_5 (RayQuery needs SM 6.5); see the .rayq.hlsl rule in
// CMakeLists.txt. Deliberately does NOT include common.hlsli: that header
// declares the full RT pipeline's resources, which this tiny set does not
// bind.
// ============================================================================

[[vk::binding(0, 0)]] RaytracingAccelerationStructure sceneTlas;

struct PickResultGpu {
    uint  hit;             // 1 = committed triangle hit
    uint  instanceIndex;   // TLAS instance index (matches InstanceIndex())
    uint  primitiveIndex;  // Triangle index within the instance's geometry
    float hitT;            // Distance along the normalized ray
    float3 worldPosition;  // origin + direction * hitT
    uint  _pad;
};

[[vk::binding(1, 0)]] RWStructuredBuffer<PickResultGpu> result;

struct PickPushConstants {
    float3 origin;   float fovScale;     // tan(fovY / 2), same as CameraData
    float3 forward;  float aspectRatio;
    float3 right;    uint  pixelX;
    float3 up;       uint  pixelY;
    uint   width;
    uint   height;
    uint   _pad0;
    uint   _pad1;
};

[[vk::push_constant]] PickPushConstants pc;

[numthreads(1, 1, 1)]
void main() {
    // Pixel center -> NDC, Y flipped: byte-for-byte the raygen.rgen mapping
    float2 pixelCenter = float2(pc.pixelX, pc.pixelY) + 0.5;
    float2 uv = pixelCenter / float2(pc.width, pc.height);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float3 direction = normalize(
        pc.forward +
        ndc.x * pc.right * pc.fovScale * pc.aspectRatio +
        ndc.y * pc.up * pc.fovScale
    );

    RayDesc ray;
    ray.Origin = pc.origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    // FORCE_OPAQUE: a pick wants the nearest surface, alpha-tested or not
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed()) {
        // FORCE_OPAQUE commits every triangle; nothing to resolve here
    }

    PickResultGpu r;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        r.hit = 1;
        r.instanceIndex = query.CommittedInstanceIndex();
        r.primitiveIndex = query.CommittedPrimitiveIndex();
        r.hitT = query.CommittedRayT();
        r.worldPosition = pc.origin + direction * r.hitT;
    } else {
        r.hit = 0;
        r.instanceIndex = 0;
        r.primitiveIndex = 0;
        r.hitT = -1.0;
        r.worldPosition = float3(0.0, 0.0, 0.0);
    }
    r._pad = 0;
    result[0] = r;
}
