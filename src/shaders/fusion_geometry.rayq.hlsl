#include "common.hlsli"
#include "hit_common.hlsli"
#include "camera_projection.hlsli"
#define QL_FUSION_DATA_ONLY
#include "fusion_transport.hlsli"

[[vk::binding(1, 0)]] RaytracingAccelerationStructure geometryScene;
[[vk::binding(3, 0)]] ByteAddressBuffer geometryVertices;
struct GeometryHit {
    uint hit;
    uint instanceIndex;
    uint primitiveIndex;
    float distance;
    float3 position;
    float depth;
    float3 normal;
    uint validity;
    float2 bary;
    float2 padding;
};
[[vk::binding(33, 0)]] RWStructuredBuffer<GeometryHit> geometryHits;
[[vk::binding(34, 0)]] StructuredBuffer<float2> geometryQueryPixels;

[numthreads(8, 8, 1)]
void main(uint3 dispatch : SV_DispatchThreadID) {
    uint width = pushConsts.sampleIndex, height = pushConsts.totalSamples;
    if (dispatch.x >= width || dispatch.y >= height)
        return;
    uint index = dispatch.y * width + dispatch.x;
    GeometryHit result = (GeometryHit)0;
    result.distance = -1;
    result.depth = -1;
    float2 pixel = float2(dispatch.xy) + 0.5;
    if (pushConsts.frameIndex != 0)
        pixel = geometryQueryPixels[index];
    float2 ndc = pixel / float2(width, height) * 2 - 1;
    float3 origin = pushConsts.camera.origin;
    float3 direction;
    if (pushConsts.camera.projection == CAMERA_PROJECTION_ORTHOGRAPHIC) {
        origin += pushConsts.camera.right * ndc.x * pushConsts.camera.orthoHeight * 0.5 *
                      pushConsts.camera.aspectRatio -
                  pushConsts.camera.up * ndc.y * pushConsts.camera.orthoHeight * 0.5;
        direction = normalize(pushConsts.camera.forward);
    } else if (projectionConfig[0].x != 0) {
        float3 local;
        if (!NativePixelDirection(pixel, local)) {
            result.validity = 2;
            geometryHits[index] = result;
            return;
        }
        direction = normalize(local.x * pushConsts.camera.right - local.y * pushConsts.camera.up +
                              local.z * pushConsts.camera.forward);
    } else {
        direction = normalize(pushConsts.camera.forward +
                              ndc.x * pushConsts.camera.right * pushConsts.camera.fovScale *
                                  pushConsts.camera.aspectRatio -
                              ndc.y * pushConsts.camera.up * pushConsts.camera.fovScale);
    }
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = 10000;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(geometryScene, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
            continue;
        const InstanceGeometryInfo geo = instanceGeometryInfo[query.CandidateInstanceIndex()];
        const MaterialData material = materials[geo.materialId];
        float2 b = query.CandidateTriangleBarycentrics();
        uint base = geo.indexOffset + query.CandidatePrimitiveIndex() * 3;
        float2 uv = uvBuffer[geo.uvOffset + indexBuffer[base]] * (1 - b.x - b.y) +
                    uvBuffer[geo.uvOffset + indexBuffer[base + 1]] * b.x +
                    uvBuffer[geo.uvOffset + indexBuffer[base + 2]] * b.y;
        if (material.alphaMode == ALPHA_MODE_MASK && SurfaceAlpha(material, uv) < material.alphaCutoff)
            continue;
        query.CommitNonOpaqueTriangleHit();
    }
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        result.hit = 1;
        result.instanceIndex = query.CommittedInstanceIndex();
        result.primitiveIndex = query.CommittedPrimitiveIndex();
        result.distance = query.CommittedRayT();
        result.position = origin + direction * result.distance;
        result.depth = dot(result.position - pushConsts.camera.origin, normalize(pushConsts.camera.forward));
        result.bary = query.CommittedTriangleBarycentrics();
        const InstanceGeometryInfo geo = instanceGeometryInfo[result.instanceIndex];
        uint base = geo.indexOffset + result.primitiveIndex * 3;
        float3 p0 = asfloat(geometryVertices.Load3((geo.vertexOffset + indexBuffer[base]) * 12));
        float3 p1 = asfloat(geometryVertices.Load3((geo.vertexOffset + indexBuffer[base + 1]) * 12));
        float3 p2 = asfloat(geometryVertices.Load3((geo.vertexOffset + indexBuffer[base + 2]) * 12));
        float3 w0 = mul(query.CommittedObjectToWorld3x4(), float4(p0, 1));
        float3 w1 = mul(query.CommittedObjectToWorld3x4(), float4(p1, 1));
        float3 w2 = mul(query.CommittedObjectToWorld3x4(), float4(p2, 1));
        result.normal = normalize(cross(w1 - w0, w2 - w0));
        const MaterialData material = materials[geo.materialId];
        result.validity = material.alphaMode == ALPHA_MODE_BLEND
                              ? 3
                              : (material.transmission > 0 || material.irTransmittance > 0 ||
                                         material.irTransmittanceCurveIndex >= 0
                                     ? 4
                                     : 1);
        if (fusionTransport[0].mode != 0) {
            if (fusionTransport[1 + result.instanceIndex].mode != 0)
                result.validity = 4;
        }
    }
    geometryHits[index] = result;
}
