#ifndef QUANTILOOM_QL_TRACE_HLSLI
#define QUANTILOOM_QL_TRACE_HLSLI

#include "shade_types.hlsli"
#include "hit_common.hlsli"

// The acceleration structure every ray in this pipeline is traced against.
// Kept here, and not in a shared resource header, so that a stage which never
// traces (the sky evaluator) does not acquire a binding it must then satisfy.
RaytracingAccelerationStructure sce// Positions, for the fusion re-hit test below (binding 3, the vertex buffer).
ByteAddressBuffer vertexBuffer : register(t3, space0);

// Every ray is traced inline (RayQuery) from raygen: a TraceRay would split
// raygen at each call and carry its live state -- the whole frame stack --
// across it. A non-opaque candidate is asked what anyhit.rahit asks, by the
// same arithmetic, and committed only where the surface is there: the fusion
// re-hit test, then MASK and BLEND coverage. Must stay equal to anyhit.rahit.
bool QlCandidatePresent(uint instanceIndex, uint primitiveIndex, float2 bary,
                        float3x3 worldToObject, float3 candidate, uint prevInstance,
                        float3 prevPosition, float3 prevNormal) {
    const InstanceGeometryInfo geoInfo = instanceGeometryInfo[instanceIndex];
    const MaterialData material = materials[geoInfo.materialId];
    if (prevInstance == instanceIndex) {
        const uint base = geoInfo.indexOffset + primitiveIndex * 3;
        const float3 p0 = asfloat(vertexBuffer.Load3((geoInfo.vertexOffset + indexBuffer[base]) * 12));
        const float3 p1 = asfloat(vertexBuffer.Load3((geoInfo.vertexOffset + indexBuffer[base + 1]) * 12));
        const float3 p2 = asfloat(vertexBuffer.Load3((geoInfo.vertexOffset + indexBuffer[base + 2]) * 12));
        const float3 normal = normalize(mul(cross(p1 - p0, p2 - p0), worldToObject));
        const float error = max(1e-5, 128 * 1.192092896e-7 *
                                          dot(abs(prevNormal), max(abs(prevPosition), 1.0)));
        if (all(abs(normal - prevNormal) < 1e-6) &&
            abs(dot(candidate - prevPosition, prevNormal)) <= error)
            return false;
    }
    if (material.alphaMode == ALPHA_MODE_OPAQUE)
        return true;
    const uint base = geoInfo.indexOffset + primitiveIndex * 3u;
    const float2 uv0 = uvBuffer[geoInfo.uvOffset + indexBuffer[base + 0u]];
    const float2 uv1 = uvBuffer[geoInfo.uvOffset + indexBuffer[base + 1u]];
    const float2 uv2 = uvBuffer[geoInfo.uvOffset + indexBuffer[base + 2u]];
    const float2 uv = uv0 * (1.0 - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;
    const float alpha = SurfaceAlpha(material, uv);
    if (material.alphaMode == ALPHA_MODE_MASK)
        return !(alpha < material.alphaCutoff);
    if (material.alphaMode == ALPHA_MODE_BLEND) {
        const uint2 pixel = DispatchRaysIndex().xy;
        const uint seed = pcg_hash(pixel.x * 73856093u ^ pixel.y * 19349663u ^
                                   pushConsts.sampleIndex * 83492791u ^
                                   instanceIndex * 2654435761u ^ primitiveIndex * 40503u);
        const float xi = float(seed) * (1.0 / 4294967296.0);
        return !(xi >= alpha);
    }
    return true;
}

// A visibility ray: true when nothing stands between the origin and TMax. Every
// shadow, sun-correction, light-sampling and medium-midpoint test in the shading
// library goes through here. It carries no interface behind it, as the
// TraceRay version's payload did not.
bool QlVisible(RayDesc ray) {
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    [loop] while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            if (QlCandidatePresent(q.CandidateInstanceIndex(), q.CandidatePrimitiveIndex(),
                                   q.CandidateTriangleBarycentrics(),
                                   (float3x3)q.CandidateWorldToObject3x4(),
                                   q.WorldRayOrigin() + q.WorldRayDirection() * q.CandidateTriangleRayT(),
                                   kFusionNoPath, float3(0, 0, 0), float3(0, 0, 0)))
                q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

eturn p.isShadowed == 0;
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
    InheritFusionMedium(child, pa// Where a ray lands, and nothing more: the closest committed triangle as a
// HitRecord, read off the query exactly as closest-hit read it off the
// intrinsics, or t = -1 for a miss. The vertex's fusionPrevious* fields feed
// the re-hit test, as the payload fed the any-hit stage.
bool QlTraceRecord(RayDesc ray, uint rayFlags, PathVertex v, out HitRecord rec) {
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(scene, rayFlags, 0xFF, ray);
    [loop] while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            if (QlCandidatePresent(q.CandidateInstanceIndex(), q.CandidatePrimitiveIndex(),
                                   q.CandidateTriangleBarycentrics(),
                                   (float3x3)q.CandidateWorldToObject3x4(),
                                   q.WorldRayOrigin() + q.WorldRayDirection() * q.CandidateTriangleRayT(),
                                   v.fusionPreviousInstance, v.fusionPreviousPosition,
                                   v.fusionPreviousNormal))
                q.CommitNonOpaqueTriangleHit();
        }
    }
    rec = (HitRecord)0;
    rec.t = -1.0;
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        rec.t = q.CommittedRayT();
        rec.bary = q.CommittedTriangleBarycentrics();
        rec.instanceIndex = q.CommittedInstanceIndex();
        rec.primitiveIndex = q.CommittedPrimitiveIndex();
        const float3x3 o2w = (float3x3)q.CommittedObjectToWorld3x4();
        const float3x3 w2o = (float3x3)q.CommittedWorldToObject3x4();
        rec.o2w0 = o2w[0]; rec.o2w1 = o2w[1]; rec.o2w2 = o2w[2];
        rec.w2o0 = w2o[0]; rec.w2o1 = w2o[1]; rec.w2o2 = w2o[2];
    }
    return rec.t >= 0.0;
}

    return rec.t >= 0.0;
}

#endif // QUANTILOOM_QL_TRACE_HLSLI
