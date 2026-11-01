// ============================================================================
// Quantiloom - structs shared by the fusion kernels and their hosts
// ============================================================================
// The fusion geometry capture (fusion_geometry.rayq.hlsl,
// RayTracingPipeline::CapturePrimaryGeometry), the fusion transport records
// (fusion_transport.hlsli, renderer/FusionTransport.hpp) and the fusion probe
// rays raygen traces (raygen.rgen, RayTracingPipeline::SetFusionProbes). Each is
// defined here once, for both languages. ql_shared_prelude.hlsli has the layout
// rules and how a shader asks for one struct (QL_SHARED_WANT_<Name>).
//
// No include guard: a shader includes this once per struct it asks for, and
// every section guards itself (QL_SHARED_HAS_<Name>).
// ============================================================================

#include "ql_shared_prelude.hlsli"

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_GeometryHit)) && !defined(QL_SHARED_HAS_GeometryHit)
#define QL_SHARED_HAS_GeometryHit
QL_SHARED_BEGIN
// One per queried pixel. The host defaults are a miss: distance and depth -1.
struct GeometryHit {
    uint hit QL_DEFAULT(0);
    uint instanceIndex QL_DEFAULT(0);
    uint primitiveIndex QL_DEFAULT(0);
    float distance QL_DEFAULT(-1.0f);
    float3 position QL_DEFAULT();
    float depth QL_DEFAULT(-1.0f);
    float3 normal QL_DEFAULT();
    uint validity QL_DEFAULT(0);
    float2 bary QL_DEFAULT();
    float2 padding QL_DEFAULT();
};
QL_SHARED_LAYOUT(GeometryHit, 64)
QL_SHARED_OFFSET(GeometryHit, hit, 0)
QL_SHARED_OFFSET(GeometryHit, instanceIndex, 4)
QL_SHARED_OFFSET(GeometryHit, primitiveIndex, 8)
QL_SHARED_OFFSET(GeometryHit, distance, 12)
QL_SHARED_OFFSET(GeometryHit, position, 16)
QL_SHARED_OFFSET(GeometryHit, depth, 28)
QL_SHARED_OFFSET(GeometryHit, normal, 32)
QL_SHARED_OFFSET(GeometryHit, validity, 44)
QL_SHARED_OFFSET(GeometryHit, bary, 48)
QL_SHARED_OFFSET(GeometryHit, padding, 56)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_FusionTransportData)) && !defined(QL_SHARED_HAS_FusionTransportData)
#define QL_SHARED_HAS_FusionTransportData
QL_SHARED_BEGIN
// One per material, plus record 0, the summary ResolveFusionTransport writes;
// the host calls it FusionTransportGpu.
struct FusionTransportData {
    uint mode QL_DEFAULT(0), nodeId QL_DEFAULT(0), materialId QL_DEFAULT(0), flags QL_DEFAULT(0);
    float absorptionPerMeter QL_DEFAULT(0.0f), sheetReflectance QL_DEFAULT(0.0f),
          sheetTransmittance QL_DEFAULT(0.0f), orientation QL_DEFAULT(1.0f);
};
QL_SHARED_LAYOUT(FusionTransportData, 32)
QL_SHARED_OFFSET(FusionTransportData, mode, 0)
QL_SHARED_OFFSET(FusionTransportData, nodeId, 4)
QL_SHARED_OFFSET(FusionTransportData, materialId, 8)
QL_SHARED_OFFSET(FusionTransportData, flags, 12)
QL_SHARED_OFFSET(FusionTransportData, absorptionPerMeter, 16)
QL_SHARED_OFFSET(FusionTransportData, sheetReflectance, 20)
QL_SHARED_OFFSET(FusionTransportData, sheetTransmittance, 24)
QL_SHARED_OFFSET(FusionTransportData, orientation, 28)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_FusionProbeRay)) && !defined(QL_SHARED_HAS_FusionProbeRay)
#define QL_SHARED_HAS_FusionProbeRay
QL_SHARED_BEGIN
// A probe ray raygen traces for the fusion export; the host calls it
// RayTracingPipeline::ProbeRay.
struct FusionProbeRay {
    float3 origin;
    float wavelength;
    float3 direction;
    uint branchMask;
};
QL_SHARED_LAYOUT(FusionProbeRay, 32)
QL_SHARED_OFFSET(FusionProbeRay, origin, 0)
QL_SHARED_OFFSET(FusionProbeRay, wavelength, 12)
QL_SHARED_OFFSET(FusionProbeRay, direction, 16)
QL_SHARED_OFFSET(FusionProbeRay, branchMask, 28)
QL_SHARED_END
#endif
