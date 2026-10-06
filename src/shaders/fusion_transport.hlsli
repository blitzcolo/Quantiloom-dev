#ifndef QL_FUSION_TRANSPORT_HLSLI
#define QL_FUSION_TRANSPORT_HLSLI
struct FusionTransportData {
    uint mode, nodeId, materialId, flags;
    float absorptionPerMeter, sheetReflectance, sheetTransmittance, orientation;
};
[[vk::binding(35, 0)]] StructuredBuffer<FusionTransportData> fusionTransport;

#ifndef QL_FUSION_DATA_ONLY
#include "fusion_records.hlsli"
bool FusionScalarTransport() {
    return fusionTransport[0].mode != 0 &&
           (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_SINGLE ||
            SPEC_SPECTRAL_MODE == SPECTRAL_MODE_CAMERA_MEASUREMENT);
}
float FusionLambda(Payload p) {
    return p.heroLambda != 0 ? abs(p.heroLambda) : pushConsts.camera.wavelength_nm;
}
void FusionAttenuateSegment(inout Payload p, float distance) {
    if (p.fusionMediumCount == 0 || !FusionScalarTransport())
        return;
    const FusionTransportData medium = fusionTransport[p.fusionMedia[p.fusionMediumCount - 1]];
    const MaterialData material = materials[medium.materialId];
    const float lambda = FusionLambda(p);
    const float n = RefractionIOR(material, lambda);
    float sigma = medium.absorptionPerMeter;
    if ((medium.flags & 1) != 0 && material.complexRefractiveIndexIndex >= 0)
        sigma = 4 * PI *
                SampleComplexRefractiveIndex(complexRefractiveIndices,
                                             material.complexRefractiveIndexIndex, lambda)
                    .y /
                (lambda * 1e-9);
    const float transmittance =
        exp(-sigma * (distance + p.fusionSegmentOffset) * lightingParams[0].worldUnitsToMeters);
    FusionRecordSegment(p, sigma, transmittance);
    const float emission = (1 - transmittance) * n * n * IRPlanckRadiance(material.irTemperature_K, lambda);
    FusionRecordRadianceTerm(p, 1, emission);
    p.radiance = p.radiance * transmittance + float4(emission, emission, emission, 0);
    p.fusionContributions = p.fusionContributions * transmittance + FusionClassify(emission, p.fusionRoute);
    p.fusionResidual *= transmittance;
}
bool TraceFusionInterface(inout Payload p, MaterialData material, uint recordIndex,
                          float3 hit, float3 faceNormal, bool backFace, float temperature) {
    if (!FusionScalarTransport())
        return false;
    const FusionTransportData interfaceData = fusionTransport[recordIndex];
    if (interfaceData.mode == 0)
        return false;
    p.fusionTerminalDepth = p.depth;
    const float lambda = FusionLambda(p);
    float rho = interfaceData.sheetReflectance, tau = interfaceData.sheetTransmittance;
    float selfEmission = 0;
    float n1 = 1, n2 = 1;
    if (p.fusionMediumCount > 0)
        n1 = RefractionIOR(
            materials[fusionTransport[p.fusionMedia[p.fusionMediumCount - 1]].materialId], lambda);
    Payload child = p;
    child.radiance = 0;
    child.fusionContributions = 0;
    child.fusionResidual = 0;
    child.primaryMaterialFlags = 0;
    child.depth = p.depth + 1;
    child.bsdfPdf = 0;
    float3 transmitted = WorldRayDirection();
    const bool entering = interfaceData.orientation > 0 ? !backFace : backFace;
    if (interfaceData.mode == 2) {
        if (entering) {
            if (p.fusionMediumCount >= kFusionMaxInitialMedia) {
                p.fusionFlags |= kFusionFlagAmbiguousMedium;
                p.radiance = 0;
                return true;
            }
            n2 = RefractionIOR(material, lambda);
            child.fusionMedia[child.fusionMediumCount++] = recordIndex;
        } else {
            if (p.fusionMediumCount == 0 ||
                fusionTransport[p.fusionMedia[p.fusionMediumCount - 1]].nodeId !=
                    interfaceData.nodeId) {
                p.fusionFlags |= kFusionFlagAmbiguousMedium;
                p.radiance = 0;
                return true;
            }
            --child.fusionMediumCount;
            n2 = 1;
            if (child.fusionMediumCount > 0) {
                const uint topMedium = child.fusionMedia[child.fusionMediumCount - 1];
                n2 = RefractionIOR(materials[fusionTransport[topMedium].materialId], lambda);
            }
        }
        transmitted = Refract(WorldRayDirection(), faceNormal, n1 / n2);
        rho = FresnelDielectric(abs(dot(faceNormal, -WorldRayDirection())), n1, n2);
        if (dot(transmitted, transmitted) < 1e-8)
            rho = 1;
        tau = 1 - rho;
    } else {
        if ((interfaceData.flags & 2) == 0 && material.spectralReflectanceCurveIndex >= 0)
            rho = EvaluateSpectralCurve(spectralCurves, material.spectralReflectanceCurveIndex, lambda);
        if ((interfaceData.flags & 4) == 0 && material.irTransmittanceCurveIndex >= 0)
            tau = EvaluateSpectralCurve(spectralCurves, material.irTransmittanceCurveIndex, lambda);
        if (rho < 0 || tau < 0 || rho + tau > 1.00001) {
            p.fusionFlags |= kFusionFlagAmbiguousMedium;
            p.radiance = 0;
            return true;
        }
        selfEmission = max(0, 1 - rho - tau) * n1 * n1 * IRPlanckRadiance(temperature, lambda) +
                       BoundEmissionRadiance(spectralCurves, material, material.emissiveFactor, 1, lambda);
    }
    const float total = rho + tau;
    FusionRecordInterface(p, 0, 0, n1, n2);
    FusionRecordRadianceTerm(p, 0, selfEmission);
    if (total > 0 && p.depth < MAX_PATH_DEPTH) {
        p.rngState = p.rngState * 747796405u + 2891336453u;
        uint word = ((p.rngState >> ((p.rngState >> 28u) + 4u)) ^ p.rngState) * 277803737u;
        word = (word >> 22u) ^ word;
        const float coin = min(float(word) / 4294967296.0, 0.99999994);
        const bool reflected =
            p.fusionForced != 0 ? ((p.fusionBranchMask >> p.depth) & 1) != 0 : coin < rho / total;
        if ((reflected ? rho : tau) <= 0) {
            p.fusionFlags |= kFusionFlagZeroBranch;
            p.radiance = 0;
            p.fusionContributions = 0;
            return true;
        }
        if (reflected) {
            child.fusionMediumCount = p.fusionMediumCount;
            [unroll] for (uint i = 0; i < kFusionMaxInitialMedia; ++i) child.fusionMedia[i] = p.fusionMedia[i];
        }
        if (p.depth == 0)
            child.fusionRoute = reflected ? 1 : 2;
        child.rngState = p.rngState;
        RayDesc ray;
        ray.Direction = reflected ? reflect(WorldRayDirection(), faceNormal) : transmitted;
        // Keep the actual interface origin and travel length. Any-hit rejects
        // only numerical re-hits on this same oriented boundary, so grazing
        // rays cannot jump across another nearby interface.
        ray.Origin = hit;
        ray.TMin = 0;
        ray.TMax = 10000;
        child.fusionSegmentOffset = 0;
        child.fusionPreviousPosition = hit;
        child.fusionPreviousNormal = backFace ? -faceNormal : faceNormal;
        child.fusionPreviousInstance = InstanceIndex();
        const FusionMediumContext context = SaveFusionMedium();
        TraceRay(scene, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, 0, 0, 0, ray, child);
        RestoreFusionMedium(context);
        const float weight =
            (p.fusionForced != 0 ? (reflected ? rho : tau) : total) *
            (interfaceData.mode == 2 && !reflected ? n1 * n1 / (n2 * n2) : 1);
        const float coefficient =
            (reflected ? rho : tau) * (interfaceData.mode == 2 && !reflected ? n1 * n1 / (n2 * n2) : 1);
        FusionRecordRadianceTerm(p, 2, weight);
        FusionRecordInterface(p, ray.Direction, coefficient, n1, n2);
        p.radiance = child.radiance * weight + float4(selfEmission, selfEmission, selfEmission, 0);
        p.fusionContributions =
            child.fusionContributions * weight + FusionClassify(selfEmission, p.fusionRoute);
        p.fusionResidual = child.fusionResidual * weight;
        p.fusionFlags |= child.fusionFlags;
        p.rngState = child.rngState;
        p.fusionTerminalDepth = child.fusionTerminalDepth;
    } else {
        p.radiance = float4(selfEmission, selfEmission, selfEmission, 0);
        p.fusionContributions = FusionClassify(selfEmission, p.fusionRoute);
        if (total > 0)
            p.fusionFlags |= kFusionFlagUnknownTail;
    }
    FusionAttenuateSegment(p, RayTCurrent());
    return true;
}
#endif
#endif
