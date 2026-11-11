#ifndef QL_FUSION_TRANSPORT_HLSLI
#define QL_FUSION_TRANSPORT_HLSLI

#include "ql_bind.hlsli"

// Shared with renderer/FusionTransport.hpp: ql_shared_fusion.hlsli.
#define QL_SHARED_WANT_FusionTransportData
#include "ql_shared_fusion.hlsli"
StructuredBuffer<FusionTransportData> fusionTransport : QL_RT_REGISTER(fusionTransport);

#ifndef QL_FUSION_DATA_ONLY
#include "fusion_records.hlsli"
bool FusionScalarTransport() {
    return fusionTransport[0].mode != 0 &&
           (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_SINGLE ||
            SPEC_SPECTRAL_MODE == SPECTRAL_MODE_CAMERA_MEASUREMENT);
}
float FusionLambda(PathVertex p) {
    return p.heroLambda != 0 ? abs(p.heroLambda) : pushConsts.camera.wavelength_nm;
}
void FusionAttenuateSegment(inout PathVertex p, float distance) {
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
// An interface vertex, cut at its one child the way ShadeVertexA cuts a surface
// vertex: BeginFusionInterface is everything up to the child, and
// CompleteFusionInterface is what followed it. What the second half reads that
// the first half formed rides in the vertex's frame as a FusionFold.
struct FusionFold {
    float3 direction;     // the child ray's direction, for the interface record
    float  weight;        // what the child's radiance and contributions scale by
    float  coefficient;   // the interface record's branch coefficient
    float  selfEmission;
    float  n1, n2;
    float  hitT;          // the segment this vertex attenuates over
};

static const uint QL_FUSION_NONE  = 0u;  // not an interface here: shade the surface
static const uint QL_FUSION_FINAL = 1u;  // answered without a child: p is final
static const uint QL_FUSION_CHILD = 2u;  // child, req and fold describe the one child

// Returns NONE without touching p where the old body returned false. For CHILD,
// child is the whole vertex the old body traced -- built from p, medium pushed or
// popped -- and req carries its ray; the driver traces it and hands the answer
// to CompleteFusionInterface.
uint BeginFusionInterface(inout PathVertex p, MaterialData material, uint recordIndex,
                          float3 hit, float3 faceNormal, bool backFace, float temperature,
                          RayDesc rayIn, float hitT, uint instanceIndex,
                          inout PathVertex child, inout ChildRequest req, out FusionFold fold) {
    fold = (FusionFold)0;
    if (!FusionScalarTransport())
        return QL_FUSION_NONE;
    const FusionTransportData interfaceData = fusionTransport[recordIndex];
    if (interfaceData.mode == 0)
        return QL_FUSION_NONE;
    p.fusionTerminalDepth = p.depth;
    const float lambda = FusionLambda(p);
    float rho = interfaceData.sheetReflectance, tau = interfaceData.sheetTransmittance;
    float selfEmission = 0;
    float n1 = 1, n2 = 1;
    if (p.fusionMediumCount > 0)
        n1 = RefractionIOR(
            materials[fusionTransport[p.fusionMedia[p.fusionMediumCount - 1]].materialId], lambda);
    child = p;
    child.radiance = 0;
    child.fusionContributions = 0;
    child.fusionResidual = 0;
    child.primaryMaterialFlags = 0;
    child.depth = p.depth + 1;
    child.bsdfPdf = 0;
    float3 transmitted = rayIn.Direction;
    const bool entering = interfaceData.orientation > 0 ? !backFace : backFace;
    if (interfaceData.mode == 2) {
        if (entering) {
            if (p.fusionMediumCount >= kFusionMaxInitialMedia) {
                p.fusionFlags |= kFusionFlagAmbiguousMedium;
                p.radiance = 0;
                return QL_FUSION_FINAL;
            }
            n2 = RefractionIOR(material, lambda);
            child.fusionMedia[child.fusionMediumCount++] = recordIndex;
        } else {
            if (p.fusionMediumCount == 0 ||
                fusionTransport[p.fusionMedia[p.fusionMediumCount - 1]].nodeId !=
                    interfaceData.nodeId) {
                p.fusionFlags |= kFusionFlagAmbiguousMedium;
                p.radiance = 0;
                return QL_FUSION_FINAL;
            }
            --child.fusionMediumCount;
            n2 = 1;
            if (child.fusionMediumCount > 0) {
                const uint topMedium = child.fusionMedia[child.fusionMediumCount - 1];
                n2 = RefractionIOR(materials[fusionTransport[topMedium].materialId], lambda);
            }
        }
        transmitted = Refract(rayIn.Direction, faceNormal, n1 / n2);
        rho = FresnelDielectric(abs(dot(faceNormal, -rayIn.Direction)), n1, n2);
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
            return QL_FUSION_FINAL;
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
            return QL_FUSION_FINAL;
        }
        if (reflected) {
            child.fusionMediumCount = p.fusionMediumCount;
            [unroll] for (uint i = 0; i < kFusionMaxInitialMedia; ++i) child.fusionMedia[i] = p.fusionMedia[i];
        }
        if (p.depth == 0)
            child.fusionRoute = reflected ? 1 : 2;
        child.rngState = p.rngState;
        RayDesc ray;
        ray.Direction = reflected ? reflect(rayIn.Direction, faceNormal) : transmitted;
        // Keep the actual interface origin and travel length. Any-hit rejects
        // only numerical re-hits on this same oriented boundary, so grazing
        // rays cannot jump across another nearby interface.
        ray.Origin = hit;
        ray.TMin = 0;
        ray.TMax = 10000;
        child.fusionSegmentOffset = 0;
        child.fusionPreviousPosition = hit;
        child.fusionPreviousNormal = backFace ? -faceNormal : faceNormal;
        child.fusionPreviousInstance = instanceIndex;
        // Formed here from values the child cannot change; the old body formed
        // them after its trace from the same values.
        fold.direction = ray.Direction;
        fold.weight =
            (p.fusionForced != 0 ? (reflected ? rho : tau) : total) *
            (interfaceData.mode == 2 && !reflected ? n1 * n1 / (n2 * n2) : 1);
        fold.coefficient =
            (reflected ? rho : tau) * (interfaceData.mode == 2 && !reflected ? n1 * n1 / (n2 * n2) : 1);
        fold.selfEmission = selfEmission;
        fold.n1 = n1;
        fold.n2 = n2;
        fold.hitT = hitT;
        req.ray = ray;
        req.rayFlags = RAY_FLAG_FORCE_NON_OPAQUE;
        req.kind = QL_CHILD_FUSION;
        req.depth = child.depth;
        req.rng = child.rngState;
        req.heroLambda = child.heroLambda;
        req.bsdfPdf = child.bsdfPdf;
        return QL_FUSION_CHILD;
    }
    p.radiance = float4(selfEmission, selfEmission, selfEmission, 0);
    p.fusionContributions = FusionClassify(selfEmission, p.fusionRoute);
    if (total > 0)
        p.fusionFlags |= kFusionFlagUnknownTail;
    FusionAttenuateSegment(p, hitT);
    return QL_FUSION_FINAL;
}

// The old body after its trace, in its order: the records, the radiance and
// contributions through the branch weight, the child's flags, stream and
// terminal depth, then the segment this vertex was reached over.
void CompleteFusionInterface(inout PathVertex p, FusionFold fold, PathVertex child) {
    FusionRecordRadianceTerm(p, 2, fold.weight);
    FusionRecordInterface(p, fold.direction, fold.coefficient, fold.n1, fold.n2);
    p.radiance = child.radiance * fold.weight +
                 float4(fold.selfEmission, fold.selfEmission, fold.selfEmission, 0);
    p.fusionContributions =
        child.fusionContributions * fold.weight + FusionClassify(fold.selfEmission, p.fusionRoute);
    p.fusionResidual = child.fusionResidual * fold.weight;
    p.fusionFlags |= child.fusionFlags;
    p.rngState = child.rngState;
    p.fusionTerminalDepth = child.fusionTerminalDepth;
    FusionAttenuateSegment(p, fold.hitT);
}
#endif
#endif
