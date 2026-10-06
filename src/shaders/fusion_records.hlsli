#ifndef QL_FUSION_RECORDS_HLSLI
#define QL_FUSION_RECORDS_HLSLI

// Byte-addressed fusion record buffer. The host side of this layout lives in
// src/libQuantiloom/renderer/FusionRecordLayout.hpp and carries the same
// names and values -- change one side, change both. All Load/Store offsets
// below are bytes; the host indexes the same fields in u32 words.
[[vk::binding(36, 0)]] RWByteAddressBuffer fusionRecords;

// Header words (byte offset = word * 4).
static const uint kFusionEnabledWord = 0;          // recording armed
static const uint kFusionRayStrideWord = 1;        // one stored ray per stride roots
static const uint kFusionStoredRaysWord = 2;
static const uint kFusionSlotsPerRayWord = 3;      // vertex slots per stored ray
static const uint kFusionWidthWord = 4;
static const uint kFusionHeightWord = 5;
static const uint kFusionSppWord = 6;
static const uint kFusionDiagnosticFlagsWord = 7;  // OR of every ray's flag bits
static const uint kFusionInitialMediumCountWord = 8;
static const uint kFusionProbeCountWord = 10;      // nonzero: probe rays replace camera rays
static const uint kFusionPixelSectionWord = 11;    // byte offset of per-pixel accumulators
static const uint kFusionQuantitativeWord = 12;    // diagnostics on: flag, never clamp
static const uint kFusionTermsSectionWord = 14;    // byte offset of vertex radiance terms
static const uint kFusionCoordSectionWord = 15;    // byte offset of native-pixel columns
static const uint kFusionInitialMediaWords = 16;   // first of eight media words (16..23)

// Section strides.
static const uint kFusionHeaderWords = 32;
static const uint kFusionHeaderBytes = 128;
static const uint kFusionRayRecordBytes = 32;
static const uint kFusionVertexRecordBytes = 80;
static const uint kFusionPixelRecordBytes = 32;
static const uint kFusionCoordColumnBytes = 12;
static const uint kFusionTermColumnBytes = 16;
static const uint kFusionMaxSlotsPerRay = 9;
// kFusionMaxInitialMedia and kFusionNoPath are declared in common.hlsli:
// the payload carries them, so they must exist before this header.

// Ray record fields (within a 32-byte record).
static const uint kFusionRayRadianceOffset = 12;
static const uint kFusionRayIdentityOffset = 16;
static const uint kFusionRayFlagsOffset = 24;
static const uint kFusionRayTerminalOffset = 28;

// Vertex record fields (within an 80-byte record).
static const uint kFusionVertexNormalOffset = 16;     // normal xyz + segment distance
static const uint kFusionVertexIdentityOffset = 32;   // kind, nodeId, primitiveId, route
static const uint kFusionVertexNodeIdOffset = 36;
static const uint kFusionVertexPrimitiveIdOffset = 40;
static const uint kFusionVertexOutgoingOffset = 48;   // direction xyz + coefficient
static const uint kFusionVertexCoefficientOffset = 60;
static const uint kFusionVertexInterfaceOffset = 64;  // n1, n2
static const uint kFusionVertexSegmentOffset = 72;    // sigma, transmittance
static const uint kFusionVertexTransmittanceOffset = 76;

// Native-pixel column fields (within a 12-byte record).
static const uint kFusionCoordResidualOffset = 8;

// Pixel accumulator fields (within a 32-byte record).
static const uint kFusionPixelValidOffset = 20;
static const uint kFusionPixelFlagsOffset = 24;

// Diagnostic flag bits, ORed into per-pixel flags and the header's diagnostic
// word.
static const uint kFusionFlagUnknownTail = 1u;
static const uint kFusionFlagAmbiguousMedium = 2u;
static const uint kFusionFlagZeroBranch = 4u;
static const uint kFusionFlagUnprojectable = 8u;
static const uint kFusionFlagNonFinite = 16u;

// The ray record's terminal depth is marked valid by the top bit.
static const uint kFusionTerminalBit = 0x80000000u;
static const uint kFusionTerminalDepthMask = 0x7FFFFFFFu;

void FusionAccumulate(uint2 pixel, uint2 size, uint sample, float4 values,
                      float total, float valid, uint flags) {
    const uint base = fusionRecords.Load(kFusionPixelSectionWord * 4);
    if (base == 0)
        return;
    const uint address = base + (pixel.y * size.x + pixel.x) * kFusionPixelRecordBytes;
    const float weight = 1.0 / float(sample + 1);
    const float4 previous = sample == 0 ? 0 : asfloat(fusionRecords.Load4(address));
    const float2 old = sample == 0 ? 0 : asfloat(fusionRecords.Load2(address + 16));
    fusionRecords.Store4(address, asuint(previous + (values - previous) * weight));
    fusionRecords.Store2(address + 16, asuint(old + (float2(total, valid) - old) * weight));
    const uint prior = fusionRecords.Load(address + kFusionPixelFlagsOffset);
    fusionRecords.Store(address + kFusionPixelFlagsOffset, prior | flags);
    if (flags != 0) {
        uint unused;
        fusionRecords.InterlockedOr(kFusionDiagnosticFlagsWord * 4, flags, unused);
    }
}

void FusionMarkUnknownTail() {
    const uint base = fusionRecords.Load(kFusionPixelSectionWord * 4);
    if (base == 0)
        return;
    const uint2 pixel = DispatchRaysIndex().xy, size = DispatchRaysDimensions().xy;
    uint unused;
    fusionRecords.InterlockedOr(
        base + (pixel.y * size.x + pixel.x) * kFusionPixelRecordBytes + kFusionPixelFlagsOffset,
        kFusionFlagUnknownTail, unused);
    fusionRecords.InterlockedOr(kFusionDiagnosticFlagsWord * 4, kFusionFlagUnknownTail, unused);
}

uint FusionVertexAddress(Payload p) {
    const uint rays = fusionRecords.Load(kFusionStoredRaysWord * 4),
               slots = fusionRecords.Load(kFusionSlotsPerRayWord * 4);
    return kFusionHeaderBytes + rays * kFusionRayRecordBytes +
           (p.fusionPathId * slots + p.depth) * kFusionVertexRecordBytes;
}

void FusionRecordVertex(Payload p, float3 hit, float3 normal, uint nodeId,
                        uint primitiveId, uint kind) {
    if (p.fusionPathId == kFusionNoPath ||
        p.depth >= fusionRecords.Load(kFusionSlotsPerRayWord * 4))
        return;
    uint address = FusionVertexAddress(p);
    fusionRecords.Store4(address, asuint(float4(hit, p.heroLambda != 0 ? abs(p.heroLambda)
                                                                     : pushConsts.camera.wavelength_nm)));
    fusionRecords.Store4(address + kFusionVertexNormalOffset,
                         asuint(float4(normal, RayTCurrent() + p.fusionSegmentOffset)));
    fusionRecords.Store4(address + kFusionVertexIdentityOffset,
                       uint4(kind, nodeId, primitiveId, p.fusionRoute));
    fusionRecords.Store4(address + kFusionVertexInterfaceOffset, asuint(float4(1, 1, 0, 1)));
}

void FusionRecordInterface(Payload p, float3 outgoing, float weight, float n1, float n2) {
    if (p.fusionPathId == kFusionNoPath ||
        p.depth >= fusionRecords.Load(kFusionSlotsPerRayWord * 4))
        return;
    uint address = FusionVertexAddress(p);
    fusionRecords.Store4(address + kFusionVertexOutgoingOffset,
                         asuint(float4(outgoing, weight)));
    fusionRecords.Store2(address + kFusionVertexInterfaceOffset, asuint(float2(n1, n2)));
}

void FusionRecordSegment(Payload p, float sigma, float transmittance) {
    if (p.fusionPathId == kFusionNoPath ||
        p.depth >= fusionRecords.Load(kFusionSlotsPerRayWord * 4))
        return;
    fusionRecords.Store2(FusionVertexAddress(p) + kFusionVertexSegmentOffset,
                         asuint(float2(sigma, transmittance)));
}

void FusionRecordRadianceTerm(Payload p, uint term, float value) {
    const uint base = fusionRecords.Load(kFusionTermsSectionWord * 4);
    if (base == 0 || p.fusionPathId == kFusionNoPath ||
        p.depth >= fusionRecords.Load(kFusionSlotsPerRayWord * 4))
        return;
    fusionRecords.Store(
        base +
            (p.fusionPathId * fusionRecords.Load(kFusionSlotsPerRayWord * 4) + p.depth) *
                kFusionTermColumnBytes +
            term * 4,
        asuint(value));
}

/// Raise the nonfinite diagnostic in the header's flag word. Quantitative
/// captures want the record to say the radiance exploded; ordinary renders
/// clamp instead and need no flag.
void FusionFlagNonFinite() {
    if (fusionRecords.Load(kFusionQuantitativeWord * 4) == 0)
        return;
    uint unused;
    fusionRecords.InterlockedOr(kFusionDiagnosticFlagsWord * 4, kFusionFlagNonFinite, unused);
}

void FusionRecordSky(Payload p) {
    if (p.fusionPathId == kFusionNoPath ||
        p.depth >= fusionRecords.Load(kFusionSlotsPerRayWord * 4))
        return;
    fusionRecords.Store4(FusionVertexAddress(p) + kFusionVertexInterfaceOffset,
                         asuint(float4(1, 1, 0, 1)));
    FusionRecordRadianceTerm(p, 0, p.radiance.x);
}
#endif
