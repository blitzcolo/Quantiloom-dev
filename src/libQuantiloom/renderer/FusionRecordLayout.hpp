#pragma once

#include "core/Types.hpp"

namespace quantiloom::rendercore {

/// The fusion record buffer: a byte-addressed blob written by raygen.rgen,
/// closesthit.rchit and miss.rmiss and read back by OfflineRenderer /
/// dataset::DecodeOpticalEndpoints / scripts/fusion_verify.py. The shader
/// side of this layout lives in src/shaders/fusion_records.hlsli, which must
/// carry the same names and values -- change one side, change both.

// 128-byte header, indexed in u32 words on the host and byte offsets in the
// shaders (word * 4).
inline constexpr u32 kFusionHeaderWords = 32;
inline constexpr u32 kFusionHeaderBytes = kFusionHeaderWords * 4;

// Header words.
inline constexpr u32 kFusionEnabledWord = 0;          // recording armed
inline constexpr u32 kFusionRayStrideWord = 1;        // one stored ray per stride roots
inline constexpr u32 kFusionStoredRaysWord = 2;
inline constexpr u32 kFusionSlotsPerRayWord = 3;      // vertex slots per stored ray
inline constexpr u32 kFusionWidthWord = 4;
inline constexpr u32 kFusionHeightWord = 5;
inline constexpr u32 kFusionSppWord = 6;
inline constexpr u32 kFusionDiagnosticFlagsWord = 7;  // OR of every ray's flag bits
inline constexpr u32 kFusionInitialMediumCountWord = 8;
inline constexpr u32 kFusionProbeCountWord = 10;      // nonzero: probe rays replace camera rays
inline constexpr u32 kFusionPixelSectionWord = 11;    // byte offset of per-pixel accumulators
inline constexpr u32 kFusionQuantitativeWord = 12;    // diagnostics on: flag, never clamp
inline constexpr u32 kFusionTermsSectionWord = 14;    // byte offset of vertex radiance terms
inline constexpr u32 kFusionCoordSectionWord = 15;    // byte offset of native-pixel columns
inline constexpr u32 kFusionInitialMediaWords = 16;   // first of eight media words (16..23)

// Section strides.
inline constexpr u32 kFusionRayRecordBytes = 32;      // per stored ray
inline constexpr u32 kFusionVertexRecordBytes = 80;   // per vertex slot
inline constexpr u32 kFusionPixelRecordBytes = 32;    // per pixel accumulator
inline constexpr u32 kFusionCoordColumnBytes = 12;    // native pixel (2f) + residual (f32)
inline constexpr u32 kFusionTermColumnBytes = 16;     // four f32 terms per vertex slot
inline constexpr u32 kFusionMaxSlotsPerRay = 9;
inline constexpr u32 kFusionMaxInitialMedia = 8;

// Ray record field offsets within a 32-byte record.
inline constexpr u32 kFusionRayRadianceOffset = 12;   // last of the value float4
inline constexpr u32 kFusionRayIdentityOffset = 16;   // pixel, sample, flags, terminal depth
inline constexpr u32 kFusionRayFlagsOffset = 24;
inline constexpr u32 kFusionRayTerminalOffset = 28;

// Vertex record field offsets within an 80-byte record.
inline constexpr u32 kFusionVertexNormalOffset = 16;      // normal xyz + segment distance
inline constexpr u32 kFusionVertexIdentityOffset = 32;    // kind, nodeId, primitiveId, route
inline constexpr u32 kFusionVertexNodeIdOffset = 36;
inline constexpr u32 kFusionVertexPrimitiveIdOffset = 40;
inline constexpr u32 kFusionVertexOutgoingOffset = 48;     // direction xyz + coefficient
inline constexpr u32 kFusionVertexCoefficientOffset = 60;  // outgoing branch weight
inline constexpr u32 kFusionVertexInterfaceOffset = 64;    // boundary n1, n2
inline constexpr u32 kFusionVertexSegmentOffset = 72;      // sigma, transmittance
inline constexpr u32 kFusionVertexTransmittanceOffset = 76;

// Native-pixel column field offsets within a 12-byte record.
inline constexpr u32 kFusionCoordResidualOffset = 8;

// Pixel accumulator field offsets within a 32-byte record.
inline constexpr u32 kFusionPixelValidOffset = 20;
inline constexpr u32 kFusionPixelFlagsOffset = 24;

// Diagnostic flag bits (payload.fusionFlags; ORed into the per-pixel flags
// word and the header's diagnostic word).
inline constexpr u32 kFusionFlagUnknownTail = 1u;      // path cut before a terminal vertex
inline constexpr u32 kFusionFlagAmbiguousMedium = 2u;  // medium stack mismatch or overflow
inline constexpr u32 kFusionFlagZeroBranch = 4u;       // selected branch carries no throughput
inline constexpr u32 kFusionFlagUnprojectable = 8u;    // pixel outside the native optics
inline constexpr u32 kFusionFlagNonFinite = 16u;       // radiance went NaN/Inf

// The ray record's terminal depth is marked valid by the top bit.
inline constexpr u32 kFusionTerminalBit = 0x80000000u;
inline constexpr u32 kFusionTerminalDepthMask = 0x7FFFFFFFu;

inline constexpr u32 kFusionNoPath = 0xFFFFFFFFu;  // payload.fusionPathId sentinel

static_assert(kFusionHeaderWords * 4 == kFusionHeaderBytes);

} // namespace quantiloom::rendercore
