/**
 * @file RtBindings.hpp
 * @brief The 38 bindings of the ray tracing descriptor set (space 0), one row each
 *
 * RayTracingPipeline creates its binding layout from this table; nothing else
 * declares the set. A row is
 *
 *     QL_RT_SLOT_(class, slot, kind, count, visibility, partially bound)
 *
 * - class: the HLSL register letter of the declaration (t, u, s). Not used by C++.
 * - slot: register number, Vulkan binding and row index at once.
 * - kind: the gpu::BindingKind the class and the HLSL type give (gpu/Binding.hpp).
 * - count: 1, or QL_BINDLESS for the two bindless arrays, which RayTracingPipeline
 *   replaces by the device's texture limit (1024 with descriptor indexing, 32 without).
 * - visibility: the final Vulkan stage set. It folds in the two loops that used to run
 *   over the hand-built table (every binding a closest-hit or miss stage sees, raygen
 *   sees too, because raygen shades every path vertex itself; and every binding is
 *   visible to compute), so it is the stage mask the layout actually carries.
 * - partially bound: 1 for the bindless arrays and for the bindings only one of the
 *   three descriptor sets ever writes (camera-only 30/31, geometry-capture-only 33/34).
 *
 * The row format is the one the binding manifest (design P05 section 5) uses, so that
 * the HLSL register declarations and this table can later be expanded from one list.
 * Each row names its HLSL declaration and the RayTracingPipeline.cpp line (at 7a875be)
 * whose hand-filled layout binding it replaces.
 */

#pragma once

#include "gpu/Binding.hpp"

#include <iterator>

namespace quantiloom::rendercore {

// Visibility column: RGen, CHit, AHit, Miss, Compute.
inline constexpr gpu::ShaderStages QL_VIS_R = gpu::ShaderStages::RayGen;
inline constexpr gpu::ShaderStages QL_VIS_C = gpu::ShaderStages::ClosestHit;
inline constexpr gpu::ShaderStages QL_VIS_A = gpu::ShaderStages::AnyHit;
inline constexpr gpu::ShaderStages QL_VIS_M = gpu::ShaderStages::Miss;
inline constexpr gpu::ShaderStages QL_VIS_X = gpu::ShaderStages::Compute;

/// Count column of a bindless array; replaced before the layout is created.
inline constexpr u32 QL_BINDLESS = gpu::kUnboundedCount;

#define QL_RT_SLOT_(cls, slot, kind, count, vis, partial) \
    gpu::BindingSlot{slot, gpu::BindingKind::kind, count, (partial) != 0, vis}

inline constexpr gpu::BindingSlot kRtBindingTemplate[] = {
    //          cls slot kind                    count        visibility                                     partial    HLSL declaration                                     was RTP.cpp
    QL_RT_SLOT_(u,  0,  StorageTexture,        1,           QL_VIS_R | QL_VIS_X,                                0),  // outputImage                 raygen.rgen:24            :397
    QL_RT_SLOT_(t,  1,  AccelerationStructure, 1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // scene                       ql_trace.hlsli:10         :409
    QL_RT_SLOT_(t,  2,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X,          0),  // lightingParams              shade_resources.hlsli:6   :416
    QL_RT_SLOT_(t,  3,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X,          0),  // vertexBuffer                ql_trace.hlsli:13         :423
    QL_RT_SLOT_(t,  4,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X,          0),  // indexBuffer                 hit_common.hlsli:31       :430
    QL_RT_SLOT_(t,  5,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X,          0),  // materials                   hit_common.hlsli:32       :437
    QL_RT_SLOT_(t,  6,  SampledTexture,        QL_BINDLESS, QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X,          1),  // textures[]                  hit_common.hlsli:33       :446
    QL_RT_SLOT_(s,  7,  Sampler,               QL_BINDLESS, QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X,          1),  // samplers[]                  hit_common.hlsli:34       :453
    QL_RT_SLOT_(t,  8,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X,          0),  // uvBuffer                    hit_common.hlsli:35       :460
    QL_RT_SLOT_(t,  9,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // tangentBuffer               shade_vertex.hlsli:30     :467
    QL_RT_SLOT_(t,  10, SampledTexture,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // prefilteredEnvMap           shade_vertex.hlsli:179    :482
    QL_RT_SLOT_(t,  11, SampledTexture,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // brdfLUT                     shade_vertex.hlsli:180    :490
    QL_RT_SLOT_(s,  12, Sampler,               1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // iblSampler                  shade_vertex.hlsli:181    :498
    QL_RT_SLOT_(t,  13, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // spectralCurves              shade_vertex.hlsli:41     :512
    QL_RT_SLOT_(t,  14, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // complexRefractiveIndices    pbr.hlsli:37              :526
    QL_RT_SLOT_(t,  15, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X,          0),  // solarSpectralLUT            shade_resources.hlsli:7   :542
    QL_RT_SLOT_(t,  16, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // normalBuffer                shade_vertex.hlsli:31     :557
    QL_RT_SLOT_(t,  17, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X,          0),  // atmosNNHeader               shade_resources.hlsli:9   :571
    QL_RT_SLOT_(t,  18, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X,          0),  // instanceGeometryInfo        hit_common.hlsli:52       :586
    QL_RT_SLOT_(t,  19, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X,          0),  // cieCMF_LUT                  shade_resources.hlsli:11  :601
    QL_RT_SLOT_(t,  20, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X,          0),  // atmosNNData                 shade_resources.hlsli:10  :614
    QL_RT_SLOT_(s,  21, Sampler,               1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // envSampler                  shade_vertex.hlsli:186    :626
    QL_RT_SLOT_(u,  22, StorageTexture,        1,           QL_VIS_R | QL_VIS_X,                                0),  // depthImage                  raygen.rgen:42            :633
    QL_RT_SLOT_(t,  23, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // emissiveTriangles           shade_vertex.hlsli:83     :643
    QL_RT_SLOT_(t,  24, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // thermalTemperatures         shade_vertex.hlsli:89     :654
    QL_RT_SLOT_(t,  25, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X,          0),  // rgbToSpectrumTable          shade_resources.hlsli:12  :668
    QL_RT_SLOT_(t,  26, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // thermalSunResponse          shade_vertex.hlsli:97     :686
    QL_RT_SLOT_(t,  27, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // thermalParameterTangent     shade_vertex.hlsli:108    :709
    QL_RT_SLOT_(u,  28, StorageTexture,        1,           QL_VIS_R | QL_VIS_X,                                0),  // cameraMeasurementImage      raygen.rgen:29            :715
    QL_RT_SLOT_(t,  29, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X,          0),  // cameraResponseTable         shade_resources.hlsli:8   :721
    QL_RT_SLOT_(u,  30, ReadWriteBuffer,       1,           QL_VIS_R | QL_VIS_X,                                1),  // cameraDynamicCounters       raygen.rgen:37            :731
    QL_RT_SLOT_(u,  31, StorageTexture,        1,           QL_VIS_R | QL_VIS_X,                                1),  // cameraMeasurementDepthImage raygen.rgen:32            :737
    QL_RT_SLOT_(t,  32, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_X,                                0),  // projectionConfig            camera_projection.hlsli:5 :745
    QL_RT_SLOT_(u,  33, ReadWriteBuffer,       1,           QL_VIS_X,                                           1),  // geometryHits                fusion_geometry.rayq.hlsl:21 :749
    QL_RT_SLOT_(t,  34, ReadOnlyBuffer,        1,           QL_VIS_X,                                           1),  // geometryQueryPixels         fusion_geometry.rayq.hlsl:22 :753
    QL_RT_SLOT_(t,  35, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // fusionTransport             fusion_transport.hlsli:7  :755
    QL_RT_SLOT_(u,  36, ReadWriteBuffer,       1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X,          0),  // fusionRecords               fusion_records.hlsli:8    :758
    QL_RT_SLOT_(t,  37, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,                     0),  // fusionProbeRays             raygen.rgen:49            :761
};

#undef QL_RT_SLOT_

static_assert(std::size(kRtBindingTemplate) == 38, "the ray tracing set declares 38 bindings");

/// RayTracingPipeline looks a binding's kind up by its number, so row i is binding i.
constexpr bool RtBindingRowsAreTheirSlots() {
    for (u32 i = 0; i < std::size(kRtBindingTemplate); ++i) {
        if (kRtBindingTemplate[i].slot != i) return false;
    }
    return true;
}
static_assert(RtBindingRowsAreTheirSlots(), "kRtBindingTemplate rows must be in binding order, one per binding");

}  // namespace quantiloom::rendercore
