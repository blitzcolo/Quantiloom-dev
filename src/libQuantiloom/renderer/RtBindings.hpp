/**
 * @file RtBindings.hpp
 * @brief The 38 bindings of the ray tracing descriptor set (space 0), expanded from the manifest
 *
 * RayTracingPipeline creates its binding layout from this table; nothing else
 * declares the set. No row is written here: each is an entry of the binding
 * manifest, src/shaders/ql_rt_bindings.hlsli, which documents the columns and
 * which the shaders read too. Every HLSL declaration of these resources takes its
 * register from the same entry, so the layout and the shaders cannot disagree on
 * a slot.
 *
 * This header gives the columns HLSL ignores their C++ meaning:
 *
 * - kind: the gpu::BindingKind the class and the HLSL type give (gpu/Binding.hpp).
 * - count: 1, or QL_BINDLESS for the two bindless arrays, which RayTracingPipeline
 *   replaces by the device's texture limit (1024 with descriptor indexing, 32 without).
 * - visibility: the QL_VIS_* stages below, the final Vulkan stage set. It folds in
 *   the two loops that used to run over the hand-built table (every binding a
 *   closest-hit or miss stage sees, raygen sees too, because raygen shades every
 *   path vertex itself; and every binding is visible to compute), so it is the
 *   stage mask the layout actually carries.
 * - partially bound: 1 for the bindless arrays and for the bindings only one of the
 *   three descriptor sets ever writes (camera-only 30/31, geometry-capture-only 33/34).
 *
 * and holds the manifest to what RayTracingPipeline relies on: 38 rows, row i is
 * binding i, and each row's register class is the one its kind is declared with.
 */

#pragma once

#include "gpu/Binding.hpp"

#include <iterator>

// By a path relative to this file, so no include directory is involved. The shader
// build depends on the same file through its *.hlsli glob (src/shaders/CMakeLists.txt),
// so an edit to it recompiles the shaders as well as this table.
#include "../../shaders/ql_rt_bindings.hlsli"

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
    gpu::BindingSlot{slot, gpu::BindingKind::kind, count, (partial) != 0, vis},
#define QL_RT_ROW(role) QL_RT_##role(QL_RT_SLOT_)

inline constexpr gpu::BindingSlot kRtBindingTemplate[] = {QL_RT_BINDINGS(QL_RT_ROW)};

#undef QL_RT_ROW
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

/// The HLSL register letter a kind is declared with (the table in gpu/Binding.hpp).
constexpr char RegisterClassOf(gpu::BindingKind kind) {
    switch (kind) {
        case gpu::BindingKind::ReadOnlyBuffer:
        case gpu::BindingKind::SampledTexture:
        case gpu::BindingKind::AccelerationStructure: return 't';
        case gpu::BindingKind::ReadWriteBuffer:
        case gpu::BindingKind::StorageTexture: return 'u';
        case gpu::BindingKind::Sampler: return 's';
        case gpu::BindingKind::ConstantBuffer: return 'b';
    }
    return '?';
}

// Only HLSL reads the class column, so check it here against the kind, entry by entry.
#define QL_RT_CLASS_(cls, slot, kind, count, vis, partial) #cls[0]
#define QL_RT_KIND_CLASS_(cls, slot, kind, count, vis, partial) RegisterClassOf(gpu::BindingKind::kind)
#define QL_RT_CHECK_CLASS(role)                                                  \
    static_assert(QL_RT_##role(QL_RT_CLASS_) == QL_RT_##role(QL_RT_KIND_CLASS_), \
                  "QL_RT_" #role ": the register class does not fit the kind");
QL_RT_BINDINGS(QL_RT_CHECK_CLASS)
#undef QL_RT_CHECK_CLASS
#undef QL_RT_KIND_CLASS_
#undef QL_RT_CLASS_

}  // namespace quantiloom::rendercore
