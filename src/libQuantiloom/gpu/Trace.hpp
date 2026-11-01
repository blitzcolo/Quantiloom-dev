/**
 * @file Trace.hpp
 * @brief Trace kernels and pipeline caches of the RHI (internal)
 *
 * A TraceKernel is what CommandList::SetTraceKernel binds and CommandList::Trace launches.
 * On Vulkan it is a ray tracing pipeline of five stages in four groups, and it owns its shader
 * binding table: the SBT never appears in a gpu/ header. On Metal there is no SBT at all: every
 * ray is traced inline from the ray generation shader (src/shaders/ql_trace.hlsli), so the
 * kernel is the Metal Shader Converter's compute kernel for the ray generation shader alone,
 * dispatched over the launch grid, and the closest-hit, any-hit and miss code goes unused.
 */

#pragma once

#include "gpu/Kernel.hpp"

#include <span>

namespace quantiloom::gpu {

/// Pipeline-creation state kept across runs. Vulkan: a pipeline cache object. Metal:
/// MTL::BinaryArchive, which records only the pipelines it is told to add; Serialize is
/// serializeToURL.
class PipelineCache {
public:
    virtual ~PipelineCache() = default;  // vkDestroyPipelineCache
    /// The bytes the next run passes to Device::CreatePipelineCache: two vkGetPipelineCacheData
    /// calls, size then data. Err when either fails or the size is 0.
    [[nodiscard]] virtual Result<Vector<std::byte>, String> Serialize() const = 0;
};

struct TraceKernelDesc {
    /// Vulkan needs all five, Metal reads rayGen only. Groups: ray generation (0), closest hit
    /// with any hit as one triangles hit group (1), miss (2), shadow miss (3) -- one hit record
    /// and two miss records, so a TraceRay's sbtRecordOffset and missIndex keep their meaning.
    const ShaderCode* rayGen = nullptr;
    const ShaderCode* closestHit = nullptr;
    const ShaderCode* anyHit = nullptr;
    const ShaderCode* miss = nullptr;
    const ShaderCode* shadowMiss = nullptr;
    const KernelLayout* layout = nullptr;
    /// QL_SPEC_CONSTANT values, laid out as for ComputeKernelDesc: 4 bytes each, in order.
    std::span<const SpecializationConstant> specialization = {};
    /// The stages the constants reach. Miss means the primary miss; the shadow miss takes none.
    ShaderStages specializedStages = ShaderStages::RayGen | ShaderStages::ClosestHit | ShaderStages::Miss;
    u32 maxRecursionDepth = 1;  // the pipeline's declared recursion limit; Metal ignores it
    const PipelineCache* cache = nullptr;  // optional
    /// What Name() returns and the creation log lines print; empty means rayGen's name.
    StringView name = {};
};

class TraceKernel {
public:
    virtual ~TraceKernel() = default;  // the pipeline, then its SBT buffer (Vulkan)
    [[nodiscard]] virtual const KernelLayout& Layout() const = 0;
    [[nodiscard]] virtual StringView Name() const = 0;
};

}  // namespace quantiloom::gpu
