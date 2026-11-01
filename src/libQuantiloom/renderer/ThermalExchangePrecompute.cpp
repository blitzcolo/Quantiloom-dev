/**
 * @file ThermalExchangePrecompute.cpp
 * @brief Who sees whom, on the GPU, once per scene
 */

#include "renderer/ThermalExchangePrecompute.hpp"

#include "core/Log.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/Device.hpp"
#include "renderer/GpuAlloc.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>


namespace quantiloom::rendercore {

namespace {

constexpr u32 kSkyRecord = 0xFFFFFFFFu;

/// The smallest distance a ray starts off its surface. The shader scales the
/// offset with each element's own size (sqrt(area) * 1e-3) and uses this only
/// as the floor, so a sliver in a metre-scale scene is not offset by the mean
/// triangle's size.
constexpr f32 kRayOffsetFloor = 1e-5f;

/// Mirrors ThermalElementGpu in thermal_exchange.rayq.hlsl. `centre`, not
/// `centroid`: the latter is an HLSL interpolation modifier and will not parse
/// as a member name, which is worth knowing before renaming it back.
struct ThermalElementGpu {
    glm::vec3 centre;
    f32 area;
    glm::vec3 normal;
    u32 materialId;
};
static_assert(sizeof(ThermalElementGpu) == 32, "ThermalElementGpu size mismatch");

struct ExchangePushConstants {
    glm::vec3 sunDirection;
    u32 elementCount;
    u32 rayCount;
    u32 sunRayCount;
    f32 sunAngularRadius;
    f32 rayOffset;
    u32 sunOutputOffset;
};
static_assert(sizeof(ExchangePushConstants) == 36, "ExchangePushConstants size mismatch");

/// Make the view factors reciprocal, and let the sky take what is left.
///
/// Each row is an independent estimate from its own element's rays, so
/// A_i F_ij and A_j F_ji disagree by the estimator's noise, and the top-K
/// truncation makes the support itself one-sided: a small element often keeps
/// a large neighbour in its row while the large element's row drops the small
/// one. The energy balance gathers along rows (element i absorbs
/// sum_j F_ij sigma T_j^4 while j emits its full sigma T_j^4), so what j emits
/// reaches the others only if A_i F_ij = A_j F_ji.
///
/// The area-weighted exchange G_ij = A_i F_ij is made symmetric once, by
/// construction: where both rows hold the pair, G is their mean; where only one
/// does, the other was truncated rather than zero (reciprocity says the value
/// is the same), so G takes the one that exists. G is then scaled by a
/// symmetric D G D toward the row sums the rays measured, A_i (1 - sky_i), with
/// D bounded to [1/2, 2] so that one pathological element cannot distort its
/// neighbours, and any row still above A_i is scaled down together with its
/// partners. Rows are F_ij = G_ij / A_i, and the sky fraction becomes whatever
/// remains, 1 - sum_j F_ij, so every row sums to one with its sky fraction and
/// the matrix is reciprocal whatever the iteration reached.
void EnforceReciprocity(thermal::CsrMatrix& viewFactors, Vector<f32>& skyFraction,
                        const Vector<thermal::ThermalElement>& elements) {
    constexpr u32 kMaxIterations = 50;
    constexpr f64 kTolerance = 1e-3;
    constexpr f64 kMinScale = 0.5;
    constexpr f64 kMaxScale = 2.0;

    const usize n = elements.size();
    if (viewFactors.column.empty() || viewFactors.rowStart.size() != n + 1) {
        return;
    }

    Vector<f64> area(n);
    Vector<f64> target(n);
    for (usize i = 0; i < n; ++i) {
        area[i] = std::max(0.0, static_cast<f64>(elements[i].area_m2));
        target[i] = area[i] * std::max(0.0, 1.0 - static_cast<f64>(skyFraction[i]));
    }

    struct Entry {
        u32 row;
        u32 column;
        f64 value;
        u32 count;
    };
    Vector<Entry> entries;
    entries.reserve(viewFactors.column.size() * 2);
    for (usize i = 0; i < n; ++i) {
        for (u32 k = viewFactors.rowStart[i]; k < viewFactors.rowStart[i + 1]; ++k) {
            const u32 j = viewFactors.column[k];
            if (j >= n || j == i || !(area[i] > 0.0) || !(area[j] > 0.0)) continue;
            const f64 flux = area[i] * static_cast<f64>(viewFactors.value[k]);
            entries.push_back({static_cast<u32>(i), j, flux, 1});
            entries.push_back({j, static_cast<u32>(i), flux, 1});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.row != b.row ? a.row < b.row : a.column < b.column;
    });
    usize merged = 0;
    for (usize k = 0; k < entries.size(); ++k) {
        if (merged > 0 && entries[merged - 1].row == entries[k].row &&
            entries[merged - 1].column == entries[k].column) {
            entries[merged - 1].value += entries[k].value;
            entries[merged - 1].count += entries[k].count;
        } else {
            entries[merged++] = entries[k];
        }
    }
    entries.resize(merged);
    // Mean where both rows hold the pair, the one value where only one does.
    for (Entry& entry : entries) entry.value /= static_cast<f64>(entry.count);

    Vector<usize> start(n + 1, 0);
    for (const Entry& entry : entries) ++start[entry.row + 1];
    for (usize i = 0; i < n; ++i) start[i + 1] += start[i];

    // Symmetric scaling toward the measured row sums. Best effort: whatever it
    // reaches, the sky below absorbs the remainder.
    Vector<f64> d(n, 1.0);
    Vector<f64> gathered(n, 0.0);
    for (u32 iteration = 0; iteration < kMaxIterations; ++iteration) {
        f64 residual = 0.0;
        for (usize i = 0; i < n; ++i) {
            f64 sum = 0.0;
            for (usize k = start[i]; k < start[i + 1]; ++k) {
                sum += entries[k].value * d[entries[k].column];
            }
            gathered[i] = sum;
            if (target[i] > 0.0 && sum > 0.0) {
                residual = std::max(residual, std::abs(d[i] * sum / target[i] - 1.0));
            }
        }
        if (residual < kTolerance) break;
        for (usize i = 0; i < n; ++i) {
            if (target[i] > 0.0 && gathered[i] > 0.0) {
                d[i] = std::clamp(std::sqrt(d[i] * target[i] / gathered[i]), kMinScale,
                                  kMaxScale);
            }
        }
    }
    for (Entry& entry : entries) entry.value *= d[entry.row] * d[entry.column];

    // A row may not exchange with more than its whole hemisphere. Scale the
    // offending rows down, each pair by the smaller of its two factors so the
    // matrix stays symmetric.
    Vector<f64> rowTotal(n, 0.0);
    for (const Entry& entry : entries) rowTotal[entry.row] += entry.value;
    Vector<f64> cap(n, 1.0);
    for (usize i = 0; i < n; ++i) {
        if (rowTotal[i] > area[i]) cap[i] = area[i] / rowTotal[i];
    }
    std::fill(rowTotal.begin(), rowTotal.end(), 0.0);
    for (Entry& entry : entries) {
        entry.value *= std::min(cap[entry.row], cap[entry.column]);
        rowTotal[entry.row] += entry.value;
    }

    thermal::CsrMatrix result;
    result.rowStart.reserve(n + 1);
    result.rowStart.push_back(0);
    for (usize i = 0; i < n; ++i) {
        for (usize k = start[i]; k < start[i + 1]; ++k) {
            const f64 value = entries[k].value / area[i];
            if (value > 0.0) {
                result.column.push_back(entries[k].column);
                result.value.push_back(static_cast<f32>(value));
            }
        }
        result.rowStart.push_back(static_cast<u32>(result.column.size()));
        if (area[i] > 0.0) {
            skyFraction[i] = static_cast<f32>(std::clamp(1.0 - rowTotal[i] / area[i], 0.0, 1.0));
        }
    }
    viewFactors = std::move(result);
}

// A transient storage buffer of the pass, in the memory class its use needs: the inputs
// host-written, the outputs host-read.
gpu::BufferDesc StorageDesc(u64 bytes, gpu::MemoryClass memory) {
    return gpu::BufferDesc{bytes, gpu::BufferUsage::Storage, memory, 0};
}

}  // namespace

struct ThermalExchangePrecompute::Impl {
    Vector<f32> materialCoverage;
    /// The caller's device; it outlives the objects below, which it creates.
    gpu::Device& gpuDevice;

    // Destroyed in reverse declaration order -- kernel, kernel layout, pool, binding
    // layout -- which is the order the native handles were destroyed in before.
    std::unique_ptr<gpu::BindingLayout> bindingLayout;
    std::unique_ptr<gpu::BindGroupPool> pool;
    std::unique_ptr<gpu::KernelLayout> kernelLayout;
    std::unique_ptr<gpu::ComputeKernel> kernel;
    gpu::BindGroup bindGroup;

    explicit Impl(gpu::Device& device) : gpuDevice(device) {}

    /// Each object is kept as soon as it exists, so a failure part-way releases what was
    /// created with the Impl. The bind group is set last: IsValid() is false on any failure.
    bool Create() {
        const StringView kShaderNames[] = {"thermal_exchange"};
        auto shaders = gpuDevice.Shaders().Load(kShaderNames);
        if (!shaders.has_value()) {
            QL_LOG_WARN("Thermal exchange: thermal_exchange.spv not found; the solver "
                        "will fall back to open-sky exchange");
            return false;
        }

        // One slot per binding thermal_exchange.rayq.hlsl declares (:31-72): t0 the TLAS,
        // t1 elements, t2 instance bases, u3 hit records, u4 sun visibility, t5 material
        // coverage. All compute-visible.
        constexpr gpu::BindingSlot kSlots[] = {
            {0, gpu::BindingKind::AccelerationStructure, 1, false},
            {1, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {2, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {3, gpu::BindingKind::ReadWriteBuffer, 1, false},
            {4, gpu::BindingKind::ReadWriteBuffer, 1, false},
            {5, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        };
        auto bindingLayoutResult = gpuDevice.CreateBindingLayout(
            gpu::BindingLayoutDesc{0, std::span<const gpu::BindingSlot>(kSlots)});
        if (!bindingLayoutResult.has_value()) {
            QL_LOG_WARN("Thermal exchange: failed to create descriptor set layout");
            return false;
        }
        bindingLayout = std::move(bindingLayoutResult.value());

        const gpu::BindingLayout* spaces[] = {bindingLayout.get()};
        auto kernelLayoutResult = gpuDevice.CreateKernelLayout(gpu::KernelLayoutDesc{
            std::span<const gpu::BindingLayout* const>(spaces), sizeof(ExchangePushConstants)});
        if (!kernelLayoutResult.has_value()) {
            QL_LOG_WARN("Thermal exchange: failed to create pipeline layout");
            return false;
        }
        kernelLayout = std::move(kernelLayoutResult.value());

        // The shader module lives only as long as the pipeline creation.
        auto kernelResult = gpuDevice.CreateComputeKernel(
            gpu::ComputeKernelDesc{&shaders.value()[0], kernelLayout.get()});
        if (!kernelResult.has_value()) {
            QL_LOG_WARN("Thermal exchange: failed to create compute pipeline ({})",
                        kernelResult.error());
            return false;
        }
        kernel = std::move(kernelResult.value());

        // Pool sizes follow the layout: {acceleration structure 1, storage buffer 5}, one set.
        auto poolResult = gpuDevice.CreateBindGroupPool(*bindingLayout, 1);
        if (!poolResult.has_value()) {
            QL_LOG_WARN("Thermal exchange: failed to create descriptor pool");
            return false;
        }
        pool = std::move(poolResult.value());

        auto groups = pool->Allocate(1);
        if (!groups.has_value()) {
            QL_LOG_WARN("Thermal exchange: failed to allocate descriptor set");
            return false;
        }
        bindGroup = groups.value()[0];
        return true;
    }
};

ThermalExchangePrecompute::ThermalExchangePrecompute(gpu::Device& device)
    : m_impl(std::make_unique<Impl>(device)) {
    // A failed Create leaves the bind group unset, so IsValid() stays false.
    m_impl->Create();
}

ThermalExchangePrecompute::~ThermalExchangePrecompute() = default;

void ThermalExchangePrecompute::SetMaterialCoverage(Vector<f32> coverage) {
    if (m_impl) {
        m_impl->materialCoverage = std::move(coverage);
    }
}

bool ThermalExchangePrecompute::IsValid() const {
    return m_impl && m_impl->kernel != nullptr && m_impl->bindGroup.IsValid();
}

thermal::ExchangeGeometry ThermalExchangePrecompute::Run(
    const gpu::RtSceneRef tlas, const Vector<thermal::ThermalElement>& elements,
    const Vector<u32>& instanceElementBase, const Params& params) {
    thermal::ExchangeGeometry exchange;
    if (!IsValid() || elements.empty() || !tlas.IsValid()) {
        return exchange;
    }

    const u32 elementCount = static_cast<u32>(elements.size());
    const u32 rays = std::max(1u, params.hemisphereRays);
    // A row keeps at least its largest entry: topK = 0 would keep none, and a
    // surface whose row sums to less than a whole hemisphere exchanges with
    // nothing and cools toward absolute zero.
    const u32 topK = std::max(1u, params.topK);

    // Hit records are elementCount * rays uints. At 15k elements and 256 rays
    // that is 15 MB, which is why they are a transient rather than something
    // the scene holds on to.
    const u64 recordBytes =
        static_cast<u64>(elementCount) * rays * sizeof(u32);

    gpu::Device& device = m_impl->gpuDevice;

    Vector<ThermalElementGpu> gpuElements(elementCount);
    for (u32 e = 0; e < elementCount; ++e) {
        gpuElements[e].centre = elements[e].centroid;
        gpuElements[e].area = elements[e].area_m2;
        gpuElements[e].normal = elements[e].normal;
        gpuElements[e].materialId = elements[e].materialId;
    }

    const auto elementBuffer = CreateBufferOrThrow(device, StorageDesc(gpuElements.size() * sizeof(ThermalElementGpu),
                                                                 gpu::MemoryClass::Upload));
    elementBuffer->Upload(gpuElements.data(), gpuElements.size() * sizeof(ThermalElementGpu));

    Vector<u32> bases = instanceElementBase;
    if (bases.empty()) bases.push_back(0);
    const auto baseBuffer =
        CreateBufferOrThrow(device, StorageDesc(bases.size() * sizeof(u32), gpu::MemoryClass::Upload));
    baseBuffer->Upload(bases.data(), bases.size() * sizeof(u32));

    const auto recordBuffer = CreateBufferOrThrow(device, StorageDesc(recordBytes, gpu::MemoryClass::Readback));
    const auto sunBuffer =
        CreateBufferOrThrow(device, StorageDesc(elementCount * sizeof(f32), gpu::MemoryClass::Readback));

    // Empty means every material is opaque, which the shader's own bounds
    // check would fall back to anyway -- but a zero-sized buffer is not a
    // legal descriptor, so it gets one entry.
    Vector<f32> coverage = m_impl->materialCoverage;
    if (coverage.empty()) coverage.push_back(1.0f);
    const auto coverageBuffer =
        CreateBufferOrThrow(device, StorageDesc(coverage.size() * sizeof(f32), gpu::MemoryClass::Upload));
    coverageBuffer->Upload(coverage.data(), coverage.size() * sizeof(f32));

    // ------------------------------------------------------------------
    // Bind and dispatch
    // ------------------------------------------------------------------
    const gpu::BindingWrite writes[] = {
        gpu::BindingWrite::Scene(0, tlas),
        gpu::BindingWrite::Buffer(1, gpu::BindingKind::ReadOnlyBuffer, elementBuffer->Ref()),
        gpu::BindingWrite::Buffer(2, gpu::BindingKind::ReadOnlyBuffer, baseBuffer->Ref()),
        gpu::BindingWrite::Buffer(3, gpu::BindingKind::ReadWriteBuffer, recordBuffer->Ref()),
        gpu::BindingWrite::Buffer(4, gpu::BindingKind::ReadWriteBuffer, sunBuffer->Ref()),
        gpu::BindingWrite::Buffer(5, gpu::BindingKind::ReadOnlyBuffer, coverageBuffer->Ref()),
    };
    m_impl->gpuDevice.WriteBindGroup(m_impl->bindGroup, writes);

    ExchangePushConstants pc{};
    pc.sunDirection = glm::normalize(params.sunDirection);
    pc.elementCount = elementCount;
    pc.rayCount = rays;
    pc.sunRayCount = params.sunRays;
    pc.sunAngularRadius = params.sunAngularRadius;
    pc.rayOffset = kRayOffsetFloor;
    pc.sunOutputOffset = 0;

    m_impl->gpuDevice.ExecuteImmediate([&](gpu::CommandList& list) {
        list.SetKernel(*m_impl->kernel);
        list.SetBindGroup(*m_impl->kernelLayout, 0, m_impl->bindGroup);
        list.PushConstants(*m_impl->kernelLayout, pc);
        list.Dispatch((elementCount + 63) / 64, 1, 1);

        // Both outputs to the host, one barrier: compute shader writes before host reads.
        const gpu::BufferBarrier barriers[2] = {
            {recordBuffer->Ref(), gpu::Access::ComputeWrite, gpu::Access::HostRead},
            {sunBuffer->Ref(), gpu::Access::ComputeWrite, gpu::Access::HostRead},
        };
        list.Barrier(gpu::BarrierBatch{{}, std::span<const gpu::BufferBarrier>(barriers), {}});
    });

    // ------------------------------------------------------------------
    // Reduce the hit records into sparse rows
    // ------------------------------------------------------------------
    const u32* records = static_cast<const u32*>(recordBuffer->MapRead());
    const f32* sun = static_cast<const f32*>(sunBuffer->MapRead());

    exchange.viewFactors.rowStart.reserve(elementCount + 1);
    exchange.viewFactors.rowStart.push_back(0);
    exchange.skyFraction.resize(elementCount);
    exchange.sunVisibility.assign(sun, sun + elementCount);

    std::unordered_map<u32, u32> counts;
    Vector<std::pair<u32, u32>> ranked;

    for (u32 e = 0; e < elementCount; ++e) {
        counts.clear();
        u32 skyHits = 0;
        u32 selfHits = 0;
        const u32* row = records + static_cast<usize>(e) * rays;
        for (u32 r = 0; r < rays; ++r) {
            const u32 hit = row[r];
            if (hit == kSkyRecord || hit >= elementCount) {
                ++skyHits;
            } else if (hit != e) {
                ++counts[hit];
            } else {
                // A ray that came back to its own element is the offset losing
                // to a grazing direction, so it says nothing about what the
                // surface sees. Counting it as sky would cool the surface and
                // counting it as self-exchange does nothing at all; the ray is
                // dropped and the rest of the row is normalised over the rays
                // that did report.
                ++selfHits;
            }
        }
        const u32 validRays = rays - selfHits;
        const f32 perRay = 1.0f / static_cast<f32>(std::max(1u, validRays));

        ranked.assign(counts.begin(), counts.end());
        if (ranked.size() > topK) {
            std::nth_element(ranked.begin(), ranked.begin() + topK, ranked.end(),
                             [](const auto& a, const auto& b) { return a.second > b.second; });
            ranked.resize(topK);
        }
        // Sorted by column so the row is scanned in order, which is what makes
        // the CSR worth being one.
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Truncation threw away part of the hemisphere. What survives is
        // rescaled to fill everything that was not sky, so the row still sums
        // to one with the sky fraction -- a row that does not is a surface
        // exchanging with less than a whole hemisphere, which cools toward
        // nothing over a long enough night.
        f32 keptWeight = 0.0f;
        for (const auto& [column, count] : ranked) {
            keptWeight += static_cast<f32>(count) * perRay;
        }
        // No ray reported at all: the open-sky answer, as for a degenerate element.
        const f32 skyFraction = validRays == 0 ? 1.0f : static_cast<f32>(skyHits) * perRay;
        const f32 target = 1.0f - skyFraction;
        const f32 rescale = keptWeight > 0.0f ? target / keptWeight : 0.0f;

        for (const auto& [column, count] : ranked) {
            exchange.viewFactors.column.push_back(column);
            exchange.viewFactors.value.push_back(static_cast<f32>(count) * perRay * rescale);
        }
        exchange.viewFactors.rowStart.push_back(
            static_cast<u32>(exchange.viewFactors.column.size()));
        exchange.skyFraction[e] = skyFraction;
    }

    recordBuffer->Unmap();
    sunBuffer->Unmap();

    EnforceReciprocity(exchange.viewFactors, exchange.skyFraction, elements);

    const f32 meanSky =
        std::accumulate(exchange.skyFraction.begin(), exchange.skyFraction.end(), 0.0f) /
        static_cast<f32>(elementCount);
    QL_LOG_INFO("  Thermal exchange: {} elements x {} rays, {} entries kept "
                "(mean sky fraction {:.3f})",
                elementCount, rays, exchange.viewFactors.NonZeros(), meanSky);

    return exchange;
}

Vector<f32> ThermalExchangePrecompute::RunSunVisibility(
    const gpu::RtSceneRef tlas, const Vector<thermal::ThermalElement>& elements,
    const Vector<u32>& instanceElementBase, std::span<const glm::vec3> directions,
    const u32 sunRays, const f32 sunAngularRadius) {
    if (!IsValid() || elements.empty() || !tlas.IsValid() || directions.empty()) {
        return {};
    }

    const u32 elementCount = static_cast<u32>(elements.size());
    const u32 K = static_cast<u32>(directions.size());

    gpu::Device& device = m_impl->gpuDevice;

    Vector<ThermalElementGpu> gpuElements(elementCount);
    for (u32 e = 0; e < elementCount; ++e) {
        gpuElements[e].centre = elements[e].centroid;
        gpuElements[e].area = elements[e].area_m2;
        gpuElements[e].normal = elements[e].normal;
        gpuElements[e].materialId = elements[e].materialId;
    }

    const auto elementBuffer = CreateBufferOrThrow(device, StorageDesc(gpuElements.size() * sizeof(ThermalElementGpu),
                                                                 gpu::MemoryClass::Upload));
    elementBuffer->Upload(gpuElements.data(), gpuElements.size() * sizeof(ThermalElementGpu));

    Vector<u32> bases = instanceElementBase;
    if (bases.empty()) bases.push_back(0);
    const auto baseBuffer =
        CreateBufferOrThrow(device, StorageDesc(bases.size() * sizeof(u32), gpu::MemoryClass::Upload));
    baseBuffer->Upload(bases.data(), bases.size() * sizeof(u32));

    // Dummy hit record buffer — the sun-only pass sets rayCount=0, so no
    // hemisphere records are written. The binding still must exist.
    const auto recordBuffer = CreateBufferOrThrow(device, StorageDesc(sizeof(u32), gpu::MemoryClass::Readback));

    // Sun visibility: K columns, each elementCount floats, direction-major.
    const u64 sunBytes =
        static_cast<u64>(K) * elementCount * sizeof(f32);
    const auto sunBuffer = CreateBufferOrThrow(device, StorageDesc(sunBytes, gpu::MemoryClass::Readback));

    Vector<f32> coverage = m_impl->materialCoverage;
    if (coverage.empty()) coverage.push_back(1.0f);
    const auto coverageBuffer =
        CreateBufferOrThrow(device, StorageDesc(coverage.size() * sizeof(f32), gpu::MemoryClass::Upload));
    coverageBuffer->Upload(coverage.data(), coverage.size() * sizeof(f32));

    // Bind descriptors
    const gpu::BindingWrite writes[] = {
        gpu::BindingWrite::Scene(0, tlas),
        gpu::BindingWrite::Buffer(1, gpu::BindingKind::ReadOnlyBuffer, elementBuffer->Ref()),
        gpu::BindingWrite::Buffer(2, gpu::BindingKind::ReadOnlyBuffer, baseBuffer->Ref()),
        gpu::BindingWrite::Buffer(3, gpu::BindingKind::ReadWriteBuffer, recordBuffer->Ref()),
        gpu::BindingWrite::Buffer(4, gpu::BindingKind::ReadWriteBuffer, sunBuffer->Ref()),
        gpu::BindingWrite::Buffer(5, gpu::BindingKind::ReadOnlyBuffer, coverageBuffer->Ref()),
    };
    m_impl->gpuDevice.WriteBindGroup(m_impl->bindGroup, writes);

    const u32 groups = (elementCount + 63) / 64;

    m_impl->gpuDevice.ExecuteImmediate([&](gpu::CommandList& list) {
        list.SetKernel(*m_impl->kernel);
        list.SetBindGroup(*m_impl->kernelLayout, 0, m_impl->bindGroup);

        for (u32 k = 0; k < K; ++k) {
            ExchangePushConstants pc{};
            pc.sunDirection = glm::normalize(directions[k]);
            pc.elementCount = elementCount;
            pc.rayCount = 0;  // no hemisphere
            pc.sunRayCount = sunRays;
            pc.sunAngularRadius = sunAngularRadius;
            pc.rayOffset = kRayOffsetFloor;
            pc.sunOutputOffset = k * elementCount;

            list.PushConstants(*m_impl->kernelLayout, pc);
            list.Dispatch(groups, 1, 1);

            if (k + 1 < K) {
                list.Barrier(gpu::GlobalBarrier{gpu::Access::ComputeWrite, gpu::Access::ComputeWrite});
            }
        }

        list.Barrier(gpu::BufferBarrier{sunBuffer->Ref(), gpu::Access::ComputeWrite,
                                        gpu::Access::HostRead});
    });

    const f32* mapped = static_cast<const f32*>(sunBuffer->MapRead());
    Vector<f32> result(mapped, mapped + static_cast<usize>(K) * elementCount);
    sunBuffer->Unmap();

    QL_LOG_INFO("  Thermal exchange: sun-only batch: {} directions x {} elements x {} rays",
                K, elementCount, sunRays);
    return result;
}

}  // namespace quantiloom::rendercore
