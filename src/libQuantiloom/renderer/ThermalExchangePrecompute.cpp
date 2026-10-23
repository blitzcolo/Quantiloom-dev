/**
 * @file ThermalExchangePrecompute.cpp
 * @brief Who sees whom, on the GPU, once per scene
 */

#include "renderer/ThermalExchangePrecompute.hpp"

#include "core/Log.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/ShaderBinary.hpp"

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

/// Make the view factors reciprocal and keep every row summing to one.
///
/// Each row is an independent estimate from its own element's rays, so
/// A_i F_ij and A_j F_ji disagree by the estimator's noise (and by the top-K
/// truncation). The energy balance gathers along rows -- element i absorbs
/// sum_j F_ij sigma T_j^4 while j emits its full sigma T_j^4 -- so what j emits
/// reaches the others only if the column sums of A_i F_ij equal A_j (1 - sky_j),
/// which is reciprocity. Without it the exchange between a pair creates or
/// destroys energy.
///
/// W = A F is symmetrised to S = (W + W^T) / 2, then scaled symmetrically,
/// M = D S D, until every row of M sums to A_i (1 - sky_i), which is the
/// constraint the rows already met. The scaling is a damped symmetric Sinkhorn
/// iteration. If it does not converge -- noise can make the targets infeasible
/// for a pathological element -- the rows are left as they were, normalised
/// but not reciprocal, which is the pre-existing behaviour.
///
/// @return true if the matrix was replaced
bool EnforceReciprocity(thermal::CsrMatrix& viewFactors, const Vector<f32>& skyFraction,
                        const Vector<thermal::ThermalElement>& elements) {
    constexpr u32 kMaxIterations = 200;
    constexpr f64 kTolerance = 1e-4;

    const usize n = elements.size();
    if (viewFactors.column.empty() || viewFactors.rowStart.size() != n + 1) {
        return false;
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
    };
    Vector<Entry> entries;
    entries.reserve(viewFactors.column.size() * 2);
    for (usize i = 0; i < n; ++i) {
        for (u32 k = viewFactors.rowStart[i]; k < viewFactors.rowStart[i + 1]; ++k) {
            const u32 j = viewFactors.column[k];
            const f64 half = 0.5 * area[i] * static_cast<f64>(viewFactors.value[k]);
            entries.push_back({static_cast<u32>(i), j, half});
            entries.push_back({j, static_cast<u32>(i), half});
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
        } else {
            entries[merged++] = entries[k];
        }
    }
    entries.resize(merged);

    Vector<usize> start(n + 1, 0);
    for (const Entry& entry : entries) ++start[entry.row + 1];
    for (usize i = 0; i < n; ++i) start[i + 1] += start[i];

    Vector<f64> d(n, 1.0);
    Vector<f64> gathered(n, 0.0);
    bool converged = false;
    for (u32 iteration = 0; iteration < kMaxIterations && !converged; ++iteration) {
        f64 residual = 0.0;
        for (usize i = 0; i < n; ++i) {
            f64 sum = 0.0;
            for (usize k = start[i]; k < start[i + 1]; ++k) {
                sum += entries[k].value * d[entries[k].column];
            }
            gathered[i] = sum;
            if (target[i] > 0.0) {
                residual = std::max(residual, sum > 0.0 ? std::abs(d[i] * sum / target[i] - 1.0)
                                                         : 1.0);
            }
        }
        if (residual < kTolerance) {
            converged = true;
            break;
        }
        for (usize i = 0; i < n; ++i) {
            d[i] = (target[i] > 0.0 && gathered[i] > 0.0)
                       ? std::sqrt(d[i] * target[i] / gathered[i])
                       : 0.0;
        }
    }
    if (!converged) {
        QL_LOG_WARN("Thermal exchange: the view factors could not be made reciprocal; "
                    "keeping the per-row estimates");
        return false;
    }

    thermal::CsrMatrix result;
    result.rowStart.reserve(n + 1);
    result.rowStart.push_back(0);
    for (usize i = 0; i < n; ++i) {
        const usize first = result.column.size();
        f64 rowSum = 0.0;
        if (area[i] > 0.0 && d[i] > 0.0) {
            for (usize k = start[i]; k < start[i + 1]; ++k) {
                const f64 value = d[i] * entries[k].value * d[entries[k].column] / area[i];
                if (value > 0.0) {
                    result.column.push_back(entries[k].column);
                    result.value.push_back(static_cast<f32>(value));
                    rowSum += value;
                }
            }
        }
        // The iteration stopped inside its tolerance; close the last fraction
        // of a percent so the row sums to one with its sky fraction exactly.
        if (rowSum > 0.0) {
            const f64 scale = (1.0 - static_cast<f64>(skyFraction[i])) / rowSum;
            for (usize k = first; k < result.value.size(); ++k) {
                result.value[k] = static_cast<f32>(result.value[k] * scale);
            }
        }
        result.rowStart.push_back(static_cast<u32>(result.column.size()));
    }
    viewFactors = std::move(result);
    return true;
}

}  // namespace

struct ThermalExchangePrecompute::Impl {
    Vector<f32> materialCoverage;
    VulkanContext& context;
    VkDevice device = VK_NULL_HANDLE;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    explicit Impl(VulkanContext& ctx) : context(ctx), device(ctx.GetDevice()) {}

    ~Impl() {
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (descriptorSetLayout) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        }
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
    }

    bool Create() {
        const Vector<u32> code = LoadSpirv("thermal_exchange.spv");
        if (code.empty()) {
            QL_LOG_WARN("Thermal exchange: thermal_exchange.spv not found; the solver "
                        "will fall back to open-sky exchange");
            return false;
        }

        VkShaderModuleCreateInfo moduleInfo{};
        moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleInfo.codeSize = code.size() * sizeof(u32);
        moduleInfo.pCode = code.data();
        if (vkCreateShaderModule(device, &moduleInfo, nullptr, &shader) != VK_SUCCESS) {
            QL_LOG_WARN("Thermal exchange: failed to create shader module");
            return false;
        }

        // 0 TLAS, 1 elements, 2 instance bases, 3 hit records, 4 sun visibility
        Vector<VkDescriptorSetLayoutBinding> bindings(6);
        for (u32 i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                        &descriptorSetLayout) != VK_SUCCESS) {
            QL_LOG_WARN("Thermal exchange: failed to create descriptor set layout");
            return false;
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(ExchangePushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                   &pipelineLayout) != VK_SUCCESS) {
            QL_LOG_WARN("Thermal exchange: failed to create pipeline layout");
            return false;
        }

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shader;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                     &pipeline) != VK_SUCCESS) {
            QL_LOG_WARN("Thermal exchange: failed to create compute pipeline");
            pipeline = VK_NULL_HANDLE;
            return false;
        }

        Vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5},
        };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 1;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            QL_LOG_WARN("Thermal exchange: failed to create descriptor pool");
            return false;
        }

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            QL_LOG_WARN("Thermal exchange: failed to allocate descriptor set");
            descriptorSet = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }
};

ThermalExchangePrecompute::ThermalExchangePrecompute(VulkanContext& context)
    : m_impl(std::make_unique<Impl>(context)) {
    if (!m_impl->Create()) {
        m_impl->pipeline = VK_NULL_HANDLE;
    }
}

ThermalExchangePrecompute::~ThermalExchangePrecompute() = default;

void ThermalExchangePrecompute::SetMaterialCoverage(Vector<f32> coverage) {
    if (m_impl) {
        m_impl->materialCoverage = std::move(coverage);
    }
}

bool ThermalExchangePrecompute::IsValid() const {
    return m_impl && m_impl->pipeline != VK_NULL_HANDLE &&
           m_impl->descriptorSet != VK_NULL_HANDLE;
}

thermal::ExchangeGeometry ThermalExchangePrecompute::Run(
    const VkAccelerationStructureKHR tlas, const Vector<thermal::ThermalElement>& elements,
    const Vector<u32>& instanceElementBase, const Params& params) {
    thermal::ExchangeGeometry exchange;
    if (!IsValid() || elements.empty() || tlas == VK_NULL_HANDLE) {
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
    const VkDeviceSize recordBytes =
        static_cast<VkDeviceSize>(elementCount) * rays * sizeof(u32);

    VmaAllocator allocator = m_impl->context.GetAllocator();

    Vector<ThermalElementGpu> gpuElements(elementCount);
    for (u32 e = 0; e < elementCount; ++e) {
        gpuElements[e].centre = elements[e].centroid;
        gpuElements[e].area = elements[e].area_m2;
        gpuElements[e].normal = elements[e].normal;
        gpuElements[e].materialId = elements[e].materialId;
    }

    GpuBuffer elementBuffer(allocator, gpuElements.size() * sizeof(ThermalElementGpu),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    elementBuffer.Upload(gpuElements.data(), gpuElements.size() * sizeof(ThermalElementGpu));

    Vector<u32> bases = instanceElementBase;
    if (bases.empty()) bases.push_back(0);
    GpuBuffer baseBuffer(allocator, bases.size() * sizeof(u32),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    baseBuffer.Upload(bases.data(), bases.size() * sizeof(u32));

    GpuBuffer recordBuffer(allocator, recordBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VMA_MEMORY_USAGE_GPU_TO_CPU);
    GpuBuffer sunBuffer(allocator, elementCount * sizeof(f32),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

    // Empty means every material is opaque, which the shader's own bounds
    // check would fall back to anyway -- but a zero-sized buffer is not a
    // legal descriptor, so it gets one entry.
    Vector<f32> coverage = m_impl->materialCoverage;
    if (coverage.empty()) coverage.push_back(1.0f);
    GpuBuffer coverageBuffer(allocator, coverage.size() * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    coverageBuffer.Upload(coverage.data(), coverage.size() * sizeof(f32));

    // ------------------------------------------------------------------
    // Bind and dispatch
    // ------------------------------------------------------------------
    VkWriteDescriptorSetAccelerationStructureKHR tlasInfo{};
    tlasInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    tlasInfo.accelerationStructureCount = 1;
    tlasInfo.pAccelerationStructures = &tlas;

    const VkDescriptorBufferInfo bufferInfos[5] = {
        {elementBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {baseBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {recordBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {sunBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {coverageBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
    };

    Vector<VkWriteDescriptorSet> writes(6);
    for (u32 i = 0; i < writes.size(); ++i) {
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_impl->descriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        if (i == 0) {
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            writes[i].pNext = &tlasInfo;
        } else {
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfos[i - 1];
        }
    }
    vkUpdateDescriptorSets(m_impl->device, static_cast<u32>(writes.size()), writes.data(), 0,
                           nullptr);

    ExchangePushConstants pc{};
    pc.sunDirection = glm::normalize(params.sunDirection);
    pc.elementCount = elementCount;
    pc.rayCount = rays;
    pc.sunRayCount = params.sunRays;
    pc.sunAngularRadius = params.sunAngularRadius;
    pc.rayOffset = kRayOffsetFloor;
    pc.sunOutputOffset = 0;

    CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipelineLayout,
                                0, 1, &m_impl->descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_impl->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(pc), &pc);
        vkCmdDispatch(cmd, (elementCount + 63) / 64, 1, 1);

        VkBufferMemoryBarrier barriers[2]{};
        for (auto& barrier : barriers) {
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.offset = 0;
            barrier.size = VK_WHOLE_SIZE;
        }
        barriers[0].buffer = recordBuffer.GetHandle();
        barriers[1].buffer = sunBuffer.GetHandle();
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 2, barriers, 0,
                             nullptr);
    });

    // ------------------------------------------------------------------
    // Reduce the hit records into sparse rows
    // ------------------------------------------------------------------
    const u32* records = static_cast<const u32*>(recordBuffer.MapRead());
    const f32* sun = static_cast<const f32*>(sunBuffer.MapRead());

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

    recordBuffer.Unmap();
    sunBuffer.Unmap();

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
    const VkAccelerationStructureKHR tlas, const Vector<thermal::ThermalElement>& elements,
    const Vector<u32>& instanceElementBase, std::span<const glm::vec3> directions,
    const u32 sunRays, const f32 sunAngularRadius) {
    if (!IsValid() || elements.empty() || tlas == VK_NULL_HANDLE || directions.empty()) {
        return {};
    }

    const u32 elementCount = static_cast<u32>(elements.size());
    const u32 K = static_cast<u32>(directions.size());

    VmaAllocator allocator = m_impl->context.GetAllocator();

    Vector<ThermalElementGpu> gpuElements(elementCount);
    for (u32 e = 0; e < elementCount; ++e) {
        gpuElements[e].centre = elements[e].centroid;
        gpuElements[e].area = elements[e].area_m2;
        gpuElements[e].normal = elements[e].normal;
        gpuElements[e].materialId = elements[e].materialId;
    }

    GpuBuffer elementBuffer(allocator, gpuElements.size() * sizeof(ThermalElementGpu),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    elementBuffer.Upload(gpuElements.data(), gpuElements.size() * sizeof(ThermalElementGpu));

    Vector<u32> bases = instanceElementBase;
    if (bases.empty()) bases.push_back(0);
    GpuBuffer baseBuffer(allocator, bases.size() * sizeof(u32),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    baseBuffer.Upload(bases.data(), bases.size() * sizeof(u32));

    // Dummy hit record buffer — the sun-only pass sets rayCount=0, so no
    // hemisphere records are written. The binding still must exist.
    GpuBuffer recordBuffer(allocator, sizeof(u32),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

    // Sun visibility: K columns, each elementCount floats, direction-major.
    const VkDeviceSize sunBytes =
        static_cast<VkDeviceSize>(K) * elementCount * sizeof(f32);
    GpuBuffer sunBuffer(allocator, sunBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_GPU_TO_CPU);

    Vector<f32> coverage = m_impl->materialCoverage;
    if (coverage.empty()) coverage.push_back(1.0f);
    GpuBuffer coverageBuffer(allocator, coverage.size() * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    coverageBuffer.Upload(coverage.data(), coverage.size() * sizeof(f32));

    // Bind descriptors
    VkWriteDescriptorSetAccelerationStructureKHR tlasInfo{};
    tlasInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    tlasInfo.accelerationStructureCount = 1;
    tlasInfo.pAccelerationStructures = &tlas;

    const VkDescriptorBufferInfo bufferInfos[5] = {
        {elementBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {baseBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {recordBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {sunBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {coverageBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
    };

    Vector<VkWriteDescriptorSet> writes(6);
    for (u32 i = 0; i < writes.size(); ++i) {
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_impl->descriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        if (i == 0) {
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            writes[i].pNext = &tlasInfo;
        } else {
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfos[i - 1];
        }
    }
    vkUpdateDescriptorSets(m_impl->device, static_cast<u32>(writes.size()), writes.data(),
                           0, nullptr);

    const u32 groups = (elementCount + 63) / 64;

    CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_impl->pipelineLayout, 0, 1,
                                &m_impl->descriptorSet, 0, nullptr);

        for (u32 k = 0; k < K; ++k) {
            ExchangePushConstants pc{};
            pc.sunDirection = glm::normalize(directions[k]);
            pc.elementCount = elementCount;
            pc.rayCount = 0;  // no hemisphere
            pc.sunRayCount = sunRays;
            pc.sunAngularRadius = sunAngularRadius;
            pc.rayOffset = kRayOffsetFloor;
            pc.sunOutputOffset = k * elementCount;

            vkCmdPushConstants(cmd, m_impl->pipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, groups, 1, 1);

            if (k + 1 < K) {
                VkMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &barrier, 0, nullptr, 0, nullptr);
            }
        }

        VkBufferMemoryBarrier hostBarrier{};
        hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.buffer = sunBuffer.GetHandle();
        hostBarrier.offset = 0;
        hostBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                             1, &hostBarrier, 0, nullptr);
    });

    const f32* mapped = static_cast<const f32*>(sunBuffer.MapRead());
    Vector<f32> result(mapped, mapped + static_cast<usize>(K) * elementCount);
    sunBuffer.Unmap();

    QL_LOG_INFO("  Thermal exchange: sun-only batch: {} directions x {} elements x {} rays",
                K, elementCount, sunRays);
    return result;
}

}  // namespace quantiloom::rendercore
