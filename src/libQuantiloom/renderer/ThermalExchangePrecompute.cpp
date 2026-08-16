/**
 * @file ThermalExchangePrecompute.cpp
 * @brief Who sees whom, on the GPU, once per scene
 */

#include "renderer/ThermalExchangePrecompute.hpp"

#include "core/Log.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <climits>
#include <unistd.h>
#endif

namespace quantiloom::rendercore {

namespace {

constexpr u32 kSkyRecord = 0xFFFFFFFFu;

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
};
static_assert(sizeof(ExchangePushConstants) == 32, "ExchangePushConstants size mismatch");

std::filesystem::path ExecutableDirectory() {
#if defined(_WIN32)
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    return std::filesystem::path(buffer).parent_path();
#elif defined(__linux__)
    char buffer[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length != -1) {
        buffer[length] = '\0';
        return std::filesystem::path(buffer).parent_path();
    }
    return std::filesystem::current_path();
#else
    return std::filesystem::current_path();
#endif
}

Vector<u32> LoadSpirv(const String& name) {
    // The same six places the pick pipeline looks, and for the same reason:
    // the .spv sits beside the executable when installed, in src/shaders when
    // run from the repository.
    const auto exeDir = ExecutableDirectory();
    const std::filesystem::path candidates[] = {
        name,
        exeDir / name,
        std::filesystem::path("shaders") / name,
        exeDir / "shaders" / name,
        std::filesystem::path("..") / "shaders" / name,
        std::filesystem::path("src") / "shaders" / name,
    };

    for (const auto& path : candidates) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) continue;
        const auto size = static_cast<usize>(file.tellg());
        if (size == 0 || size % 4 != 0) continue;
        Vector<u32> code(size / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));
        QL_LOG_DEBUG("Thermal exchange: loaded {} from {}", name, path.string());
        return code;
    }
    return {};
}

}  // namespace

struct ThermalExchangePrecompute::Impl {
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
        Vector<VkDescriptorSetLayoutBinding> bindings(5);
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
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
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

    // Hit records are elementCount * rays uints. At 15k elements and 256 rays
    // that is 15 MB, which is why they are a transient rather than something
    // the scene holds on to.
    const VkDeviceSize recordBytes =
        static_cast<VkDeviceSize>(elementCount) * rays * sizeof(u32);

    VmaAllocator allocator = m_impl->context.GetAllocator();

    Vector<ThermalElementGpu> gpuElements(elementCount);
    f32 meanArea = 0.0f;
    for (u32 e = 0; e < elementCount; ++e) {
        gpuElements[e].centre = elements[e].centroid;
        gpuElements[e].area = elements[e].area_m2;
        gpuElements[e].normal = elements[e].normal;
        gpuElements[e].materialId = elements[e].materialId;
        meanArea += elements[e].area_m2;
    }
    meanArea /= static_cast<f32>(elementCount);

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

    // ------------------------------------------------------------------
    // Bind and dispatch
    // ------------------------------------------------------------------
    VkWriteDescriptorSetAccelerationStructureKHR tlasInfo{};
    tlasInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    tlasInfo.accelerationStructureCount = 1;
    tlasInfo.pAccelerationStructures = &tlas;

    const VkDescriptorBufferInfo bufferInfos[4] = {
        {elementBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {baseBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {recordBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
        {sunBuffer.GetHandle(), 0, VK_WHOLE_SIZE},
    };

    Vector<VkWriteDescriptorSet> writes(5);
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
    // A ray has to start clear of the triangle it left, and how far that is
    // depends on how big the scene's triangles are: a fixed epsilon that works
    // in metres self-intersects in kilometres and floats a surface off itself
    // in millimetres. The mean element size is the only length available here.
    pc.rayOffset = std::max(1e-5f, std::sqrt(std::max(meanArea, 0.0f)) * 1e-3f);

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
    const u32* records = static_cast<const u32*>(recordBuffer.Map());
    const f32* sun = static_cast<const f32*>(sunBuffer.Map());

    exchange.viewFactors.rowStart.reserve(elementCount + 1);
    exchange.viewFactors.rowStart.push_back(0);
    exchange.skyFraction.resize(elementCount);
    exchange.sunVisibility.assign(sun, sun + elementCount);

    std::unordered_map<u32, u32> counts;
    Vector<std::pair<u32, u32>> ranked;
    const f32 perRay = 1.0f / static_cast<f32>(rays);

    for (u32 e = 0; e < elementCount; ++e) {
        counts.clear();
        u32 skyHits = 0;
        const u32* row = records + static_cast<usize>(e) * rays;
        for (u32 r = 0; r < rays; ++r) {
            const u32 hit = row[r];
            if (hit == kSkyRecord || hit >= elementCount) {
                ++skyHits;
            } else if (hit != e) {
                ++counts[hit];
            } else {
                // A ray that came back to its own element is the offset losing
                // to a grazing direction. Counting it as sky would cool the
                // surface; counting it as self-exchange does nothing at all,
                // which is the honest answer.
                ++skyHits;
            }
        }

        ranked.assign(counts.begin(), counts.end());
        if (ranked.size() > params.topK) {
            std::nth_element(ranked.begin(), ranked.begin() + params.topK, ranked.end(),
                             [](const auto& a, const auto& b) { return a.second > b.second; });
            ranked.resize(params.topK);
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
        const f32 skyFraction = static_cast<f32>(skyHits) * perRay;
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

    const f32 meanSky =
        std::accumulate(exchange.skyFraction.begin(), exchange.skyFraction.end(), 0.0f) /
        static_cast<f32>(elementCount);
    QL_LOG_INFO("  Thermal exchange: {} elements x {} rays, {} entries kept "
                "(mean sky fraction {:.3f})",
                elementCount, rays, exchange.viewFactors.NonZeros(), meanSky);

    return exchange;
}

}  // namespace quantiloom::rendercore
