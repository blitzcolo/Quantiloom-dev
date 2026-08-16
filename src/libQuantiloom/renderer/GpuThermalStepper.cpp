/**
 * @file GpuThermalStepper.cpp
 * @brief GPU Crank-Nicolson thermal stepper (f32)
 */

#include "renderer/GpuThermalStepper.hpp"

#include "core/Log.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <climits>
#include <unistd.h>
#endif

namespace quantiloom::rendercore {

namespace {

struct ThermalElementGpu {
    glm::vec3 centre;
    f32 area;
    glm::vec3 normal;
    u32 materialId;
};
static_assert(sizeof(ThermalElementGpu) == 32);

struct ThermalMaterialGpu {
    f32 conductivity;
    f32 density;
    f32 specificHeat;
    f32 thickness;
    f32 convection;
    f32 shortwaveAbsorptivity;
    f32 longwaveEmissivity;
    u32 interiorBcFixed;
    f32 interiorTemperature;
    f32 pad;
};
static_assert(sizeof(ThermalMaterialGpu) == 40);

struct StepPushConstants {
    glm::vec3 sunDirection;
    f32 dt_s;
    f32 airTemperature_K;
    f32 sunIrradiance;
    f32 skyTemperature_K;
    f32 sunBlend;
    u32 elementCount;
    u32 nodeCount;
    u32 parity;
    u32 sunSampleA;
    u32 sunSampleB;
    u32 pad0;
    u32 pad1;
    u32 pad2;
};
static_assert(sizeof(StepPushConstants) == 64);

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
        return code;
    }
    return {};
}

}  // namespace

struct GpuThermalStepper::Impl {
    VulkanContext& context;
    VkDevice device = VK_NULL_HANDLE;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    // GPU buffers, created per PrepareStatic / StepMany call
    std::unique_ptr<GpuBuffer> elementBuffer;
    std::unique_ptr<GpuBuffer> materialBuffer;
    std::unique_ptr<GpuBuffer> csrRowStartBuffer;
    std::unique_ptr<GpuBuffer> csrColumnBuffer;
    std::unique_ptr<GpuBuffer> csrValueBuffer;
    std::unique_ptr<GpuBuffer> skyFractionBuffer;
    std::unique_ptr<GpuBuffer> sunVisBuffer;
    std::unique_ptr<GpuBuffer> stateBuffer;
    std::unique_ptr<GpuBuffer> surfaceBuffer;

    u32 lastElementCount = 0;
    u32 lastNodeCount = 0;

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
        const Vector<u32> code = LoadSpirv("thermal_step.spv");
        if (code.empty()) {
            QL_LOG_WARN("GPU thermal stepper: thermal_step.spv not found");
            return false;
        }

        VkShaderModuleCreateInfo moduleInfo{};
        moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleInfo.codeSize = code.size() * sizeof(u32);
        moduleInfo.pCode = code.data();
        if (vkCreateShaderModule(device, &moduleInfo, nullptr, &shader) != VK_SUCCESS) {
            return false;
        }

        // 9 bindings: 0-6 SRV/SRV/SRV/SRV/SRV/SRV/SRV, 7-8 UAV
        Vector<VkDescriptorSetLayoutBinding> bindings(9);
        for (u32 i = 0; i < 9; ++i) {
            bindings[i] = {};
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 9;
        layoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                        &descriptorSetLayout) != VK_SUCCESS) {
            return false;
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(StepPushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                   &pipelineLayout) != VK_SUCCESS) {
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                     nullptr, &pipeline) != VK_SUCCESS) {
            pipeline = VK_NULL_HANDLE;
            return false;
        }

        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9};
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            return false;
        }

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            descriptorSet = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }

    void UploadAndBind(const Vector<thermal::ThermalElement>& elements,
                       const Vector<thermal::ThermalMaterial>& materials,
                       const thermal::ExchangeGeometry& exchange,
                       const thermal::SunVisibilityTable& sunTable,
                       const thermal::ThermalState& state) {
        VmaAllocator alloc = context.GetAllocator();
        const u32 n = static_cast<u32>(elements.size());
        const u32 nodes = state.nodeCount;
        lastElementCount = n;
        lastNodeCount = nodes;

        // Elements
        Vector<ThermalElementGpu> gpuElements(n);
        for (u32 e = 0; e < n; ++e) {
            gpuElements[e].centre = elements[e].centroid;
            gpuElements[e].area = elements[e].area_m2;
            gpuElements[e].normal = elements[e].normal;
            gpuElements[e].materialId = elements[e].materialId;
        }
        elementBuffer = std::make_unique<GpuBuffer>(
            alloc, n * sizeof(ThermalElementGpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        elementBuffer->Upload(gpuElements.data(), n * sizeof(ThermalElementGpu));

        // Materials
        Vector<ThermalMaterialGpu> gpuMats(materials.size());
        for (usize m = 0; m < materials.size(); ++m) {
            gpuMats[m].conductivity = materials[m].conductivity_W_mK;
            gpuMats[m].density = materials[m].density_kg_m3;
            gpuMats[m].specificHeat = materials[m].specificHeat_J_kgK;
            gpuMats[m].thickness = materials[m].thickness_m;
            gpuMats[m].convection = materials[m].convection_W_m2K;
            gpuMats[m].shortwaveAbsorptivity = materials[m].shortwaveAbsorptivity;
            gpuMats[m].longwaveEmissivity = materials[m].longwaveEmissivity;
            gpuMats[m].interiorBcFixed =
                materials[m].interiorBoundary == thermal::InteriorBoundary::FixedTemperature ? 1u : 0u;
            gpuMats[m].interiorTemperature = materials[m].interiorTemperature_K;
            gpuMats[m].pad = 0.0f;
        }
        materialBuffer = std::make_unique<GpuBuffer>(
            alloc, gpuMats.size() * sizeof(ThermalMaterialGpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        materialBuffer->Upload(gpuMats.data(), gpuMats.size() * sizeof(ThermalMaterialGpu));

        // CSR
        csrRowStartBuffer = std::make_unique<GpuBuffer>(
            alloc, exchange.viewFactors.rowStart.size() * sizeof(u32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        csrRowStartBuffer->Upload(exchange.viewFactors.rowStart.data(),
                                  exchange.viewFactors.rowStart.size() * sizeof(u32));

        const VkDeviceSize colBytes = std::max(exchange.viewFactors.column.size(), usize{1}) * sizeof(u32);
        csrColumnBuffer = std::make_unique<GpuBuffer>(
            alloc, colBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        if (!exchange.viewFactors.column.empty()) {
            csrColumnBuffer->Upload(exchange.viewFactors.column.data(),
                                    exchange.viewFactors.column.size() * sizeof(u32));
        }

        const VkDeviceSize valBytes = std::max(exchange.viewFactors.value.size(), usize{1}) * sizeof(f32);
        csrValueBuffer = std::make_unique<GpuBuffer>(
            alloc, valBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        if (!exchange.viewFactors.value.empty()) {
            csrValueBuffer->Upload(exchange.viewFactors.value.data(),
                                   exchange.viewFactors.value.size() * sizeof(f32));
        }

        // Sky fraction
        skyFractionBuffer = std::make_unique<GpuBuffer>(
            alloc, n * sizeof(f32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        skyFractionBuffer->Upload(exchange.skyFraction.data(), n * sizeof(f32));

        // Sun visibility table
        const usize sunSize = std::max(sunTable.visibility.size(), usize{n});
        sunVisBuffer = std::make_unique<GpuBuffer>(
            alloc, sunSize * sizeof(f32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        if (!sunTable.visibility.empty()) {
            sunVisBuffer->Upload(sunTable.visibility.data(),
                                 sunTable.visibility.size() * sizeof(f32));
        } else if (!exchange.sunVisibility.empty()) {
            sunVisBuffer->Upload(exchange.sunVisibility.data(), n * sizeof(f32));
        }

        // State: f64 → f32 upload
        const usize stateSize = static_cast<usize>(n) * nodes;
        Vector<f32> stateF32(stateSize);
        for (usize i = 0; i < stateSize; ++i) {
            stateF32[i] = static_cast<f32>(state.temperature_K[i]);
        }
        stateBuffer = std::make_unique<GpuBuffer>(
            alloc, stateSize * sizeof(f32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
        stateBuffer->Upload(stateF32.data(), stateSize * sizeof(f32));

        // Surface ping-pong: 2 * n floats
        surfaceBuffer = std::make_unique<GpuBuffer>(
            alloc, 2u * n * sizeof(f32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        // Seed both halves from the initial surface temperatures
        Vector<f32> surfInit(2u * n);
        for (u32 e = 0; e < n; ++e) {
            surfInit[e] = stateF32[static_cast<usize>(e) * nodes];       // parity 0
            surfInit[n + e] = stateF32[static_cast<usize>(e) * nodes];   // parity 1
        }
        surfaceBuffer->Upload(surfInit.data(), surfInit.size() * sizeof(f32));

        // Bind descriptors
        const VkDescriptorBufferInfo infos[9] = {
            {elementBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {materialBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {csrRowStartBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {csrColumnBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {csrValueBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {skyFractionBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {sunVisBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {stateBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {surfaceBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
        };
        Vector<VkWriteDescriptorSet> writes(9);
        for (u32 i = 0; i < 9; ++i) {
            writes[i] = {};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 9, writes.data(), 0, nullptr);
    }

    void ReadBack(thermal::ThermalState& state,
                  const Vector<thermal::ThermalElement>& elements,
                  const Vector<thermal::ThermalMaterial>& materials) {
        const u32 n = lastElementCount;
        const u32 nodes = lastNodeCount;

        const f32* mapped = static_cast<const f32*>(stateBuffer->Map());
        for (usize e = 0; e < n; ++e) {
            const u32 matId = elements[e].materialId;
            if (matId >= materials.size() || !materials[matId].ParticipatesInSolve()) {
                continue;
            }
            for (u32 i = 0; i < nodes; ++i) {
                state.temperature_K[e * nodes + i] =
                    static_cast<f64>(mapped[e * nodes + i]);
            }
        }
        stateBuffer->Unmap();
    }
};

GpuThermalStepper::GpuThermalStepper(VulkanContext& context)
    : m_impl(std::make_unique<Impl>(context)) {
    if (!m_impl->Create()) {
        m_impl->pipeline = VK_NULL_HANDLE;
    }
}

GpuThermalStepper::~GpuThermalStepper() = default;

bool GpuThermalStepper::IsValid() const {
    return m_impl && m_impl->pipeline != VK_NULL_HANDLE &&
           m_impl->descriptorSet != VK_NULL_HANDLE;
}

void GpuThermalStepper::Step(thermal::ThermalState& state,
                             const Vector<thermal::ThermalElement>& elements,
                             const Vector<thermal::ThermalMaterial>& materials,
                             const thermal::ExchangeGeometry& exchange,
                             const thermal::ThermalForcing& forcing, const f64 dt_s,
                             std::span<const f32> sunVisibility) {
    if (!IsValid() || state.nodeCount > kMaxNodes) {
        return;
    }

    // Wrap in a single-column sun table and single-step batch
    thermal::SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0};
    sunTable.visibility.assign(sunVisibility.begin(), sunVisibility.end());

    thermal::ThermalBatchStep bs;
    bs.forcing = forcing;
    bs.dt_s = dt_s;
    bs.sunSampleA = 0;
    bs.sunSampleB = 0;
    bs.sunBlend = 0.0;

    StepMany(state, elements, materials, exchange, sunTable, {&bs, 1});
}

void GpuThermalStepper::StepMany(thermal::ThermalState& state,
                                 const Vector<thermal::ThermalElement>& elements,
                                 const Vector<thermal::ThermalMaterial>& materials,
                                 const thermal::ExchangeGeometry& exchange,
                                 const thermal::SunVisibilityTable& sunTable,
                                 std::span<const thermal::ThermalBatchStep> steps) {
    if (!IsValid() || steps.empty() || elements.empty() || state.nodeCount > kMaxNodes) {
        return;
    }

    const u32 n = static_cast<u32>(elements.size());
    const u32 nodes = state.nodeCount;
    const u32 groups = (n + 63) / 64;

    m_impl->UploadAndBind(elements, materials, exchange, sunTable, state);

    constexpr u32 kMaxDispatchBatch = 512;
    u32 parity = 0;

    for (usize offset = 0; offset < steps.size(); offset += kMaxDispatchBatch) {
        const usize batchSize = std::min(static_cast<usize>(kMaxDispatchBatch),
                                         steps.size() - offset);

        CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    m_impl->pipelineLayout, 0, 1,
                                    &m_impl->descriptorSet, 0, nullptr);

            for (usize s = 0; s < batchSize; ++s) {
                const thermal::ThermalBatchStep& step = steps[offset + s];

                StepPushConstants pc{};
                pc.sunDirection = step.forcing.sunDirection;
                pc.dt_s = static_cast<f32>(step.dt_s);
                pc.airTemperature_K = static_cast<f32>(step.forcing.airTemperature_K);
                pc.sunIrradiance = static_cast<f32>(step.forcing.sunIrradiance_W_m2);
                pc.skyTemperature_K = static_cast<f32>(step.forcing.skyTemperature_K);
                pc.sunBlend = static_cast<f32>(step.sunBlend);
                pc.elementCount = n;
                pc.nodeCount = nodes;
                pc.parity = parity;
                pc.sunSampleA = static_cast<u32>(step.sunSampleA);
                pc.sunSampleB = static_cast<u32>(step.sunSampleB);

                vkCmdPushConstants(cmd, m_impl->pipelineLayout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(pc), &pc);
                vkCmdDispatch(cmd, groups, 1, 1);
                parity = 1 - parity;

                if (s + 1 < batchSize) {
                    VkMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    barrier.dstAccessMask =
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                         1, &barrier, 0, nullptr, 0, nullptr);
                }
            }

            // Final barrier for host readback
            VkMemoryBarrier hostBarrier{};
            hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, 0,
                                 1, &hostBarrier, 0, nullptr, 0, nullptr);
        });
    }

    m_impl->ReadBack(state, elements, materials);
}

}  // namespace quantiloom::rendercore
