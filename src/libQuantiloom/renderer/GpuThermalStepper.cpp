/**
 * @file GpuThermalStepper.cpp
 * @brief GPU Crank-Nicolson thermal stepper (f32)
 */

#include "renderer/GpuThermalStepper.hpp"

#include "core/Log.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/ShaderBinary.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>


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
    f32 wetnessFactor;
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
    f32 diffuseIrradiance;
    f32 relativeHumidity;
    /// The forcing's convective coefficient, or 0 to mean the material's own.
    /// Mirrors ThermalForcing::convection_W_m2K so that a trajectory stepped
    /// on the GPU matches one stepped on the CPU; test_thermal_step_gpu
    /// compares them over 24 h and would otherwise start disagreeing the
    /// moment a forcing file carried a ninth column.
    f32 convection_W_m2K;
    u32 hasReflectedGain;
    u32 carryTangent;
};
static_assert(sizeof(StepPushConstants) == 72);

/// Bindings the compute shader declares. Buffers 9 and 10 are the baked
/// short-wave gains and 11 is the tangent dT/dv; all three are always bound,
/// with placeholders when there is nothing to put there -- Vulkan has no
/// notion of an optional descriptor here.
constexpr u32 kBindingCount = 12;

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

    // The static inputs keep an exact copy of the bytes uploaded to Vulkan.
    // The timeline mutates its epoch/material/sun-table storage in place, so
    // pointer identity cannot say whether any of these bindings is current.
    Vector<ThermalElementGpu> cachedElements;
    Vector<ThermalMaterialGpu> cachedMaterials;
    Vector<u32> cachedCsrRowStart;
    Vector<u32> cachedCsrColumn;
    Vector<f32> cachedCsrValue;
    Vector<f32> cachedSkyFraction;
    Vector<f32> cachedSunVisibility;
    Vector<f32> cachedReflectedGain;
    Vector<f32> cachedDiffuseGain;

    // GPU buffers survive StepMany calls. Static buffers are uploaded only
    // when the exact bytes above change; dynamic buffers are uploaded on every
    // call but keep their allocations while their sizes are unchanged.
    std::unique_ptr<GpuBuffer> elementBuffer;
    std::unique_ptr<GpuBuffer> materialBuffer;
    std::unique_ptr<GpuBuffer> csrRowStartBuffer;
    std::unique_ptr<GpuBuffer> csrColumnBuffer;
    std::unique_ptr<GpuBuffer> csrValueBuffer;
    std::unique_ptr<GpuBuffer> skyFractionBuffer;
    std::unique_ptr<GpuBuffer> sunVisBuffer;
    std::unique_ptr<GpuBuffer> reflectedGainBuffer;
    std::unique_ptr<GpuBuffer> diffuseGainBuffer;
    std::unique_ptr<GpuBuffer> stateBuffer;
    std::unique_ptr<GpuBuffer> surfaceBuffer;
    std::unique_ptr<GpuBuffer> sensitivityBuffer;

    u32 lastElementCount = 0;
    u32 lastNodeCount = 0;
    /// False when binding 9 holds a placeholder, which the shader must not
    /// index past its first element.
    bool hasReflectedGain = false;
    /// False when binding 11 holds a placeholder, i.e. the caller sized no
    /// tangent into the state and does not want one stepped.
    bool carryTangent = false;
    usize staticInputGeneration = 0;
    usize bufferAllocationCount = 0;

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

        // Bindings 0-6 and 9-10 read-only, 7-8 and 11 read-write
        Vector<VkDescriptorSetLayoutBinding> bindings(kBindingCount);
        for (u32 i = 0; i < kBindingCount; ++i) {
            bindings[i] = {};
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = kBindingCount;
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

        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindingCount};
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

    template <typename T>
    static bool SameBytes(const Vector<T>& a, const Vector<T>& b) {
        static_assert(std::is_trivially_copyable_v<T>);
        return a.size() == b.size() &&
               (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
    }

    bool EnsureBuffer(std::unique_ptr<GpuBuffer>& buffer, VkDeviceSize size,
                      VkBufferUsageFlags usage) {
        size = std::max(size, VkDeviceSize{1});
        if (buffer && buffer->GetSize() == size) return false;
        buffer = std::make_unique<GpuBuffer>(context.GetAllocator(), size, usage,
                                             VMA_MEMORY_USAGE_CPU_TO_GPU);
        ++bufferAllocationCount;
        return true;
    }

    template <typename T>
    bool UpdateStaticBuffer(std::unique_ptr<GpuBuffer>& buffer, Vector<T>& cached,
                            const Vector<T>& values, bool& staticChanged) {
        const bool handleChanged = EnsureBuffer(
            buffer, std::max(values.size(), usize{1}) * sizeof(T),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if (handleChanged || !SameBytes(cached, values)) {
            if (values.empty()) {
                const T zero{};
                buffer->Upload(&zero, sizeof(T));
            } else {
                buffer->Upload(values.data(), values.size() * sizeof(T));
            }
            cached = values;
            staticChanged = true;
        }
        return handleChanged;
    }

    template <typename T>
    bool UploadDynamicBuffer(std::unique_ptr<GpuBuffer>& buffer,
                             const Vector<T>& values,
                             VkBufferUsageFlags usage) {
        const bool handleChanged = EnsureBuffer(
            buffer, std::max(values.size(), usize{1}) * sizeof(T), usage);
        if (values.empty()) {
            const T zero{};
            buffer->Upload(&zero, sizeof(T));
        } else {
            buffer->Upload(values.data(), values.size() * sizeof(T));
        }
        return handleChanged;
    }

    void UploadAndBind(const Vector<thermal::ThermalElement>& elements,
                       const Vector<thermal::ThermalMaterial>& materials,
                       const thermal::ExchangeGeometry& exchange,
                       const thermal::SunVisibilityTable& sunTable,
                       const thermal::ThermalState& state) {
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
            gpuMats[m].wetnessFactor = materials[m].wetnessFactor;
        }
        // Canonicalise the fallbacks before comparing. This makes the cache key
        // exactly the bytes the shader can read rather than every field in the
        // source objects (sample times/directions are consumed by the caller).
        Vector<f32> skyFraction(n, 0.0f);
        std::copy_n(exchange.skyFraction.begin(),
                    std::min(exchange.skyFraction.size(), usize{n}),
                    skyFraction.begin());

        const usize sunSize = std::max(sunTable.visibility.size(), usize{n});
        Vector<f32> sunVisibility(sunSize, 0.0f);
        if (!sunTable.visibility.empty()) {
            std::copy(sunTable.visibility.begin(), sunTable.visibility.end(),
                      sunVisibility.begin());
        } else if (!exchange.sunVisibility.empty()) {
            std::copy_n(exchange.sunVisibility.begin(),
                        std::min(exchange.sunVisibility.size(), usize{n}),
                        sunVisibility.begin());
        }

        hasReflectedGain = sunTable.reflectedGain.size() == sunTable.visibility.size() &&
                           !sunTable.reflectedGain.empty();
        Vector<f32> reflectedGain = hasReflectedGain
                                        ? sunTable.reflectedGain
                                        : Vector<f32>{0.0f};

        Vector<f32> diffuseGain(n, 0.0f);
        if (sunTable.diffuseGain.size() == n) {
            diffuseGain = sunTable.diffuseGain;
        } else {
            diffuseGain = skyFraction;
        }

        bool descriptorsChanged = false;
        bool staticChanged = false;
        descriptorsChanged |= UpdateStaticBuffer(elementBuffer, cachedElements,
                                                  gpuElements, staticChanged);
        descriptorsChanged |= UpdateStaticBuffer(materialBuffer, cachedMaterials,
                                                  gpuMats, staticChanged);
        descriptorsChanged |= UpdateStaticBuffer(csrRowStartBuffer, cachedCsrRowStart,
                                                  exchange.viewFactors.rowStart,
                                                  staticChanged);
        descriptorsChanged |= UpdateStaticBuffer(csrColumnBuffer, cachedCsrColumn,
                                                  exchange.viewFactors.column,
                                                  staticChanged);
        descriptorsChanged |= UpdateStaticBuffer(csrValueBuffer, cachedCsrValue,
                                                  exchange.viewFactors.value,
                                                  staticChanged);
        descriptorsChanged |= UpdateStaticBuffer(skyFractionBuffer, cachedSkyFraction,
                                                  skyFraction, staticChanged);
        descriptorsChanged |= UpdateStaticBuffer(sunVisBuffer, cachedSunVisibility,
                                                  sunVisibility, staticChanged);
        descriptorsChanged |= UpdateStaticBuffer(reflectedGainBuffer,
                                                  cachedReflectedGain,
                                                  reflectedGain, staticChanged);
        descriptorsChanged |= UpdateStaticBuffer(diffuseGainBuffer, cachedDiffuseGain,
                                                  diffuseGain, staticChanged);
        if (staticChanged) ++staticInputGeneration;

        // State: f64 → f32 upload
        const usize stateSize = static_cast<usize>(n) * nodes;
        Vector<f32> stateF32(stateSize);
        for (usize i = 0; i < stateSize; ++i) {
            stateF32[i] = static_cast<f32>(state.temperature_K[i]);
        }
        descriptorsChanged |= UploadDynamicBuffer(
            stateBuffer, stateF32,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        // Tangent: same layout as the state, uploaded only when the caller is
        // carrying one. A one-float placeholder otherwise -- the descriptor
        // must be valid either way, and pc.carryTangent is what keeps the
        // shader off it.
        carryTangent = state.HasSensitivity();
        Vector<f32> sensitivityF32(carryTangent ? stateSize : 1, 0.0f);
        if (carryTangent) {
            for (usize i = 0; i < stateSize; ++i) {
                sensitivityF32[i] = static_cast<f32>(state.sunSensitivity_K[i]);
            }
        }
        descriptorsChanged |= UploadDynamicBuffer(
            sensitivityBuffer, sensitivityF32,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        // Surface ping-pong: 2 * n floats
        // Seed both halves from the initial surface temperatures
        Vector<f32> surfInit(2u * n);
        for (u32 e = 0; e < n; ++e) {
            surfInit[e] = stateF32[static_cast<usize>(e) * nodes];       // parity 0
            surfInit[n + e] = stateF32[static_cast<usize>(e) * nodes];   // parity 1
        }
        descriptorsChanged |= UploadDynamicBuffer(
            surfaceBuffer, surfInit, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        if (!descriptorsChanged) return;

        // Buffer handles only change on creation or resize. Contents can be
        // refreshed without rewriting the descriptor set.
        const VkDescriptorBufferInfo infos[kBindingCount] = {
            {elementBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {materialBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {csrRowStartBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {csrColumnBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {csrValueBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {skyFractionBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {sunVisBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {stateBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {surfaceBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {reflectedGainBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {diffuseGainBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
            {sensitivityBuffer->GetHandle(), 0, VK_WHOLE_SIZE},
        };
        Vector<VkWriteDescriptorSet> writes(kBindingCount);
        for (u32 i = 0; i < kBindingCount; ++i) {
            writes[i] = {};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, kBindingCount, writes.data(), 0, nullptr);
    }

    void ReadBack(thermal::ThermalState& state,
                  const Vector<thermal::ThermalElement>& elements,
                  const Vector<thermal::ThermalMaterial>& materials) {
        const u32 n = lastElementCount;
        const u32 nodes = lastNodeCount;

        const f32* mapped = static_cast<const f32*>(stateBuffer->MapRead());
        if (!mapped) return;
        const f32* tangent = carryTangent && state.HasSensitivity()
                                 ? static_cast<const f32*>(sensitivityBuffer->MapRead())
                                 : nullptr;
        for (usize e = 0; e < n; ++e) {
            const u32 matId = elements[e].materialId;
            if (matId >= materials.size() || !materials[matId].ParticipatesInSolve()) {
                continue;
            }
            for (u32 i = 0; i < nodes; ++i) {
                state.temperature_K[e * nodes + i] =
                    static_cast<f64>(mapped[e * nodes + i]);
                if (tangent) {
                    state.sunSensitivity_K[e * nodes + i] =
                        static_cast<f64>(tangent[e * nodes + i]);
                }
            }
        }
        if (tangent) sensitivityBuffer->Unmap();
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

usize GpuThermalStepper::StaticInputGenerationForTesting() const {
    return m_impl ? m_impl->staticInputGeneration : 0;
}

usize GpuThermalStepper::BufferAllocationCountForTesting() const {
    return m_impl ? m_impl->bufferAllocationCount : 0;
}

void GpuThermalStepper::Step(thermal::ThermalState& state,
                             const Vector<thermal::ThermalElement>& elements,
                             const Vector<thermal::ThermalMaterial>& materials,
                             const thermal::ExchangeGeometry& exchange,
                             const thermal::ThermalForcing& forcing, const f64 dt_s,
                             const thermal::ShortwaveSample& shortwave) {
    if (!IsValid() || state.nodeCount > kMaxNodes) {
        return;
    }

    // Wrap in a single-column sun table and single-step batch. The gains are
    // already interpolated, so the column is them verbatim.
    thermal::SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0};
    sunTable.visibility.assign(shortwave.sunVisibility.begin(),
                               shortwave.sunVisibility.end());
    sunTable.reflectedGain.assign(shortwave.reflectedGain.begin(),
                                  shortwave.reflectedGain.end());
    sunTable.diffuseGain.assign(shortwave.diffuseGain.begin(),
                                shortwave.diffuseGain.end());

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
                pc.diffuseIrradiance =
                    static_cast<f32>(step.forcing.diffuseIrradiance_W_m2);
                pc.relativeHumidity = static_cast<f32>(step.forcing.relativeHumidity);
                pc.convection_W_m2K =
                    static_cast<f32>(step.forcing.convection_W_m2K);
                pc.hasReflectedGain = m_impl->hasReflectedGain ? 1u : 0u;
                pc.carryTangent = m_impl->carryTangent ? 1u : 0u;

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
