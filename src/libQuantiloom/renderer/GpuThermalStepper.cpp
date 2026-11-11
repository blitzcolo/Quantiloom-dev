/**
 * @file GpuThermalStepper.cpp
 * @brief GPU Crank-Nicolson thermal stepper (f32)
 */

#include "renderer/GpuThermalStepper.hpp"

#include "core/Log.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/Device.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <type_traits>

// The structs thermal_step.comp.hlsl reads, defined once for the shader and this
// file. By a path relative to this file, as RtBindings.hpp includes its manifest.
#include "../../shaders/ql_shared_thermal.hlsli"


namespace quantiloom::rendercore {

namespace {

using ThermalElementGpu = shader::ThermalElementGpu;
using ThermalMaterialGpu = shader::ThermalMaterialGpu;
using StepPushConstants = shader::StepPushConstants;

/// Bindings the compute shader declares. Buffers 9 and 10 are the baked
/// short-wave gains and 11 is the tangent dT/dv; all three are always bound,
/// with placeholders when there is nothing to put there -- a descriptor must
/// be valid either way. 12 is the long-wave radiosity ping-pong and 13 the
/// irradiance each element gathered, the two halves of the diffuse-grey
/// exchange CpuCrankNicolsonStepper sweeps.
constexpr u32 kBindingCount = 14;

constexpr f32 kStefanBoltzmann = 5.670374419e-8f;

}  // namespace

struct GpuThermalStepper::Impl {
    gpu::Device& device;

    // Owned gpu:: resources, committed together by Create(). Destruction order
    // is pool, kernel, layouts, so a later destruction never names an object
    // an earlier one still needs.
    std::unique_ptr<gpu::BindingLayout> bindingLayout;
    std::unique_ptr<gpu::KernelLayout> kernelLayout;
    std::unique_ptr<gpu::ComputeKernel> kernel;
    std::unique_ptr<gpu::BindGroupPool> pool;
    gpu::BindGroup bindGroup;

    // The static inputs keep an exact copy of the bytes uploaded to the GPU.
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
    std::unique_ptr<gpu::Buffer> elementBuffer;
    std::unique_ptr<gpu::Buffer> materialBuffer;
    std::unique_ptr<gpu::Buffer> csrRowStartBuffer;
    std::unique_ptr<gpu::Buffer> csrColumnBuffer;
    std::unique_ptr<gpu::Buffer> csrValueBuffer;
    std::unique_ptr<gpu::Buffer> skyFractionBuffer;
    std::unique_ptr<gpu::Buffer> sunVisBuffer;
    std::unique_ptr<gpu::Buffer> reflectedGainBuffer;
    std::unique_ptr<gpu::Buffer> diffuseGainBuffer;
    std::unique_ptr<gpu::Buffer> stateBuffer;
    std::unique_ptr<gpu::Buffer> surfaceBuffer;
    std::unique_ptr<gpu::Buffer> sensitivityBuffer;
    std::unique_ptr<gpu::Buffer> radiosityBuffer;
    std::unique_ptr<gpu::Buffer> irradianceBuffer;

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

    explicit Impl(gpu::Device& dev) : device(dev) {}

    bool Create() {
        const StringView kShaderNames[] = {"thermal_step"};
        auto shaders = device.Shaders().Load(kShaderNames);
        if (!shaders.has_value()) {
            QL_LOG_WARN("GPU thermal stepper: thermal_step.spv not found");
            return false;
        }

        // One slot per binding thermal_step.comp.hlsl declares (:52-80):
        // t0-t6 read-only, u7-u8 read-write, t9-t10 read-only, u11-u13
        // read-write.
        constexpr gpu::BindingSlot kSlots[] = {
            {0, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {1, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {2, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {3, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {4, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {5, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {6, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {7, gpu::BindingKind::ReadWriteBuffer, 1, false},
            {8, gpu::BindingKind::ReadWriteBuffer, 1, false},
            {9, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {10, gpu::BindingKind::ReadOnlyBuffer, 1, false},
            {11, gpu::BindingKind::ReadWriteBuffer, 1, false},
            {12, gpu::BindingKind::ReadWriteBuffer, 1, false},
            {13, gpu::BindingKind::ReadWriteBuffer, 1, false},
        };

        auto bindingLayoutResult = device.CreateBindingLayout(
            gpu::BindingLayoutDesc{0, std::span<const gpu::BindingSlot>(kSlots)});
        if (!bindingLayoutResult.has_value()) return false;

        const gpu::BindingLayout* spaces[] = {bindingLayoutResult.value().get()};
        auto kernelLayoutResult = device.CreateKernelLayout(gpu::KernelLayoutDesc{
            std::span<const gpu::BindingLayout* const>(spaces), sizeof(StepPushConstants)});
        if (!kernelLayoutResult.has_value()) return false;

        auto kernelResult = device.CreateComputeKernel(
            gpu::ComputeKernelDesc{&shaders.value()[0], kernelLayoutResult.value().get()});
        if (!kernelResult.has_value()) return false;

        auto poolResult = device.CreateBindGroupPool(*bindingLayoutResult.value(), 1);
        if (!poolResult.has_value()) return false;

        auto groups = poolResult.value()->Allocate(1);
        if (!groups.has_value()) return false;

        this->bindingLayout = std::move(bindingLayoutResult.value());
        this->kernelLayout = std::move(kernelLayoutResult.value());
        this->kernel = std::move(kernelResult.value());
        this->pool = std::move(poolResult.value());
        bindGroup = groups.value()[0];
        return true;
    }

    template <typename T>
    static bool SameBytes(const Vector<T>& a, const Vector<T>& b) {
        static_assert(std::is_trivially_copyable_v<T>);
        return a.size() == b.size() &&
               (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
    }

    bool EnsureBuffer(std::unique_ptr<gpu::Buffer>& buffer, u64 size,
                      gpu::BufferUsage usage) {
        size = std::max(size, u64{1});
        if (buffer && buffer->Ref().size == size) return false;
        auto created = device.CreateBuffer(gpu::BufferDesc{size, usage, gpu::MemoryClass::Upload, 0});
        if (!created.has_value()) {
            // StepMany returns void, so a failed allocation throws rather
            // than leaving the state silently unstepped.
            throw std::runtime_error(created.error());
        }
        buffer = std::move(created.value());
        ++bufferAllocationCount;
        return true;
    }

    template <typename T>
    bool UpdateStaticBuffer(std::unique_ptr<gpu::Buffer>& buffer, Vector<T>& cached,
                            const Vector<T>& values, bool& staticChanged) {
        const bool handleChanged = EnsureBuffer(
            buffer, std::max(values.size(), usize{1}) * sizeof(T),
            gpu::BufferUsage::Storage);
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
    bool UploadDynamicBuffer(std::unique_ptr<gpu::Buffer>& buffer,
                             const Vector<T>& values,
                             gpu::BufferUsage usage) {
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
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferSrc);

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
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferSrc);

        // Surface ping-pong: 2 * n floats
        // Seed both halves from the initial surface temperatures
        Vector<f32> surfInit(2u * n);
        for (u32 e = 0; e < n; ++e) {
            surfInit[e] = stateF32[static_cast<usize>(e) * nodes];       // parity 0
            surfInit[n + e] = stateF32[static_cast<usize>(e) * nodes];   // parity 1
        }
        descriptorsChanged |= UploadDynamicBuffer(
            surfaceBuffer, surfInit, gpu::BufferUsage::Storage);

        // Long-wave radiosity ping-pong, 2 * n floats, both halves seeded with
        // J = eps sigma T^4 + (1 - eps) H from the irradiance the state carries
        // -- or, for a state nothing has stepped yet, H = sigma T^4, the seed
        // CpuCrankNicolsonStepper takes, which makes the first J sigma T^4.
        // The irradiance buffer starts as that H and is rewritten by every
        // dispatch; ReadBack hands the last one back to the state, so the next
        // batch continues the same sweep rather than restarting it.
        const bool carriedIrradiance = state.longwaveIrradiance_W_m2.size() == n;
        Vector<f32> irradianceInit(n);
        Vector<f32> radiosityInit(2u * n);
        for (u32 e = 0; e < n; ++e) {
            const f32 T = stateF32[static_cast<usize>(e) * nodes];
            const f32 emitted = kStefanBoltzmann * T * T * T * T;
            const f32 received = carriedIrradiance
                                     ? static_cast<f32>(state.longwaveIrradiance_W_m2[e])
                                     : emitted;
            const u32 id = elements[e].materialId;
            const f32 eps = id < materials.size() ? materials[id].longwaveEmissivity : 1.0f;
            const f32 J = eps * emitted + (1.0f - eps) * received;
            irradianceInit[e] = received;
            radiosityInit[e] = J;       // parity 0
            radiosityInit[n + e] = J;   // parity 1
        }
        descriptorsChanged |= UploadDynamicBuffer(
            radiosityBuffer, radiosityInit, gpu::BufferUsage::Storage);
        descriptorsChanged |= UploadDynamicBuffer(
            irradianceBuffer, irradianceInit,
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferSrc);

        if (!descriptorsChanged) return;

        // Buffer handles only change on creation or resize. Contents can be
        // refreshed without rewriting the descriptor set.
        const gpu::BindingWrite writes[kBindingCount] = {
            gpu::BindingWrite::Buffer(0, gpu::BindingKind::ReadOnlyBuffer, elementBuffer->Ref()),
            gpu::BindingWrite::Buffer(1, gpu::BindingKind::ReadOnlyBuffer, materialBuffer->Ref()),
            gpu::BindingWrite::Buffer(2, gpu::BindingKind::ReadOnlyBuffer, csrRowStartBuffer->Ref()),
            gpu::BindingWrite::Buffer(3, gpu::BindingKind::ReadOnlyBuffer, csrColumnBuffer->Ref()),
            gpu::BindingWrite::Buffer(4, gpu::BindingKind::ReadOnlyBuffer, csrValueBuffer->Ref()),
            gpu::BindingWrite::Buffer(5, gpu::BindingKind::ReadOnlyBuffer, skyFractionBuffer->Ref()),
            gpu::BindingWrite::Buffer(6, gpu::BindingKind::ReadOnlyBuffer, sunVisBuffer->Ref()),
            gpu::BindingWrite::Buffer(7, gpu::BindingKind::ReadWriteBuffer, stateBuffer->Ref()),
            gpu::BindingWrite::Buffer(8, gpu::BindingKind::ReadWriteBuffer, surfaceBuffer->Ref()),
            gpu::BindingWrite::Buffer(9, gpu::BindingKind::ReadOnlyBuffer, reflectedGainBuffer->Ref()),
            gpu::BindingWrite::Buffer(10, gpu::BindingKind::ReadOnlyBuffer, diffuseGainBuffer->Ref()),
            gpu::BindingWrite::Buffer(11, gpu::BindingKind::ReadWriteBuffer, sensitivityBuffer->Ref()),
            gpu::BindingWrite::Buffer(12, gpu::BindingKind::ReadWriteBuffer, radiosityBuffer->Ref()),
            gpu::BindingWrite::Buffer(13, gpu::BindingKind::ReadWriteBuffer, irradianceBuffer->Ref()),
        };
        device.WriteBindGroup(bindGroup, std::span<const gpu::BindingWrite>(writes));
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

        // Every element's, participating or not: each one reflects, and the
        // next batch seeds its radiosity from this.
        const f32* gathered = static_cast<const f32*>(irradianceBuffer->MapRead());
        if (!gathered) return;
        state.longwaveIrradiance_W_m2.resize(n);
        for (usize e = 0; e < n; ++e) {
            state.longwaveIrradiance_W_m2[e] = static_cast<f64>(gathered[e]);
        }
        irradianceBuffer->Unmap();
    }
};

GpuThermalStepper::GpuThermalStepper(gpu::Device& device)
    : m_impl(std::make_unique<Impl>(device)) {
    if (!m_impl->Create()) {
        m_impl->kernel.reset();
        m_impl->bindGroup = {};
    }
}

GpuThermalStepper::~GpuThermalStepper() = default;

bool GpuThermalStepper::IsValid() const {
    return m_impl && m_impl->kernel && m_impl->bindGroup.IsValid();
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

        m_impl->device.ExecuteImmediate([&](gpu::CommandList& cmd) {
            cmd.SetKernel(*m_impl->kernel);
            cmd.SetBindGroup(*m_impl->kernelLayout, 0, m_impl->bindGroup);

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

                cmd.PushConstants(*m_impl->kernelLayout, pc);
                cmd.Dispatch(groups, 1, 1);
                parity = 1 - parity;

                if (s + 1 < batchSize) {
                    cmd.Barrier(gpu::GlobalBarrier{gpu::Access::ComputeWrite,
                                                   gpu::Access::ComputeRead |
                                                       gpu::Access::ComputeWrite});
                }
            }

            // Final barrier for host readback
            cmd.Barrier(gpu::GlobalBarrier{gpu::Access::ComputeWrite, gpu::Access::HostRead});
        });
    }

    m_impl->ReadBack(state, elements, materials);
}

}  // namespace quantiloom::rendercore
