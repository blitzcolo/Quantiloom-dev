/**
 * @file ThermalPreview.cpp
 * @brief Interactive thermal solve state
 */

#include "renderer/ThermalPreview.hpp"

#include "core/Log.hpp"
#include "renderer/GpuThermalStepper.hpp"
#include "renderer/ThermalExchangePrecompute.hpp"
#include "scene/Scene.hpp"
#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ThermalMesh.hpp"
#include "thermal/ThermalSolver.hpp"
#include "thermal/ThermalTimeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace quantiloom::rendercore {

namespace {

f32 EmissivityOf(const Material& material) {
    if (!material.irEmissivityCurve.empty()) {
        return material.irEmissivityCurve.front().second;
    }
    return 0.95f - 0.90f * material.metallicFactor;
}

}  // namespace

struct ThermalPreview::Impl {
    VulkanContext& context;
    ThermalSolveParams params;
    std::unordered_map<String, ThermalMaterialParams> materialParams;
    bool enabled = false;
    glm::vec3 fallbackSunDirection{0.0f, 1.0f, 0.0f};

    // Dirty flags
    bool exchangeDirty = true;
    bool sunTableDirty = true;
    bool materialTableDirty = true;
    bool timelineDirty = true;

    // Cached state
    thermal::ThermalMesh mesh;
    Vector<thermal::ThermalMaterial> materials;
    thermal::ExchangeGeometry exchange;
    thermal::SunVisibilityTable sunTable;
    std::unique_ptr<thermal::ThermalTimeline> timeline;
    Vector<std::pair<f64, thermal::ThermalForcing>> forcingSeries;
    thermal::ThermalForcing constantForcing;

    // Steppers
    std::unique_ptr<GpuThermalStepper> gpuStepper;
    thermal::CpuCrankNicolsonStepper cpuStepper;

    // Status tracking
    u32 exchangeRunCount = 0;
    u32 lastElementCount = 0;
    f64 currentTime_h = 0.0;
    f64 lastMinT = 0.0;
    f64 lastMaxT = 0.0;
    f64 lastMeanT = 0.0;
    String lastError;

    explicit Impl(VulkanContext& ctx) : context(ctx) {
        gpuStepper = std::make_unique<GpuThermalStepper>(ctx);
    }

    thermal::IThermalStepper& ChooseStepper() {
        if (gpuStepper && gpuStepper->IsValid() && params.layerCount <= GpuThermalStepper::kMaxNodes) {
            return *gpuStepper;
        }
        return cpuStepper;
    }

    void RebuildMaterialTable(const Scene& scene) {
        materials.resize(scene.materials.size());
        u32 named = 0;
        for (usize m = 0; m < scene.materials.size(); ++m) {
            materials[m] = {};
            materials[m].longwaveEmissivity = EmissivityOf(scene.materials[m]);
            auto it = materialParams.find(scene.materials[m].name);
            if (it != materialParams.end()) {
                const f32 emissivity = materials[m].longwaveEmissivity;
                materials[m].conductivity_W_mK = it->second.conductivity_W_mK;
                materials[m].density_kg_m3 = it->second.density_kg_m3;
                materials[m].specificHeat_J_kgK = it->second.specificHeat_J_kgK;
                materials[m].thickness_m = it->second.thickness_m;
                materials[m].convection_W_m2K = it->second.convection_W_m2K;
                materials[m].shortwaveAbsorptivity = it->second.shortwaveAbsorptivity;
                materials[m].longwaveEmissivity = emissivity;
                materials[m].interiorBoundary = it->second.interiorFixedTemperature
                    ? thermal::InteriorBoundary::FixedTemperature
                    : thermal::InteriorBoundary::Adiabatic;
                materials[m].interiorTemperature_K = it->second.interiorTemperature_K;
                ++named;
            }
        }
        materialTableDirty = false;
    }

    void RebuildExchange(VkAccelerationStructureKHR tlas) {
        ThermalExchangePrecompute precompute(context);
        if (precompute.IsValid() && tlas != VK_NULL_HANDLE && !mesh.elements.empty()) {
            ThermalExchangePrecompute::Params ep;
            ep.hemisphereRays = params.exchangeRays;
            ep.topK = params.exchangeTopK;
            ep.sunDirection = fallbackSunDirection;
            exchange = precompute.Run(tlas, mesh.elements,
                                      mesh.instanceElementBase, ep);
            ++exchangeRunCount;
        } else {
            exchange = thermal::MakeOpenSkyExchange(mesh.elements.size());
        }
        exchangeDirty = false;
        sunTableDirty = false;

        // Single-column sun table from the exchange
        sunTable = {};
        if (!exchange.sunVisibility.empty()) {
            sunTable.sampleTime_h = {params.startTime_h};
            sunTable.visibility = exchange.sunVisibility;
        }
    }

    void RebuildTimeline() {
        forcingSeries = thermal::LoadForcingCsv(params.forcingFile);

        constantForcing = {};
        constantForcing.airTemperature_K = params.airTemperature_K;
        constantForcing.sunIrradiance_W_m2 = params.sunIrradiance_W_m2;
        constantForcing.sunDirection = glm::vec3(fallbackSunDirection);
        constantForcing.skyTemperature_K = params.skyTemperature_K;

        thermal::SunVisibilityTable effectiveTable = sunTable;
        if (effectiveTable.SampleCount() == 0 && !exchange.sunVisibility.empty()) {
            effectiveTable.sampleTime_h = {params.startTime_h};
            effectiveTable.visibility = exchange.sunVisibility;
        }

        thermal::ThermalTimeline::Desc desc;
        desc.startTime_h = params.startTime_h;
        desc.timestep_s = params.timestep_s;
        desc.checkpointStride_h = params.checkpointStride_h;
        desc.nodeCount = params.layerCount;
        desc.initial = params.initial == ThermalInitialCondition::Steady
            ? thermal::InitialCondition::Steady
            : thermal::InitialCondition::Uniform;
        desc.initialTemperature_K = params.initialTemperature_K;

        auto& stepper = ChooseStepper();
        timeline = std::make_unique<thermal::ThermalTimeline>(
            desc, mesh.elements, materials, exchange,
            effectiveTable, forcingSeries, constantForcing, stepper);

        timelineDirty = false;
    }
};

ThermalPreview::ThermalPreview(VulkanContext& context)
    : m_impl(std::make_unique<Impl>(context)) {}

ThermalPreview::~ThermalPreview() = default;

void ThermalPreview::SetParams(const ThermalSolveParams& params) {
    auto& p = m_impl->params;
    if (params.exchangeRays != p.exchangeRays || params.exchangeTopK != p.exchangeTopK) {
        m_impl->exchangeDirty = true;
    }
    if (params.forcingFile != p.forcingFile || params.startTime_h != p.startTime_h) {
        m_impl->sunTableDirty = true;
    }
    m_impl->timelineDirty = true;
    p = params;
}

void ThermalPreview::SetMaterial(const String& name, const ThermalMaterialParams& params) {
    m_impl->materialParams[name] = params;
    m_impl->materialTableDirty = true;
    m_impl->timelineDirty = true;
}

void ThermalPreview::ClearMaterials() {
    m_impl->materialParams.clear();
    m_impl->materialTableDirty = true;
    m_impl->timelineDirty = true;
}

void ThermalPreview::SetEnabled(bool enabled) {
    m_impl->enabled = enabled;
}

void ThermalPreview::SetFallbackSunDirection(const glm::vec3& dir) {
    if (m_impl->params.forcingFile.empty()) {
        m_impl->sunTableDirty = true;
        m_impl->timelineDirty = true;
    }
    m_impl->fallbackSunDirection = glm::normalize(dir);
}

void ThermalPreview::InvalidateGeometry() {
    m_impl->exchangeDirty = true;
    m_impl->sunTableDirty = true;
    m_impl->timelineDirty = true;
}

void ThermalPreview::InvalidateMaterialEmissivity() {
    m_impl->materialTableDirty = true;
    m_impl->timelineDirty = true;
}

ThermalPreview::SolveResult ThermalPreview::SolveAt(
    const f64 time_h, const Scene& scene, const VkAccelerationStructureKHR tlas) {
    SolveResult result;

    if (!m_impl->enabled) {
        result.error = "thermal solve is disabled";
        return result;
    }

    // Rebuild mesh if geometry changed
    if (m_impl->exchangeDirty) {
        m_impl->mesh = thermal::BuildThermalMesh(scene);
    }

    if (m_impl->mesh.elements.empty()) {
        result.error = "the scene has no triangles to solve on";
        return result;
    }

    // Rebuild material table if materials changed
    if (m_impl->materialTableDirty || m_impl->exchangeDirty) {
        m_impl->RebuildMaterialTable(scene);
    }

    // Check at least one participating material
    bool anyParticipating = false;
    for (const auto& el : m_impl->mesh.elements) {
        if (el.area_m2 > 0.0f && el.materialId < m_impl->materials.size() &&
            m_impl->materials[el.materialId].ParticipatesInSolve()) {
            anyParticipating = true;
            break;
        }
    }
    if (!anyParticipating) {
        result.error = "no material in the scene has thermal properties";
        return result;
    }

    // Rebuild exchange if geometry changed
    if (m_impl->exchangeDirty) {
        m_impl->RebuildExchange(tlas);
    }

    // Rebuild timeline if anything changed
    if (m_impl->timelineDirty || !m_impl->timeline) {
        m_impl->RebuildTimeline();
    }

    // Step to the requested time
    const thermal::ThermalState& state = m_impl->timeline->StateAt(time_h);
    m_impl->currentTime_h = time_h;

    // Extract surface temperatures
    const u32 n = static_cast<u32>(m_impl->mesh.elements.size());
    result.surfaceTemperature_K.resize(n);
    for (usize e = 0; e < n; ++e) {
        const u32 id = m_impl->mesh.elements[e].materialId;
        const bool solved = m_impl->mesh.elements[e].area_m2 > 0.0f &&
                            id < m_impl->materials.size() &&
                            m_impl->materials[id].ParticipatesInSolve();
        result.surfaceTemperature_K[e] = solved ? static_cast<f32>(state.Surface(e)) : 0.0f;
    }
    result.instanceElementBase = m_impl->mesh.instanceElementBase;
    result.elementCount = n;
    result.elementCountChanged = (n != m_impl->lastElementCount);
    m_impl->lastElementCount = n;
    m_impl->lastError.clear();

    // Cache temperature stats for Status()
    f64 sum = 0.0;
    u32 count = 0;
    m_impl->lastMinT = std::numeric_limits<f64>::max();
    m_impl->lastMaxT = std::numeric_limits<f64>::lowest();
    for (usize e = 0; e < n; ++e) {
        if (result.surfaceTemperature_K[e] > 0.0f) {
            const f64 T = result.surfaceTemperature_K[e];
            sum += T;
            ++count;
            m_impl->lastMinT = std::min(m_impl->lastMinT, T);
            m_impl->lastMaxT = std::max(m_impl->lastMaxT, T);
        }
    }
    if (count > 0) {
        m_impl->lastMeanT = sum / count;
    } else {
        m_impl->lastMinT = 0.0;
        m_impl->lastMaxT = 0.0;
        m_impl->lastMeanT = 0.0;
    }

    return result;
}

ThermalSolveStatus ThermalPreview::Status() const {
    ThermalSolveStatus status;
    status.enabled = m_impl->enabled;
    status.solveValid = m_impl->timeline != nullptr && m_impl->lastError.empty();
    status.exchangeValid = !m_impl->exchangeDirty;
    status.elementCount = static_cast<u32>(m_impl->mesh.elements.size());
    status.exchangeNonZeros = static_cast<u32>(m_impl->exchange.viewFactors.NonZeros());
    status.exchangeRunCount = m_impl->exchangeRunCount;
    status.currentTime_h = m_impl->currentTime_h;
    status.stepperName = m_impl->gpuStepper && m_impl->gpuStepper->IsValid() &&
                         m_impl->params.layerCount <= GpuThermalStepper::kMaxNodes
                             ? m_impl->gpuStepper->Name()
                             : m_impl->cpuStepper.Name();
    status.error = m_impl->lastError;
    status.sliderStartTime_h = m_impl->params.startTime_h;

    if (m_impl->timeline) {
        status.lastStepCount = m_impl->timeline->LastStepCount();
        status.checkpointCount = m_impl->timeline->CheckpointCount();
        status.participatingElements = m_impl->timeline->ParticipatingElements();
        status.shortestTimeConstant_s = m_impl->timeline->ShortestTimeConstant_s();
    }

    // Determine slider range
    if (!m_impl->forcingSeries.empty()) {
        status.sliderStartTime_h = m_impl->forcingSeries.front().first;
        status.sliderEndTime_h = m_impl->forcingSeries.back().first;
    } else {
        status.sliderEndTime_h = m_impl->params.startTime_h + 24.0;
    }

    // Temperature stats from the last SolveAt call (cached, not recomputed)
    status.minTemperature_K = m_impl->lastMinT;
    status.maxTemperature_K = m_impl->lastMaxT;
    status.meanTemperature_K = m_impl->lastMeanT;

    return status;
}

}  // namespace quantiloom::rendercore
