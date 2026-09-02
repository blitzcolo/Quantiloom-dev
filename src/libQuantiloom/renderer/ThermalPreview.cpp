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
#include "thermal/ShortwaveGains.hpp"
#include "thermal/ThermalMesh.hpp"
#include "thermal/ThermalSolver.hpp"
#include "thermal/ThermalTimeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace quantiloom::rendercore {

namespace {

f32 EmissivityOf(const Material& material) {
    if (material.bandAveragedIREmissivity >= 0.0f) {
        return material.bandAveragedIREmissivity;
    }
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
    bool lateralDirty = true;

    // Cached state
    thermal::ThermalMesh mesh;
    Vector<thermal::ThermalMaterial> materials;

    /// Per material, the fraction of its area that is actually there. Built
    /// beside the material table, from CPU texels the loader retained.
    Vector<f32> materialCoverage;
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

    /// The per-element field at one instant: the temperature, the tangent the
    /// shading pass corrects a shadow with, the visibility that tangent was
    /// taken about, and where the sun was. One function because a dump has to
    /// describe the same field the viewport is showing -- reading the state
    /// twice, in two places, is how the two would come to disagree.
    ///
    /// @return the element count, so callers do not size it a second way
    u32 ExtractField(const thermal::ThermalState& state, const f64 time_h,
                     Vector<f32>& temperature_K, Vector<f32>& sunSensitivity_K,
                     Vector<f32>& visibility, glm::vec3& sunDirection,
                     Vector<f32>* lagSensitivity_K = nullptr,
                     Vector<f32>* lagVisibility = nullptr,
                     Vector<glm::vec3>* lagDirection = nullptr) const {
        const u32 n = static_cast<u32>(mesh.elements.size());
        temperature_K.assign(n, 0.0f);
        // Left empty rather than zeroed when the tangent was not carried: zero
        // is a temperature that does not move, which is a different claim from
        // not having asked for one.
        sunSensitivity_K.clear();
        visibility.clear();
        const bool haveTangent = state.HasSensitivity();
        if (haveTangent) {
            sunSensitivity_K.assign(n, 0.0f);
            visibility = thermal::SampleSunVisibilityAt(sunTable, exchange, time_h, n);
        }
        sunDirection =
            thermal::SampleForcing(forcingSeries, time_h, constantForcing).sunDirection;
        for (usize e = 0; e < n; ++e) {
            const u32 id = mesh.elements[e].materialId;
            const bool solved = mesh.elements[e].area_m2 > 0.0f &&
                                id < materials.size() &&
                                materials[id].ParticipatesInSolve();
            temperature_K[e] = solved ? static_cast<f32>(state.Surface(e)) : 0.0f;
            if (haveTangent) {
                sunSensitivity_K[e] =
                    solved ? static_cast<f32>(state.SurfaceSensitivity(e)) : 0.0f;
                // Zero where the sun is behind the element: its visibility is
                // zero at any resolution, so the correction is too. See the
                // offline solver for why the shader cannot make this test
                // itself.
                if (glm::dot(mesh.elements[e].normal, sunDirection) <= 0.0f) {
                    sunSensitivity_K[e] = 0.0f;
                }
            }
        }

        // The same three things per tracked sun column. A slot the table
        // cannot place the sun for is left at zero, and what it holds stays
        // inside sunSensitivity_K, which is where the shading pass looks when
        // a column is not carried.
        if (lagSensitivity_K == nullptr) return n;
        lagSensitivity_K->clear();
        lagVisibility->clear();
        lagDirection->clear();
        if (!haveTangent || !state.HasLagSensitivity() ||
            sunTable.sampleDirection.size() != sunTable.SampleCount()) {
            return n;
        }

        const u32 slots = state.LagSlots();
        lagSensitivity_K->assign(static_cast<usize>(slots) * n, 0.0f);
        lagVisibility->assign(static_cast<usize>(slots) * n, 0.0f);
        lagDirection->assign(slots, glm::vec3(0.0f));
        for (u32 s = 0; s < slots; ++s) {
            const u32 column = state.lagColumn[s];
            if (column >= sunTable.SampleCount()) continue;
            const glm::vec3 direction = sunTable.sampleDirection[column];
            (*lagDirection)[s] = direction;
            const f32* columnVisibility = sunTable.Column(column);
            for (usize e = 0; e < n; ++e) {
                const u32 id = mesh.elements[e].materialId;
                if (!(mesh.elements[e].area_m2 > 0.0f) || id >= materials.size() ||
                    !materials[id].ParticipatesInSolve()) {
                    continue;
                }
                (*lagVisibility)[s * n + e] = columnVisibility[e];
                if (glm::dot(mesh.elements[e].normal, direction) > 0.0f) {
                    (*lagSensitivity_K)[s * n + e] =
                        static_cast<f32>(state.SurfaceLagSensitivity(s, e));
                }
            }
        }
        return n;
    }

    /// The convection law these parameters ask for, as the solver spells it.
    [[nodiscard]] thermal::ConvectionLaw ConvectionLawFromParams() const {
        thermal::ConvectionLaw law;
        switch (params.convectionModel) {
            case ThermalConvectionModel::Wind:
                law.model = thermal::ConvectionModel::Wind;
                break;
            case ThermalConvectionModel::Stability:
                law.model = thermal::ConvectionModel::Stability;
                break;
            case ThermalConvectionModel::Constant:
                break;
        }
        law.windIntercept_W_m2K = params.convectionWindA_W_m2K;
        law.windSlope_W_s_m3K = params.convectionWindB_W_s_m3K;
        law.freeCoefficient = params.convectionFreeC;
        law.referenceHeight_m = params.convectionReferenceHeight_m;
        law.stableDamping = params.convectionStableDamping;
        return law;
    }

    thermal::IThermalStepper& ChooseStepper() {
        // The GPU stepper mirrors the constant law, one column per element and
        // one sun tangent, and nothing else -- so a run that asked for more is
        // CPU work. Deciding it here rather than letting the GPU stepper ignore
        // what it was handed is the difference between a slower solve and a
        // wrong one, and the derivatives are where "wrong" is hardest to see:
        // the state is sized by the descriptor, so a stepper that does not
        // integrate them hands back zeros, and a zero derivative is an answer
        // rather than an absence.
        const bool constantLaw = params.convectionModel == ThermalConvectionModel::Constant &&
                                 !params.lateralConduction;
        if (constantLaw && gpuStepper && gpuStepper->IsValid() &&
            params.layerCount <= GpuThermalStepper::kMaxNodes &&
            (params.sunMemoryLags == 0 || gpuStepper->CarriesLagSensitivity()) &&
            (params.parameterSensitivities.empty() ||
             gpuStepper->CarriesParameterSensitivity())) {
            return *gpuStepper;
        }
        return cpuStepper;
    }

    /// Per material, the fraction of its area that is actually there.
    ///
    /// Opaque is 1. MASK counts the texels at or above the cutoff, which is the
    /// binary test the any-hit shader applies; BLEND averages the alpha itself,
    /// which is the probability it is committed with. Both are then multiplied
    /// by baseColorFactor's alpha, exactly as the shader multiplies them.
    ///
    /// Reads CPU pixels, which the loader retains for precisely these
    /// materials -- the upload frees everyone else's.
    static Vector<f32> ComputeMaterialCoverage(const Scene& scene) {
        Vector<f32> coverage(scene.materials.size(), 1.0f);
        for (usize m = 0; m < scene.materials.size(); ++m) {
            const Material& mat = scene.materials[m];
            if (mat.alphaMode == Material::AlphaMode::Opaque) {
                continue;
            }

            f32 textureMean = 1.0f;
            const int texIdx = mat.baseColorTextureIndex;
            if (texIdx >= 0 && texIdx < static_cast<int>(scene.textures.size())) {
                const Texture& tex = scene.textures[texIdx];
                if (tex.channels == 4 && !tex.pixels.empty()) {
                    const usize texels = tex.pixels.size() / 4;
                    f64 sum = 0.0;
                    for (usize t = 0; t < texels; ++t) {
                        const f32 a = static_cast<f32>(tex.pixels[t * 4 + 3]) / 255.0f;
                        sum += (mat.alphaMode == Material::AlphaMode::Mask)
                                   ? (a >= mat.alphaCutoff ? 1.0 : 0.0)
                                   : static_cast<f64>(a);
                    }
                    textureMean = texels > 0 ? static_cast<f32>(sum / static_cast<f64>(texels))
                                             : 1.0f;
                } else if (!tex.pixels.empty()) {
                    // Fewer than four channels means no alpha to read, so the
                    // texture cannot mask anything.
                    textureMean = 1.0f;
                } else {
                    // Pixels already freed -- only reachable for a material
                    // that became non-opaque after load, where assuming solid
                    // is the conservative answer for an occluder.
                    QL_LOG_WARN("Thermal coverage: material '{}' is alpha-tested but its "
                                "base colour texture has no CPU pixels; treating it as solid",
                                mat.name);
                    textureMean = 1.0f;
                }
            }

            f32 factorAlpha = mat.baseColorFactor.a;
            if (mat.alphaMode == Material::AlphaMode::Mask) {
                factorAlpha = (factorAlpha >= mat.alphaCutoff) ? 1.0f : 0.0f;
            }
            coverage[m] = std::clamp(textureMean * factorAlpha, 0.0f, 1.0f);
        }
        return coverage;
    }

    void RebuildMaterialTable(const Scene& scene) {
        // Computed here rather than in the precompute because this is where the
        // Scene is, and it has to happen while the textures still have their
        // CPU pixels.
        materialCoverage = ComputeMaterialCoverage(scene);
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
                materials[m].wetnessFactor = it->second.wetnessFactor;
                materials[m].internalHeat_W_m2 = it->second.internalHeat_W_m2;
                materials[m].longwaveEmissivity = emissivity;
                materials[m].interiorBoundary =
                    it->second.interiorFixedTemperature
                        ? thermal::InteriorBoundary::FixedTemperature
                    : it->second.interiorAmbient
                        ? thermal::InteriorBoundary::AmbientInterior
                        : thermal::InteriorBoundary::Adiabatic;
                materials[m].interiorTemperature_K = it->second.interiorTemperature_K;
                materials[m].interiorConvection_W_m2K = it->second.interiorConvection_W_m2K;
                ++named;
            }
        }
        materialTableDirty = false;
    }

    void RebuildExchange(VkAccelerationStructureKHR tlas) {
        ThermalExchangePrecompute precompute(context);
        precompute.SetMaterialCoverage(materialCoverage);
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
        // The sun visibility the exchange carries is one direction's worth,
        // and the table is built from it, so it has to follow.
        sunTableDirty = true;
    }

    /// Where the sun is, at every hour the forcing file names. A run with a
    /// diurnal CSV gets one column per row, so the shadows move through the
    /// day the way they do offline; a run with constant forcing gets the
    /// single column the exchange already computed.
    void RebuildSunTable(VkAccelerationStructureKHR tlas) {
        forcingSeries = thermal::LoadForcingCsv(params.forcingFile);
        sunTable = {};

        if (forcingSeries.size() > 1 && tlas != VK_NULL_HANDLE) {
            Vector<glm::vec3> directions;
            directions.reserve(forcingSeries.size());
            sunTable.sampleTime_h.reserve(forcingSeries.size());
            for (const auto& [t, forcing] : forcingSeries) {
                sunTable.sampleTime_h.push_back(t);
                directions.push_back(forcing.sunDirection);
            }

            ThermalExchangePrecompute precompute(context);
            precompute.SetMaterialCoverage(materialCoverage);
            if (precompute.IsValid()) {
                sunTable.visibility = precompute.RunSunVisibility(
                    tlas, mesh.elements, mesh.instanceElementBase, directions);
            }
            if (sunTable.visibility.size() == directions.size() * mesh.elements.size()) {
                sunTable.sampleDirection = std::move(directions);
            } else {
                sunTable = {};  // the dispatch failed; fall through to one column
            }
        }

        if (sunTable.SampleCount() == 0 && !exchange.sunVisibility.empty()) {
            sunTable.sampleTime_h = {params.startTime_h};
            sunTable.visibility = exchange.sunVisibility;
            sunTable.sampleDirection = {fallbackSunDirection};
        }
        sunTableDirty = false;
    }

    void RebuildTimeline() {
        constantForcing = {};
        constantForcing.airTemperature_K = params.airTemperature_K;
        constantForcing.sunIrradiance_W_m2 = params.sunIrradiance_W_m2;
        constantForcing.diffuseIrradiance_W_m2 = params.diffuseIrradiance_W_m2;
        constantForcing.sunDirection = glm::vec3(fallbackSunDirection);
        constantForcing.skyTemperature_K = params.skyTemperature_K;
        constantForcing.relativeHumidity = params.relativeHumidity;

        // The short-wave gains depend on the geometry, the sun columns and the
        // absorptivities, and every one of those sets timelineDirty on its way
        // through -- so baking here keeps them fresh without a flag of their
        // own.
        thermal::BakeShortwaveGains(exchange, mesh.elements, materials, sunTable);

        thermal::ThermalTimeline::Desc desc;
        desc.startTime_h = params.startTime_h;
        desc.timestep_s = params.timestep_s;
        desc.checkpointStride_h = params.checkpointStride_h;
        desc.nodeCount = params.layerCount;
        desc.initial = params.initial == ThermalInitialCondition::Steady
            ? thermal::InitialCondition::Steady
            : thermal::InitialCondition::Uniform;
        desc.initialTemperature_K = params.initialTemperature_K;
        // Off sizes the tangent out of the state rather than suppressing it
        // later, so it has to reach the desc -- and SetParams marks the
        // timeline dirty for every change, which is what makes it take.
        desc.carrySunSensitivity = params.sunCorrection;
        desc.sunMemoryLags = params.sunCorrection ? params.sunMemoryLags : 0u;

        // Material tangents, by the same rule and for the same reason: they
        // size the state, so they reach the desc and SetParams rebuilds.
        desc.parameters.clear();
        desc.parameters.reserve(params.parameterSensitivities.size());
        for (const ThermalSensitivityParameter p : params.parameterSensitivities) {
            switch (p) {
                case ThermalSensitivityParameter::Convection:
                    desc.parameters.push_back(thermal::ThermalParameter::Convection);
                    break;
                case ThermalSensitivityParameter::Emissivity:
                    desc.parameters.push_back(thermal::ThermalParameter::Emissivity);
                    break;
                case ThermalSensitivityParameter::Absorptivity:
                    desc.parameters.push_back(thermal::ThermalParameter::Absorptivity);
                    break;
                case ThermalSensitivityParameter::Conductivity:
                    desc.parameters.push_back(thermal::ThermalParameter::Conductivity);
                    break;
                case ThermalSensitivityParameter::HeatCapacity:
                    desc.parameters.push_back(thermal::ThermalParameter::HeatCapacity);
                    break;
            }
        }

        // Every argument but the desc is held by reference for the timeline's
        // lifetime, so all of them are members -- a local would be read after
        // it went out of scope.
        auto& stepper = ChooseStepper();
        timeline = std::make_unique<thermal::ThermalTimeline>(
            desc, mesh.elements, materials, exchange,
            sunTable, forcingSeries, constantForcing, stepper);

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
    if (params.lateralConduction != p.lateralConduction) {
        // The mesh only carries its shared edges when it was asked to, so
        // turning this on is a geometry rebuild rather than a flag flip.
        m_impl->exchangeDirty = true;
        m_impl->lateralDirty = true;
    }
    m_impl->timelineDirty = true;
    p = params;
    m_impl->cpuStepper.SetConvection(m_impl->ConvectionLawFromParams());
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

    // Whatever stops the solve is what Status() reports, so the panel says why
    // it is showing nothing rather than only that it is.
    const auto fail = [&](const char* reason) {
        result.error = reason;
        m_impl->lastError = reason;
        return result;
    };

    if (!m_impl->enabled) {
        result.error = "thermal solve is disabled";
        return result;
    }

    // Read the flags before the rebuilds below clear them: the lateral
    // conductances depend on both the mesh and the material table, and are
    // built after each has settled.
    const bool geometryWasDirty = m_impl->exchangeDirty;
    const bool materialsWereDirty = m_impl->materialTableDirty;

    // Rebuild mesh if geometry changed
    if (m_impl->exchangeDirty) {
        m_impl->mesh = thermal::BuildThermalMesh(
            scene, {.contacts = m_impl->params.lateralConduction});
    }

    if (m_impl->mesh.elements.empty()) {
        return fail("the scene has no triangles to solve on");
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
        return fail("no material in the scene has thermal properties");
    }

    // Rebuild exchange if geometry changed
    if (m_impl->exchangeDirty) {
        m_impl->RebuildExchange(tlas);
    }

    // The lateral conductances ride in the exchange, which is what every
    // stepper is handed. They depend on the mesh and the material table, so
    // they are rebuilt whenever either was -- and cleared when the parameter
    // goes off, or a scene that turned it off would keep conducting.
    if (geometryWasDirty || materialsWereDirty || m_impl->lateralDirty) {
        m_impl->exchange.lateral =
            m_impl->params.lateralConduction
                ? thermal::BuildLateralConduction(m_impl->mesh, m_impl->materials)
                : thermal::CsrMatrix{};
        m_impl->lateralDirty = false;
    }

    // Rebuild the sun columns if the geometry, the forcing file or the sun
    // moved. Cheap next to the exchange -- no hemisphere rays, one dispatch
    // per column -- which is why it is worth having its own flag.
    if (m_impl->sunTableDirty) {
        m_impl->RebuildSunTable(tlas);
        m_impl->timelineDirty = true;
    }

    // Rebuild timeline if anything changed
    if (m_impl->timelineDirty || !m_impl->timeline) {
        m_impl->RebuildTimeline();
    }

    // Step to the requested time
    const thermal::ThermalState& state = m_impl->timeline->StateAt(time_h);
    m_impl->currentTime_h = time_h;

    // Extract surface temperatures, and beside them the tangent the shading
    // pass needs to resolve a shadow finer than one triangle.
    const u32 n = m_impl->ExtractField(state, time_h, result.surfaceTemperature_K,
                                       result.sunSensitivity_K, result.sunVisibility,
                                       result.sunDirection, &result.lagSensitivity_K,
                                       &result.lagVisibility, &result.lagDirection);
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

Vector<f32> ThermalPreview::ParameterSensitivityField(
    const ThermalSensitivityParameter parameter) const {
    Vector<f32> field;
    if (!m_impl->timeline) return field;

    // Which slot, if any: the solve carries the parameters a config or a host
    // asked for, in the order it was given them, and a parameter nobody asked
    // for has no column to read.
    const auto wanted = [parameter] {
        switch (parameter) {
            case ThermalSensitivityParameter::Convection:
                return thermal::ThermalParameter::Convection;
            case ThermalSensitivityParameter::Emissivity:
                return thermal::ThermalParameter::Emissivity;
            case ThermalSensitivityParameter::Absorptivity:
                return thermal::ThermalParameter::Absorptivity;
            case ThermalSensitivityParameter::Conductivity:
                return thermal::ThermalParameter::Conductivity;
            case ThermalSensitivityParameter::HeatCapacity:
                return thermal::ThermalParameter::HeatCapacity;
        }
        return thermal::ThermalParameter::Count;
    }();

    const thermal::ThermalState& state = m_impl->timeline->StateAt(m_impl->currentTime_h);
    if (!state.HasParameterSensitivity()) return field;

    usize slot = state.parameters.size();
    for (usize i = 0; i < state.parameters.size(); ++i) {
        if (state.parameters[i] == wanted) { slot = i; break; }
    }
    if (slot == state.parameters.size()) return field;

    const usize n = m_impl->mesh.elements.size();
    field.assign(n, 0.0f);
    for (usize e = 0; e < n; ++e) {
        const u32 id = m_impl->mesh.elements[e].materialId;
        const bool solved = m_impl->mesh.elements[e].area_m2 > 0.0f &&
                            id < m_impl->materials.size() &&
                            m_impl->materials[id].ParticipatesInSolve();
        field[e] = solved ? static_cast<f32>(state.SurfaceParameterSensitivity(slot, e)) : 0.0f;
    }
    return field;
}

bool ThermalPreview::ElementFor(const u32 instanceIndex, const u32 primitiveIndex,
                                u32& out) const {
    if (instanceIndex >= m_impl->mesh.instanceElementBase.size()) return false;
    const u32 element = m_impl->mesh.instanceElementBase[instanceIndex] + primitiveIndex;
    if (element >= m_impl->mesh.elements.size()) return false;
    out = element;
    return true;
}

Result<ThermalElementTrajectory, String> ThermalPreview::ElementTrajectory(
    const u32 element, f64 fromHour, f64 toHour, const u32 samples) {
    using TrajectoryResult = Result<ThermalElementTrajectory, String>;

    if (!m_impl->timeline) {
        return TrajectoryResult::Err("the thermal solve has not run yet");
    }
    if (element >= m_impl->mesh.elements.size()) {
        return TrajectoryResult::Err("element " + std::to_string(element) + " is past the " +
                                     std::to_string(m_impl->mesh.elements.size()) +
                                     " this scene has");
    }
    if (samples < 2) {
        return TrajectoryResult::Err("a trajectory needs at least two samples");
    }
    if (toHour < fromHour) std::swap(fromHour, toHour);

    const u32 nodes = m_impl->params.layerCount;
    ThermalElementTrajectory out;
    out.time_h.reserve(samples);
    out.surfaceTemperature_K.reserve(samples);
    out.backTemperature_K.reserve(samples);

    // Whether the fluxes come at all is a property of the stepper, and it is
    // the same stepper for every sample -- so the first answer decides, and a
    // later refusal would mean the vectors disagree in length. Which is why
    // this is a flag rather than a per-sample push.
    bool haveFluxes = true;
    const f64 step = (toHour - fromHour) / static_cast<f64>(samples - 1);
    for (u32 i = 0; i < samples; ++i) {
        const f64 t = fromHour + step * static_cast<f64>(i);
        const thermal::ThermalState& state = m_impl->timeline->StateAt(t);

        out.time_h.push_back(t);
        out.surfaceTemperature_K.push_back(state.Surface(element));
        const usize back = static_cast<usize>(element) * nodes + (nodes - 1);
        out.backTemperature_K.push_back(back < state.temperature_K.size()
                                            ? state.temperature_K[back]
                                            : state.Surface(element));

        if (haveFluxes) {
            thermal::SurfaceFluxes f;
            // The CPU stepper decomposes, whichever one stepped. The balance is
            // a pure function of the state, and evaluating it one way keeps a
            // panel's six numbers from depending on whether this machine has a
            // GPU -- the state they describe is the trajectory's either way.
            if (m_impl->timeline->SurfaceFluxesAt(t, element, f, m_impl->cpuStepper)) {
                out.fluxes.push_back(ThermalSurfaceFluxes{
                    f.shortwave_W_m2, f.longwave_W_m2, f.convection_W_m2,
                    f.latent_W_m2, f.conduction_W_m2, f.lateral_W_m2});
            } else {
                haveFluxes = false;
                out.fluxes.clear();
            }
        }
    }

    // The viewport is showing an hour, and a probe is a question about the
    // past rather than a request to move: replaying left the timeline wherever
    // the last sample was, so put it back.
    m_impl->timeline->StateAt(m_impl->currentTime_h);
    return TrajectoryResult(std::move(out));
}

Result<String, String> ThermalPreview::DumpElements(const String& pathOrEmpty) {
    using DumpResult = Result<String, String>;

    const String& path = pathOrEmpty.empty() ? m_impl->params.dumpElementsFile : pathOrEmpty;
    if (path.empty()) {
        return DumpResult::Err("no path: pass one, or set thermal.dump_elements");
    }
    if (!m_impl->timeline) {
        return DumpResult::Err("the thermal solve has not run yet");
    }

    // The instant already on screen, never a fresh one. A dump exists to be
    // compared against the image beside it, and re-solving at some other hour
    // would produce a file describing a picture nobody looked at.
    const thermal::ThermalState& state = m_impl->timeline->StateAt(m_impl->currentTime_h);

    Vector<f32> temperature_K;
    Vector<f32> sunSensitivity_K;
    Vector<f32> visibility;
    glm::vec3 sunDirection{0.0f, 1.0f, 0.0f};
    const u32 n = m_impl->ExtractField(state, m_impl->currentTime_h, temperature_K,
                                       sunSensitivity_K, visibility, sunDirection);

    // Sampled again, and unconditionally, because ExtractField answers the
    // shading pass: there, v travels with the tangent it corrects and is left
    // empty when no tangent was carried. The file's v_element column is not
    // that -- it is a property of the geometry and the sun, which an element
    // has whether or not anyone asked how its temperature responds to it. The
    // offline writer emits it either way, and two writers that disagreed about
    // one column would be the second dialect this function exists to prevent.
    if (visibility.empty()) {
        visibility = thermal::SampleSunVisibilityAt(m_impl->sunTable, m_impl->exchange,
                                                    m_impl->currentTime_h, n);
    }

    thermal::DumpThermalElements(path, m_impl->mesh.elements, m_impl->materials,
                                 m_impl->exchange, temperature_K, sunSensitivity_K,
                                 visibility);
    return path;
}

ThermalSolveStatus ThermalPreview::Status() const {
    ThermalSolveStatus status;
    status.enabled = m_impl->enabled;
    status.solveValid = m_impl->timeline != nullptr && m_impl->lastError.empty();
    status.exchangeValid = !m_impl->exchangeDirty;
    status.elementCount = static_cast<u32>(m_impl->mesh.elements.size());
    status.exchangeNonZeros = static_cast<u32>(m_impl->exchange.viewFactors.NonZeros());
    status.exchangeRunCount = m_impl->exchangeRunCount;
    status.sunSampleCount = static_cast<u32>(m_impl->sunTable.SampleCount());
    status.currentTime_h = m_impl->currentTime_h;
    // Through the same choice the steps go through, so the status cannot name
    // one stepper while another runs.
    status.stepperName = m_impl->ChooseStepper().Name();
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
