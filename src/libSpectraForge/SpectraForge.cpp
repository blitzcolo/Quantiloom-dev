#include "SpectraForge.hpp"
#include "ColorCluster.hpp"
#include "IRMaterialDatabase.hpp"
#include "core/Log.hpp"
#include <format>

namespace spectraforge {

// ============================================================================
// Helper: convert baseColorFactor RGB to LAB (no texture)
// ============================================================================

static LABColor BaseColorToLAB(const glm::vec4& bc) {
    // baseColorFactor is linear RGB
    return RGBToLAB(bc.r, bc.g, bc.b);
}

// ============================================================================
// ProcessSingle
// ============================================================================

bool SpectraForge::ProcessSingle(
    Material& mat,
    const Texture* baseColorTex,
    f32 temperature_K,
    u32 clusterCount)
{
    LABColor lab;

    if (baseColorTex && baseColorTex->IsValid() && !baseColorTex->pixels.empty()) {
        auto result = ClusterTextureColors(
            baseColorTex->pixels.data(),
            baseColorTex->width,
            baseColorTex->height,
            baseColorTex->channels,
            baseColorTex->isSRGB,
            clusterCount
        );
        lab = result.centroids[result.dominantCluster];
    } else {
        lab = BaseColorToLAB(mat.baseColorFactor);
    }

    const auto& entry = MatchMaterial(lab, mat.metallicFactor);

    GenerateIRCurves(entry,
        mat.irEmissivityCurve,
        mat.irReflectanceCurve,
        mat.irTransmittanceCurve);

    mat.irTemperature_K = temperature_K;
    mat.spectralSource = Material::SpectralSource::RGBUpsampled;

    return true;
}

// ============================================================================
// Process (batch) — delegates to ProcessSingle per material
// ============================================================================

ForgeResult SpectraForge::Process(
    const Scene& scene,
    Vector<Material>& outMaterials,
    const ForgeConfig& config)
{
    ForgeResult result;
    result.totalMaterials = static_cast<u32>(scene.materials.size());

    outMaterials = scene.materials;

    for (u32 i = 0; i < outMaterials.size(); ++i) {
        auto& mat = outMaterials[i];

        if (mat.HasIRData() && !config.overwriteExisting) {
            result.materialsSkipped++;
            if (config.verbose) {
                QL_LOG_INFO("[SpectraForge] Skip material[{}] '{}': already has IR data",
                    i, mat.name);
            }
            continue;
        }

        const Texture* tex = nullptr;
        if (mat.baseColorTextureIndex >= 0 &&
            static_cast<u32>(mat.baseColorTextureIndex) < scene.textures.size()) {
            tex = &scene.textures[static_cast<u32>(mat.baseColorTextureIndex)];
        }

        ProcessSingle(mat, tex, config.defaultTemperature_K, config.clusterCount);
        result.materialsProcessed++;

        // Identify the matched entry for logging
        LABColor lab;
        if (tex && tex->IsValid() && !tex->pixels.empty()) {
            lab = BaseColorToLAB(mat.baseColorFactor); // approximate for log only
        } else {
            lab = BaseColorToLAB(mat.baseColorFactor);
        }
        const auto& entry = MatchMaterial(lab, mat.metallicFactor);

        String msg = std::format("Material[{}] '{}' -> {}", i, mat.name, entry.name);
        result.assignments.push_back(msg);

        if (config.verbose) {
            QL_LOG_INFO("[SpectraForge] {}", msg);
        }

        if (!mat.ValidateIRKirchhoffLaw()) {
            QL_LOG_WARN("[SpectraForge] Material[{}] '{}' failed Kirchhoff validation",
                i, mat.name);
        }
    }

    if (config.verbose) {
        QL_LOG_INFO("[SpectraForge] Done: {}/{} materials processed, {} skipped",
            result.materialsProcessed, result.totalMaterials, result.materialsSkipped);
    }

    return result;
}

// ============================================================================
// GetDatabase
// ============================================================================

Span<const IRMaterialEntry> SpectraForge::GetDatabase() {
    return GetBuiltinDatabase();
}

} // namespace spectraforge
