/**
 * @file SpectraForge.hpp
 * @brief Public API for automatic IR material generation from visible textures
 *
 * SpectraForge analyzes base color textures via CIELAB clustering,
 * matches each cluster to a built-in IR material class, and fills
 * Material IR curves (emissivity/reflectance/transmittance).
 *
 * Usage:
 *   Vector<Material> irMats;
 *   auto result = SpectraForge::Process(scene, irMats);
 *   // irMats now have HasIRData() == true for all processed materials
 */

#pragma once

#include "Platform.hpp"
#include "IRMaterialDatabase.hpp"
#include "core/Types.hpp"
#include "scene/Scene.hpp"
#include "scene/Material.hpp"
#include "scene/Texture.hpp"

namespace spectraforge {

using quantiloom::f32;
using quantiloom::u32;
using quantiloom::Span;
using quantiloom::Vector;
using quantiloom::String;
using quantiloom::Scene;
using quantiloom::Material;
using quantiloom::Texture;

struct SF_API ForgeConfig {
    u32  clusterCount = 5;              // K for K-means (1-12)
    f32  defaultTemperature_K = 300.0f; // surface temperature
    bool overwriteExisting = false;     // skip materials with existing IR data
    bool verbose = false;               // log decisions to QL logger
};

struct SF_API ForgeResult {
    u32 totalMaterials = 0;
    u32 materialsProcessed = 0;
    u32 materialsSkipped = 0;
    Vector<String> assignments; // "Material[i] 'name' -> ir_class_name"
};

class SF_API SpectraForge {
public:
    // Process entire scene: read textures, cluster, assign IR materials.
    // Returns modified material copies (does NOT mutate input scene).
    static ForgeResult Process(
        const Scene& scene,
        Vector<Material>& outMaterials,
        const ForgeConfig& config = {}
    );

    // Process a single material in-place.
    static bool ProcessSingle(
        Material& mat,
        const Texture* baseColorTex,
        f32 temperature_K = 300.0f,
        u32 clusterCount = 5
    );

    // Expose database for GUI display
    static Span<const IRMaterialEntry> GetDatabase();
};

} // namespace spectraforge
