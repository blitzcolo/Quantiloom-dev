/**
 * @file UsdSurfaceTables.cpp
 * @brief The vocabulary tables
 *
 * Defaults are the node definitions', taken from the .mtlx that ships with
 * OpenUSD (lib/usd/usdMtlx/resources/libraries/bxdf/) and from the
 * UsdPreviewSurface specification. Where a table and a node definition
 * disagree, the node definition is right.
 */

#include "io/UsdSurfaceTables.hpp"

namespace quantiloom::usd {

namespace {

constexpr glm::vec4 kZero{0.0f, 0.0f, 0.0f, 0.0f};
constexpr glm::vec4 kOne{1.0f, 1.0f, 1.0f, 1.0f};

// ============================================================================
// UsdPreviewSurface
// ============================================================================
// The specification's own defaults; `ND_UsdPreviewSurface_surfaceshader`
// (MaterialX 1.39, version 2.6) repeats them exactly, which is why one table
// serves both ids.

constexpr SurfaceInputSpec kUsdPreviewSurface[] = {
    {"diffuseColor",        SurfaceSlot::BaseColor,           ValueKind::Color3, {0.18f, 0.18f, 0.18f, 1.0f}, true},
    {"emissiveColor",       SurfaceSlot::Emissive,            ValueKind::Color3, kZero,                       true},
    {"useSpecularWorkflow", SurfaceSlot::UseSpecularWorkflow, ValueKind::Int,    kZero,                       false},
    {"specularColor",       SurfaceSlot::SpecularColor,       ValueKind::Color3, kZero,                       true},
    {"metallic",            SurfaceSlot::Metallic,            ValueKind::Float,  kZero,                       false},
    {"roughness",           SurfaceSlot::Roughness,           ValueKind::Float,  {0.5f, 0.0f, 0.0f, 0.0f},    false},
    {"clearcoat",           SurfaceSlot::ClearcoatWeight,     ValueKind::Float,  kZero,                       false},
    {"clearcoatRoughness",  SurfaceSlot::ClearcoatRoughness,  ValueKind::Float,  {0.01f, 0.0f, 0.0f, 0.0f},   false},
    {"opacity",             SurfaceSlot::Opacity,             ValueKind::Float,  kOne,                        false},
    {"opacityMode",         SurfaceSlot::OpacityMode,         ValueKind::Int,    kZero,                       false},
    {"opacityThreshold",    SurfaceSlot::OpacityThreshold,    ValueKind::Float,  kZero,                       false},
    {"ior",                 SurfaceSlot::Ior,                 ValueKind::Float,  {1.5f, 0.0f, 0.0f, 0.0f},    false},
    {"normal",              SurfaceSlot::Normal,              ValueKind::Vector3,{0.0f, 0.0f, 1.0f, 0.0f},    false},
    {"occlusion",           SurfaceSlot::Occlusion,           ValueKind::Float,  kOne,                        false},
};

constexpr UnsupportedInput kUsdPreviewSurfaceUnsupported[] = {
    {"displacement", "no displacement pipeline; the mesh is rendered as authored"},
};

// ============================================================================
// MaterialX standard_surface
// ============================================================================
// ND_standard_surface_surfaceshader, MaterialX 1.39.

constexpr SurfaceInputSpec kStandardSurface[] = {
    {"base",                  SurfaceSlot::BaseWeight,         ValueKind::Float,  kOne,                        false},
    {"base_color",            SurfaceSlot::BaseColor,          ValueKind::Color3, {0.8f, 0.8f, 0.8f, 1.0f},    true},
    {"metalness",             SurfaceSlot::Metallic,           ValueKind::Float,  kZero,                       false},
    {"specular",              SurfaceSlot::Specular,           ValueKind::Float,  kOne,                        false},
    {"specular_color",        SurfaceSlot::SpecularColor,      ValueKind::Color3, kOne,                        true},
    {"specular_roughness",    SurfaceSlot::Roughness,          ValueKind::Float,  {0.2f, 0.0f, 0.0f, 0.0f},    false},
    {"specular_IOR",          SurfaceSlot::Ior,                ValueKind::Float,  {1.5f, 0.0f, 0.0f, 0.0f},    false},
    {"specular_anisotropy",   SurfaceSlot::AnisotropyStrength, ValueKind::Float,  kZero,                       false},
    {"specular_rotation",     SurfaceSlot::AnisotropyRotation, ValueKind::Float,  kZero,                       false},
    {"transmission",          SurfaceSlot::Transmission,       ValueKind::Float,  kZero,                       false},
    {"transmission_color",    SurfaceSlot::TransmissionColor,  ValueKind::Color3, kOne,                        true},
    {"transmission_depth",    SurfaceSlot::TransmissionDepth,  ValueKind::Float,  kZero,                       false},
    {"transmission_dispersion", SurfaceSlot::Dispersion,       ValueKind::Float,  kZero,                       false},
    {"sheen",                 SurfaceSlot::SheenWeight,        ValueKind::Float,  kZero,                       false},
    {"sheen_color",           SurfaceSlot::SheenColor,         ValueKind::Color3, kOne,                        true},
    {"sheen_roughness",       SurfaceSlot::SheenRoughness,     ValueKind::Float,  {0.3f, 0.0f, 0.0f, 0.0f},    false},
    {"coat",                  SurfaceSlot::ClearcoatWeight,    ValueKind::Float,  kZero,                       false},
    {"coat_color",            SurfaceSlot::ClearcoatColor,     ValueKind::Color3, kOne,                        true},
    {"coat_roughness",        SurfaceSlot::ClearcoatRoughness, ValueKind::Float,  {0.1f, 0.0f, 0.0f, 0.0f},    false},
    {"coat_IOR",              SurfaceSlot::ClearcoatIor,       ValueKind::Float,  {1.5f, 0.0f, 0.0f, 0.0f},    false},
    {"coat_normal",           SurfaceSlot::ClearcoatNormal,    ValueKind::Vector3,kZero,                       false},
    {"emission",              SurfaceSlot::EmissiveWeight,     ValueKind::Float,  kZero,                       false},
    {"emission_color",        SurfaceSlot::Emissive,           ValueKind::Color3, kOne,                        true},
    {"opacity",               SurfaceSlot::Opacity,            ValueKind::Color3, kOne,                        false},
    {"thin_walled",           SurfaceSlot::ThinWalled,         ValueKind::Bool,   kZero,                       false},
    {"normal",                SurfaceSlot::Normal,             ValueKind::Vector3,kZero,                       false},
};

constexpr UnsupportedInput kStandardSurfaceUnsupported[] = {
    {"diffuse_roughness",              "no Oren-Nayar term; the diffuse lobe is Lambertian"},
    {"subsurface",                     "no subsurface scattering model"},
    {"subsurface_color",               "no subsurface scattering model"},
    {"subsurface_radius",              "no subsurface scattering model"},
    {"subsurface_scale",               "no subsurface scattering model"},
    {"subsurface_anisotropy",          "no subsurface scattering model"},
    {"transmission_scatter",           "volume scattering takes RGB coefficients, not a scatter colour"},
    {"transmission_scatter_anisotropy","volume scattering takes RGB coefficients, not a scatter colour"},
    {"transmission_extra_roughness",   "one roughness serves both sides of the interface"},
    {"thin_film_thickness",            "no thin-film interference model"},
    {"thin_film_IOR",                  "no thin-film interference model"},
    {"coat_affect_color",              "the coat does not tint what is under it"},
    {"coat_affect_roughness",          "the coat does not roughen what is under it"},
    {"coat_anisotropy",                "the clearcoat lobe is isotropic"},
    {"coat_rotation",                  "the clearcoat lobe is isotropic"},
};

// ============================================================================
// MaterialX gltf_pbr
// ============================================================================
// ND_gltf_pbr_surfaceshader, MaterialX 1.39, version 2.0.1 -- the glTF material
// as a MaterialX node, so its inputs are the KHR extensions by name.
//
// It has no anisotropy and no dispersion input: glTF has KHR_materials_anisotropy
// and KHR_materials_dispersion, this node definition does not model them, and a
// table row for an input that cannot exist would be a promise the reader could
// never keep.
//
// attenuation_distance has no default in the node definition. Unauthored means
// no volume attenuation, which is what Material's 0 already means, so it is left
// there rather than given a number nobody wrote.

constexpr SurfaceInputSpec kGltfPbr[] = {
    {"base_color",           SurfaceSlot::BaseColor,           ValueKind::Color3, kOne,                     true},
    {"metallic",             SurfaceSlot::Metallic,            ValueKind::Float,  kOne,                     false},
    {"roughness",            SurfaceSlot::Roughness,           ValueKind::Float,  kOne,                     false},
    {"normal",               SurfaceSlot::Normal,              ValueKind::Vector3,kZero,                    false},
    {"occlusion",            SurfaceSlot::Occlusion,           ValueKind::Float,  kOne,                     false},
    {"transmission",         SurfaceSlot::Transmission,        ValueKind::Float,  kZero,                    false},
    {"specular",             SurfaceSlot::Specular,            ValueKind::Float,  kOne,                     false},
    {"specular_color",       SurfaceSlot::SpecularColor,       ValueKind::Color3, kOne,                     true},
    {"ior",                  SurfaceSlot::Ior,                 ValueKind::Float,  {1.5f, 0.0f, 0.0f, 0.0f}, false},
    {"alpha",                SurfaceSlot::Opacity,             ValueKind::Float,  kOne,                     false},
    {"alpha_mode",           SurfaceSlot::AlphaMode,           ValueKind::Int,    kZero,                    false},
    {"alpha_cutoff",         SurfaceSlot::AlphaCutoff,         ValueKind::Float,  {0.5f, 0.0f, 0.0f, 0.0f}, false},
    {"sheen_color",          SurfaceSlot::SheenColor,          ValueKind::Color3, kZero,                    true},
    {"sheen_roughness",      SurfaceSlot::SheenRoughness,      ValueKind::Float,  kZero,                    false},
    {"clearcoat",            SurfaceSlot::ClearcoatWeight,     ValueKind::Float,  kZero,                    false},
    {"clearcoat_roughness",  SurfaceSlot::ClearcoatRoughness,  ValueKind::Float,  kZero,                    false},
    {"clearcoat_normal",     SurfaceSlot::ClearcoatNormal,     ValueKind::Vector3,kZero,                    false},
    {"emissive",             SurfaceSlot::Emissive,            ValueKind::Color3, kZero,                    true},
    {"emissive_strength",    SurfaceSlot::EmissiveWeight,      ValueKind::Float,  kOne,                     false},
    {"thickness",            SurfaceSlot::Thickness,           ValueKind::Float,  kZero,                    false},
    {"attenuation_distance", SurfaceSlot::AttenuationDistance, ValueKind::Float,  kZero,                    false},
    {"attenuation_color",    SurfaceSlot::AttenuationColor,    ValueKind::Color3, kOne,                     true},
};

constexpr UnsupportedInput kGltfPbrUnsupported[] = {
    {"iridescence",           "no thin-film interference model"},
    {"iridescence_ior",       "no thin-film interference model"},
    {"iridescence_thickness", "no thin-film interference model"},
};

// ============================================================================
// MaterialX open_pbr_surface
// ============================================================================
// ND_open_pbr_surface_surfaceshader, MaterialX 1.39, version 1.1.

constexpr SurfaceInputSpec kOpenPbrSurface[] = {
    {"base_weight",                         SurfaceSlot::BaseWeight,         ValueKind::Float,  kOne,                      false},
    {"base_color",                          SurfaceSlot::BaseColor,          ValueKind::Color3, {0.8f, 0.8f, 0.8f, 1.0f},  true},
    {"base_metalness",                      SurfaceSlot::Metallic,           ValueKind::Float,  kZero,                     false},
    {"specular_weight",                     SurfaceSlot::Specular,           ValueKind::Float,  kOne,                      false},
    {"specular_color",                      SurfaceSlot::SpecularColor,      ValueKind::Color3, kOne,                      true},
    {"specular_roughness",                  SurfaceSlot::Roughness,          ValueKind::Float,  {0.3f, 0.0f, 0.0f, 0.0f},  false},
    {"specular_ior",                        SurfaceSlot::Ior,                ValueKind::Float,  {1.5f, 0.0f, 0.0f, 0.0f},  false},
    {"specular_roughness_anisotropy",       SurfaceSlot::AnisotropyStrength, ValueKind::Float,  kZero,                     false},
    {"transmission_weight",                 SurfaceSlot::Transmission,       ValueKind::Float,  kZero,                     false},
    {"transmission_color",                  SurfaceSlot::TransmissionColor,  ValueKind::Color3, kOne,                      true},
    {"transmission_depth",                  SurfaceSlot::TransmissionDepth,  ValueKind::Float,  kZero,                     false},
    {"transmission_dispersion_scale",       SurfaceSlot::DispersionScale,    ValueKind::Float,  kZero,                     false},
    {"transmission_dispersion_abbe_number", SurfaceSlot::Dispersion,         ValueKind::Float,  {20.0f, 0.0f, 0.0f, 0.0f}, false},
    {"fuzz_weight",                         SurfaceSlot::SheenWeight,        ValueKind::Float,  kZero,                     false},
    {"fuzz_color",                          SurfaceSlot::SheenColor,         ValueKind::Color3, kOne,                      true},
    {"fuzz_roughness",                      SurfaceSlot::SheenRoughness,     ValueKind::Float,  {0.5f, 0.0f, 0.0f, 0.0f},  false},
    {"coat_weight",                         SurfaceSlot::ClearcoatWeight,    ValueKind::Float,  kZero,                     false},
    {"coat_color",                          SurfaceSlot::ClearcoatColor,     ValueKind::Color3, kOne,                      true},
    {"coat_roughness",                      SurfaceSlot::ClearcoatRoughness, ValueKind::Float,  kZero,                     false},
    {"coat_ior",                            SurfaceSlot::ClearcoatIor,       ValueKind::Float,  {1.6f, 0.0f, 0.0f, 0.0f},  false},
    {"emission_luminance",                  SurfaceSlot::EmissiveWeight,     ValueKind::Float,  kZero,                     false},
    {"emission_color",                      SurfaceSlot::Emissive,           ValueKind::Color3, kOne,                      true},
    {"geometry_opacity",                    SurfaceSlot::Opacity,            ValueKind::Float,  kOne,                      false},
    {"geometry_thin_walled",                SurfaceSlot::ThinWalled,         ValueKind::Bool,   kZero,                     false},
    {"geometry_normal",                     SurfaceSlot::Normal,             ValueKind::Vector3,kZero,                     false},
    {"geometry_coat_normal",                SurfaceSlot::ClearcoatNormal,    ValueKind::Vector3,kZero,                     false},
};

constexpr UnsupportedInput kOpenPbrSurfaceUnsupported[] = {
    {"base_diffuse_roughness",           "no Oren-Nayar term; the diffuse lobe is Lambertian"},
    {"transmission_scatter",             "volume scattering takes RGB coefficients, not a scatter colour"},
    {"transmission_scatter_anisotropy",  "volume scattering takes RGB coefficients, not a scatter colour"},
    {"subsurface_weight",                "no subsurface scattering model"},
    {"subsurface_color",                 "no subsurface scattering model"},
    {"subsurface_radius",                "no subsurface scattering model"},
    {"subsurface_radius_scale",          "no subsurface scattering model"},
    {"subsurface_scatter_anisotropy",    "no subsurface scattering model"},
    {"coat_roughness_anisotropy",        "the clearcoat lobe is isotropic"},
    {"coat_darkening",                   "the coat does not darken what is under it"},
    {"thin_film_weight",                 "no thin-film interference model"},
    {"thin_film_thickness",              "no thin-film interference model"},
    {"thin_film_ior",                    "no thin-film interference model"},
};

}  // namespace

// ============================================================================
// Lookup
// ============================================================================

SurfaceVocabulary ClassifyShaderId(StringView shaderId) {
    if (shaderId == "UsdPreviewSurface" ||
        shaderId == "ND_UsdPreviewSurface_surfaceshader") {
        return SurfaceVocabulary::UsdPreviewSurface;
    }
    if (shaderId == "ND_standard_surface_surfaceshader") {
        return SurfaceVocabulary::StandardSurface;
    }
    if (shaderId == "ND_gltf_pbr_surfaceshader") {
        return SurfaceVocabulary::GltfPbr;
    }
    if (shaderId == "ND_open_pbr_surface_surfaceshader") {
        return SurfaceVocabulary::OpenPbrSurface;
    }
    return SurfaceVocabulary::Unknown;
}

SurfaceTable TableFor(SurfaceVocabulary vocabulary) {
    switch (vocabulary) {
        case SurfaceVocabulary::UsdPreviewSurface:
            return {kUsdPreviewSurface, std::size(kUsdPreviewSurface)};
        case SurfaceVocabulary::StandardSurface:
            return {kStandardSurface, std::size(kStandardSurface)};
        case SurfaceVocabulary::GltfPbr:
            return {kGltfPbr, std::size(kGltfPbr)};
        case SurfaceVocabulary::OpenPbrSurface:
            return {kOpenPbrSurface, std::size(kOpenPbrSurface)};
        case SurfaceVocabulary::Unknown:
            break;
    }
    return {};
}

UnsupportedTable UnsupportedFor(SurfaceVocabulary vocabulary) {
    switch (vocabulary) {
        case SurfaceVocabulary::UsdPreviewSurface:
            return {kUsdPreviewSurfaceUnsupported, std::size(kUsdPreviewSurfaceUnsupported)};
        case SurfaceVocabulary::StandardSurface:
            return {kStandardSurfaceUnsupported, std::size(kStandardSurfaceUnsupported)};
        case SurfaceVocabulary::GltfPbr:
            return {kGltfPbrUnsupported, std::size(kGltfPbrUnsupported)};
        case SurfaceVocabulary::OpenPbrSurface:
            return {kOpenPbrSurfaceUnsupported, std::size(kOpenPbrSurfaceUnsupported)};
        case SurfaceVocabulary::Unknown:
            break;
    }
    return {};
}

const char* VocabularyName(SurfaceVocabulary vocabulary) {
    switch (vocabulary) {
        case SurfaceVocabulary::UsdPreviewSurface: return "UsdPreviewSurface";
        case SurfaceVocabulary::StandardSurface:   return "standard_surface";
        case SurfaceVocabulary::GltfPbr:           return "gltf_pbr";
        case SurfaceVocabulary::OpenPbrSurface:    return "open_pbr_surface";
        case SurfaceVocabulary::Unknown:           break;
    }
    return "unknown";
}

}  // namespace quantiloom::usd
