/**
 * @file UsdSurfaceTables.hpp
 * @brief The surface-shader vocabularies USD scenes are written in, as data
 *
 * A USD material's surface shader is a `UsdShadeShader` whose `info:id` names
 * one of a handful of vocabularies: `UsdPreviewSurface`, or one of the
 * MaterialX node definitions that ship with OpenUSD. They differ in their input
 * names, their defaults and their parameterisation, and not at all in how they
 * are read -- so the reading is one walker over a table, and each vocabulary is
 * a table plus the scalar conversions that only it needs.
 *
 * The tables are the record of what is supported. An input that is not in one
 * is not read, which is why UnsupportedFor() exists beside them: an input
 * Quantiloom has no model for warns rather than being silently ignored.
 *
 * Internal header. No pxr types appear here, so the tables are testable without
 * a stage.
 */

#pragma once

#include "core/Types.hpp"

#include <glm/glm.hpp>

namespace quantiloom::usd {

// ============================================================================
// Slots
// ============================================================================

/**
 * @enum SurfaceSlot
 * @brief What an input means, independent of which vocabulary named it
 *
 * Four vocabularies spell the same quantity four ways -- `roughness`,
 * `specular_roughness`, `roughness`, `specular_roughness` -- and a slot is the
 * name they all resolve to. `Material` is the destination, so a slot exists
 * only where something in `Material` can receive it; everything else warns from
 * UnsupportedFor().
 */
enum class SurfaceSlot : u8 {
    BaseColor,
    BaseWeight,
    Metallic,
    Roughness,
    Normal,
    Emissive,
    EmissiveWeight,
    Opacity,
    OpacityThreshold,
    OpacityMode,
    AlphaMode,
    AlphaCutoff,
    Ior,
    Transmission,
    TransmissionColor,
    TransmissionDepth,
    Thickness,
    AttenuationColor,
    AttenuationDistance,
    Dispersion,
    DispersionScale,
    SheenWeight,
    SheenColor,
    SheenRoughness,
    Specular,
    SpecularColor,
    AnisotropyStrength,
    AnisotropyRotation,
    ClearcoatWeight,
    ClearcoatColor,
    ClearcoatRoughness,
    ClearcoatNormal,
    ClearcoatIor,
    ThinWalled,
    Occlusion,
    UseSpecularWorkflow,
    Count
};

/// How to read an input's constant value. USD's typed Get() needs to be told.
enum class ValueKind : u8 { Float, Color3, Color4, Vector3, Int, Bool };

enum class SurfaceVocabulary : u8 {
    UsdPreviewSurface,
    StandardSurface,
    GltfPbr,
    OpenPbrSurface,
    Unknown
};

/**
 * @struct SurfaceInputSpec
 * @brief One row of a vocabulary: an input name, what it means, how to read it
 *
 * @var SurfaceInputSpec::specDefault
 *   The value the node definition gives the input, not Quantiloom's. It is what
 *   an unauthored input reads as, and what "authored away from the default"
 *   is measured against.
 * @var SurfaceInputSpec::colour
 *   Whether a texture bound here is colour (decoded from sRGB unless the file
 *   says otherwise) or data (linear). This is the fallback when nothing in the
 *   shader graph states a colour space, and getting it wrong is the one error
 *   that corrupts spectral upsampling silently.
 */
struct SurfaceInputSpec {
    const char* input;
    SurfaceSlot slot;
    ValueKind kind;
    glm::vec4 specDefault;
    bool colour;
};

struct SurfaceTable {
    const SurfaceInputSpec* specs = nullptr;
    usize count = 0;

    [[nodiscard]] const SurfaceInputSpec* begin() const { return specs; }
    [[nodiscard]] const SurfaceInputSpec* end() const { return specs + count; }
};

/**
 * @struct UnsupportedInput
 * @brief An input a vocabulary defines and Quantiloom has no model for
 *
 * Warned once per material when the scene actually authored it. A material that
 * left it alone says nothing, because a default nobody set is not a loss.
 */
struct UnsupportedInput {
    const char* input;
    const char* why;
};

struct UnsupportedTable {
    const UnsupportedInput* items = nullptr;
    usize count = 0;

    [[nodiscard]] const UnsupportedInput* begin() const { return items; }
    [[nodiscard]] const UnsupportedInput* end() const { return items + count; }
};

// ============================================================================
// Lookup
// ============================================================================

/**
 * @brief Which vocabulary an `info:id` names
 *
 * Exact token match. The substring test this replaced routed
 * `ND_gltf_pbr_surfaceshader` into the UsdPreviewSurface reader, where none of
 * its input names matched and every material came out the default grey.
 */
[[nodiscard]] SurfaceVocabulary ClassifyShaderId(StringView shaderId);

[[nodiscard]] SurfaceTable TableFor(SurfaceVocabulary vocabulary);

[[nodiscard]] UnsupportedTable UnsupportedFor(SurfaceVocabulary vocabulary);

[[nodiscard]] const char* VocabularyName(SurfaceVocabulary vocabulary);

}  // namespace quantiloom::usd
