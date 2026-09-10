/**
 * @file UsdShadeGraph.hpp
 * @brief Reading one USD surface shader: the graph walk, the textures, the scalars
 *
 * Three steps, in this order, and only the last one knows which vocabulary it is
 * looking at:
 *
 *  1. ReadSurface walks the shader graph once and fills a SurfaceReading -- a
 *     value and possibly a texture binding for every slot the vocabulary
 *     defines. Nothing is written to a Material yet, which is what lets Pass 0
 *     collect every texture path before a single file is decoded.
 *  2. ApplySlotTextures turns those bindings into `Texture` entries and texture
 *     indices. It is vocabulary-independent: a roughness map is packed into G
 *     whatever the input that carried it was called.
 *  3. ConvertSurface applies the scalar semantics that differ between
 *     vocabularies -- weight times colour, Abbe number to its reciprocal,
 *     opacity to an alpha mode.
 *
 * Internal header. Shader pointers cross as `const void*`, the same convention
 * UsdLoader.hpp uses, so no pxr type appears here.
 */

#pragma once

#include "io/UsdSurfaceTables.hpp"
#include "io/UsdTextureBank.hpp"
#include "scene/Material.hpp"

#include <array>
#include <optional>
#include <unordered_set>

namespace quantiloom::usd {

/**
 * @struct SlotValue
 * @brief What one slot ended up bound to
 *
 * @var SlotValue::value
 *   The constant, or the node definition's default when nothing was authored.
 *   Scalars use .x, colours .rgb.
 * @var SlotValue::authored
 *   Whether the scene said anything at all. A converter that treats "authored 0"
 *   differently from "defaulted 0" asks this.
 */
struct SlotValue {
    glm::vec4 value{0.0f, 0.0f, 0.0f, 0.0f};
    bool authored = false;
    std::optional<TextureRef> texture;

    [[nodiscard]] bool HasTexture() const { return texture.has_value(); }
};

/**
 * @struct SurfaceReading
 * @brief One shader, read
 */
struct SurfaceReading {
    SurfaceVocabulary vocabulary = SurfaceVocabulary::Unknown;
    String materialPath;
    std::array<SlotValue, static_cast<usize>(SurfaceSlot::Count)> slots;

    [[nodiscard]] const SlotValue& operator[](SurfaceSlot slot) const {
        return slots[static_cast<usize>(slot)];
    }
    [[nodiscard]] SlotValue& operator[](SurfaceSlot slot) {
        return slots[static_cast<usize>(slot)];
    }
    [[nodiscard]] f32 Scalar(SurfaceSlot slot) const { return (*this)[slot].value.x; }
    [[nodiscard]] glm::vec3 Colour(SurfaceSlot slot) const {
        return glm::vec3((*this)[slot].value);
    }
    [[nodiscard]] bool HasTexture(SurfaceSlot slot) const { return (*this)[slot].HasTexture(); }
};

/**
 * @struct BindingContext
 * @brief What the walker needs besides the shader, and what it has already said
 *
 * `warned` is per material: a node type nobody supports should be reported once
 * for the material that used it, not once per input that reached it.
 */
struct BindingContext {
    String usdDir;
    f64 timeCode = 0.0;
    bool useDefaultTime = true;
    bool loadTextures = true;
    String materialPath;
    std::unordered_set<String> warned;

    /// True the first time a (topic) is seen for this material.
    bool WarnOnce(const String& topic) { return warned.insert(topic).second; }
};

/// Walk one surface shader and record what every slot of `vocabulary` is bound to.
[[nodiscard]] SurfaceReading ReadSurface(const void* shaderPtr,
                                         SurfaceVocabulary vocabulary,
                                         BindingContext& context);

/// Every image file the reading refers to, for the bank to decode up front.
void CollectTextureSources(const SurfaceReading& reading,
                           std::unordered_set<String>& outPaths);

/// Build the texture entries the reading needs and point the material's slots at
/// them. Vocabulary-independent; the channel routing is `Material`'s, not USD's.
void ApplySlotTextures(const SurfaceReading& reading, Material& material,
                       UsdTextureBank& bank, std::vector<Texture>& textures);

/// The scalar semantics, per vocabulary. Call after ApplySlotTextures: a factor
/// depends on whether its slot ended up textured.
void ConvertSurface(const SurfaceReading& reading, Material& material);

}  // namespace quantiloom::usd
