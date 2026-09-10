/**
 * @file UsdTextureBank.hpp
 * @brief Decoded image files, and the repacking that turns them into texture slots
 *
 * USD binds one image to one input. `Material` binds one texture to a slot with
 * a fixed channel meaning -- metallicRoughness is G = roughness, B = metallic,
 * and the GPU side has no per-slot channel selector to point elsewhere
 * (MaterialGpuData is a 656-byte struct a static_assert pins). So a scene with a
 * metallic map and a separate roughness map has to be repacked at load, and a
 * scene that binds the same file to two slots has to produce two entries with
 * different colour spaces.
 *
 * The bank owns the decoded sources for the length of one load, hands out
 * `Texture` indices for recipes, and deduplicates by what the recipe asks for
 * rather than by file, so two materials repacking the same pair of maps the same
 * way share one entry.
 *
 * Internal header. No pxr types, so the repacking is testable without a stage.
 */

#pragma once

#include "core/Types.hpp"
#include "scene/Material.hpp"
#include "scene/Texture.hpp"

#include <array>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace quantiloom::usd {

/// Which channels of an image an input reads. `UsdUVTexture` says so with the
/// output it connects to (`r`, `g`, `b`, `a`, `rgb`); MaterialX says so with a
/// `separate`/`extract` node.
enum class ChannelSel : u8 { R, G, B, A, RGB, RGBA };

/// What the scene said about an image's colour space, which is not always
/// anything. Unspecified means the destination slot decides.
enum class ColourSpaceHint : u8 { Unspecified, Srgb, Linear };

/**
 * @struct TextureRef
 * @brief One image binding, as the shader graph described it
 *
 * Produced by the graph walker, consumed by ApplySlotTextures. The path is
 * absolute so that a `.usdz` layer's internal path and an asset resolver's
 * answer both work, and so that it can be the bank's key.
 */
struct TextureRef {
    String absolutePath;
    ChannelSel channels = ChannelSel::RGBA;
    ColourSpaceHint colourSpace = ColourSpaceHint::Unspecified;
    TextureSampler sampler;
    glm::vec4 scale{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 bias{0.0f, 0.0f, 0.0f, 0.0f};
    String uvPrimvar = "st";
    UvTransform uv;
    f32 normalScale = 1.0f;
    String nodePath;
};

/**
 * @struct SlotRecipe
 * @brief How to build one `Texture` entry out of one or more decoded images
 *
 * `dst[i]` is where the entry's channel i comes from; an empty one is `fill[i]`.
 * A recipe whose four channels are the four channels of one image in order, with
 * no affine, is an identity and reuses the decoded source rather than walking
 * every texel.
 */
struct SlotRecipe {
    struct Src {
        String path;
        ChannelSel channel = ChannelSel::R;
        f32 scale = 1.0f;
        f32 bias = 0.0f;

        [[nodiscard]] bool IsAffineIdentity() const { return scale == 1.0f && bias == 0.0f; }
    };

    std::array<std::optional<Src>, 4> dst;
    std::array<u8, 4> fill{255, 255, 255, 255};
    bool srgb = false;
    bool retainCpuPixels = false;
    TextureSampler sampler;
    String debugName;

    /// Everything that makes two entries different. Two recipes with the same
    /// key produce byte-identical pixels and the same upload format, so they are
    /// one entry.
    [[nodiscard]] String Key() const;

    /// Whether this is the four channels of one image, in order, untouched.
    [[nodiscard]] bool IsIdentityOf(const String& path) const;

    /// The distinct files this recipe reads.
    [[nodiscard]] std::vector<String> Sources() const;
};

/**
 * @brief Decode one image file to RGBA8
 *
 * The single decode path: `UsdLoader::ParseTexture` resolves a path and calls
 * this, and the bank calls it directly for the paths it preloads.
 */
[[nodiscard]] Texture DecodeTextureFile(const String& absolutePath);

/**
 * @brief Decode an image already in memory
 *
 * For assets that are not files. A texture inside a .usdz is one: the resolver
 * hands back a package-relative path and the bytes live in the zip, so there is
 * nothing for the file path to open.
 */
[[nodiscard]] Texture DecodeTextureBytes(const String& name, const u8* data, usize size);

/**
 * @class UsdTextureBank
 * @brief Decoded sources for one load, plus the entries built from them
 */
class UsdTextureBank {
public:
    /// Decode every path in parallel. Called once, before any material is read,
    /// because decoding is the load's I/O cost and it parallelises. Paths
    /// already adopted -- as a decoded source or as encoded bytes -- are left
    /// alone.
    void Preload(const std::unordered_set<String>& absolutePaths);

    /// Hand the bank an image that is not a file, to be decoded by the next
    /// Preload along with everything else.
    void AdoptEncoded(const String& path, std::vector<u8> bytes) {
        m_encoded.emplace(path, std::move(bytes));
    }

    /// Build (or find) the entry a recipe describes and return its index in
    /// `textures`, or -1 when none of its sources decoded.
    i32 Materialise(const SlotRecipe& recipe, std::vector<Texture>& textures);

    /// Drop the decoded sources. After this only the entries in `scene.textures`
    /// hold pixels; a 4K source is 64 MB and there is no reason to keep it once
    /// every material has been read.
    void ReleaseSources();

    [[nodiscard]] usize SourceCount() const { return m_sources.size(); }
    [[nodiscard]] bool HasSource(const String& path) const {
        const auto it = m_sources.find(path);
        return it != m_sources.end() && it->second.width > 0;
    }

    /// For tests: seed a source without touching the filesystem.
    void AdoptSource(const String& path, Texture texture) {
        m_sources.emplace(path, std::move(texture));
    }

private:
    std::unordered_map<String, Texture> m_sources;
    std::unordered_map<String, std::vector<u8>> m_encoded;
    std::unordered_map<String, i32> m_byKey;
};

// ============================================================================
// Conversions the walker and the bank share
// ============================================================================

/**
 * @brief Whether `st` is flipped in V on the way in
 *
 * USD's `st` origin is the image's lower-left corner (the UsdUVTexture spec says
 * so), while stb decodes top-down and the shaders sample with glTF's upper-left
 * convention. Flipping V at load is what puts a USD texture the right way up.
 *
 * It lives here rather than beside the mesh reader because the vertex flip and
 * the UV-transform conjugation have to agree: flip one without the other and a
 * transformed texture is upside down again.
 */
inline constexpr bool kFlipUsdV = true;

[[nodiscard]] inline glm::vec2 UsdStToUv(f32 s, f32 t) {
    return kFlipUsdV ? glm::vec2(s, 1.0f - t) : glm::vec2(s, t);
}

/**
 * @brief The same UV transform, expressed for V-flipped coordinates
 *
 * A transform authored against USD's `st` has to be conjugated by the flip that
 * `UsdStToUv` applies to the vertices, or the two disagree. With F(u,v) =
 * (u, 1-v), the answer is F o M o F: the scale is unchanged, the rotation
 * negates, and the offset picks up the flip's translation.
 */
[[nodiscard]] UvTransform ConjugateByVFlip(const UvTransform& transform);

/// `black`, `clamp`, `periodic`/`repeat`, `mirror`, `useMetadata`. Sets
/// `outUnsupported` for `black`, which has no wrap mode here and clamps instead.
[[nodiscard]] TextureSampler::WrapMode WrapFromToken(StringView token, bool& outUnsupported);

/// `closest` is nearest; everything else is linear.
[[nodiscard]] TextureSampler::Filter FilterFromToken(StringView token);

}  // namespace quantiloom::usd
