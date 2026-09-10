// ============================================================================
// Quantiloom - Unit Tests for io/UsdTextureBank.hpp
// ============================================================================
// The bank is the part of USD material loading that needs no stage: recipes,
// deduplication, channel repacking and the UV conjugation are all decided from
// data. They are tested here directly so that a repacking bug is a failure with
// a line number rather than a texture that looks slightly wrong in a render.
// ============================================================================

#include <gtest/gtest.h>

#include "io/UsdLoader.hpp"
#include "io/UsdSurfaceTables.hpp"
#include "io/UsdTextureBank.hpp"

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::usd;

namespace {

/// A solid RGBA8 image, without touching the filesystem.
Texture SolidSource(const String& name, u8 r, u8 g, u8 b, u8 a, u32 size = 2) {
    Texture texture;
    texture.width = size;
    texture.height = size;
    texture.channels = 4;
    texture.name = name;
    texture.sourceUri = name;
    texture.pixels.reserve(static_cast<usize>(size) * size * 4);
    for (u32 i = 0; i < size * size; ++i) {
        texture.pixels.insert(texture.pixels.end(), {r, g, b, a});
    }
    return texture;
}

SlotRecipe::Src SourceFrom(const String& path, ChannelSel channel) {
    SlotRecipe::Src src;
    src.path = path;
    src.channel = channel;
    return src;
}

/// uv' = Translate(offset) * Rotate(rotation) * Scale(scale) * uv, which is what
/// Material::UvTransform documents and what the GPU-side packing does.
glm::vec2 ApplyUv(const UvTransform& transform, glm::vec2 uv) {
    const glm::vec2 scaled = uv * transform.scale;
    const f32 c = std::cos(transform.rotation);
    const f32 s = std::sin(transform.rotation);
    return glm::vec2(c * scaled.x - s * scaled.y, s * scaled.x + c * scaled.y) +
           transform.offset;
}

glm::vec2 FlipV(glm::vec2 uv) { return glm::vec2(uv.x, 1.0f - uv.y); }

}  // namespace

TEST(UsdTextureBank, RecipeKeyDedupsIdenticalRepacks) {
    UsdTextureBank bank;
    bank.AdoptSource("metal.png", SolidSource("metal.png", 0x40, 0, 0, 0xFF));
    bank.AdoptSource("rough.png", SolidSource("rough.png", 0, 0xC0, 0, 0xFF));

    SlotRecipe recipe;
    recipe.dst[1] = SourceFrom("rough.png", ChannelSel::G);
    recipe.dst[2] = SourceFrom("metal.png", ChannelSel::R);
    recipe.debugName = "metallicRoughness";

    std::vector<Texture> textures;
    const i32 first = bank.Materialise(recipe, textures);
    const i32 second = bank.Materialise(recipe, textures);

    ASSERT_GE(first, 0);
    EXPECT_EQ(first, second) << "two materials repacking the same pair the same way "
                                "are one entry";
    EXPECT_EQ(textures.size(), 1u);
    EXPECT_EQ(textures[0].pixels[1], 0xC0);
    EXPECT_EQ(textures[0].pixels[2], 0x40);
    EXPECT_EQ(textures[0].pixels[0], 0xFF) << "an unbound channel takes its fill";
}

TEST(UsdTextureBank, ADifferentColourSpaceIsADifferentEntry) {
    UsdTextureBank bank;
    bank.AdoptSource("shared.png", SolidSource("shared.png", 90, 120, 150, 255));

    SlotRecipe asColour;
    for (u32 i = 0; i < 4; ++i) {
        asColour.dst[i] = SourceFrom("shared.png",
                                     i == 0   ? ChannelSel::R
                                     : i == 1 ? ChannelSel::G
                                     : i == 2 ? ChannelSel::B
                                              : ChannelSel::A);
    }
    SlotRecipe asData = asColour;
    asColour.srgb = true;
    asData.srgb = false;

    std::vector<Texture> textures;
    const i32 colour = bank.Materialise(asColour, textures);
    const i32 data = bank.Materialise(asData, textures);

    ASSERT_GE(colour, 0);
    ASSERT_GE(data, 0);
    EXPECT_NE(colour, data) << "the same pixels uploaded in two formats are two entries";
    EXPECT_TRUE(textures[colour].isSRGB);
    EXPECT_FALSE(textures[data].isSRGB);
}

TEST(UsdTextureBank, IdentityRecipeReusesTheDecodedSource) {
    UsdTextureBank bank;
    const Texture source = SolidSource("normal.png", 128, 129, 255, 254, 4);
    bank.AdoptSource("normal.png", source);

    SlotRecipe recipe;
    recipe.dst[0] = SourceFrom("normal.png", ChannelSel::R);
    recipe.dst[1] = SourceFrom("normal.png", ChannelSel::G);
    recipe.dst[2] = SourceFrom("normal.png", ChannelSel::B);
    recipe.dst[3] = SourceFrom("normal.png", ChannelSel::A);
    ASSERT_TRUE(recipe.IsIdentityOf("normal.png"));

    std::vector<Texture> textures;
    const i32 index = bank.Materialise(recipe, textures);

    ASSERT_GE(index, 0);
    EXPECT_EQ(textures[index].width, source.width);
    EXPECT_EQ(textures[index].height, source.height);
    EXPECT_EQ(textures[index].pixels, source.pixels)
        << "a pass-through must not go through the repacking loop at all";
}

TEST(UsdTextureBank, AnAffineIsBakedIntoThePixels) {
    UsdTextureBank bank;
    bank.AdoptSource("data.png", SolidSource("data.png", 0x80, 0, 0, 0xFF));

    SlotRecipe recipe;
    SlotRecipe::Src src = SourceFrom("data.png", ChannelSel::R);
    src.scale = 0.5f;
    src.bias = 0.25f;
    recipe.dst[0] = src;

    std::vector<Texture> textures;
    const i32 index = bank.Materialise(recipe, textures);

    ASSERT_GE(index, 0);
    // 0x80 is 0.502; 0.502 * 0.5 + 0.25 = 0.501, which quantises back to 0x80.
    const f32 expected = (static_cast<f32>(0x80) / 255.0f) * 0.5f + 0.25f;
    EXPECT_EQ(textures[index].pixels[0],
              static_cast<u8>(expected * 255.0f + 0.5f));
}

TEST(UsdTextureBank, ARecipeWithNoDecodedSourceIsNotAnEntry) {
    UsdTextureBank bank;

    SlotRecipe recipe;
    recipe.dst[0] = SourceFrom("missing.png", ChannelSel::R);

    std::vector<Texture> textures;
    EXPECT_EQ(bank.Materialise(recipe, textures), -1);
    EXPECT_TRUE(textures.empty());
}

TEST(UsdTextureBank, OneMissingSourceLeavesTheRestOfTheSlot) {
    UsdTextureBank bank;
    bank.AdoptSource("colour.png", SolidSource("colour.png", 10, 20, 30, 255));

    // A base colour whose opacity map failed to decode should still be a base
    // colour.
    SlotRecipe recipe;
    recipe.dst[0] = SourceFrom("colour.png", ChannelSel::R);
    recipe.dst[1] = SourceFrom("colour.png", ChannelSel::G);
    recipe.dst[2] = SourceFrom("colour.png", ChannelSel::B);
    recipe.dst[3] = SourceFrom("missing.png", ChannelSel::A);

    std::vector<Texture> textures;
    const i32 index = bank.Materialise(recipe, textures);

    ASSERT_GE(index, 0);
    EXPECT_EQ(textures[index].pixels[0], 10);
    EXPECT_EQ(textures[index].pixels[3], 255) << "the missing channel takes its fill";
}

TEST(UsdTextureBank, ReleaseSourcesKeepsTheEntries) {
    UsdTextureBank bank;
    bank.AdoptSource("colour.png", SolidSource("colour.png", 10, 20, 30, 255));

    SlotRecipe recipe;
    recipe.dst[0] = SourceFrom("colour.png", ChannelSel::R);

    std::vector<Texture> textures;
    ASSERT_GE(bank.Materialise(recipe, textures), 0);

    bank.ReleaseSources();
    EXPECT_EQ(bank.SourceCount(), 0u);
    EXPECT_FALSE(textures[0].pixels.empty()) << "the entry owns its own pixels";
}

TEST(UsdTextureBank, ConjugateByVFlipMatchesNumericComposition) {
    // F o M o F, with F(u, v) = (u, 1 - v). Checked by composition rather than by
    // re-deriving the algebra, because the algebra is what the function is.
    const UvTransform cases[] = {
        UvTransform{},
        UvTransform{glm::vec2(0.1f, 0.2f), 0.0f, glm::vec2(2.0f, 3.0f)},
        UvTransform{glm::vec2(-0.3f, 0.7f), 0.6f, glm::vec2(1.5f, 0.5f)},
        UvTransform{glm::vec2(0.0f, 0.0f), -1.2f, glm::vec2(1.0f, 1.0f)},
        UvTransform{glm::vec2(2.0f, -1.0f), 3.0f, glm::vec2(0.25f, 4.0f)},
    };
    const glm::vec2 probes[] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f},
                                {0.37f, 0.62f}, {-0.5f, 1.75f}};

    for (const UvTransform& transform : cases) {
        const UvTransform conjugate = ConjugateByVFlip(transform);
        for (const glm::vec2& probe : probes) {
            const glm::vec2 expected = FlipV(ApplyUv(transform, FlipV(probe)));
            const glm::vec2 actual = ApplyUv(conjugate, probe);
            EXPECT_NEAR(actual.x, expected.x, 1e-4f);
            EXPECT_NEAR(actual.y, expected.y, 1e-4f);
        }
    }
}

TEST(UsdTextureBank, WrapAndFilterTokens) {
    bool unsupported = false;
    EXPECT_EQ(WrapFromToken("clamp", unsupported), TextureSampler::WrapMode::ClampToEdge);
    EXPECT_FALSE(unsupported);
    EXPECT_EQ(WrapFromToken("mirror", unsupported), TextureSampler::WrapMode::MirroredRepeat);
    EXPECT_FALSE(unsupported);
    EXPECT_EQ(WrapFromToken("periodic", unsupported), TextureSampler::WrapMode::Repeat);
    EXPECT_FALSE(unsupported);
    EXPECT_EQ(WrapFromToken("useMetadata", unsupported), TextureSampler::WrapMode::Repeat);
    EXPECT_FALSE(unsupported);

    // `black` has no equivalent, so it clamps and says so.
    EXPECT_EQ(WrapFromToken("black", unsupported), TextureSampler::WrapMode::ClampToEdge);
    EXPECT_TRUE(unsupported);

    EXPECT_EQ(FilterFromToken("closest"), TextureSampler::Filter::Nearest);
    EXPECT_EQ(FilterFromToken("linear"), TextureSampler::Filter::Linear);
    EXPECT_EQ(FilterFromToken("cubic"), TextureSampler::Filter::Linear);
}

TEST(UsdSurfaceTables, ClassifyShaderIdMatchesExactTokens) {
    EXPECT_EQ(ClassifyShaderId("UsdPreviewSurface"), SurfaceVocabulary::UsdPreviewSurface);
    EXPECT_EQ(ClassifyShaderId("ND_UsdPreviewSurface_surfaceshader"),
              SurfaceVocabulary::UsdPreviewSurface);
    EXPECT_EQ(ClassifyShaderId("ND_standard_surface_surfaceshader"),
              SurfaceVocabulary::StandardSurface);
    EXPECT_EQ(ClassifyShaderId("ND_gltf_pbr_surfaceshader"), SurfaceVocabulary::GltfPbr);
    EXPECT_EQ(ClassifyShaderId("ND_open_pbr_surface_surfaceshader"),
              SurfaceVocabulary::OpenPbrSurface);

    // The substring test this replaced routed gltf_pbr into the UsdPreviewSurface
    // reader, where no input name matched and every material came out grey.
    EXPECT_EQ(ClassifyShaderId("ND_disney_principled_surfaceshader"),
              SurfaceVocabulary::Unknown);
    EXPECT_EQ(ClassifyShaderId(""), SurfaceVocabulary::Unknown);
}

// ============================================================================
// The variant spec a config carries
// ============================================================================
// pxr-free, so these run whether or not this build has OpenUSD.

TEST(UsdVariantSpec, ParsesPrimScopedAndBareSelections) {
    auto parsed = ParseUsdVariantSpec("/Root/Car{color=red}, lod=low ,/Root/Trailer{lod=high}");
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    const auto& selections = parsed.value();

    ASSERT_EQ(selections.size(), 3u);
    EXPECT_EQ(selections.at("/Root/Car").at("color"), "red");
    EXPECT_EQ(selections.at("/Root/Trailer").at("lod"), "high");

    // The empty prim path is the wildcard: that set, on every prim that owns it.
    EXPECT_EQ(selections.at("").at("lod"), "low");
}

TEST(UsdVariantSpec, AnEmptySpecSelectsNothing) {
    auto parsed = ParseUsdVariantSpec("");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_TRUE(parsed.value().empty());

    auto blank = ParseUsdVariantSpec("   ");
    ASSERT_TRUE(blank.has_value());
    EXPECT_TRUE(blank.value().empty());
}

TEST(UsdVariantSpec, RejectsAMalformedEntry) {
    // An unknown variant *name* is a typo in the data and warns; the syntax is
    // the contract, so a spec that does not parse fails the load.
    EXPECT_FALSE(ParseUsdVariantSpec("/Root/Car").has_value());
    EXPECT_FALSE(ParseUsdVariantSpec("/Root/Car{color}").has_value());
    EXPECT_FALSE(ParseUsdVariantSpec("/Root/Car{color=red").has_value());
    EXPECT_FALSE(ParseUsdVariantSpec("{color=red}").has_value());
    EXPECT_FALSE(ParseUsdVariantSpec("color=").has_value());
    EXPECT_FALSE(ParseUsdVariantSpec("=red").has_value());
    EXPECT_FALSE(ParseUsdVariantSpec("a=b=c").has_value());
    EXPECT_FALSE(ParseUsdVariantSpec("lod=low,").has_value());
}
