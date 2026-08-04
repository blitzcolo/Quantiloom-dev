/**
 * @file test_spectral_unmixer.cpp
 * @brief Base colour texels to endmember weights
 *
 * The unmixer decides, per texel, what a measured surface is made of. It has
 * no visible failure mode: wrong weights render as a plausible material that
 * is not the one asked for, so the properties worth pinning are the ones an
 * image would never show -- that the average reflectance still matches the
 * measurement, and that a texel which IS one endmember gets that endmember.
 */

#include <gtest/gtest.h>

#include "renderer/SpectralUnmixer.hpp"

#include <vector>

using namespace quantiloom;
using namespace quantiloom::rendercore;

namespace {

constexpr f32 kWeightScale = 2.0f;

f32 DecodeWeight(u8 encoded) {
    return kWeightScale * static_cast<f32>(encoded) / 255.0f;
}

u8 EncodeLinear(f32 v) {
    return static_cast<u8>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

void PushTexel(std::vector<u8>& out, const glm::vec3& linear) {
    out.push_back(EncodeLinear(linear.r));
    out.push_back(EncodeLinear(linear.g));
    out.push_back(EncodeLinear(linear.b));
    out.push_back(255);
}

}  // namespace

// A texture made of exactly two endmember colours must come back as exactly
// those two endmembers -- the one case where the right answer is known without
// argument. Half and half, so the normalisation leaves the weights at 1.
TEST(SpectralUnmixer, CheckerboardOfTwoEndmembersSeparatesCleanly) {
    const glm::vec3 colors[2] = {glm::vec3(0.20f, 0.35f, 0.10f),   // green
                                 glm::vec3(0.45f, 0.35f, 0.15f)};  // dry

    std::vector<u8> src;
    for (i32 i = 0; i < 64; ++i) {
        PushTexel(src, colors[i % 2]);
    }
    std::vector<u8> dst(src.size());
    f32 meanWeights[Material::MAX_ENDMEMBERS] = {};

    UnmixTexels(src.data(), 64, /*srgb=*/false, glm::vec3(1.0f), colors, 2, dst.data(),
                meanWeights);

    // Texel 0 is pure green, texel 1 is pure dry.
    EXPECT_NEAR(DecodeWeight(dst[0]), 1.0f, 0.02f);
    EXPECT_NEAR(DecodeWeight(dst[1]), 0.0f, 0.02f);
    EXPECT_NEAR(DecodeWeight(dst[4]), 0.0f, 0.02f);
    EXPECT_NEAR(DecodeWeight(dst[5]), 1.0f, 0.02f);

    // And the surface reads as half of each.
    EXPECT_NEAR(meanWeights[0], 0.5f, 0.02f);
    EXPECT_NEAR(meanWeights[1], 0.5f, 0.02f);
}

// The quantitative promise: binding a measured curve means the surface HAS
// that reflectance on average. Without the normalisation a base colour painted
// darker than the material would darken the material, and the measurement
// would stop meaning anything.
TEST(SpectralUnmixer, MeanWeightIsOneHoweverBrightTheTextureIs) {
    const glm::vec3 colors[1] = {glm::vec3(0.30f, 0.40f, 0.20f)};

    for (const f32 scale : {0.25f, 0.5f, 1.0f}) {
        std::vector<u8> src;
        for (i32 i = 0; i < 32; ++i) {
            // Varying brightness around `scale`, so there is something to
            // preserve the mean of.
            const f32 t = scale * (0.6f + 0.8f * static_cast<f32>(i % 4) / 3.0f);
            PushTexel(src, colors[0] * t);
        }
        std::vector<u8> dst(src.size());
        f32 meanWeights[Material::MAX_ENDMEMBERS] = {};

        UnmixTexels(src.data(), 32, false, glm::vec3(1.0f), colors, 1, dst.data(), meanWeights);

        EXPECT_NEAR(meanWeights[0], 1.0f, 1e-3f) << "scale " << scale;

        f32 sum = 0.0f;
        for (usize i = 0; i < 32; ++i) {
            sum += DecodeWeight(dst[i * 4]);
        }
        EXPECT_NEAR(sum / 32.0f, 1.0f, 0.02f) << "encoded mean, scale " << scale;
    }
}

// What the mean preserves is the average; the variation is what the texture
// contributes, and it has to survive. A flat texture gives flat weights, a
// varying one does not -- this is the difference between the old behaviour and
// the new one.
TEST(SpectralUnmixer, VariationInTheTextureSurvivesAsVariationInTheWeights) {
    const glm::vec3 colors[1] = {glm::vec3(0.30f, 0.40f, 0.20f)};

    std::vector<u8> flat, varied;
    for (i32 i = 0; i < 32; ++i) {
        PushTexel(flat, colors[0] * 0.5f);
        PushTexel(varied, colors[0] * (i % 2 == 0 ? 0.25f : 0.75f));
    }
    std::vector<u8> flatOut(flat.size()), variedOut(varied.size());

    UnmixTexels(flat.data(), 32, false, glm::vec3(1.0f), colors, 1, flatOut.data());
    UnmixTexels(varied.data(), 32, false, glm::vec3(1.0f), colors, 1, variedOut.data());

    EXPECT_NEAR(DecodeWeight(flatOut[0]), DecodeWeight(flatOut[4]), 0.02f);
    EXPECT_GT(std::abs(DecodeWeight(variedOut[0]) - DecodeWeight(variedOut[4])), 0.5f);
}

// sRGB 0.5 is linear 0.21, so treating one as the other biases every weight by
// more than a factor of two. The flag comes from how the texture was tagged at
// load, which is a different file from this one.
TEST(SpectralUnmixer, SrgbSourceIsLinearisedBeforeSolving) {
    const glm::vec3 colors[1] = {glm::vec3(1.0f, 1.0f, 1.0f)};

    std::vector<u8> src;
    for (i32 i = 0; i < 16; ++i) {
        src.insert(src.end(), {128, 128, 128, 255});
    }
    std::vector<u8> asSrgb(src.size()), asLinear(src.size());
    f32 srgbMean[Material::MAX_ENDMEMBERS] = {};
    f32 linearMean[Material::MAX_ENDMEMBERS] = {};

    UnmixTexels(src.data(), 16, true, glm::vec3(1.0f), colors, 1, asSrgb.data(), srgbMean);
    UnmixTexels(src.data(), 16, false, glm::vec3(1.0f), colors, 1, asLinear.data(), linearMean);

    // Both normalise to a mean of 1 -- a uniform texture cannot show the
    // difference in the weights, which is exactly why this is worth a test
    // that looks at something else.
    EXPECT_NEAR(srgbMean[0], 1.0f, 1e-3f);
    EXPECT_NEAR(linearMean[0], 1.0f, 1e-3f);

    // The difference shows when the texture is not uniform: linearising
    // stretches the dark end, so the same byte range spans a wider ratio.
    std::vector<u8> ramp;
    for (i32 i = 0; i < 16; ++i) {
        const u8 v = static_cast<u8>(64 + i * 8);
        ramp.insert(ramp.end(), {v, v, v, 255});
    }
    std::vector<u8> rampSrgb(ramp.size()), rampLinear(ramp.size());
    UnmixTexels(ramp.data(), 16, true, glm::vec3(1.0f), colors, 1, rampSrgb.data());
    UnmixTexels(ramp.data(), 16, false, glm::vec3(1.0f), colors, 1, rampLinear.data());

    const f32 srgbRatio = DecodeWeight(rampSrgb[15 * 4]) / DecodeWeight(rampSrgb[0]);
    const f32 linearRatio = DecodeWeight(rampLinear[15 * 4]) / DecodeWeight(rampLinear[0]);
    EXPECT_GT(srgbRatio, linearRatio * 1.5f);
}

TEST(SpectralUnmixer, BaseColorFactorModulatesTheTexture) {
    const glm::vec3 colors[1] = {glm::vec3(0.5f, 0.5f, 0.5f)};

    std::vector<u8> src;
    for (i32 i = 0; i < 16; ++i) {
        PushTexel(src, glm::vec3(i % 2 == 0 ? 0.25f : 0.5f));
    }
    std::vector<u8> plain(src.size()), tinted(src.size());

    UnmixTexels(src.data(), 16, false, glm::vec3(1.0f), colors, 1, plain.data());
    UnmixTexels(src.data(), 16, false, glm::vec3(0.5f), colors, 1, tinted.data());

    // A uniform factor scales every texel, and the normalisation then divides
    // it back out -- the factor changes the level, which the mean anchors, not
    // the pattern.
    EXPECT_NEAR(DecodeWeight(plain[0]), DecodeWeight(tinted[0]), 0.02f);
}

TEST(SpectralUnmixer, EmptyInputIsHandled) {
    const glm::vec3 colors[1] = {glm::vec3(0.5f)};
    std::vector<u8> dst(4);

    EXPECT_EQ(UnmixTexels(nullptr, 0, false, glm::vec3(1.0f), colors, 1, dst.data()), 0u);
    EXPECT_EQ(UnmixTexels(dst.data(), 0, false, glm::vec3(1.0f), colors, 1, dst.data()), 0u);
}

// A texture far brighter than its endmember wants weights past the 2x the
// encoding holds. The count is reported so the loader can say so rather than
// silently flattening the highlights.
TEST(SpectralUnmixer, ClippedTexelsAreCounted) {
    const glm::vec3 colors[1] = {glm::vec3(0.1f, 0.1f, 0.1f)};

    // One very bright texel among dark ones: after normalisation it needs a
    // weight far above 2.
    std::vector<u8> src;
    PushTexel(src, glm::vec3(1.0f));
    for (i32 i = 0; i < 31; ++i) {
        PushTexel(src, glm::vec3(0.02f));
    }
    std::vector<u8> dst(src.size());

    const u64 clipped = UnmixTexels(src.data(), 32, false, glm::vec3(1.0f), colors, 1,
                                    dst.data());
    EXPECT_GE(clipped, 1u);
}
