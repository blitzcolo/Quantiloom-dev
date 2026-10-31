// ============================================================================
// Quantiloom - CLAHE display path, driven through the ExternalRenderContext facade
// ============================================================================
// Thermography rule (repository AGENTS.md): on an achromatic image, Linear and Equalize must
// show zero inverted pairs -- a pixel that was darker in the source is never brighter on
// screen -- and Clahe must stay finite and inside [0, 1]. The extraction to renderer/GpuClahe
// moves this test from the facade to the class; the claim stays.
// ============================================================================

#include <gtest/gtest.h>

#include "support/FacadeScene.hpp"
#include "support/VulkanTestDevice.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

using namespace quantiloom;
using quantiloom::testing::VulkanDeviceTest;

namespace {

class GpuClaheTest : public VulkanDeviceTest {};

f64 Luminance(const Image& image, size_t pixel) {
    const f32* p = &image.data[pixel * image.channels];
    if (image.channels < 3) return p[0];
    return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
}

// Pixels sorted by source luminance; a group of equal source values must not dip below the
// brightest display value of any strictly darker group.
size_t CountInvertedGroups(const Image& source, const Image& shown) {
    std::vector<std::pair<f64, f64>> pairs;
    const size_t pixels = static_cast<size_t>(source.width) * source.height;
    pairs.reserve(pixels);
    for (size_t i = 0; i < pixels; ++i) pairs.emplace_back(Luminance(source, i), Luminance(shown, i));
    std::sort(pairs.begin(), pairs.end());

    size_t inverted = 0;
    f64 brightestBelow = -1.0;
    size_t begin = 0;
    while (begin < pairs.size()) {
        size_t end = begin;
        f64 groupMin = pairs[begin].second;
        f64 groupMax = pairs[begin].second;
        while (end < pairs.size() && pairs[end].first == pairs[begin].first) {
            groupMin = std::min(groupMin, pairs[end].second);
            groupMax = std::max(groupMax, pairs[end].second);
            ++end;
        }
        if (brightestBelow >= 0.0 && groupMin < brightestBelow - 1e-6) ++inverted;
        brightestBelow = std::max(brightestBelow, groupMax);
        begin = end;
    }
    return inverted;
}

}  // namespace

TEST_F(GpuClaheTest, MonotoneTonesNeverInvertAnAchromaticImage) {
    if (!quantiloom::testing::FacadeScene::HasAssets()) GTEST_SKIP() << "cornell_box.gltf not in assets";
    quantiloom::testing::FacadeScene scene(Device());
    ASSERT_TRUE(scene.Ready()) << scene.Error();
    ASSERT_TRUE(scene.ApplyScene()) << scene.Error();

    for (const auto tone : {DisplayToneMode::Linear, DisplayToneMode::Equalize}) {
        DisplayEnhancementParams params;
        params.enabled = true;
        params.toneMode = tone;
        params.palette = DisplayPalette::Grey;
        scene.Context().SetDisplayEnhancementParams(params);
        scene.DrawFrame();
        scene.DrawFrame();
        const auto raw = scene.Context().CaptureScreenshot();
        const auto shown = scene.Context().CaptureDisplayImage();
        ASSERT_TRUE(raw.has_value()) << raw.error();
        ASSERT_TRUE(shown.has_value()) << shown.error();
        ASSERT_EQ(raw.value().width, shown.value().width);
        EXPECT_EQ(CountInvertedGroups(raw.value(), shown.value()), 0u)
            << "tone mode " << static_cast<int>(tone);
    }
}

TEST_F(GpuClaheTest, ClaheStaysFiniteAndInRange) {
    if (!quantiloom::testing::FacadeScene::HasAssets()) GTEST_SKIP() << "cornell_box.gltf not in assets";
    quantiloom::testing::FacadeScene scene(Device());
    ASSERT_TRUE(scene.Ready()) << scene.Error();
    ASSERT_TRUE(scene.ApplyScene()) << scene.Error();

    DisplayEnhancementParams params;
    params.enabled = true;
    params.toneMode = DisplayToneMode::Clahe;
    params.palette = DisplayPalette::Grey;
    scene.Context().SetDisplayEnhancementParams(params);
    scene.DrawFrame();
    scene.DrawFrame();
    const auto shown = scene.Context().CaptureDisplayImage();
    ASSERT_TRUE(shown.has_value()) << shown.error();
    const Image& image = shown.value();
    const u32 colourChannels = std::min<u32>(image.channels, 3u);
    for (size_t pixel = 0; pixel < static_cast<size_t>(image.width) * image.height; ++pixel) {
        for (u32 c = 0; c < colourChannels; ++c) {
            const f32 value = image.data[pixel * image.channels + c];
            ASSERT_TRUE(std::isfinite(value)) << "pixel " << pixel;
            EXPECT_GE(value, -1e-4f) << "pixel " << pixel;
            EXPECT_LE(value, 1.0f + 1e-4f) << "pixel " << pixel;
        }
    }
}
