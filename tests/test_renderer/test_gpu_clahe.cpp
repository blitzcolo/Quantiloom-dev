// ============================================================================
// Quantiloom - CLAHE display path, driven through the ExternalRenderContext facade
// ============================================================================
// What the tone operators guarantee (src/shaders/clahe.comp.hlsl):
//   - With luminanceOnly off, every channel goes through ONE fixed function of its own
//     value: the percentile window (Linear) or the global CDF (Equalize, where all four
//     tiles' CDFs are identical, so the blend is exact). So equal inputs give equal
//     outputs and a brighter input is never darker on screen, per channel.
//   - With luminanceOnly on, only grey pixels are luminance-monotone; a coloured pixel is
//     scaled per channel and clamped per channel, which preserves chroma and not order.
// The Cornell box has red and green walls, so luminance order is not a property of this
// image and is not asserted. Clahe blends per-tile CDFs and is only required to stay
// finite and inside [0, 1]. The extraction to renderer/GpuClahe moves this test from the
// facade to the class; the claims stay.
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

// Pixels of one channel sorted by source value. A group of equal sources must show one
// value, and no group may show less than the brightest of any strictly darker group.
// 1e-6 is far below the smallest step the mapping can take (one CDF bin of a 4096-pixel
// image is 2.4e-4), so it absorbs rounding and nothing else.
size_t CountBrokenGroups(const Image& source, const Image& shown, u32 channel) {
    std::vector<std::pair<f32, f32>> pairs;
    const size_t pixels = static_cast<size_t>(source.width) * source.height;
    pairs.reserve(pixels);
    for (size_t i = 0; i < pixels; ++i) {
        const f32 raw = source.data[i * source.channels + channel];
        if (!std::isfinite(raw)) continue;
        pairs.emplace_back(raw, shown.data[i * shown.channels + channel]);
    }
    std::sort(pairs.begin(), pairs.end());

    constexpr f32 kTolerance = 1e-6f;
    size_t broken = 0;
    bool haveBelow = false;
    f32 brightestBelow = 0.0f;
    size_t begin = 0;
    while (begin < pairs.size()) {
        size_t end = begin;
        f32 groupMin = pairs[begin].second;
        f32 groupMax = pairs[begin].second;
        while (end < pairs.size() && pairs[end].first == pairs[begin].first) {
            groupMin = std::min(groupMin, pairs[end].second);
            groupMax = std::max(groupMax, pairs[end].second);
            ++end;
        }
        if (groupMax - groupMin > kTolerance) ++broken;
        if (haveBelow && groupMin < brightestBelow - kTolerance) ++broken;
        brightestBelow = haveBelow ? std::max(brightestBelow, groupMax) : groupMax;
        haveBelow = true;
        begin = end;
    }
    return broken;
}

}  // namespace

TEST_F(GpuClaheTest, PerChannelTonesNeverInvertTheirOwnChannel) {
    if (!quantiloom::testing::FacadeScene::HasAssets()) GTEST_SKIP() << "cornell_box.gltf not in assets";
    quantiloom::testing::FacadeScene scene(Device());
    ASSERT_TRUE(scene.Ready()) << scene.Error();
    ASSERT_TRUE(scene.ApplyScene()) << scene.Error();

    for (const auto tone : {DisplayToneMode::Linear, DisplayToneMode::Equalize}) {
        DisplayEnhancementParams params;
        params.enabled = true;
        params.toneMode = tone;
        params.palette = DisplayPalette::Grey;
        params.luminanceOnly = false;
        scene.Context().SetDisplayEnhancementParams(params);
        scene.DrawFrame();
        scene.DrawFrame();
        const auto raw = scene.Context().CaptureScreenshot();
        const auto shown = scene.Context().CaptureDisplayImage();
        ASSERT_TRUE(raw.has_value()) << raw.error();
        ASSERT_TRUE(shown.has_value()) << shown.error();
        ASSERT_EQ(raw.value().width, shown.value().width);
        ASSERT_EQ(raw.value().height, shown.value().height);
        ASSERT_GE(raw.value().channels, 3u);
        ASSERT_GE(shown.value().channels, 3u);
        for (u32 channel = 0; channel < 3u; ++channel) {
            EXPECT_EQ(CountBrokenGroups(raw.value(), shown.value(), channel), 0u)
                << "tone mode " << static_cast<int>(tone) << ", channel " << channel;
        }
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
