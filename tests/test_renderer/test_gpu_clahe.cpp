// ============================================================================
// Quantiloom - CLAHE display path, driven directly against rendercore::GpuClahe
// ============================================================================
// What the tone operators guarantee (src/shaders/clahe.comp.hlsl):
//   - With luminanceOnly off, every channel goes through ONE fixed function of its own
//     value: the percentile window (Linear) or the global CDF (Equalize, where all four
//     tiles' CDFs are identical, so the blend is exact). So equal inputs give equal
//     outputs and a brighter input is never darker on screen, per channel.
//   - With luminanceOnly on, only grey pixels are luminance-monotone; a coloured pixel is
//     scaled per channel and clamped per channel, which preserves chroma and not order.
// This test feeds a synthetic quantised image through GpuClahe directly (P0.3 step 4a
// extracted the passes from ExternalRenderContext), so luminanceOnly stays off: the image
// is not achromatic, and the claim that holds for it is the per-channel one. Clahe blends
// per-tile CDFs and is only required to stay finite and inside [0, 1].
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/CommandHelper.hpp"
#include "gpu/vulkan/VulkanBridge.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuClahe.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/VulkanContext.hpp"
#include "support/VulkanTestDevice.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::rendercore;
using quantiloom::testing::VulkanDeviceTest;

namespace {

class GpuClaheTest : public VulkanDeviceTest {};

std::unique_ptr<GpuImage> UploadRgba32f(VulkanContext& context, u32 width, u32 height,
                                        const std::vector<f32>& rgba) {
    auto image = std::make_unique<GpuImage>(
        context.GetAllocator(), context.GetDevice(), width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    if (!image->IsValid()) return {};
    GpuBuffer staging(context.GetAllocator(), rgba.size() * sizeof(f32), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);
    staging.Upload(rgba.data(), rgba.size() * sizeof(f32));
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        CommandHelper::TransitionImageLayout(cmd, image->GetImage(), VK_FORMAT_R32G32B32A32_SFLOAT,
                                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, staging.GetHandle(), image->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &copy);
        CommandHelper::TransitionImageLayout(cmd, image->GetImage(), VK_FORMAT_R32G32B32A32_SFLOAT,
                                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    });
    return image;
}

// A quantised image: 37x23 pixels over 41 discrete values, so equal-input
// groups exist to be checked, with negatives and an asymmetric channel mix.
std::vector<f32> MakeInput(u32 width, u32 height) {
    std::vector<f32> rgba(static_cast<size_t>(width) * height * 4);
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            const size_t i = (static_cast<size_t>(y) * width + x) * 4;
            const f32 base = static_cast<f32>((x * 7 + y * 13) % 41) * 0.05f - 1.0f;
            rgba[i + 0] = base;
            rgba[i + 1] = base * 0.5f + 0.1f;
            rgba[i + 2] = 0.3f - base * 0.25f;
            rgba[i + 3] = 1.0f;
        }
    }
    return rgba;
}

std::pair<f32, f32> FiniteRgbRange(const std::vector<f32>& rgba) {
    f32 lo = std::numeric_limits<f32>::max();
    f32 hi = std::numeric_limits<f32>::lowest();
    for (size_t i = 0; i + 2 < rgba.size(); i += 4) {
        for (u32 c = 0; c < 3; ++c) {
            const f32 v = rgba[i + c];
            if (!std::isfinite(v)) continue;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    return {lo, hi};
}

struct ClaheFixture {
    VulkanContext& context;
    std::unique_ptr<GpuImage> input;
    std::unique_ptr<GpuImage> output;
    std::unique_ptr<GpuClahe> clahe;
    f32 inputMin = 0.0f;
    f32 inputMax = 1.0f;
};

ClaheFixture MakeClahe(gpu::Device& device, VulkanContext& context, u32 width, u32 height,
                       const std::vector<f32>& rgba) {
    ClaheFixture fixture{context, nullptr, nullptr, nullptr, 0.0f, 1.0f};
    fixture.input = UploadRgba32f(context, width, height, rgba);
    fixture.output = std::make_unique<GpuImage>(
        context.GetAllocator(), context.GetDevice(), width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    if (!fixture.input || !fixture.output->IsValid()) return fixture;
    auto created = GpuClahe::Create(device);
    if (created) {
        fixture.clahe = std::move(created.value());
        fixture.clahe->BindAll(gpu::vulkan::Ref(*fixture.input), gpu::vulkan::Ref(*fixture.output));
        const auto range = FiniteRgbRange(rgba);
        fixture.inputMin = range.first;
        fixture.inputMax = range.second;
    }
    return fixture;
}

std::vector<f32> RunClahe(ClaheFixture& fixture, u32 width, u32 height,
                          const DisplayEnhancementParams& params) {
    ClaheRun run;
    run.width = width;
    run.height = height;
    run.params = params;
    run.inputMin = fixture.inputMin;
    run.inputMax = fixture.inputMax;
    run.inputProducer = gpu::Access::ComputeWrite;
    CommandHelper::ExecuteImmediate(fixture.context, [&](VkCommandBuffer cmd) {
        auto list = gpu::vulkan::Wrap(cmd);
        fixture.clahe->Record(list, gpu::vulkan::Ref(*fixture.input), run);
    });
    return CommandHelper::ReadbackImage(fixture.context, fixture.output->GetImage(),
                                        VK_FORMAT_R32G32B32A32_SFLOAT, width, height);
}

// Pixels of one channel sorted by source value. A group of equal sources must show one
// value, and no group may show less than the brightest of any strictly darker group.
// 1e-6 is far below the smallest step the mapping can take (one CDF bin of a 4096-pixel
// image is 2.4e-4), so it absorbs rounding and nothing else.
size_t CountBrokenGroups(const std::vector<f32>& source, const std::vector<f32>& shown,
                         u32 channels, u32 channel) {
    std::vector<std::pair<f32, f32>> pairs;
    const size_t pixels = source.size() / channels;
    pairs.reserve(pixels);
    for (size_t i = 0; i < pixels; ++i) {
        const f32 raw = source[i * channels + channel];
        if (!std::isfinite(raw)) continue;
        pairs.emplace_back(raw, shown[i * channels + channel]);
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
    constexpr u32 kWidth = 37;
    constexpr u32 kHeight = 23;
    const std::vector<f32> rgba = MakeInput(kWidth, kHeight);
    auto fixture = MakeClahe(GpuDevice(), Device(), kWidth, kHeight, rgba);
    if (!fixture.clahe) GTEST_SKIP() << "clahe shaders unavailable";

    for (const auto tone : {DisplayToneMode::Linear, DisplayToneMode::Equalize}) {
        DisplayEnhancementParams params;
        params.enabled = true;
        params.toneMode = tone;
        params.palette = DisplayPalette::Grey;
        params.luminanceOnly = false;
        const std::vector<f32> shown = RunClahe(fixture, kWidth, kHeight, params);
        ASSERT_EQ(shown.size(), rgba.size()) << "tone mode " << static_cast<int>(tone);
        for (u32 channel = 0; channel < 3u; ++channel) {
            EXPECT_EQ(CountBrokenGroups(rgba, shown, 4, channel), 0u)
                << "tone mode " << static_cast<int>(tone) << ", channel " << channel;
        }
    }
}

TEST_F(GpuClaheTest, ClaheStaysFiniteAndInRange) {
    constexpr u32 kWidth = 37;
    constexpr u32 kHeight = 23;
    const std::vector<f32> rgba = MakeInput(kWidth, kHeight);
    auto fixture = MakeClahe(GpuDevice(), Device(), kWidth, kHeight, rgba);
    if (!fixture.clahe) GTEST_SKIP() << "clahe shaders unavailable";

    DisplayEnhancementParams params;
    params.enabled = true;
    params.toneMode = DisplayToneMode::Clahe;
    params.palette = DisplayPalette::Grey;
    params.luminanceOnly = false;
    const std::vector<f32> shown = RunClahe(fixture, kWidth, kHeight, params);
    ASSERT_EQ(shown.size(), rgba.size());
    for (size_t pixel = 0; pixel < shown.size() / 4; ++pixel) {
        for (u32 c = 0; c < 3u; ++c) {
            const f32 value = shown[pixel * 4 + c];
            ASSERT_TRUE(std::isfinite(value)) << "pixel " << pixel;
            EXPECT_GE(value, -1e-4f) << "pixel " << pixel;
            EXPECT_LE(value, 1.0f + 1e-4f) << "pixel " << pixel;
        }
    }
}
