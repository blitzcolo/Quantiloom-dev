#include <gtest/gtest.h>

#include "support/VulkanTestDevice.hpp"

#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuDisplayRange.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/VulkanContext.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace quantiloom;
using quantiloom::testing::VulkanDeviceTest;

namespace {

using rendercore::DisplayRange;
using rendercore::GpuDisplayRange;

DisplayRange CpuReference(const std::vector<f32>& rgba,
                          f32 percentileLow, f32 percentileHigh) {
    std::vector<f32> luminances;
    luminances.reserve(rgba.size() / 4);
    f32 absMin = std::numeric_limits<f32>::max();
    f32 absMax = std::numeric_limits<f32>::lowest();
    for (usize i = 0; i < rgba.size(); i += 4) {
        const f32 r = rgba[i];
        const f32 g = rgba[i + 1];
        const f32 b = rgba[i + 2];
        if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b))
            continue;
        const f32 lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        luminances.push_back(lum);
        absMin = std::min(absMin, lum);
        absMax = std::max(absMax, lum);
    }
    if (luminances.empty() || absMin >= absMax) return {};

    constexpr usize binsCount = GpuDisplayRange::kHistogramBins;
    std::vector<u32> histogram(binsCount, 0);
    const f32 scale = static_cast<f32>(binsCount - 1) / (absMax - absMin);
    for (const f32 lum : luminances) {
        usize bin = static_cast<usize>((lum - absMin) * scale);
        bin = std::min(bin, binsCount - 1);
        ++histogram[bin];
    }

    const f64 lowFraction =
        std::clamp(static_cast<f64>(percentileLow), 0.0, 100.0) / 100.0;
    const f64 highFraction =
        std::clamp(static_cast<f64>(percentileHigh), 0.0, 100.0) / 100.0;
    const usize targetLow =
        static_cast<usize>(static_cast<f64>(luminances.size()) * lowFraction);
    const usize targetHigh =
        static_cast<usize>(static_cast<f64>(luminances.size()) * highFraction);
    usize cumulative = 0;
    usize binLow = 0;
    usize binHigh = binsCount - 1;
    bool haveLow = false;
    for (usize i = 0; i < binsCount; ++i) {
        cumulative += histogram[i];
        if (!haveLow && cumulative >= targetLow) {
            binLow = i;
            haveLow = true;
        }
        if (cumulative >= targetHigh) {
            binHigh = i;
            break;
        }
    }
    const f32 invScale = (absMax - absMin) / static_cast<f32>(binsCount - 1);
    DisplayRange result{absMin + static_cast<f32>(binLow) * invScale,
                        absMin + static_cast<f32>(binHigh) * invScale};
    if (!(result.max - result.min > (absMax - absMin) * 1e-3f))
        result = {absMin, absMax};
    return result;
}

std::unique_ptr<GpuImage> UploadRgba32f(VulkanContext& context,
                                        u32 width, u32 height,
                                        const std::vector<f32>& rgba) {
    if (rgba.size() != static_cast<usize>(width) * height * 4) return {};
    auto image = std::make_unique<GpuImage>(
        context.GetAllocator(), context.GetDevice(), width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    if (!image->IsValid()) return {};
    GpuBuffer staging(context.GetAllocator(), rgba.size() * sizeof(f32),
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);
    staging.Upload(rgba.data(), rgba.size() * sizeof(f32));
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        CommandHelper::TransitionImageLayout(
            cmd, image->GetImage(), VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, staging.GetHandle(), image->GetImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        VkImageMemoryBarrier ready{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ready.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ready.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.image = image->GetImage();
        ready.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ready.subresourceRange.levelCount = 1;
        ready.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &ready);
    });
    return image;
}

void ExpectExactRange(GpuDisplayRange& gpu, VulkanContext& context,
                      u32 width, u32 height, const std::vector<f32>& rgba,
                      f32 low, f32 high) {
    auto image = UploadRgba32f(context, width, height, rgba);
    ASSERT_NE(image, nullptr);
    const auto actual = gpu.Compute(*image, width, height, low, high);
    ASSERT_TRUE(actual.has_value()) << actual.error();
    const DisplayRange expected = CpuReference(rgba, low, high);
    EXPECT_EQ(actual.value().min, expected.min)
        << "percentiles " << low << ", " << high;
    EXPECT_EQ(actual.value().max, expected.max)
        << "percentiles " << low << ", " << high;
}

std::vector<f32> Rgba(const std::vector<std::array<f32, 4>>& pixels) {
    std::vector<f32> result;
    result.reserve(pixels.size() * 4);
    for (const auto& pixel : pixels)
        result.insert(result.end(), pixel.begin(), pixel.end());
    return result;
}

}  // namespace

class DisplayRangeGpuTest : public VulkanDeviceTest {};

TEST_F(DisplayRangeGpuTest, FiltersInvalidRgbAndFallsBackForEmptyOrConstantImages) {
    auto created = GpuDisplayRange::Create(Device());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());
    const f32 nan = std::numeric_limits<f32>::quiet_NaN();
    const f32 inf = std::numeric_limits<f32>::infinity();
    const auto invalid = Rgba({{nan, 1.0f, 1.0f, 7.0f},
                               {1.0f, inf, 1.0f, 8.0f},
                               {1.0f, 1.0f, -inf, 9.0f}});
    ExpectExactRange(*gpu, Device(), 3, 1, invalid, 1.0f, 99.0f);

    const auto constant = Rgba({{-4.0f, -4.0f, -4.0f, nan},
                                {-4.0f, -4.0f, -4.0f, inf},
                                {-4.0f, -4.0f, -4.0f, 0.0f}});
    ExpectExactRange(*gpu, Device(), 3, 1, constant, 1.0f, 99.0f);

    const auto mixed = Rgba({{nan, 0.0f, 0.0f, 0.0f},
                             {-8.0f, 1.0f, 3.0f, nan},
                             {10.0f, -4.0f, 2.0f, inf},
                             {1.0f, 2.0f, 30.0f, -inf},
                             {inf, 0.0f, 0.0f, 0.0f}});
    ExpectExactRange(*gpu, Device(), 5, 1, mixed, 1.0f, 99.0f);
}

TEST_F(DisplayRangeGpuTest, RequestsCpuFallbackForUnsafeSmallRgb) {
    auto created = GpuDisplayRange::Create(Device());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());
    const f32 subnormal = std::numeric_limits<f32>::denorm_min() * 4096.0f;
    const f32 tinyNormal = std::numeric_limits<f32>::min() * 2.0f;
    const auto rgba = Rgba({{subnormal, subnormal, subnormal, 0.0f},
                            {tinyNormal, tinyNormal, tinyNormal, 0.0f},
                            {tinyNormal, -tinyNormal, tinyNormal, 0.0f},
                            {1.0f, 1.0f, 1.0f, 0.0f}});
    ASSERT_EQ(std::fpclassify(CpuReference(rgba, 0.0f, 100.0f).min),
              FP_SUBNORMAL);
    auto image = UploadRgba32f(Device(), 4, 1, rgba);
    ASSERT_NE(image, nullptr);
    const auto computed = gpu->Compute(*image, 4, 1, 0.0f, 100.0f);
    ASSERT_FALSE(computed.has_value());
    EXPECT_NE(computed.error().find("CPU small-value arithmetic"),
              String::npos);
}

TEST_F(DisplayRangeGpuTest, RequestsCpuFallbackForExceptionalScale) {
    auto created = GpuDisplayRange::Create(Device());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());
    const f32 large = std::numeric_limits<f32>::max() * 0.75f;
    const auto rgba = Rgba({{-large, -large, -large, 0.0f},
                            {large, large, large, 0.0f}});
    auto image = UploadRgba32f(Device(), 2, 1, rgba);
    ASSERT_NE(image, nullptr);
    const auto computed = gpu->Compute(*image, 2, 1, 0.0f, 100.0f);
    ASSERT_FALSE(computed.has_value());
    EXPECT_NE(computed.error().find("CPU scale arithmetic"), String::npos);
}

TEST_F(DisplayRangeGpuTest, PreservesPercentileEdgesAndCollapsedWindowFallback) {
    auto created = GpuDisplayRange::Create(Device());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());
    std::vector<std::array<f32, 4>> pixels;
    for (int i = -100; i <= 100; ++i) {
        const f32 v = static_cast<f32>(i) * 0.25f;
        pixels.push_back({v, v * -0.5f, v * 3.0f, static_cast<f32>(i)});
    }
    const auto rgba = Rgba(pixels);
    constexpr std::array<std::pair<f32, f32>, 8> windows{{
        {-50.0f, 150.0f}, {0.0f, 100.0f}, {1.0f, 99.0f},
        {0.5f, 99.5f}, {50.0f, 50.0f}, {99.9f, 100.0f},
        {100.0f, 0.0f}, {12.345f, 87.654f},
    }};
    for (const auto [low, high] : windows)
        ExpectExactRange(*gpu, Device(), static_cast<u32>(pixels.size()), 1,
                         rgba, low, high);
}

TEST_F(DisplayRangeGpuTest, PreservesCpuRoundingWhenMaximumFallsInsideLastBin) {
    auto created = GpuDisplayRange::Create(Device());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());

    std::mt19937 random(0x45504e44u);
    std::uniform_real_distribution<f32> value(-1.0e6f, 1.0e6f);
    std::vector<f32> rgba;
    usize maxBin = GpuDisplayRange::kHistogramBins - 1;
    for (u32 attempt = 0; attempt < 100000u && rgba.empty(); ++attempt) {
        const std::array<f32, 3> a{value(random), value(random), value(random)};
        const std::array<f32, 3> b{value(random), value(random), value(random)};
        const f32 lumA = 0.2126f * a[0] + 0.7152f * a[1] + 0.0722f * a[2];
        const f32 lumB = 0.2126f * b[0] + 0.7152f * b[1] + 0.0722f * b[2];
        const f32 absMin = std::min(lumA, lumB);
        const f32 absMax = std::max(lumA, lumB);
        if (!(absMin < absMax)) continue;
        const f32 scale = static_cast<f32>(GpuDisplayRange::kHistogramBins - 1) /
                          (absMax - absMin);
        maxBin = static_cast<usize>((absMax - absMin) * scale);
        if (maxBin < GpuDisplayRange::kHistogramBins - 1)
            rgba = Rgba({{a[0], a[1], a[2], 0.0f},
                         {b[0], b[1], b[2], 0.0f}});
    }
    ASSERT_FALSE(rgba.empty());
    ASSERT_LT(maxBin, GpuDisplayRange::kHistogramBins - 1);
    ExpectExactRange(*gpu, Device(), 2, 1, rgba, 0.0f, 100.0f);
}

TEST_F(DisplayRangeGpuTest, MatchesCpuAtHistogramBoundariesAndFmaSensitiveInputs) {
    auto created = GpuDisplayRange::Create(Device());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());
    constexpr u32 width = 131;
    constexpr u32 height = 37;
    std::vector<f32> rgba(static_cast<usize>(width) * height * 4);
    std::mt19937 bits(0x52414e47u);
    std::uniform_real_distribution<f32> mantissa(-4096.0f, 4096.0f);
    for (usize pixel = 0; pixel < static_cast<usize>(width) * height; ++pixel) {
        f32 r = mantissa(bits);
        f32 g = mantissa(bits);
        f32 b = mantissa(bits);
        if (pixel % 17 == 0) {
            const f32 edge = -100.0f + 200.0f *
                static_cast<f32>(pixel % 65536u) / 65535.0f;
            r = std::nextafter(edge / 0.2126f,
                               pixel % 34 ? INFINITY : -INFINITY);
            g = 0.0f;
            b = 0.0f;
        }
        rgba[pixel * 4 + 0] = r;
        rgba[pixel * 4 + 1] = g;
        rgba[pixel * 4 + 2] = b;
        rgba[pixel * 4 + 3] = mantissa(bits);
    }
    constexpr std::array<std::pair<f32, f32>, 5> windows{{
        {0.0f, 100.0f}, {0.5f, 99.5f}, {1.0f, 99.0f},
        {17.0f, 83.0f}, {49.0f, 51.0f},
    }};
    for (const auto [low, high] : windows)
        ExpectExactRange(*gpu, Device(), width, height, rgba, low, high);
}

TEST_F(DisplayRangeGpuTest, MatchesCpuAcrossNontrivialRandomImages) {
    auto created = GpuDisplayRange::Create(Device());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());
    constexpr std::array<std::pair<u32, u32>, 6> extents{{
        {31, 19}, {64, 33}, {127, 17}, {257, 9}, {19, 113}, {73, 71},
    }};
    std::mt19937 random(0x44455601u);
    std::uniform_real_distribution<f32> value(-2000.0f, 8000.0f);
    for (usize imageIndex = 0; imageIndex < extents.size(); ++imageIndex) {
        const auto [width, height] = extents[imageIndex];
        std::vector<f32> rgba(static_cast<usize>(width) * height * 4);
        for (usize pixel = 0; pixel < static_cast<usize>(width) * height; ++pixel) {
            rgba[pixel * 4 + 0] = value(random);
            rgba[pixel * 4 + 1] = value(random) * 0.03125f;
            rgba[pixel * 4 + 2] = value(random) * 16.0f;
            rgba[pixel * 4 + 3] = value(random);
        }
        ExpectExactRange(*gpu, Device(), width, height, rgba,
                         0.1f + static_cast<f32>(imageIndex),
                         99.9f - static_cast<f32>(imageIndex));
    }
}

// Manual measurement: the full readback deliberately moves 12 MiB at 1K and
// 126.6 MiB at 4K, so this stays disabled in the normal eight-second suite.
// Run with --gtest_also_run_disabled_tests and this test's full name.
TEST_F(DisplayRangeGpuTest, DISABLED_BenchmarkGpuHistogramAgainstFullImageReadback) {
    auto created = GpuDisplayRange::Create(Device());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());

    enum class Distribution { Uniform, Peaked, Thermal };
    struct Case { u32 width; u32 height; Distribution distribution; const char* name; };
    constexpr std::array<Case, 6> cases{{
        {1024, 768, Distribution::Uniform, "1k uniform"},
        {1024, 768, Distribution::Peaked, "1k peaked"},
        {1024, 768, Distribution::Thermal, "1k thermal"},
        {3840, 2160, Distribution::Uniform, "4k uniform"},
        {3840, 2160, Distribution::Peaked, "4k peaked"},
        {3840, 2160, Distribution::Thermal, "4k thermal"},
    }};
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    const auto mean = [](const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) /
               static_cast<double>(values.size());
    };

    for (usize caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
        const Case& bench = cases[caseIndex];
        const usize pixelCount = static_cast<usize>(bench.width) * bench.height;
        std::vector<f32> rgba(pixelCount * 4, 1.0f);
        std::mt19937 random(0x42454e43u + static_cast<u32>(caseIndex));
        std::uniform_real_distribution<f32> uniform(-250.0f, 4000.0f);
        std::normal_distribution<f32> thermal(300.0f, 4.0f);
        for (usize pixel = 0; pixel < pixelCount; ++pixel) {
            if (bench.distribution == Distribution::Uniform) {
                rgba[pixel * 4 + 0] = uniform(random);
                rgba[pixel * 4 + 1] = uniform(random);
                rgba[pixel * 4 + 2] = uniform(random);
            } else if (bench.distribution == Distribution::Peaked) {
                const f32 spike = pixel % 4096 == 0
                    ? 100.0f + static_cast<f32>(pixel % 97) : 0.25f;
                rgba[pixel * 4 + 0] = spike;
                rgba[pixel * 4 + 1] = spike;
                rgba[pixel * 4 + 2] = spike;
            } else {
                const f32 temperature = thermal(random);
                rgba[pixel * 4 + 0] = temperature * 0.98f;
                rgba[pixel * 4 + 1] = temperature;
                rgba[pixel * 4 + 2] = temperature * 1.02f;
            }
        }
        auto image = UploadRgba32f(Device(), bench.width, bench.height, rgba);
        ASSERT_NE(image, nullptr);

        // Warm shader modules, pipelines, allocations and the fixed input.
        auto warmGpu = gpu->Compute(*image, bench.width, bench.height, 1.0f, 99.0f);
        ASSERT_TRUE(warmGpu.has_value()) << warmGpu.error();
        auto warmPixels = CommandHelper::ReadbackImage(
            Device(), image->GetImage(), VK_FORMAT_R32G32B32A32_SFLOAT,
            bench.width, bench.height);
        const DisplayRange warmCpu = CpuReference(warmPixels, 1.0f, 99.0f);
        ASSERT_EQ(warmGpu.value().min, warmCpu.min);
        ASSERT_EQ(warmGpu.value().max, warmCpu.max);

        std::vector<double> gpuMs;
        std::vector<double> cpuMs;
        for (u32 iteration = 0; iteration < 5; ++iteration) {
            const auto cpuStart = std::chrono::steady_clock::now();
            auto pixels = CommandHelper::ReadbackImage(
                Device(), image->GetImage(), VK_FORMAT_R32G32B32A32_SFLOAT,
                bench.width, bench.height);
            const DisplayRange cpu = CpuReference(pixels, 1.0f, 99.0f);
            const auto cpuEnd = std::chrono::steady_clock::now();

            const auto gpuStart = std::chrono::steady_clock::now();
            const auto computed = gpu->Compute(
                *image, bench.width, bench.height, 1.0f, 99.0f);
            const auto gpuEnd = std::chrono::steady_clock::now();
            ASSERT_TRUE(computed.has_value()) << computed.error();
            EXPECT_EQ(computed.value().min, cpu.min);
            EXPECT_EQ(computed.value().max, cpu.max);
            cpuMs.push_back(std::chrono::duration<double, std::milli>(
                                cpuEnd - cpuStart).count());
            gpuMs.push_back(std::chrono::duration<double, std::milli>(
                                gpuEnd - gpuStart).count());
        }
        const double oldBytes = static_cast<double>(pixelCount) * 4.0 * sizeof(f32);
        std::cout << "display-range benchmark " << bench.name
                  << ": old full-readback mean/median "
                  << mean(cpuMs) << "/" << median(cpuMs)
                  << " ms, GPU histogram mean/median "
                  << mean(gpuMs) << "/" << median(gpuMs)
                  << " ms, readback " << oldBytes / (1024.0 * 1024.0)
                  << " MiB -> "
                  << (GpuDisplayRange::kHistogramBins * sizeof(u32) + 4 * sizeof(u32)) /
                         1024.0
                  << " KiB\n";
    }
}
