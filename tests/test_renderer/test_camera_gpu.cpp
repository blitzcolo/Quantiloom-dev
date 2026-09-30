#include <gtest/gtest.h>

#include "postprocess/CpuCameraPipeline.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/ExternalRenderContext.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuCameraPipeline.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/VulkanContext.hpp"
#include "support/VulkanTestDevice.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#ifndef QUANTILOOM_SOURCE_ROOT
#define QUANTILOOM_SOURCE_ROOT "."
#endif

using namespace quantiloom;
using namespace quantiloom::camera;
using namespace quantiloom::rendercore;

namespace {

CameraConfig CameraConfigFor(u32 width, u32 height) {
    CameraConfig config;
    config.enabled = true;
    config.device.id = "gpu_operator_fixture";
    config.device.detector = DetectorKind::Photon;
    config.device.cfa = CfaPattern::Mono;
    ResponseCurve qe;
    qe.kind = ResponseKind::AbsoluteQE;
    qe.wavelengthNm = {500.0, 600.0};
    qe.value = {0.5, 0.5};
    ResponseStack stack;
    stack.quantumEfficiency = qe;
    config.device.channels.push_back({"Mono", stack});
    config.optics.sensorWidthPx = width;
    config.optics.sensorHeightPx = height;
    config.optics.pixelPitchUm = 5.0;
    config.optics.fNumber = 2.0;
    // Strict CPU/GPU operator checks start from the same measured image.
    // Their different optical PSF approximations are tested separately.
    config.optics.psfSigmaPixelsOverride = 0.0;
    config.readout.exposureSeconds = 0.01;
    config.readout.framePeriodSeconds = 0.1;
    config.readout.electronsPerDn = 1.0;
    config.readout.analogGain = 1.0;
    config.readout.adcBits = 16;
    config.readout.outputBits = 16;
    config.photon.fullWellElectrons = 50000.0;
    config.quality.backend = ProcessingBackend::GpuPreview;
    config.quality.noiseFree = true;
    config.products.bandMeasurement = true;
    config.products.rawDn = true;
    config.products.correctedDeviceSignal = true;
    config.products.display = true;
    return config;
}

CameraConfig ThermalConfigFor(u32 width, u32 height) {
    auto config = CameraConfigFor(width, height);
    config.device.id = "gpu_thermal_fixture";
    config.device.detector = DetectorKind::Thermal;
    config.device.channels.clear();
    ResponseCurve absorptance;
    absorptance.kind = ResponseKind::ThermalAbsorptance;
    absorptance.wavelengthNm = {8000.0, 14000.0};
    absorptance.value = {1.0, 1.0};
    ResponseStack response;
    response.thermalAbsorptance = absorptance;
    config.device.channels.push_back({"Mono", response});
    config.optics.pixelPitchUm = 10.0;
    config.thermal.timeConstantSeconds = 0.008;
    config.thermal.responsivityDnPerWatt = 1e13;
    return config;
}

// Bayer RGB camera over the photon fixture's single QE curve: channel
// separation comes from the CFA sampling, exactly as in the M4-3 ISP tests.
CameraConfig BayerColorConfigFor(u32 width, u32 height) {
    auto config = CameraConfigFor(width, height);
    config.device.cfa = CfaPattern::RGGB;
    const auto baseChannel = config.device.channels.front();
    config.device.channels = {baseChannel, baseChannel, baseChannel};
    config.device.channels[0].name = "R";
    config.device.channels[1].name = "G";
    config.device.channels[2].name = "B";
    return config;
}

// A Bayer measurement image: the latent R/G/B rates ride in rgb, alpha
// records the CFA response identity of the pixel, as the readout pass
// expects. The scene itself is achromatic; display chroma comes from the
// configured white balance, so the CPU reference (which models the scene as
// one scalar rate) sees exactly the same image.
void FillBayerMeasurement(u32 width, u32 height, std::vector<f32>& rgba,
                          f32 baseRate = 3000.0f) {
    rgba.assign(static_cast<size_t>(width) * height * 4u, 0.0f);
    for (u32 y = 0; y < height; ++y)
        for (u32 x = 0; x < width; ++x) {
            const u32 i = y * width + x;
            const f32 rate = baseRate + 40.0f * x + 25.0f * y;
            rgba[i * 4 + 0] = rate;
            rgba[i * 4 + 1] = rate;
            rgba[i * 4 + 2] = rate;
            rgba[i * 4 + 3] = static_cast<f32>(
                (y % 2 == 0) ? (x % 2 == 0 ? 0u : 1u)
                             : (x % 2 == 0 ? 1u : 2u));
        }
}

std::unique_ptr<GpuImage> UploadMeasurement(
    VulkanContext& context, u32 width, u32 height,
    const std::vector<f32>& rgba) {
    const VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    auto image = std::make_unique<GpuImage>(
        context.GetAllocator(), context.GetDevice(), width, height, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    if (!image->IsValid() || rgba.size() != static_cast<size_t>(width) * height * 4)
        return {};
    const VkDeviceSize bytes = rgba.size() * sizeof(f32);
    GpuBuffer staging(context.GetAllocator(), bytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);
    if (!staging.IsValid()) return {};
    staging.Upload(rgba.data(), bytes);
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        CommandHelper::TransitionImageLayout(
            cmd, image->GetImage(), format, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, staging.GetHandle(), image->GetImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier ready{};
        ready.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
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

bool UpdateMeasurement(VulkanContext& context, GpuImage& image,
                       u32 width, u32 height,
                       const std::vector<f32>& rgba) {
    if (!image.IsValid() || image.GetExtent().width != width ||
        image.GetExtent().height != height ||
        rgba.size() != static_cast<size_t>(width) * height * 4)
        return false;
    const VkDeviceSize bytes = rgba.size() * sizeof(f32);
    GpuBuffer staging(context.GetAllocator(), bytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);
    if (!staging.IsValid()) return false;
    staging.Upload(rgba.data(), bytes);
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier toDst{};
        toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toDst.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDst.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image = image.GetImage();
        toDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toDst.subresourceRange.levelCount = 1;
        toDst.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toDst);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, staging.GetHandle(), image.GetImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier ready{};
        ready.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ready.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ready.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.image = image.GetImage();
        ready.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ready.subresourceRange.levelCount = 1;
        ready.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &ready);
    });
    return true;
}

Result<void, String> RecordCamera(VulkanContext& context,
                                  GpuCameraPipeline& pipeline,
                                  const GpuImage& measurement,
                                  u64 acquisitionIndex,
                                  f64 timeSeconds) {
    Result<void, String> status = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        status = pipeline.RecordMeasurement(
            cmd, measurement, acquisitionIndex, timeSeconds);
    });
    return status;
}

std::vector<f32> ReadFloat(VulkanContext& context, const GpuImage* image,
                           u32 width, u32 height) {
    if (!image) return {};
    return CommandHelper::ReadbackImage(
        context, image->GetImage(), image->GetFormat(), width, height);
}

std::vector<u32> ReadUint(VulkanContext& context, const GpuImage* image,
                          u32 width, u32 height) {
    if (!image || image->GetFormat() != VK_FORMAT_R32G32B32A32_UINT) return {};
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(u32);
    GpuBuffer staging(context.GetAllocator(), bytes,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    if (!staging.IsValid()) return {};
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        CommandHelper::TransitionImageLayout(
            cmd, image->GetImage(), image->GetFormat(),
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, image->GetImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.GetHandle(), 1, &region);
        CommandHelper::TransitionImageLayout(
            cmd, image->GetImage(), image->GetFormat(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    });
    std::vector<u32> out(static_cast<size_t>(width) * height * 4);
    const void* data = staging.Map();
    if (!data) return {};
    std::memcpy(out.data(), data, bytes);
    staging.Unmap();
    return out;
}

class CameraGpuTest : public quantiloom::testing::VulkanDeviceTest {};

std::pair<double, double> RedMeanVariance(const std::vector<f32>& rgba) {
    const size_t pixels = rgba.size() / 4;
    double sum = 0.0;
    for (size_t i = 0; i < pixels; ++i) sum += rgba[i * 4];
    const double mean = sum / pixels;
    double squared = 0.0;
    for (size_t i = 0; i < pixels; ++i)
        squared += (rgba[i * 4] - mean) * (rgba[i * 4] - mean);
    return {mean, squared / pixels};
}

} // namespace

TEST_F(CameraGpuTest, NoiseFreePhotonOperatorMatchesCpuBeforeAndAfterAdc) {
    const auto config = CameraConfigFor(6, 1);
    Image cpuRate(6, 1, 1);
    cpuRate.data = {0.0f, 49.999f, 50.001f, 1000.0f, 12345.0f, 1e7f};
    std::vector<f32> rgba(6 * 4, 0.0f);
    for (u32 i = 0; i < 6; ++i) rgba[i * 4] = cpuRate.data[i];
    auto input = UploadMeasurement(Device(), 6, 1, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 6, 1);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    const auto configured = pipeline->Configure(config);
    ASSERT_TRUE(configured.has_value());
    const auto recorded = RecordCamera(Device(), *pipeline, *input, 0, 0.0);
    ASSERT_TRUE(recorded.has_value());
    const auto outputs = pipeline->GetOutputs();
    const auto gpuExpected = ReadFloat(Device(), outputs.expectedElectrons, 6, 1);
    const auto gpuPreAdc = ReadFloat(Device(), outputs.preAdcElectrons, 6, 1);
    const auto gpuRaw = ReadFloat(Device(), outputs.rawDn, 6, 1);
    const auto gpuCorrected = ReadFloat(Device(), outputs.corrected, 6, 1);
    ASSERT_EQ(gpuExpected.size(), 24u);
    ASSERT_EQ(gpuPreAdc.size(), 24u);
    ASSERT_EQ(gpuRaw.size(), 24u);
    ASSERT_EQ(gpuCorrected.size(), 24u);

    CaptureState cpuState;
    const auto cpu = CpuCameraPipeline(config).CaptureMeasured(cpuState, 0.0, cpuRate);
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(cpu.value().rawDn.has_value());
    ASSERT_TRUE(cpu.value().correctedDeviceSignal.has_value());
    for (u32 pixel = 0; pixel < 6; ++pixel) {
        const double expected = static_cast<double>(cpuRate.data[pixel]) *
                                config.readout.exposureSeconds;
        const double clamped = std::min(expected, config.photon.fullWellElectrons);
        const double tolerance = std::max(0.001, 1e-5 * expected);
        EXPECT_NEAR(gpuExpected[pixel * 4], expected, tolerance)
            << "pixel " << pixel;
        EXPECT_NEAR(gpuPreAdc[pixel * 4], clamped,
                    std::max(0.001, 1e-5 * clamped))
            << "pixel " << pixel;
        EXPECT_NEAR(gpuRaw[pixel * 4], cpu.value().rawDn->image.data[pixel], 1.0)
            << "pixel " << pixel;
        EXPECT_NEAR(gpuCorrected[pixel * 4],
                    cpu.value().correctedDeviceSignal->image.data[pixel], 1.0)
            << "pixel " << pixel;
    }
    auto replacementView = UploadMeasurement(Device(), 6, 1, rgba);
    ASSERT_NE(replacementView, nullptr);
    const auto rejected = RecordCamera(
        Device(), *pipeline, *replacementView, 1, 0.1);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_NE(rejected.error().find("changed"), String::npos);
    // The rejected bind must leave the capture counter and thermal history
    // untouched, so the original image can still serve acquisition 1.
    EXPECT_TRUE(RecordCamera(Device(), *pipeline, *input, 1, 0.1).has_value());
}

TEST_F(CameraGpuTest, FastRgbApproximationUsesSharedReadoutAndAbsoluteScale) {
    auto config = CameraConfigFor(2, 1);
    config.inputKind = CameraInputKind::FastRgbApproximation;
    config.fastRgbRadianceScale = 2e-4;
    config.calibratedFastRgbInput = true;
    config.fastRgbToDevice = {0.5, 0.25, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0};
    std::vector<f32> rgba = {0.2f, 0.4f, 0.1f, 1.0f,
                             0.8f, 0.1f, 0.3f, 1.0f};
    auto input = UploadMeasurement(Device(), 2, 1, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 2, 1);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    Result<void, String> recorded = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        recorded = pipeline->RecordFastRgbMeasurement(cmd, *input, 0, 0.0);
    });
    ASSERT_TRUE(recorded.has_value()) << recorded.error();
    const auto gpuExpected = ReadFloat(
        Device(), pipeline->GetOutputs().expectedElectrons, 2, 1);
    const auto gpuRaw = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 2, 1);
    ASSERT_EQ(gpuExpected.size(), 8u);
    ASSERT_EQ(gpuRaw.size(), 8u);

    Image rgb(2, 1, 3);
    rgb.data = {0.2f, 0.4f, 0.1f, 0.8f, 0.1f, 0.3f};
    CaptureState state;
    const auto cpu = CpuCameraPipeline(config).CaptureFastRgb(state, 0.0, rgb);
    ASSERT_TRUE(cpu.has_value()) << cpu.error();
    ASSERT_TRUE(cpu.value().rawDn.has_value());
    ASSERT_TRUE(cpu.value().bandMeasurement.has_value());
    // expectedElectrons is after exposure; compare against the CPU's device
    // rate before pinning the common ADC result to 1 DN.
    for (u32 pixel = 0; pixel < 2; ++pixel) {
        const double gpu = gpuExpected[pixel * 4u];
        const double cpuExpected =
            cpu.value().bandMeasurement->image.data[pixel] *
            config.readout.exposureSeconds;
        const double cpuRaw = cpu.value().rawDn->image.data[pixel];
        EXPECT_NEAR(gpu, cpuExpected,
                    std::max(0.001, 1e-5 * cpuExpected)) << "pixel " << pixel;
        EXPECT_NEAR(gpuRaw[pixel * 4u], cpuRaw, 1.0) << "pixel " << pixel;
    }
}

TEST_F(CameraGpuTest, IntegerCounterVectorsMatchIndependentGoldens) {
    auto created = GpuCameraPipeline::Create(Device(), 64, 64);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    const auto configured = pipeline->Configure(CameraConfigFor(64, 64));
    ASSERT_TRUE(configured.has_value());
    Result<void, String> status = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        status = pipeline->RecordRandomVectors(
            cmd, 0x548cu, 7u, NoiseClass::PhotonShot, 0u);
    });
    ASSERT_TRUE(status.has_value());
    const auto first = ReadUint(Device(), pipeline->GetRandomVectors(), 64, 64);
    ASSERT_EQ(first.size(), 64u * 64u * 4u);
    const size_t base = 1234u * 4u;
    EXPECT_EQ(first[base + 0], 0xf3472877u);
    EXPECT_EQ(first[base + 1], 0x7b3e1d5fu);
    EXPECT_EQ(first[base + 2], 0x349a82a3u);
    EXPECT_EQ(first[base + 3], 0xe0588ed0u);

    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        status = pipeline->RecordRandomVectors(
            cmd, 0x548cu, 0x100000007ull, NoiseClass::PhotonShot, 0u);
    });
    ASSERT_TRUE(status.has_value());
    const auto highCapture = ReadUint(Device(), pipeline->GetRandomVectors(), 64, 64);
    ASSERT_EQ(highCapture.size(), first.size());
    EXPECT_EQ(highCapture[base], 0x0877f273u);

    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        status = pipeline->RecordRandomVectors(
            cmd, 0x548cu, 7u, NoiseClass::FixedPrnu, 0u);
    });
    ASSERT_TRUE(status.has_value());
    const auto fixedFirst = ReadUint(Device(), pipeline->GetRandomVectors(), 64, 64);
    ASSERT_EQ(fixedFirst.size(), first.size());
    EXPECT_EQ(fixedFirst[base], 0x3c64784eu);
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        status = pipeline->RecordRandomVectors(
            cmd, 0x548cu, 0x1000003e7ull, NoiseClass::FixedPrnu, 0u);
    });
    ASSERT_TRUE(status.has_value());
    const auto fixedLater = ReadUint(Device(), pipeline->GetRandomVectors(), 64, 64);
    ASSERT_EQ(fixedLater.size(), first.size());
    EXPECT_EQ(fixedLater[base], fixedFirst[base]);
}

TEST_F(CameraGpuTest, BayerMeasuredRateProducesOneRawNumberPerPixel) {
    auto config = CameraConfigFor(2, 2);
    config.device.cfa = CfaPattern::RGGB;
    const auto baseChannel = config.device.channels.front();
    config.device.channels = {baseChannel, baseChannel, baseChannel};
    config.device.channels[0].name = "R";
    config.device.channels[1].name = "G";
    config.device.channels[2].name = "B";
    Image cpuRate(2, 2, 1);
    cpuRate.data = {5000.0f, 10000.0f, 20000.0f, 40000.0f};
    std::vector<f32> rgba(2 * 2 * 4, 0.0f);
    const std::array<f32, 4> cfa = {0.0f, 1.0f, 1.0f, 2.0f};
    for (u32 i = 0; i < 4; ++i) {
        // Bayer measurement input carries three latent response fields;
        // alpha chooses the one physical filter read by this pixel.
        for (u32 channel = 0; channel < 3; ++channel)
            rgba[i * 4 + channel] = cpuRate.data[i];
        rgba[i * 4 + 3] = cfa[i];
    }
    auto input = UploadMeasurement(Device(), 2, 2, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 2, 2);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    const auto gpuRaw = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 2, 2);
    ASSERT_EQ(gpuRaw.size(), 16u);
    CaptureState state;
    const auto cpu = CpuCameraPipeline(config).CaptureMeasured(state, 0.0, cpuRate);
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(cpu.value().rawDn.has_value());
    EXPECT_EQ(cpu.value().rawDn->image.channels, 1u);
    for (u32 i = 0; i < 4; ++i)
        EXPECT_NEAR(gpuRaw[i * 4], cpu.value().rawDn->image.data[i], 1.0)
            << "Bayer pixel " << i;
}

TEST_F(CameraGpuTest, BayerBlursEachLatentChannelBeforeChoosingItsCfaFilter) {
    auto bayer = CameraConfigFor(5, 5);
    bayer.device.cfa = CfaPattern::RGGB;
    const auto baseChannel = bayer.device.channels.front();
    bayer.device.channels = {baseChannel, baseChannel, baseChannel};
    bayer.device.channels[0].name = "R";
    bayer.device.channels[1].name = "G";
    bayer.device.channels[2].name = "B";
    bayer.optics.psfSigmaPixelsOverride = 1.0;
    std::vector<f32> bayerRgba(5 * 5 * 4, 0.0f);
    for (u32 y = 0; y < 5; ++y)
        for (u32 x = 0; x < 5; ++x) {
            const u32 pixel = y * 5 + x;
            const u32 cfa = (y % 2 == 0) ? (x % 2 == 0 ? 0u : 1u)
                                         : (x % 2 == 0 ? 1u : 2u);
            bayerRgba[pixel * 4 + 3] = static_cast<f32>(cfa);
        }
    // The center is a red physical pixel, but its latent G/B response
    // fields are also illuminated. After optical blur they reach nearby
    // green and blue physical pixels.
    for (u32 channel = 0; channel < 3; ++channel)
        bayerRgba[(2 * 5 + 2) * 4 + channel] = 100000.0f;
    auto bayerInput = UploadMeasurement(Device(), 5, 5, bayerRgba);
    ASSERT_NE(bayerInput, nullptr);
    auto bayerCreated = GpuCameraPipeline::Create(Device(), 5, 5);
    ASSERT_TRUE(bayerCreated.has_value());
    auto bayerGpu = std::move(bayerCreated.value());
    ASSERT_TRUE(bayerGpu->Configure(bayer).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *bayerGpu, *bayerInput, 0, 0.0).has_value());
    const auto bayerExpected =
        ReadFloat(Device(), bayerGpu->GetOutputs().expectedElectrons, 5, 5);
    ASSERT_EQ(bayerExpected.size(), 5u * 5u * 4u);
    EXPECT_GT(bayerExpected[(2 * 5 + 3) * 4], 1.0f); // G neighbor.
    EXPECT_GT(bayerExpected[(3 * 5 + 3) * 4], 1.0f); // B diagonal.
    EXPECT_LT(bayerExpected[(2 * 5 + 2) * 4], 1000.0f);

    auto mono = CameraConfigFor(5, 5);
    mono.optics.psfSigmaPixelsOverride = 1.0;
    std::vector<f32> monoRgba(5 * 5 * 4, 0.0f);
    monoRgba[(2 * 5 + 2) * 4] = 100000.0f;
    auto monoInput = UploadMeasurement(Device(), 5, 5, monoRgba);
    ASSERT_NE(monoInput, nullptr);
    auto monoCreated = GpuCameraPipeline::Create(Device(), 5, 5);
    ASSERT_TRUE(monoCreated.has_value());
    auto monoGpu = std::move(monoCreated.value());
    ASSERT_TRUE(monoGpu->Configure(mono).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *monoGpu, *monoInput, 0, 0.0).has_value());
    const auto monoExpected =
        ReadFloat(Device(), monoGpu->GetOutputs().expectedElectrons, 5, 5);
    ASSERT_EQ(monoExpected.size(), bayerExpected.size());
    for (size_t pixel = 0; pixel < 25; ++pixel)
        EXPECT_NEAR(bayerExpected[pixel * 4], monoExpected[pixel * 4],
                    std::max(0.001f, monoExpected[pixel * 4] * 1e-5f))
            << "pixel " << pixel;
}

TEST_F(CameraGpuTest, ThreeDeviceChannelsHaveIndependentRawComponents) {
    auto config = CameraConfigFor(2, 1);
    config.device.cfa = CfaPattern::MultiChannel;
    const auto baseChannel = config.device.channels.front();
    config.device.channels = {baseChannel, baseChannel, baseChannel};
    config.device.channels[0].name = "R";
    config.device.channels[1].name = "G";
    config.device.channels[2].name = "B";
    Image cpuRate(2, 1, 3);
    cpuRate.data = {1000.0f, 2000.0f, 3000.0f,
                    5000.0f, 7000.0f, 11000.0f};
    std::vector<f32> rgba = {1000.0f, 2000.0f, 3000.0f, 0.0f,
                             5000.0f, 7000.0f, 11000.0f, 0.0f};
    auto input = UploadMeasurement(Device(), 2, 1, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 2, 1);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    const auto gpuRaw = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 2, 1);
    const auto gpuExpected =
        ReadFloat(Device(), pipeline->GetOutputs().expectedElectrons, 2, 1);
    ASSERT_EQ(gpuRaw.size(), 8u);
    ASSERT_EQ(gpuExpected.size(), 8u);
    CaptureState state;
    const auto cpu = CpuCameraPipeline(config).CaptureMeasured(state, 0.0, cpuRate);
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(cpu.value().rawDn.has_value());
    ASSERT_EQ(cpu.value().rawDn->image.channels, 3u);
    for (u32 pixel = 0; pixel < 2; ++pixel)
        for (u32 channel = 0; channel < 3; ++channel) {
            const u32 index = pixel * 3 + channel;
            EXPECT_NEAR(gpuExpected[pixel * 4 + channel],
                        cpuRate.data[index] * config.readout.exposureSeconds,
                        0.001);
            EXPECT_NEAR(gpuRaw[pixel * 4 + channel],
                        cpu.value().rawDn->image.data[index], 1.0);
        }
}

TEST_F(CameraGpuTest, MultiChannelNoiseKeysSeparateChannelsAndFixPatternOverTime) {
    auto config = CameraConfigFor(64, 64);
    config.device.cfa = CfaPattern::MultiChannel;
    const auto baseChannel = config.device.channels.front();
    config.device.channels = {baseChannel, baseChannel, baseChannel};
    config.device.channels[0].name = "R";
    config.device.channels[1].name = "G";
    config.device.channels[2].name = "B";
    config.quality.noiseFree = false;
    std::vector<f32> rgba(64u * 64u * 4u, 0.0f);
    for (size_t pixel = 0; pixel < 64u * 64u; ++pixel)
        for (u32 channel = 0; channel < 3; ++channel)
            rgba[pixel * 4 + channel] = 10000.0f; // 100 e- each.
    auto input = UploadMeasurement(Device(), 64, 64, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 64, 64);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    const auto shot = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 64, 64);
    ASSERT_EQ(shot.size(), 64u * 64u * 4u);
    size_t sameShot = 0;
    for (size_t pixel = 0; pixel < 64u * 64u; ++pixel)
        if (shot[pixel * 4] == shot[pixel * 4 + 1]) ++sameShot;
    EXPECT_LT(static_cast<double>(sameShot) / (64.0 * 64.0), 0.08);

    config.photon.enableShotNoise = false;
    config.photon.enableDarkShotNoise = false;
    config.photon.enableReadNoise = false;
    config.photon.enableFpn = true;
    config.photon.prnuSigma = 0.05;
    std::vector<f32> brightRgba = rgba;
    for (size_t pixel = 0; pixel < 64u * 64u; ++pixel)
        for (u32 channel = 0; channel < 3; ++channel)
            brightRgba[pixel * 4 + channel] = 100000.0f; // 1000 e-.
    auto brightInput = UploadMeasurement(Device(), 64, 64, brightRgba);
    ASSERT_NE(brightInput, nullptr);
    auto fixedCreated = GpuCameraPipeline::Create(Device(), 64, 64);
    ASSERT_TRUE(fixedCreated.has_value());
    auto fixed = std::move(fixedCreated.value());
    ASSERT_TRUE(fixed->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *fixed, *brightInput, 0, 0.0).has_value());
    const auto first = ReadFloat(Device(), fixed->GetOutputs().rawDn, 64, 64);
    ASSERT_EQ(first.size(), shot.size());
    ASSERT_TRUE(RecordCamera(Device(), *fixed, *brightInput, 1, 0.1).has_value());
    const auto second = ReadFloat(Device(), fixed->GetOutputs().rawDn, 64, 64);
    ASSERT_EQ(second.size(), first.size());
    size_t differentChannels = 0;
    for (size_t pixel = 0; pixel < 64u * 64u; ++pixel) {
        for (u32 channel = 0; channel < 3; ++channel)
            EXPECT_FLOAT_EQ(first[pixel * 4 + channel],
                            second[pixel * 4 + channel]);
        if (first[pixel * 4] != first[pixel * 4 + 1])
            ++differentChannels;
    }
    EXPECT_GT(differentChannels, 3u * 64u * 64u / 4u);
}

TEST_F(CameraGpuTest, GpuShotNoiseHasExpectedMeanVarianceAndLowCountZeroMass) {
    auto config = CameraConfigFor(64, 64);
    config.quality.noiseFree = false;
    const auto run = [&](f32 rate, u64 capture) {
        std::vector<f32> rgba(64 * 64 * 4, 0.0f);
        for (size_t i = 0; i < 64u * 64u; ++i) rgba[i * 4] = rate;
        auto input = UploadMeasurement(Device(), 64, 64, rgba);
        EXPECT_NE(input, nullptr);
        auto created = GpuCameraPipeline::Create(Device(), 64, 64);
        EXPECT_TRUE(created.has_value());
        if (!input || !created) return std::vector<f32>{};
        auto pipeline = std::move(created.value());
        const auto configured = pipeline->Configure(config);
        EXPECT_TRUE(configured.has_value());
        if (!configured) return std::vector<f32>{};
        const auto recorded = RecordCamera(
            Device(), *pipeline, *input, capture, 0.0);
        EXPECT_TRUE(recorded.has_value());
        if (!recorded) return std::vector<f32>{};
        return ReadFloat(Device(), pipeline->GetOutputs().rawDn, 64, 64);
    };
    const auto bright = run(10000.0f, 0);
    ASSERT_EQ(bright.size(), 64u * 64u * 4u);
    const auto [mean, variance] = RedMeanVariance(bright);
    EXPECT_NEAR(mean, 100.0, 1.0);
    EXPECT_NEAR(variance, 100.0, 10.0);
    const auto low = run(100.0f, 1);
    ASSERT_EQ(low.size(), bright.size());
    size_t zeros = 0;
    for (size_t pixel = 0; pixel < 64u * 64u; ++pixel)
        if (low[pixel * 4] == 0.0f) ++zeros;
    EXPECT_NEAR(static_cast<double>(zeros) / (64.0 * 64.0),
                std::exp(-1.0), 0.03);
}

TEST_F(CameraGpuTest, PoissonHandlesNearlyZeroMeanAndItsExactNormalBoundary) {
    auto config = CameraConfigFor(64, 64);
    config.quality.noiseFree = false;
    const auto run = [&](f32 rate, u64 capture) {
        std::vector<f32> rgba(64u * 64u * 4u, 0.0f);
        for (size_t pixel = 0; pixel < 64u * 64u; ++pixel)
            rgba[pixel * 4] = rate;
        auto input = UploadMeasurement(Device(), 64, 64, rgba);
        EXPECT_NE(input, nullptr);
        auto created = GpuCameraPipeline::Create(Device(), 64, 64);
        EXPECT_TRUE(created.has_value());
        if (!input || !created) return std::vector<f32>{};
        auto pipeline = std::move(created.value());
        const auto configured = pipeline->Configure(config);
        EXPECT_TRUE(configured.has_value());
        if (!configured) return std::vector<f32>{};
        const auto recorded = RecordCamera(
            Device(), *pipeline, *input, capture, 0.0);
        EXPECT_TRUE(recorded.has_value());
        if (!recorded) return std::vector<f32>{};
        return ReadFloat(Device(), pipeline->GetOutputs().rawDn, 64, 64);
    };
    const auto dark = run(0.0f, 0);
    const auto nearlyDark = run(1e-8f, 1); // mu = 1e-10 e-.
    ASSERT_EQ(dark.size(), 64u * 64u * 4u);
    ASSERT_EQ(nearlyDark.size(), dark.size());
    for (size_t pixel = 0; pixel < 64u * 64u; ++pixel) {
        EXPECT_FLOAT_EQ(dark[pixel * 4], 0.0f);
        EXPECT_FLOAT_EQ(nearlyDark[pixel * 4], 0.0f);
    }
    // The implementation switches from exact inversion to the accepted
    // high-count normal approximation at 32 expected electrons.
    for (const auto& [mean, capture] :
         std::array<std::pair<double, u64>, 2>{{{31.9, 2u}, {32.1, 3u}}}) {
        const auto samples = run(static_cast<f32>(mean / 0.01), capture);
        ASSERT_EQ(samples.size(), dark.size());
        const auto [observedMean, observedVariance] = RedMeanVariance(samples);
        EXPECT_NEAR(observedMean, mean, 0.8);
        EXPECT_NEAR(observedVariance, mean, 5.0);
    }
}

TEST_F(CameraGpuTest, HighCountNormalApproximationMatchesPoissonMoments) {
    auto config = CameraConfigFor(256, 256);
    config.quality.noiseFree = false;
    std::vector<f32> rgba(256u * 256u * 4u, 0.0f);
    for (size_t pixel = 0; pixel < 256u * 256u; ++pixel)
        rgba[pixel * 4] = 1e6f; // exposure 0.01 s -> 10000 expected e-.
    auto input = UploadMeasurement(Device(), 256, 256, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 256, 256);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    const auto raw = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 256, 256);
    ASSERT_EQ(raw.size(), 256u * 256u * 4u);
    const auto [mean, variance] = RedMeanVariance(raw);
    EXPECT_NEAR(mean, 10000.0, 3.0);
    EXPECT_NEAR(variance, 10000.0, 250.0);
}

TEST_F(CameraGpuTest, FixedPatternIsStableAndIndependentCalibrationActsOnCorrected) {
    auto config = CameraConfigFor(64, 64);
    config.quality.noiseFree = false;
    config.photon.enableShotNoise = false;
    config.photon.enableDarkShotNoise = false;
    config.photon.enableReadNoise = false;
    config.photon.enableFpn = true;
    config.photon.prnuSigma = 0.05;
    std::vector<f32> rgba(64 * 64 * 4, 0.0f);
    for (size_t i = 0; i < 64u * 64u; ++i) rgba[i * 4] = 100000.0f;
    auto input = UploadMeasurement(Device(), 64, 64, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 64, 64);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    const auto first = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 64, 64);
    ASSERT_EQ(first.size(), 64u * 64u * 4u);
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 1, 0.1).has_value());
    const auto second = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 64, 64);
    ASSERT_EQ(second.size(), first.size());
    for (size_t pixel = 0; pixel < 64u * 64u; ++pixel)
        EXPECT_FLOAT_EQ(first[pixel * 4], second[pixel * 4]);
    EXPECT_GT(RedMeanVariance(first).second, 100.0);

    auto calibrated = CameraConfigFor(2, 1);
    calibrated.photon.applyNuc = true;
    calibrated.photon.nucGainMap = {1.0, 0.5};
    calibrated.photon.nucOffsetElectronsMap = {0.0, 10.0};
    std::vector<f32> calibratedRgba = {10000.0f, 0.0f, 0.0f, 0.0f,
                                        10000.0f, 0.0f, 0.0f, 0.0f};
    auto calibratedInput = UploadMeasurement(Device(), 2, 1, calibratedRgba);
    ASSERT_NE(calibratedInput, nullptr);
    auto calibratedCreated = GpuCameraPipeline::Create(Device(), 2, 1);
    ASSERT_TRUE(calibratedCreated.has_value());
    auto calibratedGpu = std::move(calibratedCreated.value());
    ASSERT_TRUE(calibratedGpu->Configure(calibrated).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *calibratedGpu,
                             *calibratedInput, 0, 0.0).has_value());
    const auto gpuRaw = ReadFloat(Device(), calibratedGpu->GetOutputs().rawDn, 2, 1);
    const auto gpuCorrected = ReadFloat(
        Device(), calibratedGpu->GetOutputs().corrected, 2, 1);
    ASSERT_EQ(gpuRaw.size(), 8u);
    ASSERT_EQ(gpuCorrected.size(), 8u);
    Image cpuRate(2, 1, 1);
    cpuRate.data = {10000.0f, 10000.0f};
    CaptureState state;
    const auto cpu = CpuCameraPipeline(calibrated).CaptureMeasured(
        state, 0.0, cpuRate);
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(cpu.value().correctedDeviceSignal.has_value());
    for (u32 pixel = 0; pixel < 2; ++pixel) {
        EXPECT_NEAR(gpuRaw[pixel * 4], cpu.value().rawDn->image.data[pixel], 1.0);
        EXPECT_NEAR(gpuCorrected[pixel * 4],
                    cpu.value().correctedDeviceSignal->image.data[pixel], 1.0);
    }
    EXPECT_NEAR(gpuCorrected[0], 100.0, 1.0);
    EXPECT_NEAR(gpuCorrected[4], 60.0, 1.0);
}

TEST_F(CameraGpuTest, ThermalHistoryAndNetdBandwidthMatchCpuAndStatistics) {
    auto config = ThermalConfigFor(1, 1);
    auto created = GpuCameraPipeline::Create(Device(), 1, 1);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    auto input = UploadMeasurement(Device(), 1, 1,
                                   {0.0f, 0.0f, 0.0f, 0.0f});
    const f32 power = 1.0e-9f;
    ASSERT_NE(input, nullptr);
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    ASSERT_TRUE(UpdateMeasurement(Device(), *input, 1, 1,
                                  {power, 0.0f, 0.0f, 0.0f}));
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 1, 0.008).has_value());
    const auto gpuRaw = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 1, 1);
    ASSERT_EQ(gpuRaw.size(), 4u);
    Image darkRate(1, 1, 1);
    Image warmRate(1, 1, 1);
    warmRate.data[0] = power;
    CpuCameraPipeline cpu(config);
    CaptureState state;
    ASSERT_TRUE(cpu.CaptureMeasured(state, 0.0, darkRate).has_value());
    const auto expected = cpu.CaptureMeasured(state, 0.008, warmRate);
    ASSERT_TRUE(expected.has_value());
    EXPECT_NEAR(gpuRaw[0], expected.value().rawDn->image.data[0], 1.0);
    EXPECT_NEAR(gpuRaw[0],
                power * (1.0 - std::exp(-1.0)) *
                    config.thermal.responsivityDnPerWatt,
                1.0);

    auto noisy = ThermalConfigFor(64, 64);
    noisy.quality.noiseFree = false;
    noisy.thermal.netdKelvin = 0.04;
    noisy.thermal.netdReferenceTemperatureK = 300.0;
    noisy.thermal.netdNoiseBandwidthHz = 100.0;
    noisy.thermal.netdOpticalCondition = "f/2 reference lens";
    noisy.thermal.readoutWindowSeconds = 0.005; // 100 Hz effective bandwidth.
    const auto run = [&](const CameraConfig& settings) {
        std::vector<f32> rgba(64u * 64u * 4u, 0.0f);
        for (size_t i = 0; i < 64u * 64u; ++i)
            rgba[i * 4] = 1.0151679938e-9f; // 300 K blackbody through 8-14 um.
        auto input = UploadMeasurement(Device(), 64, 64, rgba);
        EXPECT_NE(input, nullptr);
        auto gpuCreated = GpuCameraPipeline::Create(Device(), 64, 64);
        EXPECT_TRUE(gpuCreated.has_value());
        if (!input || !gpuCreated) return std::vector<f32>{};
        auto gpu = std::move(gpuCreated.value());
        const auto configured = gpu->Configure(settings);
        EXPECT_TRUE(configured.has_value());
        if (!configured) return std::vector<f32>{};
        const auto recorded = RecordCamera(Device(), *gpu, *input, 0, 0.0);
        EXPECT_TRUE(recorded.has_value());
        if (!recorded) return std::vector<f32>{};
        return ReadFloat(Device(), gpu->GetOutputs().rawDn, 64, 64);
    };
    const auto reference = run(noisy);
    ASSERT_EQ(reference.size(), 64u * 64u * 4u);
    EXPECT_NEAR(RedMeanVariance(reference).second,
                6.19316049 * 6.19316049, 5.0);
    noisy.thermal.readoutWindowSeconds = 0.02; // 25 Hz: quarter variance.
    const auto slow = run(noisy);
    ASSERT_EQ(slow.size(), reference.size());
    EXPECT_NEAR(RedMeanVariance(slow).second /
                    RedMeanVariance(reference).second,
                0.25, 0.08);
}

TEST_F(CameraGpuTest, SameAcquisitionRecomputesThermalStateFromItsPreCaptureHistory) {
    const auto config = ThermalConfigFor(1, 1);
    auto created = GpuCameraPipeline::Create(Device(), 1, 1);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    auto input = UploadMeasurement(Device(), 1, 1,
                                   {0.0f, 0.0f, 0.0f, 0.0f});
    ASSERT_NE(input, nullptr);
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    ASSERT_TRUE(UpdateMeasurement(Device(), *input, 1, 1,
                                  {1e-9f, 0.0f, 0.0f, 0.0f}));
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 1, 0.008).has_value());
    ASSERT_TRUE(UpdateMeasurement(Device(), *input, 1, 1,
                                  {2e-9f, 0.0f, 0.0f, 0.0f}));
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 1, 0.008).has_value());
    const auto reprocessed = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 1, 1);
    ASSERT_EQ(reprocessed.size(), 4u);
    Result<void, String> readoutReplay = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        readoutReplay = pipeline->RecordReadoutReprocess(cmd, config);
    });
    ASSERT_TRUE(readoutReplay.has_value()) << readoutReplay.error();
    EXPECT_EQ(ReadFloat(Device(), pipeline->GetOutputs().rawDn, 1, 1), reprocessed);
    const double firstStep = 2e-9 * (1.0 - std::exp(-1.0));
    EXPECT_NEAR(reprocessed[0], firstStep * 1e13, 1.0);

    ASSERT_TRUE(UpdateMeasurement(Device(), *input, 1, 1,
                                  {1e-9f, 0.0f, 0.0f, 0.0f}));
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 2, 0.016).has_value());
    const auto later = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 1, 1);
    ASSERT_EQ(later.size(), 4u);
    Image zero(1, 1, 1);
    Image high(1, 1, 1);
    high.data[0] = 2e-9f;
    Image middle(1, 1, 1);
    middle.data[0] = 1e-9f;
    CpuCameraPipeline cpu(config);
    CaptureState state;
    ASSERT_TRUE(cpu.CaptureMeasured(state, 0.0, zero).has_value());
    ASSERT_TRUE(cpu.CaptureMeasured(state, 0.008, high).has_value());
    const auto cpuLater = cpu.CaptureMeasured(state, 0.016, middle);
    ASSERT_TRUE(cpuLater.has_value());
    EXPECT_NEAR(later[0], cpuLater.value().rawDn->image.data[0], 1.0);
}

TEST_F(CameraGpuTest, ReadoutReprocessMatchesFreshReadoutOnSameMeasurement) {
    auto original = CameraConfigFor(4, 4);
    std::vector<f32> rgba(4u * 4u * 4u, 0.0f);
    for (u32 pixel = 0; pixel < 16; ++pixel)
        rgba[pixel * 4] = 10000.0f + static_cast<f32>(pixel) * 200.0f;
    auto input = UploadMeasurement(Device(), 4, 4, rgba);
    ASSERT_NE(input, nullptr);

    auto first = GpuCameraPipeline::Create(Device(), 4, 4);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(first.value()->Configure(original).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *first.value(), *input, 0, 0.0).has_value());

    auto edited = original;
    edited.readout.blackLevelDn += 128.0;
    Result<void, String> reapplied = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        reapplied = first.value()->RecordReadoutReprocess(cmd, edited);
    });
    ASSERT_TRUE(reapplied.has_value()) << reapplied.error();
    const auto replayRaw = ReadFloat(Device(), first.value()->GetOutputs().rawDn, 4, 4);
    const auto replayDisplay = ReadFloat(Device(), first.value()->GetOutputs().display, 4, 4);

    auto fresh = GpuCameraPipeline::Create(Device(), 4, 4);
    ASSERT_TRUE(fresh.has_value());
    ASSERT_TRUE(fresh.value()->Configure(edited).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *fresh.value(), *input, 0, 0.0).has_value());
    EXPECT_EQ(replayRaw, ReadFloat(Device(), fresh.value()->GetOutputs().rawDn, 4, 4));
    EXPECT_EQ(replayDisplay,
              ReadFloat(Device(), fresh.value()->GetOutputs().display, 4, 4));
}

TEST_F(CameraGpuTest, PreviewPixelCentersChoosePhysicalNucPixel) {
    auto config = CameraConfigFor(2, 1); // Physical array remains 2x1.
    config.quality.backend = ProcessingBackend::GpuPreview;
    config.photon.applyNuc = true;
    config.photon.nucGainMap = {1.0, 2.0};
    std::vector<f32> rgba = {10000.0f, 0.0f, 0.0f, 0.0f,
                             10000.0f, 0.0f, 0.0f, 0.0f,
                             10000.0f, 0.0f, 0.0f, 0.0f};
    auto input = UploadMeasurement(Device(), 3, 1, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 3, 1);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    const auto raw = ReadFloat(Device(), pipeline->GetOutputs().rawDn, 3, 1);
    const auto corrected =
        ReadFloat(Device(), pipeline->GetOutputs().corrected, 3, 1);
    ASSERT_EQ(raw.size(), 12u);
    ASSERT_EQ(corrected.size(), 12u);
    // Centers x=(0.5,1.5,2.5) in a 3-column preview map to physical
    // columns (0,1,1). An edge-based floor(x*2/3) would map x=1 to 0.
    for (u32 pixel = 0; pixel < 3; ++pixel)
        EXPECT_NEAR(raw[pixel * 4], 100.0, 1.0);
    EXPECT_NEAR(corrected[0], 100.0, 1.0);
    EXPECT_NEAR(corrected[4], 200.0, 1.0);
    EXPECT_NEAR(corrected[8], 200.0, 1.0);
}

// ---------------------------------------------------------------------------
// Interactive viewport scheduling (M4-0)
//
// With the camera enabled, RenderFrame drives acquisitions from the timeline
// clock instead of tracing the visibility image: one committed acquisition
// per reached frame-period slot, a same-tick re-record (convergence, no state
// advance) on redraws with the clock unmoved, and a history reset when the
// clock scrubs back before a committed acquisition. These cases run the full
// ExternalRenderContext against the shared headless device, the same fixture
// pattern as test_depth_aov.cpp.
// ---------------------------------------------------------------------------

namespace {

constexpr u32 kViewportSize = 64;
constexpr u32 kSensorSize = 16;

class CameraSchedulerGpuTest : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        testDir = std::filesystem::temp_directory_path() / "quantiloom_camera_scheduler";
        std::filesystem::create_directories(testDir);

        auto* shared = quantiloom::testing::SharedVulkanDevice();
        ASSERT_NE(shared, nullptr);

        ExternalRenderContext::InitParams params{};
        params.instance = shared->GetInstance();
        params.physicalDevice = shared->GetPhysicalDevice();
        params.device = shared->GetDevice();
        params.graphicsQueue = shared->GetGraphicsQueue();
        params.graphicsQueueFamily = shared->GetGraphicsQueueFamily();
        params.targetColorFormat = VK_FORMAT_B8G8R8A8_SRGB;
        params.width = kViewportSize;
        params.height = kViewportSize;
        params.pipelineCacheDir = testDir.string();

        auto created = ExternalRenderContext::Create(params);
        ASSERT_TRUE(created.has_value()) << created.error();
        context = std::move(created.value());

        target = std::make_unique<GpuImage>(
            shared->GetAllocator(), shared->GetDevice(), kViewportSize, kViewportSize,
            VK_FORMAT_B8G8R8A8_SRGB,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    }

    void TearDown() override {
        target.reset();
        context.reset();
        std::error_code ignored;
        std::filesystem::remove_all(testDir, ignored);
        VulkanDeviceTest::TearDown();
    }

    void ApplyScene(const std::array<f64, 3>& albedo = {0.8, 0.8, 0.8}) {
        const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
        const auto gltf = root / "assets" / "models" / "cornell_box" / "cornell_box.gltf";
        const auto path = testDir / "scene.toml";
        {
            std::ofstream file(path);
            file << "[renderer]\nresolution = [64, 64]\n"
                 << "[camera]\n"
                    "position = [278.0, 274.0, -800.0]\n"
                    "look_at = [278.0, 274.0, 0.0]\n"
                 << "[spectral]\n"
                 << "[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                    "sun_radiance = [1.0, 1.0, 1.0]\nsky_radiance = [0.1, 0.1, 0.1]\n"
                 << "[material]\nalbedo = [" << albedo[0] << ", " << albedo[1]
                 << ", " << albedo[2] << "]\n"
                 << "[scene]\ngltf = \"" << gltf.generic_string() << "\"\n";
        }
        auto loaded = Config::Load(path.string());
        ASSERT_TRUE(loaded.has_value()) << "fixture TOML did not parse";
        const auto report = context->ApplyConfig(loaded.value());
        ASSERT_TRUE(report.ok()) << report.FirstError();
    }

    Result<void, String> EnableCamera() {
        auto config = CameraConfigFor(kSensorSize, kSensorSize);
        return context->SetCameraConfig(config);
    }

    void DrawFrame() {
        CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
            context->RenderFrame(cmd, target->GetImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                                 kViewportSize, kViewportSize);
        });
    }

    std::filesystem::path testDir;
    std::unique_ptr<ExternalRenderContext> context;
    std::unique_ptr<GpuImage> target;
};

bool CornellBoxAvailable() {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    return std::filesystem::exists(
        root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
}

} // namespace

TEST_F(CameraSchedulerGpuTest, ScheduledFramesAdvanceAcquisitionIndex) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    const auto enabled = EnableCamera();
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    // The first frame takes scheduling ownership and commits acquisition 0
    // at the current clock time.
    DrawFrame();
    auto status = context->GetCameraHistoryStatus();
    EXPECT_EQ(status.acquisitionIndex, 0u);
    EXPECT_EQ(status.epoch, 0u);
    EXPECT_FALSE(status.historyReset);

    // Advance the clock one frame period (the scheduler's cadence) per draw;
    // each draw commits the next scheduled acquisition.
    constexpr int kFrames = 4;
    for (int i = 1; i <= kFrames; ++i) {
        ASSERT_TRUE(context->SetTimelineTime(static_cast<f64>(i) / 30.0).has_value());
        DrawFrame();
        status = context->GetCameraHistoryStatus();
        EXPECT_EQ(status.acquisitionIndex, static_cast<u64>(i)) << "frame " << i;
        EXPECT_EQ(status.epoch, 0u);
    }

    // The viewport display product is readable at the physical sensor
    // extent, independent of the target extent.
    auto display = context->CaptureDisplayImage();
    ASSERT_TRUE(display.has_value()) << display.error();
    EXPECT_EQ(display.value().width, kSensorSize);
    EXPECT_EQ(display.value().height, kSensorSize);
    ASSERT_EQ(display.value().data.size(),
              static_cast<size_t>(kSensorSize) * kSensorSize * 4u);
    for (const f32 value : display.value().data)
        EXPECT_TRUE(std::isfinite(value));
}

TEST_F(CameraSchedulerGpuTest, RedrawWithoutClockAdvanceDoesNotAdvanceAcquisition) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    const auto enabled = EnableCamera();
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    DrawFrame();
    ASSERT_TRUE(context->SetTimelineTime(1.0 / 30.0).has_value());
    DrawFrame();
    const auto before = context->GetCameraHistoryStatus();
    ASSERT_EQ(before.acquisitionIndex, 1u);

    // Paused redraws: the host draws the same instant again. Extra samples
    // may re-record for convergence, but the acquisition index, the frame
    // time and the history epoch must not move -- state advances only on a
    // new acquisition tick.
    DrawFrame();
    DrawFrame();
    const auto after = context->GetCameraHistoryStatus();
    EXPECT_EQ(after.acquisitionIndex, before.acquisitionIndex);
    EXPECT_EQ(after.frameTimeSeconds, before.frameTimeSeconds);
    EXPECT_EQ(after.epoch, before.epoch);
}

TEST_F(CameraSchedulerGpuTest, HistoryEpochResetsOnScrubBackAndHostReset) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    const auto enabled = EnableCamera();
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    auto status = context->GetCameraHistoryStatus();
    EXPECT_EQ(status.epoch, 0u);
    EXPECT_EQ(status.acquisitionIndex, 0u);
    EXPECT_FALSE(status.historyReset);
    EXPECT_TRUE(status.lastResetReason.empty());

    DrawFrame();
    ASSERT_TRUE(context->SetTimelineTime(1.0 / 30.0).has_value());
    DrawFrame();
    ASSERT_EQ(context->GetCameraHistoryStatus().acquisitionIndex, 1u);

    // Scrubbing back before the committed acquisition invalidates the
    // detector's temporal history: a reset is pending and the reason is
    // recorded.
    ASSERT_TRUE(context->SetTimelineTime(0.0).has_value());
    status = context->GetCameraHistoryStatus();
    EXPECT_TRUE(status.historyReset);
    EXPECT_FALSE(status.lastResetReason.empty());

    // The reset frame records no acquisition but applies the reset: the
    // epoch increments, the pending flag clears, and the committed index
    // restarts at 0.
    DrawFrame();
    status = context->GetCameraHistoryStatus();
    EXPECT_EQ(status.epoch, 1u);
    EXPECT_FALSE(status.historyReset);
    EXPECT_EQ(status.acquisitionIndex, 0u);

    // A host-requested reset starts another epoch.
    ASSERT_TRUE(context->ResetCameraHistory().has_value());
    status = context->GetCameraHistoryStatus();
    EXPECT_EQ(status.epoch, 2u);
    EXPECT_EQ(status.acquisitionIndex, 0u);
    EXPECT_FALSE(status.historyReset);
}

// ---------------------------------------------------------------------------
// M4-2: acquisition history on the GPU. A RecordStateCheckpoint snapshot
// (thermal ping-pong pair + previous-frame AGC window + host advance scalars)
// must rewind the device so that replaying the same tick sequence -- the
// noise streams are keyed on the acquisition index -- is bit-identical to
// the uninterrupted run. The facade pairs the snapshot with the host
// feedback state, bumps the history epoch on restore, and runs warmup /
// single-tick advances on the same frame grid as the offline CPU reference.
// ---------------------------------------------------------------------------

TEST_F(CameraGpuTest, CheckpointRestoreReplaysThermalHistoryBitIdentically) {
    constexpr u32 kSize = 16;
    auto config = ThermalConfigFor(kSize, kSize);
    // Exercise the acquisition-keyed noise streams, not only the
    // deterministic operator: identical indices must reproduce identical
    // noise, or the replay below cannot be bit-identical.
    config.quality.noiseFree = false;
    config.thermal.readNoiseDnRms = 3.0;
    config.thermal.driftDnPerSecond = 7.0;
    std::vector<f32> rgba(static_cast<size_t>(kSize) * kSize * 4, 0.0f);
    for (u32 pixel = 0; pixel < kSize * kSize; ++pixel)
        rgba[pixel * 4] = 2.5e-11f;
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());

    const auto readProducts = [&]() {
        return std::make_pair(
            ReadFloat(Device(), pipeline->GetOutputs().rawDn, kSize, kSize),
            ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize));
    };
    const auto checkpointNow = [&]() {
        Result<GpuCameraPipeline::GpuCheckpoint, String> result =
            Result<GpuCameraPipeline::GpuCheckpoint, String>::Err(
            "checkpoint was not recorded");
        CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
            result = pipeline->RecordStateCheckpoint(cmd);
        });
        return result;
    };

    // Ticks 0 and 1, then the checkpoint at acquisition 1's exit.
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 1, 0.1).has_value());
    auto checkpoint = checkpointNow();
    ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error();
    EXPECT_EQ(checkpoint.value().acquisitionIndex, 1u);
    EXPECT_TRUE(checkpoint.value().hasCapture);

    // The uninterrupted continuation: ticks 2 and 3.
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 2, 0.2).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 3, 0.3).has_value());
    const auto continuous = readProducts();

    // Restore rewinds to acquisition 1's exit; replaying 2 and 3 -- with a
    // same-tick re-record of 2 in between, the redraw/convergence case --
    // must reproduce the uninterrupted products exactly.
    Result<void, String> restored = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        restored = pipeline->RestoreStateCheckpoint(checkpoint.value(), cmd);
    });
    ASSERT_TRUE(restored.has_value()) << restored.error();
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 2, 0.2).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 2, 0.2).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 3, 0.3).has_value());
    const auto replayed = readProducts();

    ASSERT_EQ(replayed.first.size(), continuous.first.size());
    ASSERT_EQ(replayed.second.size(), continuous.second.size());
    EXPECT_TRUE(std::equal(replayed.first.begin(), replayed.first.end(),
                           continuous.first.begin()));
    EXPECT_TRUE(std::equal(replayed.second.begin(), replayed.second.end(),
                           continuous.second.begin()));

    // The restored state accepts the next tick like the uninterrupted run.
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 4, 0.4).has_value());
    const auto after = readProducts();
    ASSERT_EQ(after.first.size(), continuous.first.size());

    // A checkpoint taken before any capture round-trips as the initial
    // condition: restore accepts acquisition 0 as the first capture again.
    auto fresh = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(fresh.has_value());
    ASSERT_TRUE(fresh.value()->Configure(config).has_value());
    Result<GpuCameraPipeline::GpuCheckpoint, String> freshCheckpoint =
        Result<GpuCameraPipeline::GpuCheckpoint, String>::Err(
        "checkpoint was not recorded");
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        freshCheckpoint = fresh.value()->RecordStateCheckpoint(cmd);
    });
    ASSERT_TRUE(freshCheckpoint.has_value());
    EXPECT_FALSE(freshCheckpoint.value().hasCapture);
    ASSERT_TRUE(RecordCamera(Device(), *fresh.value(), *input, 0, 0.0).has_value());
    Result<void, String> freshRestored = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        freshRestored = fresh.value()->RestoreStateCheckpoint(*freshCheckpoint, cmd);
    });
    ASSERT_TRUE(freshRestored.has_value());
    ASSERT_TRUE(RecordCamera(Device(), *fresh.value(), *input, 0, 0.0).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *fresh.value(), *input, 1, 0.1).has_value());
}

TEST_F(CameraSchedulerGpuTest, CheckpointRestoreRewindsEpochAndReplaysIdentically) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    auto config = CameraConfigFor(kSensorSize, kSensorSize);
    config.quality.noiseFree = false;  // replay must reproduce the keyed noise
    const auto enabled = context->SetCameraConfig(config);
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    DrawFrame();  // acquisition 0 at the current clock time
    ASSERT_TRUE(context->SetTimelineTime(1.0 / 30.0).has_value());
    DrawFrame();  // acquisition 1
    ASSERT_EQ(context->GetCameraHistoryStatus().acquisitionIndex, 1u);
    ASSERT_TRUE(context->CheckpointCameraHistory().has_value());

    ASSERT_TRUE(context->SetTimelineTime(2.0 / 30.0).has_value());
    DrawFrame();  // acquisition 2
    const auto before = context->CaptureDisplayImage();
    ASSERT_TRUE(before.has_value());

    // Scrub back before the committed acquisition. Without a restore the
    // next frame would discard the detector history; the checkpoint keeps
    // it instead and the epoch moves on.
    ASSERT_TRUE(context->SetTimelineTime(1.0 / 30.0).has_value());
    ASSERT_TRUE(context->RestoreCameraHistoryCheckpoint().has_value());
    auto status = context->GetCameraHistoryStatus();
    EXPECT_EQ(status.epoch, 1u);
    EXPECT_EQ(status.acquisitionIndex, 1u);
    EXPECT_FALSE(status.historyReset);
    EXPECT_EQ(status.lastResetReason, "checkpoint restore");

    // Replay: acquisition 1 re-records as a same-tick convergence pass,
    // acquisition 2 advances the restored state. Both products must match
    // the original run bit for bit.
    DrawFrame();
    EXPECT_EQ(context->GetCameraHistoryStatus().acquisitionIndex, 1u);
    ASSERT_TRUE(context->SetTimelineTime(2.0 / 30.0).has_value());
    DrawFrame();
    status = context->GetCameraHistoryStatus();
    EXPECT_EQ(status.acquisitionIndex, 2u);
    EXPECT_EQ(status.epoch, 1u);
    const auto after = context->CaptureDisplayImage();
    ASSERT_TRUE(after.has_value());
    ASSERT_EQ(after.value().data.size(), before.value().data.size());
    EXPECT_TRUE(std::equal(after.value().data.begin(), after.value().data.end(),
                           before.value().data.begin()));

    // The stack is depth one and consumed by the restore.
    const auto again = context->RestoreCameraHistoryCheckpoint();
    ASSERT_FALSE(again.has_value());
}

TEST_F(CameraSchedulerGpuTest, WarmupAndAdvanceRunSyntheticAcquisitionsOnTheFrameGrid) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    const auto enabled = EnableCamera();
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    // Zero warmup is a quiet no-op.
    ASSERT_TRUE(context->WarmUpCamera(0.0).has_value());

    // Three warmup periods at the 0.1 s fixture frame period, anchored at
    // the current acquisition time 0: the grid clamps to the clock origin,
    // so all three synthetic acquisitions land on t=0 (indices 0..2) and
    // move the index and the noise streams without advancing the detector's
    // elapsed time.
    ASSERT_TRUE(context->WarmUpCamera(0.3).has_value());
    auto status = context->GetCameraHistoryStatus();
    EXPECT_EQ(status.acquisitionIndex, 2u);
    EXPECT_DOUBLE_EQ(status.frameTimeSeconds, 0.0);
    EXPECT_EQ(status.epoch, 0u);

    // The first real acquisition then sees a full frame period of elapsed
    // history, exactly like the offline CPU warmup grid.
    ASSERT_TRUE(context->AdvanceCameraTo(0.1).has_value());
    status = context->GetCameraHistoryStatus();
    EXPECT_EQ(status.acquisitionIndex, 3u);
    EXPECT_DOUBLE_EQ(status.frameTimeSeconds, 0.1);

    ASSERT_TRUE(context->AdvanceCameraTo(0.2).has_value());
    status = context->GetCameraHistoryStatus();
    EXPECT_EQ(status.acquisitionIndex, 4u);
    EXPECT_DOUBLE_EQ(status.frameTimeSeconds, 0.2);

    const auto bad = context->WarmUpCamera(-1.0);
    ASSERT_FALSE(bad.has_value());
}

// ---------------------------------------------------------------------------
// M4-1: GPU dynamic exposure (time-stratified trace + reprojection composite)
//
// The measurement image is a layer-per-time-stratum array; the dynamic
// compositor reprojects the anchor depth into the strata bracketing each
// row's integration time. These cases drive GpuCameraPipeline directly with
// synthetic strata (an analytic box scene rasterized per layer, so the
// checker can recomposite the same math in numpy) and drive the full
// ExternalRenderContext scheduler for the report, metadata, rolling shutter,
// and degenerate T=1 behaviour.
// ---------------------------------------------------------------------------

namespace {

constexpr u32 kDynamicWidth = 32;
constexpr u32 kDynamicHeight = 8;
constexpr u32 kDynamicStrata = 8;

// A slab-test box: [min,max] on each axis plus a constant rate, optionally a
// linear rate gradient along world x (rate = rate + rateSlopeX * hitX) so a
// reprojected sample differs from the anchor even where depth agrees.
struct DynamicBox {
    std::array<f32, 3> min{};
    std::array<f32, 3> max{};
    f32 rate = 0.0f;
    f32 rateSlopeX = 0.0f;
};

struct DynamicCameraFrame {
    std::array<f32, 3> origin{};
    std::array<f32, 3> forward{0.0f, 0.0f, -1.0f};
    std::array<f32, 3> right{1.0f, 0.0f, 0.0f};
    std::array<f32, 3> up{0.0f, 1.0f, 0.0f};
    f32 fovScale = 0.1f;
    f32 aspect = 4.0f;
    f64 timeSeconds = 0.0;
};

// Exact twin of raygen.rgen's primary-ray construction for a pixel centre.
void DynamicPixelRay(const DynamicCameraFrame& camera, u32 x, u32 y,
                     std::array<f32, 3>& origin, std::array<f32, 3>& direction) {
    const f32 u = (static_cast<f32>(x) + 0.5f) / kDynamicWidth;
    const f32 v = (static_cast<f32>(y) + 0.5f) / kDynamicHeight;
    f32 ndcX = u * 2.0f - 1.0f;
    f32 ndcY = -(v * 2.0f - 1.0f);
    origin = camera.origin;
    for (int axis = 0; axis < 3; ++axis) {
        direction[axis] = camera.forward[axis] +
            ndcX * camera.right[axis] * camera.fovScale * camera.aspect +
            ndcY * camera.up[axis] * camera.fovScale;
    }
    f32 length = 0.0f;
    for (f32 component : direction) length += component * component;
    length = std::sqrt(length);
    for (f32& component : direction) component /= length;
}

// Nearest slab hit along the ray; returns t or -1. On hit, `rate` receives
// the box rate including its x-gradient evaluated at the hit point.
f32 DynamicIntersectBox(const DynamicCameraFrame& camera,
                        const std::array<f32, 3>& origin,
                        const std::array<f32, 3>& direction,
                        const DynamicBox& box, bool& hit, f32& rate) {
    f32 tNear = -1e30f, tFar = 1e30f;
    hit = true;
    for (int axis = 0; axis < 3; ++axis) {
        const f32 originAxis = origin[axis];
        const f32 directionAxis = direction[axis];
        const f32 slabMin = box.min[axis] - originAxis;
        const f32 slabMax = box.max[axis] - originAxis;
        if (std::abs(directionAxis) < 1e-12f) {
            if (slabMin > 0.0f || slabMax < 0.0f) {
                hit = false;
                return -1.0f;
            }
            continue;
        }
        f32 t0 = slabMin / directionAxis;
        f32 t1 = slabMax / directionAxis;
        if (t0 > t1) std::swap(t0, t1);
        tNear = std::max(tNear, t0);
        tFar = std::min(tFar, t1);
    }
    if (tFar < tNear || tFar < 0.0f || tNear < 0.0f) {
        hit = false;
        return -1.0f;
    }
    hit = true;
    const f32 hitX = origin[0] + direction[0] * tNear;
    rate = box.rate + box.rateSlopeX * hitX;
    return tNear;
}

// Rasterize one stratum of the synthetic scene exactly as a pinhole trace of
// that layer's camera would: nearest box hit wins, depth is the parametric
// distance, miss writes depth -1 and zero rate.
void DynamicRasterizeLayer(const DynamicCameraFrame& camera,
                           const std::vector<DynamicBox>& boxes,
                           std::vector<f32>& rateRgba,
                           std::vector<f32>& depth) {
    for (u32 y = 0; y < kDynamicHeight; ++y) {
        for (u32 x = 0; x < kDynamicWidth; ++x) {
            const size_t pixel = static_cast<size_t>(y) * kDynamicWidth + x;
            std::array<f32, 3> origin, direction;
            DynamicPixelRay(camera, x, y, origin, direction);
            f32 bestT = 1e30f;
            f32 bestRate = 0.0f;
            bool any = false;
            for (const DynamicBox& box : boxes) {
                bool hit = false;
                f32 rate = 0.0f;
                const f32 t = DynamicIntersectBox(camera, origin, direction,
                                                  box, hit, rate);
                if (hit && t < bestT) {
                    bestT = t;
                    bestRate = rate;
                    any = true;
                }
            }
            rateRgba[pixel * 4 + 0] = any ? bestRate : 0.0f;
            rateRgba[pixel * 4 + 1] = 0.0f;
            rateRgba[pixel * 4 + 2] = 0.0f;
            rateRgba[pixel * 4 + 3] = 0.0f;
            depth[pixel] = any ? bestT : -1.0f;
        }
    }
}

std::unique_ptr<GpuImage> UploadStrataImage(
    VulkanContext& context, u32 width, u32 height, u32 layers, VkFormat format,
    const std::vector<f32>& texelData, u32 componentsPerTexel) {
    auto image = std::make_unique<GpuImage>(
        context.GetAllocator(), context.GetDevice(), width, height, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY, 1u, layers, 0, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    if (!image->IsValid() ||
        texelData.size() != static_cast<size_t>(width) * height * layers *
                                componentsPerTexel)
        return {};
    const VkDeviceSize bytes = texelData.size() * sizeof(f32);
    GpuBuffer staging(context.GetAllocator(), bytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);
    if (!staging.IsValid()) return {};
    staging.Upload(texelData.data(), bytes);
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        CommandHelper::TransitionImageLayout(
            cmd, image->GetImage(), format, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, layers);
        for (u32 layer = 0; layer < layers; ++layer) {
            VkBufferImageCopy region{};
            region.bufferOffset = static_cast<VkDeviceSize>(layer) * width *
                                  height * componentsPerTexel * sizeof(f32);
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = layer;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {width, height, 1};
            vkCmdCopyBufferToImage(cmd, staging.GetHandle(), image->GetImage(),
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                   &region);
        }
        VkImageMemoryBarrier ready{};
        ready.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT;
        ready.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ready.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.image = image->GetImage();
        ready.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ready.subresourceRange.levelCount = 1;
        ready.subresourceRange.layerCount = layers;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &ready);
    });
    return image;
}

// The synthetic scene: a flat far wall and a near occluder that slides with
// the stratum (world-unit shift per layer), viewed by a camera that drifts
// sideways per layer. Static parts keep every reprojection consistent; the
// occluder's silhouette and the camera drift produce known disocclusions.
struct DynamicSceneSpec {
    std::vector<DynamicCameraFrame> cameras;
    std::vector<DynamicBox> boxes;
    f64 t0 = 0.5;
    f64 exposureSeconds = 0.125;
    f64 rowDelaySeconds = 0.0;
};

DynamicSceneSpec MakeDynamicSceneSpec() {
    DynamicSceneSpec spec;
    const auto times = camera::ExposureStratumTimes(
        spec.t0, spec.exposureSeconds, kDynamicStrata);
    for (u32 k = 0; k < kDynamicStrata; ++k) {
        DynamicCameraFrame frame;
        // Slow drift: the wall stays consistent between strata (blends pass
        // the depth check) while the rate gradient makes each stratum's value
        // visibly different, so the composite is not just the anchor.
        frame.origin = {0.05f * static_cast<f32>(k), 0.0f, 0.0f};
        frame.timeSeconds = times[k];
        spec.cameras.push_back(frame);
    }
    // Far wall with a linear rate gradient along x: reprojected samples read
    // different rates than the anchor even where depth is consistent.
    spec.boxes.push_back({{-20.0f, -20.0f, -5.2f},
                          {20.0f, 20.0f, -5.0f},
                          0.5f,
                          0.25f});
    // Near occluder sliding 0.05 world units (1 px at its depth) per stratum:
    // its silhouette edges turn over between strata and disocclude.
    for (u32 k = 0; k < kDynamicStrata; ++k) {
        const f32 shift = 0.05f * static_cast<f32>(k);
        spec.boxes.push_back({{-1.2f + shift, -0.25f, -2.1f},
                              {-0.2f + shift, 0.25f, -2.0f},
                              2.0f,
                              0.0f});
    }
    return spec;
}

void RasterizeSpec(const DynamicSceneSpec& spec,
                   std::vector<f32>& rateRgba, std::vector<f32>& depth) {
    rateRgba.assign(static_cast<size_t>(kDynamicStrata) * kDynamicWidth *
                        kDynamicHeight * 4,
                    0.0f);
    depth.assign(static_cast<size_t>(kDynamicStrata) * kDynamicWidth *
                     kDynamicHeight,
                 -1.0f);
    // Each stratum sees only the wall plus ITS occluder position.
    std::vector<DynamicBox> common{spec.boxes.front()};
    for (u32 k = 0; k < kDynamicStrata; ++k) {
        std::vector<DynamicBox> layerBoxes = common;
        layerBoxes.push_back(spec.boxes[1 + k]);
        std::vector<f32> layerRate(kDynamicWidth * kDynamicHeight * 4, 0.0f);
        std::vector<f32> layerDepth(kDynamicWidth * kDynamicHeight, -1.0f);
        DynamicRasterizeLayer(spec.cameras[k], layerBoxes, layerRate,
                              layerDepth);
        std::copy(layerRate.begin(), layerRate.end(),
                  rateRgba.begin() + static_cast<size_t>(k) * kDynamicWidth *
                                         kDynamicHeight * 4);
        std::copy(layerDepth.begin(), layerDepth.end(),
                  depth.begin() + static_cast<size_t>(k) * kDynamicWidth *
                                      kDynamicHeight);
    }
}

std::filesystem::path DynamicArtifactDir() {
    const char* env = std::getenv("QUANTILOOM_CAMERA_DYNAMIC_DIR");
    if (!env || !*env) return {};
    std::filesystem::create_directories(env);
    return env;
}

void WriteDynamicArtifacts(const std::filesystem::path& dir,
                           const DynamicSceneSpec& spec,
                           const std::vector<f32>& rateRgba,
                           const std::vector<f32>& depth,
                           const std::vector<f32>& composite,
                           const std::array<u32, 4>& counters) {
    if (dir.empty()) return;
    const auto write = [&](const char* name, const void* data, size_t bytes) {
        std::ofstream file(dir / name, std::ios::binary);
        file.write(static_cast<const char*>(data),
                   static_cast<std::streamsize>(bytes));
    };
    write("strata_rate.f32", rateRgba.data(), rateRgba.size() * sizeof(f32));
    write("strata_depth.f32", depth.data(), depth.size() * sizeof(f32));
    write("composite.f32", composite.data(),
          composite.size() * sizeof(f32));
    write("counters.u32", counters.data(), counters.size() * sizeof(u32));

    std::ofstream specFile(dir / "spec.json");
    specFile << "{\n"
             << "  \"width\": " << kDynamicWidth << ",\n"
             << "  \"height\": " << kDynamicHeight << ",\n"
             << "  \"strata\": " << kDynamicStrata << ",\n"
             << "  \"t0\": " << spec.t0 << ",\n"
             << "  \"exposure_seconds\": " << spec.exposureSeconds << ",\n"
             << "  \"row_delay_seconds\": " << spec.rowDelaySeconds << ",\n"
             << "  \"cameras\": [\n";
    for (size_t k = 0; k < spec.cameras.size(); ++k) {
        const auto& camera = spec.cameras[k];
        specFile << "    {\"origin\": [" << camera.origin[0] << ", "
                 << camera.origin[1] << ", " << camera.origin[2]
                 << "], \"forward\": [" << camera.forward[0] << ", "
                 << camera.forward[1] << ", " << camera.forward[2]
                 << "], \"right\": [" << camera.right[0] << ", "
                 << camera.right[1] << ", " << camera.right[2]
                 << "], \"up\": [" << camera.up[0] << ", " << camera.up[1]
                 << ", " << camera.up[2] << "], \"fov_scale\": "
                 << camera.fovScale << ", \"aspect\": " << camera.aspect
                 << ", \"time\": " << camera.timeSeconds << "}"
                 << (k + 1 < spec.cameras.size() ? ",\n" : "\n");
    }
    specFile << "  ],\n  \"boxes\": [\n";
    // The occluder position is per stratum; the checker rebuilds the layer
    // list the same way the test does: common boxes, then one occluder per
    // layer. So publish wall + occluder template list and the per-layer index
    // mapping is implicit in rasterization order.
    specFile << "    {\"min\": [" << spec.boxes[0].min[0] << ", "
             << spec.boxes[0].min[1] << ", " << spec.boxes[0].min[2]
             << "], \"max\": [" << spec.boxes[0].max[0] << ", "
             << spec.boxes[0].max[1] << ", " << spec.boxes[0].max[2]
             << "], \"rate\": " << spec.boxes[0].rate
             << ", \"rate_slope_x\": " << spec.boxes[0].rateSlopeX << "},\n";
    specFile << "    {\"min\": [" << spec.boxes[1].min[0] << ", "
             << spec.boxes[1].min[1] << ", " << spec.boxes[1].min[2]
             << "], \"max\": [" << spec.boxes[1].max[0] << ", "
             << spec.boxes[1].max[1] << ", " << spec.boxes[1].max[2]
             << "], \"rate\": " << spec.boxes[1].rate
             << ", \"shift_per_stratum\": 0.05}\n";
    specFile << "  ]\n}\n";
}

Result<void, String> RecordDynamicMeasurement(
    VulkanContext& context, GpuCameraPipeline& pipeline,
    const GpuImage& strataRate, const GpuImage& strataDepth,
    const DynamicSceneSpec& spec, u64 acquisitionIndex,
    VkQueryPool timingPool) {
    std::array<DynamicLayerCamera, kCameraTimeStrataMax> frames{};
    const u32 count = std::min<u32>(kDynamicStrata, kCameraTimeStrataMax);
    for (u32 k = 0; k < count; ++k) {
        const auto& camera = spec.cameras[k];
        frames[k].origin = {camera.origin[0], camera.origin[1],
                            camera.origin[2], 0.0f};
        frames[k].forward = {camera.forward[0], camera.forward[1],
                             camera.forward[2], 0.0f};
        frames[k].right = {camera.right[0], camera.right[1],
                           camera.right[2], 0.0f};
        frames[k].up = {camera.up[0], camera.up[1], camera.up[2], 0.0f};
        frames[k].params = {camera.fovScale, camera.aspect,
                            static_cast<f32>(camera.timeSeconds), 0.0f};
    }
    DynamicExposureInput dynamic;
    dynamic.strataRate = &strataRate;
    dynamic.strataDepth = &strataDepth;
    dynamic.layerCameras = frames.data();
    dynamic.strataCount = count;
    dynamic.firstRowMidSeconds = spec.t0;
    dynamic.exposureSeconds = spec.exposureSeconds;
    dynamic.rowDelaySeconds = spec.rowDelaySeconds;
    Result<void, String> status = Result<void, String>::Ok();
    GpuCameraPipeline::TimingQueries timing{timingPool, 0u};
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        status = pipeline.RecordMeasurement(cmd, strataRate, acquisitionIndex,
                                            spec.t0, timing, dynamic);
    });
    return status;
}

} // namespace

TEST_F(CameraGpuTest, StratifiedCompositorMatchesAnchorsAndCountsDisocclusion) {
    const auto spec = MakeDynamicSceneSpec();
    std::vector<f32> rateRgba, depth;
    RasterizeSpec(spec, rateRgba, depth);
    auto strataRate = UploadStrataImage(Device(), kDynamicWidth, kDynamicHeight,
                                        kDynamicStrata,
                                        VK_FORMAT_R32G32B32A32_SFLOAT, rateRgba,
                                        4u);
    auto strataDepth = UploadStrataImage(Device(), kDynamicWidth,
                                         kDynamicHeight, kDynamicStrata,
                                         VK_FORMAT_R32_SFLOAT, depth, 1u);
    ASSERT_NE(strataRate, nullptr);
    ASSERT_NE(strataDepth, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), kDynamicWidth,
                                             kDynamicHeight);
    ASSERT_TRUE(created.has_value()) << created.error();
    auto pipeline = std::move(created.value());
    auto configured = pipeline->Configure(CameraConfigFor(kDynamicWidth,
                                                          kDynamicHeight));
    ASSERT_TRUE(configured.has_value()) << configured.error();

    VkQueryPool timingPool = VK_NULL_HANDLE;
    VkQueryPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    poolInfo.queryCount = GpuCameraPipeline::kTimingQueryCount;
    ASSERT_EQ(vkCreateQueryPool(Device().GetDevice(), &poolInfo, nullptr,
                                &timingPool),
              VK_SUCCESS);

    // One stratified acquisition over the synthetic scene.
    const auto recorded = RecordDynamicMeasurement(
        Device(), *pipeline, *strataRate, *strataDepth, spec, 0u, timingPool);
    ASSERT_TRUE(recorded.has_value()) << recorded.error();
    const auto outputs = pipeline->GetOutputs();
    ASSERT_NE(outputs.compositedRate, nullptr);
    const auto counters = pipeline->ReadDynamicCounters();
    ASSERT_TRUE(counters.has_value()) << counters.error();

    // The occluder's silhouette and the per-layer camera drift must produce
    // disoccluded pixels; the wall must keep most of the frame trustworthy.
    EXPECT_GT((*counters)[0], 0u);
    EXPECT_LT((*counters)[0], kDynamicWidth * kDynamicHeight / 2u);

    // Timestamp stamps resolve after the synchronous record and stay ordered.
    std::array<u64, 14> stamps{};
    EXPECT_EQ(vkGetQueryPoolResults(
                  Device().GetDevice(), timingPool, 0,
                  GpuCameraPipeline::kTimingQueryCount,
                  stamps.size() * sizeof(u64), stamps.data(), 2 * sizeof(u64),
                  VK_QUERY_RESULT_64_BIT |
                      VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
              VK_SUCCESS);
    for (u32 i = 0; i < 7; ++i) {
        ASSERT_NE(stamps[2 * i + 1], 0u) << "stamp " << i;
        if (i > 0) EXPECT_GE(stamps[2 * i], stamps[2 * i - 2]);
    }

    const auto globalComposite = ReadFloat(Device(), outputs.compositedRate,
                                           kDynamicWidth, kDynamicHeight);
    ASSERT_EQ(globalComposite.size(),
              static_cast<size_t>(kDynamicWidth) * kDynamicHeight * 4u);

    // Rolling shutter: rows integrate at different layer pairs, so the
    // composite must move where the scene moves.
    DynamicSceneSpec rollingSpec = spec;
    rollingSpec.rowDelaySeconds =
        spec.exposureSeconds / kDynamicHeight;
    const auto rollingRecorded = RecordDynamicMeasurement(
        Device(), *pipeline, *strataRate, *strataDepth, rollingSpec, 1u,
        timingPool);
    ASSERT_TRUE(rollingRecorded.has_value()) << rollingRecorded.error();
    const auto rollingOutputs = pipeline->GetOutputs();
    const auto rollingComposite = ReadFloat(
        Device(), rollingOutputs.compositedRate, kDynamicWidth,
        kDynamicHeight);
    double difference = 0.0;
    for (size_t i = 0; i < globalComposite.size(); ++i)
        difference += std::abs(static_cast<double>(globalComposite[i]) -
                               rollingComposite[i]);
    EXPECT_GT(difference, 1e-6)
        << "rolling shutter must change the composite for a moving scene";

    // T=1 degenerate: a single stratum must composite to the anchor layer
    // exactly. Feed the same arrays with strataCount 1.
    DynamicSceneSpec singleSpec = spec;
    singleSpec.cameras.resize(1);
    singleSpec.cameras[0].timeSeconds = spec.t0;
    const auto singleRecorded = RecordDynamicMeasurement(
        Device(), *pipeline, *strataRate, *strataDepth, singleSpec, 2u,
        VK_NULL_HANDLE);
    ASSERT_TRUE(singleRecorded.has_value()) << singleRecorded.error();
    const auto singleOutputs = pipeline->GetOutputs();
    const auto singleComposite = ReadFloat(
        Device(), singleOutputs.compositedRate, kDynamicWidth,
        kDynamicHeight);
    // The anchor layer content, from the synthetic rasterization itself.
    const auto anchorBegin = rateRgba.begin();
    for (size_t pixel = 0; pixel < kDynamicWidth * kDynamicHeight; ++pixel) {
        for (u32 channel = 0; channel < 4; ++channel) {
            EXPECT_NEAR(
                singleComposite[pixel * 4 + channel],
                *(anchorBegin + (pixel * 4 + channel)), 1e-5)
                << "pixel " << pixel << " channel " << channel;
        }
    }

    // Same-tick re-record keeps converging: the same acquisition index at the
    // same frame time is legal and must not disturb the detector state.
    const auto reRecorded = RecordDynamicMeasurement(
        Device(), *pipeline, *strataRate, *strataDepth, spec, 2u,
        VK_NULL_HANDLE);
    ASSERT_TRUE(reRecorded.has_value()) << reRecorded.error();

    // Artifacts for scripts/render-tests/check_camera_dynamic.py: the strata,
    // the global composite, the counters, and the scene spec the checker
    // recomposites from.
    if (const auto dir = DynamicArtifactDir(); !dir.empty()) {
        WriteDynamicArtifacts(dir, spec, rateRgba, depth, globalComposite,
                              *counters);
    }

    vkDestroyQueryPool(Device().GetDevice(), timingPool, nullptr);
}

TEST_F(CameraSchedulerGpuTest, StratifiedAcquisitionReportsDynamicExposure) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    auto config = CameraConfigFor(kSensorSize, kSensorSize);
    // Fast sideways motion, authored across t=0 so the acquisition's whole
    // exposure window (t0 +/- E/2 around the first acquisition at t=0) sits
    // inside the track: CameraPoseAt clamps outside the key range, and keys
    // starting at 0 would freeze every stratum before t=0 into one pose. The
    // anchor-to-candidate parallax (several px on the 16 px sensor) pushes
    // the reprojection of the frame's border columns outside the strata
    // images, so the composite must fall back to the anchor there and report
    // real disocclusions.
    config.motion.keys = {
        {-1.0, {128.0, 274.0, -800.0}, {278.0, 274.0, 0.0}},
        {1.0, {428.0, 274.0, -800.0}, {278.0, 274.0, 0.0}},
    };
    const auto enabled = context->SetCameraConfig(config);
    ASSERT_TRUE(enabled.has_value()) << enabled.error();
    context->SetSPP(8);

    DrawFrame();
    const auto report = context->GetLastDynamicExposureReport();
    EXPECT_TRUE(report.valid);
    EXPECT_EQ(report.strataCount, 8u);
    EXPECT_DOUBLE_EQ(report.timeSampleCoverage, 1.0);
    EXPECT_GT(report.disoccludedFraction, 0.0);
    EXPECT_GE(report.viewDependentFraction, report.disoccludedFraction);
    EXPECT_GE(report.viewDependentFraction, report.transparentFraction);
    EXPECT_LE(report.viewDependentFraction, 1.0);
    // No timeline: object motion is the scene's, and the scene is static.
    EXPECT_DOUBLE_EQ(report.objectMotionApproximation, 0.0);

    // Per-pass timings resolve once the frame has completed.
    const auto timings = context->GetLastCameraGpuTimings();
    EXPECT_TRUE(timings.valid);
    EXPECT_GE(timings.dynamicMs, 0.0f);
    EXPECT_GE(timings.psfMs, 0.0f);
    EXPECT_GE(timings.detectorMs, 0.0f);

    // The band product is annotated as a preview approximation.
    auto products = context->CaptureCameraProducts();
    ASSERT_TRUE(products.has_value()) << products.error();
    ASSERT_TRUE(products.value().bandMeasurement.has_value());
    const auto& metadata =
        products.value().bandMeasurement.value().image.metadata;
    EXPECT_NE(metadata.find("camera_dynamic_approximation"), metadata.end());

    // Same-tick redraw: convergence continues, the report describes the same
    // acquisition, and no state advances (M4-0 owns the scheduler invariants;
    // here the report must simply stay valid and identical in structure).
    DrawFrame();
    const auto redrawReport = context->GetLastDynamicExposureReport();
    EXPECT_TRUE(redrawReport.valid);
    EXPECT_EQ(redrawReport.strataCount, report.strataCount);
}

TEST_F(CameraSchedulerGpuTest, StaticSceneDegeneratesToSingleStratum) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    const auto enabled = EnableCamera();
    ASSERT_TRUE(enabled.has_value()) << enabled.error();
    context->SetSPP(8);

    DrawFrame();
    const auto report = context->GetLastDynamicExposureReport();
    EXPECT_TRUE(report.valid);
    // No camera motion and no timeline: the exposure is one stratum no
    // matter how many time positions were requested.
    EXPECT_EQ(report.strataCount, 1u);
    EXPECT_DOUBLE_EQ(report.disoccludedFraction, 0.0);
    EXPECT_DOUBLE_EQ(report.timeSampleCoverage, 0.125);

    auto products = context->CaptureCameraProducts();
    ASSERT_TRUE(products.has_value()) << products.error();
    ASSERT_TRUE(products.value().bandMeasurement.has_value());
    const auto& metadata =
        products.value().bandMeasurement.value().image.metadata;
    EXPECT_EQ(metadata.find("camera_dynamic_approximation"), metadata.end());
}

// ---------------------------------------------------------------------------
// M4-3: full classic ISP on the GPU (statistics, CDF, demosaic, color,
// display/AGC) against the CPU RunIsp reference, plus the CLAHE persistent
// tone wiring.
//
// The CPU chain computes in f64, the GPU in f32; every comparison below uses
// the task tolerance max(1e-3, 1e-5 * signal) on encoded-sRGB values, which
// sits three orders of magnitude above the float-rounding differences.
// ---------------------------------------------------------------------------

namespace {

// Record the same acquisition twice: the first pass leaves the persistent
// statistics buffer (and hence the next Linear-AGC window) describing the
// image, the second pass renders through that window.
Result<void, String> RecordTwice(VulkanContext& context,
                                 GpuCameraPipeline& pipeline,
                                 const GpuImage& measurement, u64 index,
                                 f64 timeSeconds) {
    auto first = RecordCamera(context, pipeline, measurement, index, timeSeconds);
    if (!first) return first;
    return RecordCamera(context, pipeline, measurement, index, timeSeconds);
}

} // namespace

TEST_F(CameraGpuTest, IspDisplayMatchesCpuMonoChain) {
    auto config = CameraConfigFor(16, 16);
    config.isp.whiteBalance = {1.1, 1.0, 0.9};
    config.isp.deviceToLinearSrgb = {0.90, 0.10, 0.00,
                                     0.05, 0.85, 0.10,
                                     0.00, 0.15, 0.85};
    config.isp.denoise = true;
    config.isp.denoiseStrength = 0.5;
    config.isp.sharpen = true;
    config.isp.sharpenStrength = 0.3;
    config.isp.toneGamma = 2.2;
    config.isp.clipOutOfGamut = false;

    Image cpuRate(16, 16, 1);
    std::vector<f32> rgba(16u * 16u * 4u, 0.0f);
    for (u32 y = 0; y < 16; ++y)
        for (u32 x = 0; x < 16; ++x) {
            const u32 i = y * 16 + x;
            const f32 rate = 25.0f * (2000.0f + 500.0f * x + 4000.0f * y +
                                      100.0f * x * y);
            cpuRate.data[i] = rate;
            rgba[i * 4] = rate;
        }
    auto input = UploadMeasurement(Device(), 16, 16, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 16, 16);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());

    const auto gpuDisplay =
        ReadFloat(Device(), pipeline->GetOutputs().display, 16, 16);
    ASSERT_EQ(gpuDisplay.size(), 16u * 16u * 4u);
    CaptureState state;
    const auto cpu = CpuCameraPipeline(config).CaptureMeasured(state, 0.0, cpuRate);
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(cpu.value().display.has_value());
    const auto& cpuDisplay = cpu.value().display.value().image;
    ASSERT_EQ(cpuDisplay.channels, 3u);

    for (u32 pixel = 0; pixel < 16u * 16u; ++pixel)
        for (u32 c = 0; c < 3; ++c) {
            const double expected = cpuDisplay.data[pixel * 3 + c];
            EXPECT_NEAR(gpuDisplay[pixel * 4 + c], expected,
                        std::max(1e-3, 1e-5 * std::abs(expected)))
                << "pixel " << pixel << " channel " << c;
        }
}

TEST_F(CameraGpuTest, IspBayerDemosaicAndDefectRepairMatchCpu) {
    auto config = CameraConfigFor(8, 8);
    config.device.cfa = CfaPattern::RGGB;
    const auto baseChannel = config.device.channels.front();
    config.device.channels = {baseChannel, baseChannel, baseChannel};
    config.device.channels[0].name = "R";
    config.device.channels[1].name = "G";
    config.device.channels[2].name = "B";
    config.isp.defectPixels = {{3, 3}};

    Image cpuRate(8, 8, 1);
    std::vector<f32> rgba(8u * 8u * 4u, 0.0f);
    for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x) {
            const u32 i = y * 8 + x;
            const f32 rate = 3000.0f + 700.0f * x * x + 900.0f * y +
                             50.0f * x * y;
            cpuRate.data[i] = rate;
            for (u32 channel = 0; channel < 3; ++channel)
                rgba[i * 4 + channel] = rate;
            rgba[i * 4 + 3] = static_cast<f32>(
                (y % 2 == 0) ? (x % 2 == 0 ? 0u : 1u)
                             : (x % 2 == 0 ? 1u : 2u));
        }
    auto input = UploadMeasurement(Device(), 8, 8, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), 8, 8);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());

    const auto gpuDisplay =
        ReadFloat(Device(), pipeline->GetOutputs().display, 8, 8);
    ASSERT_EQ(gpuDisplay.size(), 8u * 8u * 4u);
    CaptureState state;
    const auto cpu = CpuCameraPipeline(config).CaptureMeasured(state, 0.0, cpuRate);
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(cpu.value().display.has_value());
    const auto& cpuDisplay = cpu.value().display.value().image;

    for (u32 pixel = 0; pixel < 8u * 8u; ++pixel)
        for (u32 c = 0; c < 3; ++c) {
            const double expected = cpuDisplay.data[pixel * 3 + c];
            EXPECT_NEAR(gpuDisplay[pixel * 4 + c], expected,
                        std::max(1e-3, 1e-5 * std::abs(expected)))
                << "pixel " << pixel << " channel " << c;
        }
}

TEST_F(CameraGpuTest, IspInfraredLinearAgcMatchesCpuAndRawIsPaletteInvariant) {
    constexpr u32 kSize = 8;
    auto makeRate = [](u32 saturatedFrom = 0xffffffffu) {
        Image rate(kSize, kSize, 1);
        for (u32 i = 0; i < kSize * kSize; ++i)
            rate.data[i] = i >= saturatedFrom ? 5e-8f
                                              : static_cast<f32>(i + 1) * 1e-10f;
        return rate;
    };
    auto configFor = [](DisplayPalette palette) {
        auto config = ThermalConfigFor(kSize, kSize);
        config.isp.infraredTone = DisplayToneMode::Linear;
        config.isp.infraredPalette = palette;
        config.isp.contrastLowPercentile = 0.0;
        config.isp.contrastHighPercentile = 100.0;
        return config;
    };
    const auto cpuRate = makeRate();
    std::vector<f32> rgba(kSize * kSize * 4, 0.0f);
    for (u32 i = 0; i < kSize * kSize; ++i) rgba[i * 4] = cpuRate.data[i];
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());

    // Pass 1: ironbow palette over the Linear AGC window.
    const auto ironbow = configFor(DisplayPalette::Ironbow);
    ASSERT_TRUE(pipeline->Configure(ironbow).has_value());
    ASSERT_TRUE(RecordTwice(Device(), *pipeline, *input, 0, 0.0).has_value());
    const auto firstRaw =
        ReadFloat(Device(), pipeline->GetOutputs().rawDn, kSize, kSize);
    const auto firstCorrected =
        ReadFloat(Device(), pipeline->GetOutputs().corrected, kSize, kSize);
    const auto firstAgc =
        ReadFloat(Device(), pipeline->GetOutputs().agcSource, kSize, kSize);
    const auto firstDisplay =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    ASSERT_EQ(firstDisplay.size(), kSize * kSize * 4u);

    // The AGC source is the pre-AGC scalar: exactly the corrected DN.
    for (u32 pixel = 0; pixel < kSize * kSize; ++pixel)
        EXPECT_FLOAT_EQ(firstAgc[pixel * 4], firstCorrected[pixel * 4])
            << "pixel " << pixel;

    CaptureState state;
    const auto cpu =
        CpuCameraPipeline(ironbow).CaptureMeasured(state, 0.0, cpuRate);
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(cpu.value().display.has_value());
    const auto& cpuDisplay = cpu.value().display.value().image;
    for (u32 pixel = 0; pixel < kSize * kSize; ++pixel)
        for (u32 c = 0; c < 3; ++c) {
            const double expected = cpuDisplay.data[pixel * 3 + c];
            EXPECT_NEAR(firstDisplay[pixel * 4 + c], expected,
                        std::max(1e-3, 1e-5 * std::abs(expected)))
                << "pixel " << pixel << " channel " << c;
        }

    // Pass 2: a different palette changes the display but must not touch the
    // RAW or corrected products.
    const auto inverted = configFor(DisplayPalette::GreyInverted);
    ASSERT_TRUE(pipeline->Configure(inverted).has_value());
    ASSERT_TRUE(RecordTwice(Device(), *pipeline, *input, 0, 0.0).has_value());
    const auto secondRaw =
        ReadFloat(Device(), pipeline->GetOutputs().rawDn, kSize, kSize);
    const auto secondCorrected =
        ReadFloat(Device(), pipeline->GetOutputs().corrected, kSize, kSize);
    const auto secondDisplay =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    ASSERT_EQ(secondRaw.size(), firstRaw.size());
    ASSERT_EQ(secondDisplay.size(), firstDisplay.size());
    for (size_t i = 0; i < firstRaw.size(); ++i) {
        EXPECT_FLOAT_EQ(secondRaw[i], firstRaw[i]) << "raw slot " << i;
        EXPECT_FLOAT_EQ(secondCorrected[i], firstCorrected[i])
            << "corrected slot " << i;
    }
    size_t changed = 0;
    for (size_t i = 0; i < firstDisplay.size(); ++i)
        if (secondDisplay[i] != firstDisplay[i]) ++changed;
    EXPECT_GT(changed, firstDisplay.size() / 2u);
}

TEST_F(CameraGpuTest, IspInfraredEqualizeMatchesCpu) {
    constexpr u32 kSize = 8;
    auto config = ThermalConfigFor(kSize, kSize);
    config.isp.infraredTone = DisplayToneMode::Equalize;
    config.isp.infraredPalette = DisplayPalette::Viridis;

    Image cpuRate(kSize, kSize, 1);
    std::vector<f32> rgba(kSize * kSize * 4, 0.0f);
    for (u32 i = 0; i < kSize * kSize; ++i) {
        cpuRate.data[i] = static_cast<f32>(i + 1) * 1e-10f;
        rgba[i * 4] = cpuRate.data[i];
    }
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    // Equalize consumes this frame's own statistics; one pass suffices, the
    // second keeps the same-tick convergence contract honest.
    ASSERT_TRUE(RecordTwice(Device(), *pipeline, *input, 0, 0.0).has_value());

    const auto gpuDisplay =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    ASSERT_EQ(gpuDisplay.size(), kSize * kSize * 4u);
    CaptureState state;
    const auto cpu = CpuCameraPipeline(config).CaptureMeasured(state, 0.0, cpuRate);
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(cpu.value().display.has_value());
    const auto& cpuDisplay = cpu.value().display.value().image;
    for (u32 pixel = 0; pixel < kSize * kSize; ++pixel)
        for (u32 c = 0; c < 3; ++c) {
            const double expected = cpuDisplay.data[pixel * 3 + c];
            EXPECT_NEAR(gpuDisplay[pixel * 4 + c], expected,
                        std::max(1e-3, 1e-5 * std::abs(expected)))
                << "pixel " << pixel << " channel " << c;
        }
}

TEST_F(CameraGpuTest, IspStatsCountSaturationAndHistogram) {
    constexpr u32 kSize = 8;
    auto config = ThermalConfigFor(kSize, kSize);
    constexpr u32 kSaturatedFrom = 56; // last 8 pixels deliberately hot
    Image cpuRate(kSize, kSize, 1);
    std::vector<f32> rgba(kSize * kSize * 4, 0.0f);
    for (u32 i = 0; i < kSize * kSize; ++i) {
        cpuRate.data[i] = i >= kSaturatedFrom ? 5e-8f
                                              : static_cast<f32>(i + 1) * 1e-10f;
        rgba[i * 4] = cpuRate.data[i];
    }
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());

    const auto stats = pipeline->ReadIspStats();
    ASSERT_TRUE(stats.has_value()) << stats.error();
    EXPECT_EQ(stats.value().saturatedCount, kSize * kSize - kSaturatedFrom);
    EXPECT_EQ(stats.value().unsaturatedCount, kSaturatedFrom);
    const u32 histogramTotal =
        std::accumulate(stats.value().histogram.begin(),
                        stats.value().histogram.end(), 0u);
    EXPECT_EQ(histogramTotal, stats.value().unsaturatedCount);

    // The expected scalar is the CPU corrected product after one thermal
    // step of tau at t=0.
    CaptureState state;
    const auto cpu = CpuCameraPipeline(config).CaptureMeasured(state, 0.0, cpuRate);
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(cpu.value().correctedDeviceSignal.has_value());
    const auto& corrected = cpu.value().correctedDeviceSignal.value().image;
    double expectedMin = std::numeric_limits<double>::max(), expectedMax = 0.0;
    double expectedSum = 0.0;
    for (u32 i = 0; i < kSaturatedFrom; ++i) {
        const double v = corrected.data[i];
        expectedMin = std::min(expectedMin, v);
        expectedMax = std::max(expectedMax, v);
        expectedSum += v;
    }
    EXPECT_NEAR(stats.value().minValue, expectedMin, 2.0);
    EXPECT_NEAR(stats.value().maxValue, expectedMax, 2.0);
    EXPECT_NEAR(stats.value().meanValue, expectedSum / kSaturatedFrom,
                (expectedMax - expectedMin) * 0.05 + 1.0);
}

TEST_F(CameraGpuTest, IspClahePersistentTonePassesAgcScalarThrough) {
    constexpr u32 kSize = 8;
    auto config = ThermalConfigFor(kSize, kSize);
    config.isp.infraredTone = DisplayToneMode::Clahe;
    config.isp.infraredPalette = DisplayPalette::Ironbow;

    Image cpuRate(kSize, kSize, 1);
    std::vector<f32> rgba(kSize * kSize * 4, 0.0f);
    for (u32 i = 0; i < kSize * kSize; ++i) {
        cpuRate.data[i] = static_cast<f32>(i + 1) * 1e-10f;
        rgba[i * 4] = cpuRate.data[i];
    }
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);
    auto created = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());

    // Persistent CLAHE mode: no tone or palette in the camera chain; the
    // display and agcSource both carry the raw scalar for the host's CLAHE
    // pipeline to consume.
    const auto display =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    const auto agc =
        ReadFloat(Device(), pipeline->GetOutputs().agcSource, kSize, kSize);
    const auto corrected =
        ReadFloat(Device(), pipeline->GetOutputs().corrected, kSize, kSize);
    ASSERT_EQ(display.size(), kSize * kSize * 4u);
    for (u32 pixel = 0; pixel < kSize * kSize; ++pixel) {
        EXPECT_FLOAT_EQ(display[pixel * 4], corrected[pixel * 4]);
        EXPECT_FLOAT_EQ(agc[pixel * 4], corrected[pixel * 4]);
    }
}

TEST_F(CameraSchedulerGpuTest, PersistentClaheToneDrivesTheViewportDisplay) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    auto config = ThermalConfigFor(kSensorSize, kSensorSize);
    config.isp.infraredTone = DisplayToneMode::Clahe;
    config.isp.infraredPalette = DisplayPalette::Ironbow;
    const auto enabled = context->SetCameraConfig(config);
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    DrawFrame();
    DrawFrame(); // same-tick redraw: convergence, no state advance
    auto display = context->CaptureDisplayImage();
    ASSERT_TRUE(display.has_value()) << display.error();
    EXPECT_EQ(display.value().width, kSensorSize);
    EXPECT_EQ(display.value().height, kSensorSize);
    for (const f32 value : display.value().data)
        EXPECT_TRUE(std::isfinite(value));

    // The ISP segment of the GPU timing breakdown is wired to the stamps.
    const auto timings = context->GetLastCameraGpuTimings();
    EXPECT_TRUE(timings.valid);
    EXPECT_GE(timings.ispMs, 0.0f);
}

// ---------------------------------------------------------------------------
// M4-4: HSV display grading, AE/AWB closed loop, display reprocess, and the
// five-segment GPU timing breakdown.
//
// HSV identity is cross-checked against the CPU chain: with every HSV
// parameter at its default the GPU never dispatches the pass, so the display
// must match the CPU RunIsp output (which skips ApplyHsv the same way) to the
// standard ISP tolerance. RAW and corrected products must never see the HSV
// stage: they are compared bitwise.
// ---------------------------------------------------------------------------

TEST_F(CameraGpuTest, HsvDefaultsAreIdentityAndGradingChangesDisplayOnly) {
    constexpr u32 kSize = 16;
    std::vector<f32> rgba;
    FillBayerMeasurement(kSize, kSize, rgba);
    Image cpuRate(kSize, kSize, 1);
    for (u32 i = 0; i < kSize * kSize; ++i)
        cpuRate.data[i] = rgba[i * 4];
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);

    auto makeConfig = [] {
        auto config = BayerColorConfigFor(kSize, kSize);
        // Chromatic white balance: without it the achromatic scene would
        // leave the hue offset with nothing to rotate.
        config.isp.whiteBalance = {1.3, 1.0, 0.65};
        return config;
    };
    const auto defaultConfig = makeConfig();
    auto hueConfig = makeConfig();
    hueConfig.isp.hsv.hueOffsetDegrees = 40.0;

    auto defaultPipeline = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(defaultPipeline.has_value());
    ASSERT_TRUE(defaultPipeline.value()->Configure(defaultConfig).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *defaultPipeline.value(), *input, 0, 0.0)
                    .has_value());

    auto huePipeline = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(huePipeline.has_value());
    ASSERT_TRUE(huePipeline.value()->Configure(hueConfig).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *huePipeline.value(), *input, 0, 0.0)
                    .has_value());

    const auto gpuDefault =
        ReadFloat(Device(), defaultPipeline.value()->GetOutputs().display,
                  kSize, kSize);
    const auto gpuHue = ReadFloat(Device(), huePipeline.value()->GetOutputs().display,
                                  kSize, kSize);
    ASSERT_EQ(gpuDefault.size(), kSize * kSize * 4u);
    ASSERT_EQ(gpuHue.size(), kSize * kSize * 4u);

    CaptureState cpuState;
    const auto cpuDefault =
        CpuCameraPipeline(defaultConfig).CaptureMeasured(cpuState, 0.0, cpuRate);
    ASSERT_TRUE(cpuDefault.has_value());
    ASSERT_TRUE(cpuDefault.value().display.has_value());
    cpuState = CaptureState{};
    const auto cpuHue =
        CpuCameraPipeline(hueConfig).CaptureMeasured(cpuState, 0.0, cpuRate);
    ASSERT_TRUE(cpuHue.has_value());
    ASSERT_TRUE(cpuHue.value().display.has_value());

    // Default HSV: the GPU never ran the pass, so both chains reduce to the
    // plain ISP and agree to the shared tolerance.
    for (u32 pixel = 0; pixel < kSize * kSize; ++pixel)
        for (u32 c = 0; c < 3u; ++c) {
            const double expected =
                cpuDefault.value().display.value().image.data[pixel * 3 + c];
            EXPECT_NEAR(gpuDefault[pixel * 4 + c], expected,
                        std::max(1e-3, 1e-5 * std::abs(expected)))
                << "pixel " << pixel << " channel " << c;
        }

    // The grading pass changed the display...
    u32 changedPixels = 0;
    for (u32 i = 0; i < kSize * kSize * 4u; ++i)
        if (std::abs(gpuHue[i] - gpuDefault[i]) > 1e-4f) ++changedPixels;
    EXPECT_GT(changedPixels, kSize * kSize) << "hue offset changed nothing";

    // ... to exactly what the CPU ApplyHsv produces.
    for (u32 pixel = 0; pixel < kSize * kSize; ++pixel)
        for (u32 c = 0; c < 3u; ++c) {
            const double expected =
                cpuHue.value().display.value().image.data[pixel * 3 + c];
            EXPECT_NEAR(gpuHue[pixel * 4 + c], expected,
                        std::max(1e-3, 1e-5 * std::abs(expected)))
                << "pixel " << pixel << " channel " << c;
        }

    // RAW and corrected never pass through HSV: bitwise identical between
    // the default chain and the graded one.
    for (const GpuImage* product : {defaultPipeline.value()->GetOutputs().rawDn,
                                    defaultPipeline.value()->GetOutputs().corrected}) {
        const auto a = ReadFloat(Device(), product, kSize, kSize);
        const GpuImage* graded = product == defaultPipeline.value()->GetOutputs().rawDn
            ? huePipeline.value()->GetOutputs().rawDn
            : huePipeline.value()->GetOutputs().corrected;
        const auto b = ReadFloat(Device(), graded, kSize, kSize);
        ASSERT_EQ(a.size(), b.size());
        for (u32 i = 0; i < a.size(); ++i)
            EXPECT_FLOAT_EQ(a[i], b[i]) << "element " << i;
    }
}

TEST_F(CameraGpuTest, HsvNoiseIsDeterministicPerAcquisitionAndKeysOnIndex) {
    constexpr u32 kSize = 16;
    std::vector<f32> rgba;
    FillBayerMeasurement(kSize, kSize, rgba);
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);

    auto config = BayerColorConfigFor(kSize, kSize);
    config.isp.hsv.empiricalNoise = true;
    config.isp.hsv.temporalDrift = true;
    auto created = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());

    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 5, 0.0).has_value());
    const auto first =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    const auto rawFirst =
        ReadFloat(Device(), pipeline->GetOutputs().rawDn, kSize, kSize);
    // Same acquisition index, same tick: a same-tick reprocess must be
    // bit-identical.
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 5, 0.0).has_value());
    const auto repeated =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    ASSERT_EQ(first.size(), repeated.size());
    for (u32 i = 0; i < first.size(); ++i)
        EXPECT_FLOAT_EQ(first[i], repeated[i]) << "element " << i;

    // The next acquisition draws a different pattern from both streams.
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 6, 0.1).has_value());
    const auto next =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    u32 changedPixels = 0;
    for (u32 i = 0; i < first.size(); ++i)
        if (std::abs(next[i] - first[i]) > 1e-4f) ++changedPixels;
    EXPECT_GT(changedPixels, kSize) << "acquisition 6 repeated the pattern";

    // The empirical effects are display-only: with the noise-free detector
    // chain the RAW product is the same quantized readout no matter which
    // acquisition drew the display noise.
    const auto rawNext =
        ReadFloat(Device(), pipeline->GetOutputs().rawDn, kSize, kSize);
    ASSERT_EQ(rawFirst.size(), rawNext.size());
    for (u32 i = 0; i < rawFirst.size(); ++i)
        EXPECT_FLOAT_EQ(rawFirst[i], rawNext[i]) << "element " << i;
}

TEST_F(CameraGpuTest, DisplayReprocessIsStatelessAndAppliesNewDisplayParams) {
    constexpr u32 kSize = 16;
    std::vector<f32> rgba;
    FillBayerMeasurement(kSize, kSize, rgba);
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);
    auto config = BayerColorConfigFor(kSize, kSize);
    config.isp.whiteBalance = {1.2, 1.0, 0.8};
    auto created = GpuCameraPipeline::Create(Device(), kSize, kSize);
    ASSERT_TRUE(created.has_value());
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    ASSERT_TRUE(RecordTwice(Device(), *pipeline, *input, 0, 0.0).has_value());

    const auto display =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    const auto raw =
        ReadFloat(Device(), pipeline->GetOutputs().rawDn, kSize, kSize);
    const auto statsBefore = pipeline->ReadIspStats();
    ASSERT_TRUE(statsBefore.has_value());

    const auto reprocess = [&](const CameraConfig& with) {
        Result<void, String> status = Result<void, String>::Ok();
        CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
            status = pipeline->RecordDisplayReprocess(cmd, with);
        });
        return status;
    };

    // Unchanged parameters: the reprocess is bit-identical and the
    // statistics buffer (the cached AGC window) is untouched.
    ASSERT_TRUE(reprocess(config).has_value());
    const auto displayAgain =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    ASSERT_EQ(display.size(), displayAgain.size());
    for (u32 i = 0; i < display.size(); ++i)
        EXPECT_FLOAT_EQ(display[i], displayAgain[i]) << "element " << i;
    const auto statsAfter = pipeline->ReadIspStats();
    ASSERT_TRUE(statsAfter.has_value());
    EXPECT_FLOAT_EQ(statsBefore.value().minValue, statsAfter.value().minValue);
    EXPECT_FLOAT_EQ(statsBefore.value().maxValue, statsAfter.value().maxValue);
    EXPECT_FLOAT_EQ(statsBefore.value().meanValue, statsAfter.value().meanValue);
    EXPECT_EQ(statsBefore.value().histogram, statsAfter.value().histogram);

    // A white-balance change reaches the display through the reprocess,
    // while RAW/corrected and the statistics stay exactly where they were.
    auto rebalanced = config;
    rebalanced.isp.whiteBalance = {0.7, 1.0, 1.4};
    ASSERT_TRUE(reprocess(rebalanced).has_value());
    const auto displayRebalanced =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    u32 changedPixels = 0;
    for (u32 i = 0; i < display.size(); ++i)
        if (std::abs(displayRebalanced[i] - display[i]) > 1e-4f)
            ++changedPixels;
    EXPECT_GT(changedPixels, kSize) << "rebalanced display is unchanged";
    const auto rawAfter =
        ReadFloat(Device(), pipeline->GetOutputs().rawDn, kSize, kSize);
    ASSERT_EQ(raw.size(), rawAfter.size());
    for (u32 i = 0; i < raw.size(); ++i)
        EXPECT_FLOAT_EQ(raw[i], rawAfter[i]) << "element " << i;

    // And so does an HSV grading change.
    auto graded = config;
    graded.isp.hsv.hueOffsetDegrees = 30.0;
    ASSERT_TRUE(reprocess(graded).has_value());
    const auto displayGraded =
        ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize);
    changedPixels = 0;
    for (u32 i = 0; i < display.size(); ++i)
        if (std::abs(displayGraded[i] - display[i]) > 1e-4f) ++changedPixels;
    EXPECT_GT(changedPixels, kSize) << "graded display is unchanged";
}

TEST_F(CameraSchedulerGpuTest, AutoExposureConvergesAcrossCommittedTicks) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    auto config = CameraConfigFor(kSensorSize, kSensorSize);
    config.readout.framePeriodSeconds = 0.1;
    config.readout.exposureSeconds = 1e-4; // deliberately dim starting point
    config.isp.autoExposure = true;
    config.isp.autoControl = {0.18, 0.6, 1e-5, 0.5, 16.0};
    const auto enabled = context->SetCameraConfig(config);
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    constexpr u32 kAcquisitions = 12;
    std::vector<f64> nextExposure;
    for (u32 i = 0; i < kAcquisitions; ++i) {
        const auto advanced =
            context->AdvanceCameraTo(0.05 + 0.1 * static_cast<f64>(i));
        ASSERT_TRUE(advanced.has_value()) << advanced.error();
        nextExposure.push_back(
            context->GetCameraHistoryStatus().nextExposureSeconds);
    }
    // [0] is the authored value (fresh state); [k] is the feedback computed
    // from tick k-1's statistics during tick k's record.
    ASSERT_EQ(nextExposure.size(), kAcquisitions);
    for (const f64 exposure : nextExposure) {
        EXPECT_GE(exposure, 1e-5 - 1e-12);
        EXPECT_LE(exposure, 0.5 + 1e-12);
    }
    const f64 authored = config.readout.exposureSeconds;
    EXPECT_GT(std::abs(nextExposure.back() - authored), 1e-6)
        << "auto exposure never moved from the authored value";
    const f64 lastStep =
        std::abs(nextExposure.back() - nextExposure[kAcquisitions - 2]);
    EXPECT_LT(lastStep / std::max(nextExposure.back(), 1e-12), 0.05)
        << "auto exposure did not settle: last step " << lastStep;

    // A same-tick re-record must not move the loop: re-queue the committed
    // index and time, record and complete again.
    const auto status = context->GetCameraHistoryStatus();
    const auto queued =
        context->QueueCameraAcquisition(status.acquisitionIndex,
                                        status.frameTimeSeconds);
    ASSERT_TRUE(queued.has_value()) << queued.error();
    Result<void, String> recorded = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        recorded = context->RecordQueuedCameraAcquisition(cmd);
    });
    ASSERT_TRUE(recorded.has_value()) << recorded.error();
    ASSERT_TRUE(context->CompleteQueuedCameraAcquisition().has_value());
    EXPECT_DOUBLE_EQ(context->GetCameraHistoryStatus().nextExposureSeconds,
                     nextExposure.back());
}

TEST_F(CameraSchedulerGpuTest, AutoWhiteBalanceConvergesOnTintedScene) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    // A Bayer camera whose three CFA responses are flat over the same span
    // at 1.0 / 0.25 / 0.5: the scene spectrum cancels in the grey-world
    // ratio, so the measured channel means are exactly 4:1:2 and the
    // balanced gains are 1/4 : 1 : 1/2 no matter what the scene radiates.
    auto config = CameraConfigFor(kSensorSize, kSensorSize);
    config.device.cfa = CfaPattern::RGGB;
    config.device.channels.clear();
    const auto makeQe = [](f64 value) {
        ResponseCurve qe;
        qe.kind = ResponseKind::AbsoluteQE;
        qe.wavelengthNm = {420.0, 700.0};
        qe.value = {value, value};
        return qe;
    };
    ResponseStack red, green, blue;
    red.quantumEfficiency = makeQe(1.0);
    green.quantumEfficiency = makeQe(0.25);
    blue.quantumEfficiency = makeQe(0.5);
    config.device.channels.push_back({"R", red});
    config.device.channels.push_back({"G", green});
    config.device.channels.push_back({"B", blue});
    config.readout.framePeriodSeconds = 0.1;
    // Short exposure: saturation would exclude a CFA class from the
    // unsaturated means and distort the ratio the loop balances.
    config.readout.exposureSeconds = 1e-6;
    config.isp.autoWhiteBalance = true;
    config.isp.autoControl.smoothing = 0.5;
    const auto enabled = context->SetCameraConfig(config);
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    constexpr u32 kAcquisitions = 10;
    std::array<f64, 3> whiteBalance{1.0, 1.0, 1.0};
    for (u32 i = 0; i < kAcquisitions; ++i) {
        const auto advanced =
            context->AdvanceCameraTo(0.05 + 0.1 * static_cast<f64>(i));
        ASSERT_TRUE(advanced.has_value()) << advanced.error();
        whiteBalance =
            context->GetCameraHistoryStatus().nextWhiteBalance;
    }
    // Grey world: the bright R channel's gain falls, the dark B channel's
    // rises, G is the reference and stays at unity.
    EXPECT_LT(whiteBalance[0], 0.4) << "R gain did not fall: " << whiteBalance[0];
    EXPECT_DOUBLE_EQ(whiteBalance[1], 1.0);
    EXPECT_LT(whiteBalance[2], 0.75) << "B gain did not rise: " << whiteBalance[2];
    EXPECT_GT(whiteBalance[2], whiteBalance[0]);
}

TEST_F(CameraSchedulerGpuTest, CameraGpuTimingsReportAllFiveStages) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    auto config = CameraConfigFor(kSensorSize, kSensorSize);
    config.isp.hsv.hueOffsetDegrees = 25.0; // force the HSV pass to run
    config.isp.hsv.empiricalNoise = true;
    const auto enabled = context->SetCameraConfig(config);
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    DrawFrame();
    DrawFrame(); // same-tick convergence pass
    const auto timings = context->GetLastCameraGpuTimings();
    EXPECT_TRUE(timings.valid);
    EXPECT_GT(timings.traceMs, 0.0f);
    EXPECT_GE(timings.postMs, 0.0f);
    EXPECT_GE(timings.dynamicMs, 0.0f);
    EXPECT_GE(timings.psfMs, 0.0f);
    EXPECT_GT(timings.detectorMs, 0.0f);
    EXPECT_GT(timings.ispMs, 0.0f);
    EXPECT_GT(timings.hsvMs, 0.0f);
    const f32 stages = timings.dynamicMs + timings.psfMs + timings.detectorMs +
                       timings.ispMs + timings.hsvMs;
    EXPECT_LE(stages, timings.postMs + 0.05f)
        << "stage spans " << stages << " exceed the post window "
        << timings.postMs;
}

TEST_F(CameraSchedulerGpuTest, ReprocessCameraDisplayRefreshesWithoutAdvancing) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    auto config = CameraConfigFor(kSensorSize, kSensorSize);
    config.isp.hsv.hueOffsetDegrees = 20.0;
    const auto enabled = context->SetCameraConfig(config);
    ASSERT_TRUE(enabled.has_value()) << enabled.error();

    DrawFrame();
    const auto before = context->GetCameraHistoryStatus();
    const auto products = context->CaptureCameraProducts();
    ASSERT_TRUE(products.has_value()) << products.error();
    ASSERT_TRUE(products.value().display.has_value());
    const std::vector<f32> displayBefore =
        products.value().display.value().image.data;
    ASSERT_TRUE(products.value().rawDn.has_value());
    const std::vector<f32> rawBefore = products.value().rawDn.value().image.data;

    // Studio keeps the authored zero effective-span fields; SetCameraConfig
    // resolves them inside the SDK. The display path must accept that copy.
    auto displayConfig = config;
    displayConfig.isp.toneGamma = 2.0;
    const auto updated = context->UpdateCameraDisplayConfig(displayConfig);
    ASSERT_TRUE(updated.has_value()) << updated.error();
    EXPECT_EQ(context->GetCameraHistoryStatus().acquisitionIndex,
              before.acquisitionIndex);

    auto readoutChange = displayConfig;
    readoutChange.readout.exposureSeconds *= 2.0;
    EXPECT_FALSE(context->UpdateCameraDisplayConfig(readoutChange).has_value());

    const auto reprocessed = context->ReprocessCameraDisplay();
    ASSERT_TRUE(reprocessed.has_value()) << reprocessed.error();
    const auto productsAgain = context->CaptureCameraProducts();
    ASSERT_TRUE(productsAgain.has_value()) << productsAgain.error();
    ASSERT_TRUE(productsAgain.value().display.has_value());
    ASSERT_TRUE(productsAgain.value().rawDn.has_value());
    const std::vector<f32>& displayAfter =
        productsAgain.value().display.value().image.data;
    EXPECT_EQ(productsAgain.value().rawDn.value().image.data, rawBefore);
    ASSERT_EQ(displayBefore.size(), displayAfter.size());
    EXPECT_NE(displayBefore, displayAfter);

    const auto sameAgain = context->ReprocessCameraDisplay();
    ASSERT_TRUE(sameAgain.has_value()) << sameAgain.error();
    const auto productsThird = context->CaptureCameraProducts();
    ASSERT_TRUE(productsThird.has_value()) << productsThird.error();
    EXPECT_EQ(productsThird.value().display.value().image.data, displayAfter);

    const auto after = context->GetCameraHistoryStatus();
    EXPECT_EQ(after.acquisitionIndex, before.acquisitionIndex);
    EXPECT_EQ(after.epoch, before.epoch);

    // The same reprocess also runs inside ReprocessAccumulated's camera
    // branch, which must stay a valid present.
    bool presented = false;
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        presented = context->ReprocessAccumulated(
            cmd, target->GetImage(), VK_IMAGE_LAYOUT_UNDEFINED,
            kViewportSize, kViewportSize);
    });
    EXPECT_TRUE(presented);
}

TEST_F(CameraSchedulerGpuTest, ReadoutEditRerecordsSameTickWithoutHistoryReset) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
    ApplyScene();
    auto config = CameraConfigFor(kSensorSize, kSensorSize);
    ASSERT_TRUE(context->SetCameraConfig(config).has_value());
    DrawFrame();
    const auto before = context->GetCameraHistoryStatus();
    const auto first = context->CaptureCameraProducts();
    ASSERT_TRUE(first.has_value()) << first.error();
    ASSERT_TRUE(first.value().rawDn.has_value());
    const auto rawBefore = first.value().rawDn->image.data;
    const u32 samplesBefore = context->GetAccumulatedSamples();

    auto readout = config; // authored effective span is still zero
    readout.readout.blackLevelDn += 256.0;
    const auto updated = context->TryUpdateCameraReadoutConfig(readout);
    ASSERT_TRUE(updated.has_value()) << updated.error();
    ASSERT_TRUE(updated.value());
    EXPECT_EQ(context->GetAccumulatedSamples(), samplesBefore);
    const auto reprocessed = context->CaptureCameraProducts();
    ASSERT_TRUE(reprocessed.has_value()) << reprocessed.error();
    ASSERT_TRUE(reprocessed.value().rawDn.has_value());
    EXPECT_NE(reprocessed.value().rawDn->image.data, rawBefore);
    auto exposure = readout;
    exposure.readout.exposureSeconds *= 2.0;
    const auto needsMeasurement = context->TryUpdateCameraReadoutConfig(exposure);
    ASSERT_TRUE(needsMeasurement.has_value()) << needsMeasurement.error();
    EXPECT_FALSE(needsMeasurement.value());

    const auto after = context->GetCameraHistoryStatus();
    EXPECT_EQ(after.epoch, before.epoch);
    EXPECT_EQ(after.acquisitionIndex, before.acquisitionIndex);
    EXPECT_EQ(context->GetAccumulatedSamples(), samplesBefore);
}


TEST_F(CameraSchedulerGpuTest, ViewportBatchDoesNotMultiplyPhysicalAcquisitions) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf unavailable";
    ApplyScene();
    ASSERT_TRUE(EnableCamera().has_value());
    context->SetViewportSampleBatch(16);
    DrawFrame();
    EXPECT_EQ(context->GetAccumulatedSamples(), 1u);
    const auto first = context->GetCameraHistoryStatus();
    EXPECT_EQ(first.acquisitionIndex, 0u);
    ASSERT_TRUE(context->SetTimelineTime(1.0 / 30.0).has_value());
    DrawFrame();
    const auto second = context->GetCameraHistoryStatus();
    EXPECT_EQ(second.acquisitionIndex, 1u);
    EXPECT_EQ(second.epoch, first.epoch);
}
