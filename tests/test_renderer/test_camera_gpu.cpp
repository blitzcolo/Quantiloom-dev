#include <gtest/gtest.h>

#include "postprocess/CpuCameraPipeline.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuCameraPipeline.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/VulkanContext.hpp"
#include "support/VulkanTestDevice.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

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
