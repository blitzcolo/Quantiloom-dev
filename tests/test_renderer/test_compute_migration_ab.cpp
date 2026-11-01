// ============================================================================
// Quantiloom - ComputeMigrationAB: bit-exact output hashes across the P0.3 migration
// ============================================================================
// Each case runs one compute module on fixed, seeded inputs and hashes every output
// byte with SHA-256. The suite skips unless QUANTILOOM_GPU_AB is set:
//
//   QUANTILOOM_GPU_AB=record:<file>    merge this run's "case sha256" lines into <file>
//   QUANTILOOM_GPU_AB=compare:<file>   fail on any case whose hash differs from <file>
//
// Record twice on the pre-migration binary; the two files must be equal (run-to-run
// determinism). Later steps edit only the call syntax of a case, never its inputs.
// ============================================================================

#include <gtest/gtest.h>

#include "support/FacadeScene.hpp"
#include "support/VulkanTestDevice.hpp"

#include "gpu/Device.hpp"
#include "gpu/vulkan/VulkanCommandList.hpp"
#include "gpu/vulkan/VulkanResources.hpp"

#include "core/Sha256.hpp"
#include "core/SpectralCube.hpp"
#include "hs_core/GpuSpectralReconstructor.hpp"
#include "postprocess/CameraPhysics.hpp"
#include "gpu/vulkan/CommandHelper.hpp"
#include "renderer/ExternalRenderContext.hpp"
#include "gpu/vulkan/GpuBuffer.hpp"
#include "renderer/GpuCameraPipeline.hpp"
#include "renderer/GpuDisplayRange.hpp"
#include "gpu/vulkan/GpuImage.hpp"
#include "renderer/GpuThermalStepper.hpp"
#include "gpu/vulkan/VulkanContext.hpp"
#include "thermal/ThermalSolver.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>

using namespace quantiloom;
using namespace quantiloom::camera;
using namespace quantiloom::rendercore;
using quantiloom::testing::VulkanDeviceTest;

namespace {

struct AbMode {
    enum class Kind { Off, Record, Compare } kind = Kind::Off;
    std::string file;
};

AbMode ReadMode() {
    AbMode mode;
    const char* raw = std::getenv("QUANTILOOM_GPU_AB");
    if (raw == nullptr) return mode;
    const std::string value(raw);
    if (value.rfind("record:", 0) == 0) {
        mode.kind = AbMode::Kind::Record;
        mode.file = value.substr(7);
    } else if (value.rfind("compare:", 0) == 0) {
        mode.kind = AbMode::Kind::Compare;
        mode.file = value.substr(8);
    }
    return mode;
}

std::map<std::string, std::string> LoadRecords(const std::string& path) {
    std::map<std::string, std::string> records;
    std::ifstream in(path);
    std::string name;
    std::string hash;
    while (in >> name >> hash) records[name] = hash;
    return records;
}

void Check(const AbMode& mode, const std::string& name, const std::string& hash) {
    if (mode.kind == AbMode::Kind::Record) {
        auto records = LoadRecords(mode.file);
        records[name] = hash;
        std::ofstream out(mode.file, std::ios::trunc);
        for (const auto& [key, value] : records) out << key << ' ' << value << '\n';
        ASSERT_TRUE(out.good()) << "cannot write " << mode.file;
        return;
    }
    const auto records = LoadRecords(mode.file);
    const auto it = records.find(name);
    ASSERT_TRUE(it != records.end()) << name << " is not in " << mode.file;
    EXPECT_EQ(it->second, hash) << name;
}

template <class T>
std::string HashOf(const std::vector<T>& values) {
    core::Sha256 sha;
    sha.UpdateSized(values.data(), values.size() * sizeof(T));
    return sha.FinalizeHex();
}

class ComputeMigrationAB : public VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (IsSkipped()) return;
        mode_ = ReadMode();
        if (mode_.kind == AbMode::Kind::Off) {
            GTEST_SKIP() << "set QUANTILOOM_GPU_AB=record:<file> or compare:<file>";
        }
    }
    AbMode mode_;
};

std::unique_ptr<GpuImage> UploadRgba32f(VulkanContext& context, u32 width, u32 height,
                                        const std::vector<f32>& rgba) {
    auto image = std::make_unique<GpuImage>(
        context.GetAllocator(), context.GetDevice(), width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
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
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &ready);
    });
    return image;
}

}  // namespace

TEST_F(ComputeMigrationAB, SpectralReconstruct) {
    constexpr u32 kWidth = 37;
    constexpr u32 kHeight = 23;
    const std::vector<f32> sourceWavelengths{400.f, 430.f, 490.f, 560.f, 600.f, 680.f, 750.f, 820.f, 900.f};
    SpectralCube sparse(kWidth, kHeight, static_cast<u32>(sourceWavelengths.size()), 400.0f, 900.0f);
    sparse.wavelengths = sourceWavelengths;
    std::mt19937 rng(1);
    std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
    for (f32& v : sparse.data) v = dist(rng);

    std::vector<f32> targets(64);
    for (usize i = 0; i < targets.size(); ++i) {
        targets[i] = 400.0f + 500.0f * static_cast<f32>(i) / static_cast<f32>(targets.size() - 1);
    }

    GpuSpectralReconstructor reconstructor(GpuDevice());
    if (reconstructor.Initialize() != GpuReconstructorStatus::Success) {
        GTEST_SKIP() << "spectral_reconstruct shader unavailable";
    }
    const std::pair<const char*, InterpolationMethod> methods[] = {
        {"Linear", InterpolationMethod::Linear},
        {"CatmullRom", InterpolationMethod::CatmullRom},
        {"Akima", InterpolationMethod::Akima},
    };
    for (const auto& [label, method] : methods) {
        GpuReconstructorConfig config;
        config.method = method;
        auto [status, result] = reconstructor.ReconstructCustom(sparse, targets, config);
        ASSERT_EQ(status, GpuReconstructorStatus::Success) << label;
        Check(mode_, std::string("SpectralReconstruct.") + label, HashOf(result.data));
    }
}

TEST_F(ComputeMigrationAB, ThermalStep) {
    using namespace quantiloom::thermal;
    using quantiloom::rendercore::GpuThermalStepper;

    GpuThermalStepper stepper(GpuDevice());
    if (!stepper.IsValid()) GTEST_SKIP() << "thermal_step.spv unavailable";

    Vector<ThermalElement> elements(2);
    for (auto& el : elements) {
        el.area_m2 = 1.0f;
        el.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        el.materialId = 0;
    }
    ThermalElement inert;
    inert.area_m2 = 1.0f;
    inert.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    inert.materialId = 1;
    elements.push_back(inert);

    ThermalMaterial concrete;
    concrete.conductivity_W_mK = 1.4f;
    concrete.density_kg_m3 = 2300.0f;
    concrete.specificHeat_J_kgK = 880.0f;
    concrete.thickness_m = 0.2f;
    concrete.convection_W_m2K = 10.0f;
    concrete.shortwaveAbsorptivity = 0.6f;
    concrete.longwaveEmissivity = 0.92f;
    ThermalMaterial inertMat;
    inertMat.conductivity_W_mK = 0.0f;
    const Vector<ThermalMaterial> materials{concrete, inertMat};

    ExchangeGeometry exchange;
    exchange.viewFactors.rowStart = {0, 1, 2, 2};
    exchange.viewFactors.column = {1, 0};
    exchange.viewFactors.value = {0.3f, 0.3f};
    exchange.skyFraction = {0.7f, 0.7f, 1.0f};
    exchange.sunVisibility = {1.0f, 0.5f, 1.0f};

    SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0};
    sunTable.visibility = exchange.sunVisibility;
    sunTable.reflectedGain = {0.2f, 0.1f, 0.0f};
    sunTable.diffuseGain = {0.8f, 0.75f, 1.0f};

    constexpr u32 kNodes = 10;
    ThermalState state;
    state.nodeCount = kNodes;
    state.temperature_K.assign(3 * kNodes, 290.0);
    state.sunSensitivity_K.assign(3 * kNodes, 0.0);

    ThermalForcing forcing;
    forcing.airTemperature_K = 293.15;
    forcing.sunIrradiance_W_m2 = 900.0;
    forcing.diffuseIrradiance_W_m2 = 150.0;
    forcing.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    forcing.skyTemperature_K = 268.0;

    constexpr i32 kBatchSteps = 300;
    for (int batchIndex = 0; batchIndex < 2; ++batchIndex) {
        Vector<ThermalBatchStep> batch(kBatchSteps);
        for (auto& b : batch) {
            b.forcing = forcing;
            b.dt_s = 60.0;
        }
        stepper.StepMany(state, elements, materials, exchange, sunTable, batch);
    }

    Check(mode_, "ThermalStep.temperature_K", HashOf(state.temperature_K));
    Check(mode_, "ThermalStep.sunSensitivity_K", HashOf(state.sunSensitivity_K));
    Check(mode_, "ThermalStep.longwaveIrradiance_W_m2", HashOf(state.longwaveIrradiance_W_m2));
}

TEST_F(ComputeMigrationAB, DisplayRange) {
    constexpr u32 kWidth = 257;
    constexpr u32 kHeight = 131;
    std::vector<f32> rgba(static_cast<usize>(kWidth) * kHeight * 4);
    std::mt19937 rng(7);
    std::uniform_real_distribution<f32> dist(-2.0f, 40.0f);
    for (f32& v : rgba) v = dist(rng);
    auto image = UploadRgba32f(Device(), kWidth, kHeight, rgba);
    ASSERT_TRUE(image->IsValid());

    auto created = rendercore::GpuDisplayRange::Create(GpuDevice());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());

    const std::pair<f32, f32> percentiles[] = {{1.0f, 99.0f}, {0.0f, 100.0f}, {5.0f, 95.0f}};
    std::vector<f32> windows;
    for (const auto& [low, high] : percentiles) {
        const auto range = gpu->Compute(gpu::vulkan::Ref(*image), kWidth, kHeight, low, high);
        ASSERT_TRUE(range.has_value()) << range.error();
        windows.push_back(range.value().min);
        windows.push_back(range.value().max);
    }
    Check(mode_, "DisplayRange.windows", HashOf(windows));
}

// ----------------------------------------------------------------------------
// Camera pipeline and CLAHE facade cases
// ----------------------------------------------------------------------------

namespace {

CameraConfig CameraConfigFor(u32 width, u32 height) {
    CameraConfig config;
    config.enabled = true;
    config.device.id = "gpu_ab_fixture";
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
    config.device.id = "gpu_ab_thermal_fixture";
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

void FillBayerMeasurement(u32 width, u32 height, std::vector<f32>& rgba, f32 baseRate = 3000.0f) {
    rgba.assign(static_cast<size_t>(width) * height * 4u, 0.0f);
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            const u32 i = y * width + x;
            const f32 rate = baseRate + 40.0f * x + 25.0f * y;
            rgba[i * 4 + 0] = rate;
            rgba[i * 4 + 1] = rate;
            rgba[i * 4 + 2] = rate;
            rgba[i * 4 + 3] = static_cast<f32>((y % 2 == 0) ? (x % 2 == 0 ? 0u : 1u) : (x % 2 == 0 ? 1u : 2u));
        }
    }
}

std::unique_ptr<GpuImage> UploadMeasurement(VulkanContext& context, u32 width, u32 height,
                                            const std::vector<f32>& rgba) {
    const VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    auto image = std::make_unique<GpuImage>(
        context.GetAllocator(), context.GetDevice(), width, height, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    if (!image->IsValid() || rgba.size() != static_cast<size_t>(width) * height * 4) return {};
    const VkDeviceSize bytes = rgba.size() * sizeof(f32);
    GpuBuffer staging(context.GetAllocator(), bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    if (!staging.IsValid()) return {};
    staging.Upload(rgba.data(), bytes);
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        CommandHelper::TransitionImageLayout(cmd, image->GetImage(), format, VK_IMAGE_LAYOUT_UNDEFINED,
                                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, staging.GetHandle(), image->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);
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
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &ready);
    });
    return image;
}

std::unique_ptr<GpuImage> UploadStrataImage(VulkanContext& context, u32 width, u32 height, u32 layers,
                                            VkFormat format, const std::vector<f32>& texelData,
                                            u32 componentsPerTexel) {
    auto image = std::make_unique<GpuImage>(
        context.GetAllocator(), context.GetDevice(), width, height, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY, 1u, layers, 0, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    if (!image->IsValid() ||
        texelData.size() != static_cast<size_t>(width) * height * layers * componentsPerTexel) {
        return {};
    }
    const VkDeviceSize bytes = texelData.size() * sizeof(f32);
    GpuBuffer staging(context.GetAllocator(), bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    if (!staging.IsValid()) return {};
    staging.Upload(texelData.data(), bytes);
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        CommandHelper::TransitionImageLayout(cmd, image->GetImage(), format, VK_IMAGE_LAYOUT_UNDEFINED,
                                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, layers);
        for (u32 layer = 0; layer < layers; ++layer) {
            VkBufferImageCopy region{};
            region.bufferOffset = static_cast<VkDeviceSize>(layer) * width * height * componentsPerTexel * sizeof(f32);
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = layer;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {width, height, 1};
            vkCmdCopyBufferToImage(cmd, staging.GetHandle(), image->GetImage(),
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        VkImageMemoryBarrier ready{};
        ready.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        ready.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ready.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.image = image->GetImage();
        ready.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ready.subresourceRange.levelCount = 1;
        ready.subresourceRange.layerCount = layers;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &ready);
    });
    return image;
}

Result<void, String> RecordCamera(VulkanContext& context, GpuCameraPipeline& pipeline, const GpuImage& measurement,
                                  u64 acquisitionIndex, f64 timeSeconds) {
    Result<void, String> status = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        auto list = gpu::vulkan::Wrap(cmd);
        status = pipeline.RecordMeasurement(list, gpu::vulkan::Ref(measurement),
                                            acquisitionIndex, timeSeconds);
    });
    return status;
}

std::vector<f32> ReadFloat(VulkanContext& context, gpu::TextureRef image, u32 width, u32 height) {
    if (!image.IsValid()) return {};
    return CommandHelper::ReadbackImage(context, gpu::vulkan::NativeImage(image),
                                        gpu::vulkan::ToVk(image.format), width, height);
}

std::vector<u32> ReadUint(VulkanContext& context, gpu::TextureRef image, u32 width, u32 height) {
    if (!image.IsValid() || image.format != gpu::Format::RGBA32Uint) return {};
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(u32);
    GpuBuffer staging(context.GetAllocator(), bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    if (!staging.IsValid()) return {};
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        CommandHelper::TransitionImageLayout(cmd, gpu::vulkan::NativeImage(image),
                                             gpu::vulkan::ToVk(image.format), VK_IMAGE_LAYOUT_GENERAL,
                                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, gpu::vulkan::NativeImage(image), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.GetHandle(), 1, &region);
        CommandHelper::TransitionImageLayout(cmd, gpu::vulkan::NativeImage(image),
                                             gpu::vulkan::ToVk(image.format),
                                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    });
    std::vector<u32> out(static_cast<size_t>(width) * height * 4);
    const void* data = staging.Map();
    if (!data) return {};
    std::memcpy(out.data(), data, bytes);
    staging.Unmap();
    return out;
}

void HashProducts(const AbMode& mode, VulkanContext& device, GpuCameraPipeline& pipeline, u32 width, u32 height,
                  const std::string& prefix) {
    const auto outputs = pipeline.GetOutputs();
    const std::pair<const char*, gpu::TextureRef> products[] = {
        {"rawDn", outputs.rawDn},
        {"corrected", outputs.corrected},
        {"display", outputs.display},
        {"expectedElectrons", outputs.expectedElectrons},
        {"preAdcElectrons", outputs.preAdcElectrons},
        {"compositedRate", outputs.compositedRate},
        {"linearRgb", outputs.linearRgb},
        {"colorRgb", outputs.colorRgb},
        {"agcSource", outputs.agcSource},
    };
    for (const auto& [name, image] : products) {
        Check(mode, prefix + "." + name, HashOf(ReadFloat(device, image, width, height)));
    }
}

void HashStats(const AbMode& mode, GpuCameraPipeline& pipeline, const std::string& prefix) {
    const auto stats = pipeline.ReadIspStats();
    ASSERT_TRUE(stats.has_value()) << stats.error();
    const auto& s = stats.value();
    std::vector<f32> scalars{s.minValue, s.maxValue, s.meanValue, s.channelSums[0], s.channelSums[1],
                             s.channelSums[2]};
    std::vector<u32> counts{s.saturatedCount, s.unsaturatedCount, s.channelCounts[0], s.channelCounts[1],
                            s.channelCounts[2]};
    std::vector<u32> histogram(s.histogram.begin(), s.histogram.end());
    Check(mode, prefix + ".stats.scalars", HashOf(scalars));
    Check(mode, prefix + ".stats.counts", HashOf(counts));
    Check(mode, prefix + ".stats.histogram", HashOf(histogram));
}

void RecordReprocess(VulkanContext& device, GpuCameraPipeline& pipeline, const CameraConfig& readoutConfig,
                     const CameraConfig& displayConfig) {
    Result<void, String> readout = Result<void, String>::Ok();
    Result<void, String> display = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(device, [&](VkCommandBuffer cmd) {
        auto list = gpu::vulkan::Wrap(cmd);
        readout = pipeline.RecordReadoutReprocess(list, readoutConfig);
    });
    ASSERT_TRUE(readout.has_value()) << readout.error();
    CommandHelper::ExecuteImmediate(device, [&](VkCommandBuffer cmd) {
        auto list = gpu::vulkan::Wrap(cmd);
        display = pipeline.RecordDisplayReprocess(list, displayConfig);
    });
    ASSERT_TRUE(display.has_value()) << display.error();
}

}  // namespace

TEST_F(ComputeMigrationAB, CameraPhoton) {
    constexpr u32 kSize = 16;
    auto config = CameraConfigFor(kSize, kSize);
    config.quality.noiseFree = false;

    std::vector<f32> rgba(static_cast<size_t>(kSize) * kSize * 4, 0.0f);
    std::mt19937 rng(11);
    std::uniform_real_distribution<f32> dist(0.0f, 4000.0f);
    for (u32 i = 0; i < kSize * kSize; ++i) rgba[i * 4] = dist(rng);
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);

    auto created = GpuCameraPipeline::Create(GpuDevice(), kSize, kSize);
    ASSERT_TRUE(created.has_value()) << created.error();
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    for (u64 i = 0; i < 3; ++i) {
        ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, i, 0.1 * static_cast<f64>(i)).has_value());
    }
    HashProducts(mode_, Device(), *pipeline, kSize, kSize, "CameraPhoton.acquisition");
    HashStats(mode_, *pipeline, "CameraPhoton");

    Result<void, String> randomStatus = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        auto list = gpu::vulkan::Wrap(cmd);
        randomStatus = pipeline->RecordRandomVectors(list, 1234u, 2, NoiseClass::PhotonShot, 0);
    });
    ASSERT_TRUE(randomStatus.has_value()) << randomStatus.error();
    Check(mode_, "CameraPhoton.randomVectors",
          HashOf(ReadUint(Device(), pipeline->GetRandomVectors(), kSize, kSize)));

    RecordReprocess(Device(), *pipeline, config, config);
    HashProducts(mode_, Device(), *pipeline, kSize, kSize, "CameraPhoton.reprocess");
}

TEST_F(ComputeMigrationAB, CameraBayerHsv) {
    constexpr u32 kSize = 16;
    auto config = BayerColorConfigFor(kSize, kSize);
    config.quality.noiseFree = false;
    config.isp.whiteBalance = {1.3, 1.0, 0.65};
    config.isp.hsv.hueOffsetDegrees = 40.0;
    config.isp.hsv.empiricalNoise = true;
    config.isp.hsv.temporalDrift = true;

    std::vector<f32> rgba;
    FillBayerMeasurement(kSize, kSize, rgba);
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);

    auto created = GpuCameraPipeline::Create(GpuDevice(), kSize, kSize);
    ASSERT_TRUE(created.has_value()) << created.error();
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());
    for (u64 i = 0; i < 3; ++i) {
        ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, i, 0.1 * static_cast<f64>(i)).has_value());
    }
    HashProducts(mode_, Device(), *pipeline, kSize, kSize, "CameraBayerHsv.acquisition");
    HashStats(mode_, *pipeline, "CameraBayerHsv");

    auto edited = config;
    edited.isp.hsv.hueOffsetDegrees = 70.0;
    RecordReprocess(Device(), *pipeline, config, edited);
    HashProducts(mode_, Device(), *pipeline, kSize, kSize, "CameraBayerHsv.reprocess");
}

TEST_F(ComputeMigrationAB, CameraThermalCheckpoint) {
    constexpr u32 kSize = 16;
    auto config = ThermalConfigFor(kSize, kSize);
    config.quality.noiseFree = false;
    config.thermal.readNoiseDnRms = 3.0;
    config.thermal.driftDnPerSecond = 7.0;
    std::vector<f32> rgba(static_cast<size_t>(kSize) * kSize * 4, 0.0f);
    for (u32 pixel = 0; pixel < kSize * kSize; ++pixel) rgba[pixel * 4] = 2.5e-11f;
    auto input = UploadMeasurement(Device(), kSize, kSize, rgba);
    ASSERT_NE(input, nullptr);

    auto created = GpuCameraPipeline::Create(GpuDevice(), kSize, kSize);
    ASSERT_TRUE(created.has_value()) << created.error();
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());

    const auto hashPair = [&](const std::string& label) {
        Check(mode_, "CameraThermalCheckpoint." + label + ".rawDn",
              HashOf(ReadFloat(Device(), pipeline->GetOutputs().rawDn, kSize, kSize)));
        Check(mode_, "CameraThermalCheckpoint." + label + ".display",
              HashOf(ReadFloat(Device(), pipeline->GetOutputs().display, kSize, kSize)));
    };

    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 0, 0.0).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 1, 0.1).has_value());
    Result<GpuCameraPipeline::GpuCheckpoint, String> checkpoint =
        Result<GpuCameraPipeline::GpuCheckpoint, String>::Err("checkpoint was not recorded");
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        auto list = gpu::vulkan::Wrap(cmd);
        checkpoint = pipeline->RecordStateCheckpoint(list);
    });
    ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error();

    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 2, 0.2).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 3, 0.3).has_value());
    hashPair("continuous");

    Result<void, String> restored = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        auto list = gpu::vulkan::Wrap(cmd);
        restored = pipeline->RestoreStateCheckpoint(checkpoint.value(), list);
    });
    ASSERT_TRUE(restored.has_value()) << restored.error();
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 2, 0.2).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 2, 0.2).has_value());
    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 3, 0.3).has_value());
    hashPair("replayed");

    ASSERT_TRUE(RecordCamera(Device(), *pipeline, *input, 4, 0.4).has_value());
    hashPair("after");
    HashStats(mode_, *pipeline, "CameraThermalCheckpoint");
}

TEST_F(ComputeMigrationAB, CameraStratified) {
    constexpr u32 kWidth = 32;
    constexpr u32 kHeight = 8;
    constexpr u32 kLayers = 8;
    auto config = CameraConfigFor(kWidth, kHeight);
    config.quality.noiseFree = false;

    std::vector<f32> rate(static_cast<size_t>(kWidth) * kHeight * kLayers * 4);
    std::vector<f32> depth(static_cast<size_t>(kWidth) * kHeight * kLayers);
    std::mt19937 rng(5);
    std::uniform_real_distribution<f32> jitter(0.9f, 1.1f);
    for (u32 layer = 0; layer < kLayers; ++layer) {
        for (u32 y = 0; y < kHeight; ++y) {
            for (u32 x = 0; x < kWidth; ++x) {
                const size_t i = (static_cast<size_t>(layer) * kHeight + y) * kWidth + x;
                const f32 base = 1500.0f + 30.0f * x + 10.0f * y + 40.0f * layer;
                rate[i * 4 + 0] = base * jitter(rng);
                rate[i * 4 + 1] = base * jitter(rng);
                rate[i * 4 + 2] = base * jitter(rng);
                rate[i * 4 + 3] = 0.0f;
                depth[i] = 4.0f + 0.05f * x + 0.01f * layer;
            }
        }
    }
    auto strataRate = UploadStrataImage(Device(), kWidth, kHeight, kLayers, VK_FORMAT_R32G32B32A32_SFLOAT, rate, 4);
    auto strataDepth = UploadStrataImage(Device(), kWidth, kHeight, kLayers, VK_FORMAT_R32_SFLOAT, depth, 1);
    ASSERT_NE(strataRate, nullptr);
    ASSERT_NE(strataDepth, nullptr);

    std::array<DynamicLayerCamera, kCameraTimeStrataMax> frames{};
    const u32 count = std::min<u32>(kLayers, kCameraTimeStrataMax);
    for (u32 k = 0; k < count; ++k) {
        frames[k].origin = {0.05f * static_cast<f32>(k), 0.0f, 0.0f, 0.0f};
        frames[k].forward = {0.0f, 0.0f, -1.0f, 0.0f};
        frames[k].right = {1.0f, 0.0f, 0.0f, 0.0f};
        frames[k].up = {0.0f, 1.0f, 0.0f, 0.0f};
        frames[k].params = {0.5f, 4.0f, 0.5f + 0.01f * static_cast<f32>(k), 0.0f};
    }

    auto created = GpuCameraPipeline::Create(GpuDevice(), kWidth, kHeight);
    ASSERT_TRUE(created.has_value()) << created.error();
    auto pipeline = std::move(created.value());
    ASSERT_TRUE(pipeline->Configure(config).has_value());

    for (u64 acquisition = 0; acquisition < 2; ++acquisition) {
        DynamicExposureInput dynamic;
        dynamic.strataRate = gpu::vulkan::Ref(*strataRate);
        dynamic.strataDepth = gpu::vulkan::Ref(*strataDepth);
        dynamic.layerCameras = frames.data();
        dynamic.strataCount = count;
        dynamic.strataLayers = kLayers;
        dynamic.firstRowMidSeconds = 0.5;
        dynamic.exposureSeconds = 0.125;
        dynamic.rowDelaySeconds = acquisition == 0 ? 0.0 : 0.0005;
        Result<void, String> status = Result<void, String>::Ok();
        CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
            auto list = gpu::vulkan::Wrap(cmd);
            status = pipeline->RecordMeasurement(list, gpu::vulkan::Ref(*strataRate),
                                                 acquisition, 0.5, {}, dynamic);
        });
        ASSERT_TRUE(status.has_value()) << status.error();
    }
    HashProducts(mode_, Device(), *pipeline, kWidth, kHeight, "CameraStratified");
    HashStats(mode_, *pipeline, "CameraStratified");
    const auto counters = pipeline->ReadDynamicCounters();
    ASSERT_TRUE(counters.has_value()) << counters.error();
    Check(mode_, "CameraStratified.counters",
          HashOf(std::vector<u32>(counters.value().begin(), counters.value().end())));
}

TEST_F(ComputeMigrationAB, ClaheFacade) {
    if (!quantiloom::testing::FacadeScene::HasAssets()) GTEST_SKIP() << "cornell_box.gltf not in assets";
    quantiloom::testing::FacadeScene scene(Device());
    ASSERT_TRUE(scene.Ready()) << scene.Error();
    ASSERT_TRUE(scene.ApplyScene()) << scene.Error();

    const std::pair<const char*, DisplayToneMode> tones[] = {
        {"Linear", DisplayToneMode::Linear},
        {"Equalize", DisplayToneMode::Equalize},
        {"Clahe", DisplayToneMode::Clahe},
    };
    const std::pair<const char*, DisplayPalette> palettes[] = {
        {"Grey", DisplayPalette::Grey},
        {"Ironbow", DisplayPalette::Ironbow},
    };
    for (const auto& [toneName, tone] : tones) {
        for (const auto& [paletteName, palette] : palettes) {
            DisplayEnhancementParams params;
            params.enabled = true;
            params.toneMode = tone;
            params.palette = palette;
            scene.Context().SetDisplayEnhancementParams(params);
            scene.DrawFrame();
            scene.DrawFrame();

            const auto display = scene.Context().CaptureDisplayImage();
            ASSERT_TRUE(display.has_value()) << display.error();
            const auto raw = scene.Context().CaptureScreenshot();
            ASSERT_TRUE(raw.has_value()) << raw.error();
            const std::string label = std::string("ClaheFacade.") + toneName + "." + paletteName;
            Check(mode_, label + ".input", HashOf(raw.value().data));
            Check(mode_, label + ".display", HashOf(display.value().data));
        }
    }
}
