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

#include "support/VulkanTestDevice.hpp"

#include "core/Sha256.hpp"
#include "core/SpectralCube.hpp"
#include "hs_core/GpuSpectralReconstructor.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuDisplayRange.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/GpuThermalStepper.hpp"
#include "renderer/VulkanContext.hpp"
#include "thermal/ThermalSolver.hpp"

#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>

using namespace quantiloom;
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

    GpuSpectralReconstructor reconstructor(Device());
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

    GpuThermalStepper stepper(Device());
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

    auto created = rendercore::GpuDisplayRange::Create(Device());
    ASSERT_TRUE(created.has_value()) << created.error();
    auto gpu = std::move(created.value());

    const std::pair<f32, f32> percentiles[] = {{1.0f, 99.0f}, {0.0f, 100.0f}, {5.0f, 95.0f}};
    std::vector<f32> windows;
    for (const auto& [low, high] : percentiles) {
        const auto range = gpu->Compute(*image, kWidth, kHeight, low, high);
        ASSERT_TRUE(range.has_value()) << range.error();
        windows.push_back(range.value().min);
        windows.push_back(range.value().max);
    }
    Check(mode_, "DisplayRange.windows", HashOf(windows));
}
