/**
 * @file FacadeScene.hpp
 * @brief An ExternalRenderContext over the shared headless device and the Cornell box
 *
 * The same setup CameraSchedulerGpuTest uses, as a plain object so the compute-migration
 * evidence (ComputeMigrationAB.ClaheFacade, GpuClaheTest) drives today's code through the
 * facade's real entry points.
 */

#pragma once

#include "gpu/Device.hpp"
#include "renderer/ExternalRenderContext.hpp"
#include "renderer/GpuAlloc.hpp"
#include "support/GpuInteropTestUtils.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#ifndef QUANTILOOM_SOURCE_ROOT
#define QUANTILOOM_SOURCE_ROOT "."
#endif

namespace quantiloom::testing {

class FacadeScene {
public:
    static constexpr u32 kViewportSize = 64;

    explicit FacadeScene(gpu::Device& device) : device_(device) {
        static int counter = 0;
        testDir_ = std::filesystem::temp_directory_path() /
                   ("quantiloom_facade_scene_" + std::to_string(counter++));
        std::filesystem::create_directories(testDir_);

        ExternalRenderContext::InitParams params{};
        params.device = GpuHandles(device_);
        params.width = kViewportSize;
        params.height = kViewportSize;
        params.pipelineCacheDir = testDir_.string();
        auto created = ExternalRenderContext::Create(params);
        if (!created.has_value()) {
            error_ = created.error();
            return;
        }
        context_ = std::move(created.value());
        target_ = CreateTextureOrThrow(
            device_, gpu::TextureDesc{.width = kViewportSize, .height = kViewportSize,
                                      .format = gpu::Format::BGRA8Srgb,
                                      .usage = gpu::TextureUsage::TransferDst | gpu::TextureUsage::Sampled});
    }

    ~FacadeScene() {
        target_.reset();
        context_.reset();
        std::error_code ignored;
        std::filesystem::remove_all(testDir_, ignored);
    }

    FacadeScene(const FacadeScene&) = delete;
    FacadeScene& operator=(const FacadeScene&) = delete;

    [[nodiscard]] bool Ready() const { return context_ != nullptr; }
    [[nodiscard]] const String& Error() const { return error_; }
    [[nodiscard]] ExternalRenderContext& Context() { return *context_; }

    [[nodiscard]] static bool HasAssets() {
        const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
        return std::filesystem::exists(root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
    }

    [[nodiscard]] bool ApplyScene() {
        const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
        const auto gltf = root / "assets" / "models" / "cornell_box" / "cornell_box.gltf";
        const auto path = testDir_ / "scene.toml";
        {
            std::ofstream file(path);
            file << "[renderer]\nresolution = [64, 64]\n"
                 << "[camera]\n"
                    "position = [278.0, 274.0, -800.0]\n"
                    "look_at = [278.0, 274.0, 0.0]\n"
                 << "[spectral]\n"
                 << "[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                    "sun_radiance = [1.0, 1.0, 1.0]\nsky_radiance = [0.1, 0.1, 0.1]\n"
                 << "[material]\nalbedo = [0.8, 0.8, 0.8]\n"
                 << "[scene]\ngltf = \"" << gltf.generic_string() << "\"\n";
        }
        auto loaded = Config::Load(path.string());
        if (!loaded.has_value()) {
            error_ = "fixture TOML did not parse";
            return false;
        }
        const auto report = context_->ApplyConfig(loaded.value());
        if (!report.ok()) {
            error_ = report.FirstError();
            return false;
        }
        return true;
    }

    void DrawFrame() {
        device_.ExecuteImmediate([&](gpu::CommandList& list) {
            context_->RenderFrame(GpuRecording(list),
                                  GpuTarget(device_.Info().backend, target_->Ref(), GpuTargetFormat::Bgra8Srgb,
                                            kHostLayoutUndefined, kHostLayoutGeneral));
        });
    }

private:
    gpu::Device& device_;
    std::filesystem::path testDir_;
    std::unique_ptr<ExternalRenderContext> context_;
    std::unique_ptr<gpu::Texture> target_;
    String error_;
};

}  // namespace quantiloom::testing
