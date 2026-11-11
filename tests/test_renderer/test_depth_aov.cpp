// ============================================================================
// Quantiloom - Unit Tests for the primary-hit depth AOV and BlitDepthTo
// ============================================================================
// The depth AOV feeds the GUI's occluded grid overlay: hit distance along the
// normalized primary ray, -1 where it missed. These cover the whole exported
// path -- render a frame, BlitDepthTo into a caller-owned image, read it back
// -- with structural asserts (sentinel vs. plausible distance), not pixels.
//
// Skipped rather than failed on a machine with no ray-tracing GPU.
// ============================================================================

#include <gtest/gtest.h>

#include "gpu/Device.hpp"
#include "renderer/ExternalRenderContext.hpp"
#include "renderer/GpuAlloc.hpp"
#include "support/GpuInteropTestUtils.hpp"
#include "support/GpuTestDevice.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>

using namespace quantiloom;

namespace {

constexpr u32 kSize = 64;

class DepthAovTest : public quantiloom::testing::GpuDeviceTest {
protected:
    void SetUp() override {
        GpuDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        testDir = std::filesystem::temp_directory_path() / "quantiloom_depth_aov";
        std::filesystem::create_directories(testDir);

        auto* shared = quantiloom::testing::SharedGpuDevice(Backend());
        ASSERT_NE(shared, nullptr);

        ExternalRenderContext::InitParams params{};
        params.device = quantiloom::testing::GpuHandles(*shared);
        params.width = kSize;
        params.height = kSize;
        params.pipelineCacheDir = testDir.string();

        auto created = ExternalRenderContext::Create(params);
        ASSERT_TRUE(created.has_value()) << created.error();
        context = std::move(created.value());
    }

    void TearDown() override {
        context.reset();
        if (!testDir.empty() && std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
        GpuDeviceTest::TearDown();
    }

    /// Same in-repo cornell box fixture as test_apply_config.cpp, with the
    /// camera injectable so a case can aim at the box or at empty sky.
    void ApplyScene(const std::string& cameraKeys) {
        const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
        const auto gltf = (root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");

        const auto path = testDir / "scene.toml";
        {
            std::ofstream file(path);
            file << "[renderer]\nresolution = [64, 64]\n"
                 << "[camera]\n" << cameraKeys
                 << "[spectral]\n"
                 << "[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                    "sun_radiance = [1.0, 1.0, 1.0]\nsky_radiance = [0.1, 0.1, 0.1]\n"
                 << "[material]\nalbedo = [0.8, 0.8, 0.8]\n"
                 << "[scene]\ngltf = \"" << gltf.generic_string() << "\"\n";
        }
        auto loaded = Config::Load(path.string());
        ASSERT_TRUE(loaded.has_value()) << "fixture TOML did not parse";
        const auto report = context->ApplyConfig(loaded.value());
        ASSERT_TRUE(report.ok()) << report.FirstError();
    }

    /// Render one frame and pull the whole depth AOV back to the CPU through
    /// the exported BlitDepthTo -- the exact sequence the GUI overlay records.
    std::vector<f32> RenderAndReadDepth() {
        auto& device = Gpu();

        auto colorTarget = CreateTextureOrThrow(
            device, gpu::TextureDesc{.width = kSize, .height = kSize, .format = gpu::Format::BGRA8Srgb,
                                     .usage = gpu::TextureUsage::TransferDst | gpu::TextureUsage::Sampled});
        auto depthTarget = CreateTextureOrThrow(
            device, gpu::TextureDesc{.width = kSize, .height = kSize, .format = gpu::Format::R32Float,
                                     .usage = gpu::TextureUsage::TransferDst | gpu::TextureUsage::TransferSrc |
                                              gpu::TextureUsage::Sampled});
        auto readback = CreateBufferOrThrow(
            device, gpu::BufferDesc{kSize * kSize * sizeof(f32), gpu::BufferUsage::TransferDst,
                                    gpu::MemoryClass::Readback, 0});

        device.ExecuteImmediate([&](gpu::CommandList& list) {
            context->RenderFrame(quantiloom::testing::GpuRecording(list),
                                 quantiloom::testing::GpuTarget(Backend(), colorTarget->Ref(),
                                                                GpuTargetFormat::Bgra8Srgb,
                                                                quantiloom::testing::kHostLayoutUndefined,
                                                                quantiloom::testing::kHostLayoutGeneral));
            context->BlitDepthTo(quantiloom::testing::GpuRecording(list),
                             quantiloom::testing::GpuTarget(Backend(), depthTarget->Ref(),
                                                            GpuTargetFormat::R32Float,
                                                            quantiloom::testing::kHostLayoutUndefined,
                                                            gpu::kHostLayoutShaderReadOnly));

            // BlitDepthTo leaves the target SHADER_READ_ONLY; copy it out. The target
            // has no storage usage, so its policy is Optimal: a fragment read names
            // SHADER_READ_ONLY_OPTIMAL and a transfer read TRANSFER_SRC_OPTIMAL.
            const gpu::TextureRef depth = depthTarget->Ref().WithLayout(gpu::TextureLayoutPolicy::Optimal);
            gpu::TextureBarrier toSrc;
            toSrc.texture = depth;
            toSrc.from = gpu::Access::FragmentRead;
            toSrc.to = gpu::Access::TransferRead;
            toSrc.layerCount = 1;
            list.Barrier(toSrc);
            gpu::BufferTextureCopy region;
            region.width = kSize;
            region.height = kSize;
            list.CopyTextureToBuffer(depth, readback->Ref(), region);
        });

        std::vector<f32> depths(kSize * kSize);
        void* mapped = readback->Map();
        std::memcpy(depths.data(), mapped, depths.size() * sizeof(f32));
        readback->Unmap();
        return depths;
    }

    std::filesystem::path testDir;
    std::unique_ptr<ExternalRenderContext> context;
};
QL_INSTANTIATE_GPU_SUITE(DepthAovTest);

/// The scene the fixture names has to exist for any of this to mean anything.
bool CornellBoxAvailable() {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    return std::filesystem::exists(root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
}

}  // namespace

TEST_P(DepthAovTest, HitPixelsCarryAPlausibleDistance) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    // The classic cornell view. The box is in its original units -- roughly
    // 556 x 549 x 559 spanning from the origin -- so the center ray from
    // z = -800 crosses several hundred units before any wall.
    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    const auto depths = RenderAndReadDepth();

    const f32 center = depths[(kSize / 2) * kSize + (kSize / 2)];
    EXPECT_GT(center, 100.0f);
    EXPECT_LT(center, 5000.0f);
}

TEST_P(DepthAovTest, MissPixelsCarryTheSentinel) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    // Aim horizontally from far above the box (top is at y ~ 549): with a
    // 45-degree FOV no ray in the frustum can dip low enough to reach it, so
    // every primary ray misses. (Not straight up -- forward parallel to the
    // up vector degenerates the camera basis.)
    ApplyScene("position = [0.0, 2000.0, 0.0]\nlook_at = [100.0, 2000.0, 0.0]\n");
    const auto depths = RenderAndReadDepth();

    for (const f32 d : depths) {
        ASSERT_FLOAT_EQ(d, -1.0f);
    }
}
