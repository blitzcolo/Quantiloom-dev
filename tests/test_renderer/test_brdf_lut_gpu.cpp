/**
 * @file test_brdf_lut_gpu.cpp
 * @brief Cover for rendercore::BrdfLut, the first device-bound stage shared by the
 *        CLI and ExternalRenderContext
 *
 * Uses a tiny LUT written to a temporary path rather than the 512x1024 default: the
 * default takes seconds to generate, and its cache under assets/luts/ is produced at
 * runtime rather than committed, so depending on it would make the case either slow
 * or machine-dependent.
 */

#include <gtest/gtest.h>

#include "support/GpuTestDevice.hpp"

#include "gpu/GpuTypes.hpp"
#include "renderer/RenderCore.hpp"

#include <filesystem>

using namespace quantiloom;
using quantiloom::testing::GpuDeviceTest;

namespace {

BRDFLutGenerator::Config TinyConfig() {
    BRDFLutGenerator::Config cfg;
    cfg.resolution = 16;
    cfg.sampleCount = 16;
    return cfg;
}

std::string TempCachePath(const char* name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return path.string();
}

}  // namespace

TEST_P(GpuDeviceTest, BrdfLutUploadsAndOwnsItsSampler) {
    const auto cache = TempCachePath("quantiloom_brdf_lut_upload.bin");
    auto lut = rendercore::BrdfLut::Create(Gpu(), cache, TinyConfig());

    ASSERT_TRUE(lut.IsValid());
    EXPECT_TRUE(lut.View().IsValid());
    EXPECT_TRUE(lut.Sampler().IsValid());
}

// The image used to be created at a hardcoded 512 while the generator read the
// resolution from the config, so a non-default resolution would have uploaded into a
// mismatched image. Both call sites had that shape.
TEST_P(GpuDeviceTest, BrdfLutSizesTheImageFromTheConfig) {
    const auto cache = TempCachePath("quantiloom_brdf_lut_resolution.bin");
    auto lut = rendercore::BrdfLut::Create(Gpu(), cache, TinyConfig());

    ASSERT_TRUE(lut.IsValid());
    EXPECT_EQ(lut.Resolution(), 16u);
}

// Generation costs seconds at the shipped resolution, so the cache is what keeps
// start-up bearable. Create() must write one when it finds none, and read it back.
TEST_P(GpuDeviceTest, BrdfLutWritesAndReusesItsCache) {
    const auto cache = TempCachePath("quantiloom_brdf_lut_cache.bin");
    ASSERT_FALSE(std::filesystem::exists(cache));

    { auto first = rendercore::BrdfLut::Create(Gpu(), cache, TinyConfig());
      ASSERT_TRUE(first.IsValid()); }

    ASSERT_TRUE(std::filesystem::exists(cache));
    EXPECT_TRUE(BRDFLutGenerator::LoadFromBinary(cache, nullptr).has_value());

    auto second = rendercore::BrdfLut::Create(Gpu(), cache, TinyConfig());
    EXPECT_TRUE(second.IsValid());
}

// Owning the sampler is the point of the type: both call sites tracked it by hand,
// and one of them ignored the create result. Moving must not double-destroy it.
TEST_P(GpuDeviceTest, BrdfLutSurvivesAMove) {
    const auto cache = TempCachePath("quantiloom_brdf_lut_move.bin");
    auto lut = rendercore::BrdfLut::Create(Gpu(), cache, TinyConfig());
    ASSERT_TRUE(lut.IsValid());

    const gpu::SamplerRef sampler = lut.Sampler();
    rendercore::BrdfLut moved = std::move(lut);

    EXPECT_EQ(moved.Sampler(), sampler);
    EXPECT_TRUE(moved.IsValid());
    EXPECT_FALSE(lut.IsValid());  // NOLINT(bugprone-use-after-move) -- the point
}
