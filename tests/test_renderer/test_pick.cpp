// ============================================================================
// Quantiloom - Unit Tests for ExternalRenderContext::Pick
// ============================================================================
// Pick drives click-to-select in the GUI: a 1x1 inline ray-query dispatch
// whose ray must agree with the raygen shader's ray for the same pixel.
// Structural asserts: hits map to a valid node, sky misses, and the reported
// distance is a plausible camera-to-wall length -- not pixels.
//
// Skipped rather than failed on a machine with no ray-tracing GPU.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/ExternalRenderContext.hpp"
#include "gpu/Device.hpp"
#include "renderer/GpuAlloc.hpp"
#include "support/GpuInteropTestUtils.hpp"
#include "support/GpuTestDevice.hpp"
#include "support/LogCapture.hpp"
#include "support/vulkan/QueueBlocker.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <chrono>
#include <functional>
#include <thread>

#include <filesystem>
#include <fstream>

using namespace quantiloom;

namespace {

constexpr u32 kSize = 64;

class PickTest : public quantiloom::testing::GpuDeviceTest {
protected:
    void SetUp() override {
        GpuDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        testDir = std::filesystem::temp_directory_path() / "quantiloom_pick";
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

    /// Same cornell fixture as test_depth_aov.cpp: the box spans roughly
    /// 556 x 549 x 559 from the origin, in its original units.
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

    void LoadEmitterScene() {
        const std::array<float, 9> positions{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                             0.0f, 1.0f, 0.0f};
        const std::array<u32, 3> indices{0, 1, 2};
        {
            std::ofstream binary(testDir / "triangle.bin", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(positions.data()), sizeof(positions));
            binary.write(reinterpret_cast<const char*>(indices.data()), sizeof(indices));
        }
        const auto file = testDir / "emitters.gltf";
        {
            std::ofstream json(file);
            json << R"({
                "asset":{"version":"2.0"},
                "buffers":[{"byteLength":48,"uri":"triangle.bin"}],
                "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},
                               {"buffer":0,"byteOffset":36,"byteLength":12}],
                "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
                             {"bufferView":1,"componentType":5125,"count":3,"type":"SCALAR"}],
                "materials":[{"name":"light","emissiveFactor":[0.5,0.5,0.5]},
                             {"name":"dark"}],
                "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1,"material":0}]},
                          {"primitives":[{"attributes":{"POSITION":0},"indices":1,"material":1}]}],
                "nodes":[{"name":"lamp","mesh":0},
                         {"name":"rock","mesh":1,"translation":[2,0,0]}],
                "scenes":[{"nodes":[0,1]}],"scene":0
            })";
        }
        const auto loaded = context->LoadSceneFromGltf(file.string());
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
    }

    std::unique_ptr<gpu::Texture> MakeTarget(gpu::TextureUsage usage) {
        return CreateTextureOrThrow(
            Gpu(), gpu::TextureDesc{.width = kSize, .height = kSize, .format = gpu::Format::BGRA8Srgb,
                                    .usage = usage});
    }

    GpuImageTarget FrameTarget(const gpu::Texture& target) {
        return quantiloom::testing::GpuTarget(Backend(), target.Ref(), GpuTargetFormat::Bgra8Srgb,
                                              quantiloom::testing::kHostLayoutUndefined,
                                              quantiloom::testing::kHostLayoutGeneral);
    }

    /// A frame held in the queue behind a gate the host has not released: Block() opens
    /// the gated command buffer, the frame is recorded into it, and it is submitted.
    std::unique_ptr<quantiloom::testing::QueueBlocker> SubmitGatedFrame(const gpu::Texture& target) {
        auto created = quantiloom::testing::QueueBlocker::Create(Gpu());
        EXPECT_TRUE(created.has_value()) << (created.has_value() ? "" : created.error());
        if (!created.has_value()) return nullptr;
        auto blocker = std::move(created.value());
        const auto recording = blocker->Block();
        EXPECT_TRUE(recording.has_value()) << (recording.has_value() ? "" : recording.error());
        if (!recording.has_value()) return nullptr;
        context->RenderFrame(recording.value(), FrameTarget(target));
        const auto submitted = blocker->Submit();
        EXPECT_TRUE(submitted.has_value()) << (submitted.has_value() ? "" : submitted.error());
        if (!submitted.has_value()) return nullptr;
        return blocker;
    }

    void ExpectUploadCompletesPriorReader(const std::function<void()>& upload) {
        if (!quantiloom::testing::QueueBlocker::IsSupported(Backend())) {
            GTEST_SKIP() << "no queue blocker on this backend";
        }
        const auto target = MakeTarget(gpu::TextureUsage::TransferDst);
        // Hold a real old trace in the queue. This deterministically exposes
        // a setter that writes shared bytes while that reader is still pending.
        const auto blocker = SubmitGatedFrame(*target);
        ASSERT_NE(blocker, nullptr);
        std::jthread release([&blocker] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            blocker->Release();
        });
        upload();
        const auto status = blocker->Status();
        EXPECT_TRUE(status.has_value() && status.value() == gpu::FenceStatus::Signaled)
            << (status.has_value() ? "the gated frame had not completed" : status.error());
    }

    // The deferred-upload contract: an in-place write to a shared buffer is
    // recorded into the next submission, so the setter returns while the
    // gated frame is still in flight -- it must not drain the queue. The
    // watchdog keeps a regression that reintroduces the wait from deadlocking
    // the suite: it releases the reader after 10 s, letting the blocked
    // setter return with the fence already signaled, which the assertion then
    // catches as a failure instead of a hang.
    void ExpectUploadDoesNotWaitForPriorReader(const std::function<void()>& upload) {
        if (!quantiloom::testing::QueueBlocker::IsSupported(Backend())) {
            GTEST_SKIP() << "no queue blocker on this backend";
        }
        const auto target = MakeTarget(gpu::TextureUsage::TransferDst);
        // Hold a real old trace in the queue, exactly as the waiting variant
        // does: the gated reader is what used to force the host wait.
        const auto blocker = SubmitGatedFrame(*target);
        ASSERT_NE(blocker, nullptr);
        std::jthread watchdog([&blocker] {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            blocker->Release();
        });
        upload();
        // The setter must have returned without draining the queue: the
        // gated frame has not been released yet (10 s watchdog), so its fence
        // is still unsignaled unless the setter waited.
        const auto status = blocker->Status();
        EXPECT_TRUE(status.has_value() && status.value() == gpu::FenceStatus::NotReady)
            << "the setter waited on submitted readers -- the deferred queue "
               "exists precisely to avoid that wait";
        blocker->Release();
        ASSERT_TRUE(blocker->Wait(10ull * 1000 * 1000 * 1000).has_value());
        // The deferred write lands with the next recorded frame.
        RenderRawFrame();
        EXPECT_GE(context->GetAccumulatedSamples(), 1u);
    }

    void RenderRawFrame() {
        const auto target = MakeTarget(gpu::TextureUsage::TransferDst);
        Gpu().ExecuteImmediate([&](gpu::CommandList& list) {
            context->RenderFrame(quantiloom::testing::GpuRecording(list), FrameTarget(*target));
        });
    }

    std::filesystem::path testDir;
    std::unique_ptr<ExternalRenderContext> context;
};
QL_INSTANTIATE_GPU_SUITE(PickTest);

bool CornellBoxAvailable() {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    return std::filesystem::exists(root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
}

}  // namespace

TEST_P(PickTest, CenterPixelHitsAValidNode) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");

    const auto result = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(result.has_value()) << result.error();
    ASSERT_TRUE(result.value().hit);

    const Scene* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    EXPECT_LT(result.value().nodeIndex, scene->nodes.size());

    // Camera is 800 units in front of the box; any wall is hundreds away
    EXPECT_GT(result.value().hitT, 100.0f);
    EXPECT_LT(result.value().hitT, 5000.0f);

    // worldPosition must be origin + direction * hitT, i.e. its distance from
    // the camera equals hitT
    const glm::vec3 camPos(278.0f, 274.0f, -800.0f);
    EXPECT_NEAR(glm::length(result.value().worldPosition - camPos),
                result.value().hitT, 0.5f);
}

TEST_P(PickTest, SkyPixelMisses) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    // High above the box aimed horizontally: nothing in the frustum
    ApplyScene("position = [0.0, 2000.0, 0.0]\nlook_at = [100.0, 2000.0, 0.0]\n");

    const auto result = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_FALSE(result.value().hit);
}

TEST_P(PickTest, OutOfBoundsIsAnError) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    EXPECT_FALSE(context->Pick(kSize, 0).has_value());
    EXPECT_FALSE(context->Pick(0, kSize).has_value());
}

TEST_P(PickTest, AsynchronousPixelMatchesSynchronousReadingAndCarriesItsCoordinates) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf unavailable";
    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    context->SetRenderScale(0.5f);
    context->SetDebugMode(DebugVisualizationMode::ShadedNormal);
    RenderRawFrame();
    const auto expected = context->ReadPixelValue(kSize - 1, kSize / 2);
    ASSERT_TRUE(expected.has_value());
    const auto requested = context->RequestPixelValue(kSize - 1, kSize / 2, 73);
    ASSERT_TRUE(requested.has_value()) << requested.error();
    ASSERT_TRUE(requested.value());
    // Waiting belongs to this deterministic test, never to Request/Poll.
    ASSERT_TRUE(Gpu().WaitQueueIdle().has_value());
    const auto completed = context->PollPixelValue();
    ASSERT_TRUE(completed.has_value()) << completed.error();
    ASSERT_TRUE(completed.value().has_value());
    const auto& reading = *completed.value();
    EXPECT_EQ(reading.requestId, 73u);
    EXPECT_EQ(reading.x, kSize - 1);
    EXPECT_EQ(reading.y, kSize / 2);
    EXPECT_EQ(reading.accumulatedSamples, 1u);
    for (u32 c = 0; c < 4; ++c) EXPECT_EQ(reading.value[c], expected.value()[c]);
    EXPECT_FALSE(context->PollPixelValue().value().has_value());
}

TEST_P(PickTest, AsynchronousPixelRingCoalescesCompletedCopiesAndReusesSlots) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf unavailable";
    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    RenderRawFrame();
    for (u32 i = 0; i < 3; ++i) {
        const auto requested = context->RequestPixelValue(i, i + 1, 100 + i);
        ASSERT_TRUE(requested.has_value());
        ASSERT_TRUE(requested.value());
    }
    const auto busy = context->RequestPixelValue(10, 11, 103);
    ASSERT_TRUE(busy.has_value());
    EXPECT_FALSE(busy.value());
    ASSERT_TRUE(Gpu().WaitQueueIdle().has_value());
    const auto completed = context->PollPixelValue();
    ASSERT_TRUE(completed.has_value());
    ASSERT_TRUE(completed.value().has_value());
    EXPECT_EQ(completed.value()->requestId, 102u);
    EXPECT_EQ(completed.value()->x, 2u);
    EXPECT_EQ(completed.value()->y, 3u);
    const auto reused = context->RequestPixelValue(10, 11, 103);
    ASSERT_TRUE(reused.has_value());
    EXPECT_TRUE(reused.value());
}

TEST_P(PickTest, AsynchronousPixelDropsResultsWhenTheAccumulationChanges) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf unavailable";
    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    EXPECT_FALSE(context->RequestPixelValue(0, 0, 1).has_value());
    RenderRawFrame();
    ASSERT_TRUE(context->RequestPixelValue(0, 0, 1).value());
    context->ResetAccumulation();
    ASSERT_TRUE(Gpu().WaitQueueIdle().has_value());
    const auto stale = context->PollPixelValue();
    ASSERT_TRUE(stale.has_value());
    EXPECT_FALSE(stale.value().has_value());
    EXPECT_FALSE(context->RequestPixelValue(0, 0, 2).has_value());
    RenderRawFrame();
    ASSERT_TRUE(context->RequestPixelValue(0, 0, 2).value());
    ASSERT_TRUE(Gpu().WaitQueueIdle().has_value());
    const auto fresh = context->PollPixelValue();
    ASSERT_TRUE(fresh.has_value());
    ASSERT_TRUE(fresh.value().has_value());
    EXPECT_EQ(fresh.value()->requestId, 2u);
    EXPECT_FALSE(context->RequestPixelValue(kSize, 0, 3).has_value());
}

TEST_P(PickTest, OrdinaryTransformsDoNotRebuildSampledEmission) {
    LoadEmitterScene();
    const auto* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    u32 lamp = 0, rock = 0;
    bool haveLamp = false, haveRock = false;
    for (u32 i = 0; i < scene->nodes.size(); ++i) {
        if (scene->nodes[i].name == "lamp") { lamp = i; haveLamp = true; }
        if (scene->nodes[i].name == "rock") { rock = i; haveRock = true; }
    }
    ASSERT_TRUE(haveLamp && haveRock);
    {
        quantiloom::support::ScopedLogCapture capture;
        context->SetNodeTransform(rock, glm::translate(glm::mat4(1.0f), {3.0f, 0.0f, 0.0f}));
        context->RefitAccelerationStructure();
        EXPECT_EQ(capture.Count(Log::Level::Info, "Emissive geometry:"), 0) << capture.Dump();
    }
    {
        quantiloom::support::ScopedLogCapture capture;
        context->SetNodeTransform(lamp, glm::translate(glm::mat4(1.0f), {0.0f, 2.0f, 0.0f}));
        context->RefitAccelerationStructure();
        EXPECT_EQ(capture.Count(Log::Level::Info, "Emissive geometry:"), 1) << capture.Dump();
    }
    {
        quantiloom::support::ScopedLogCapture capture;
        context->RefitAccelerationStructure();
        EXPECT_EQ(capture.Count(Log::Level::Info, "Emissive geometry:"), 0) << capture.Dump();
    }
}

TEST_P(PickTest, WavelengthChangesPreserveSampledEmitterGeometry) {
    LoadEmitterScene();
    quantiloom::support::ScopedLogCapture capture;
    context->SetWavelength(612.0f);
    EXPECT_EQ(capture.Count(Log::Level::Info, "Emissive geometry:"), 0) << capture.Dump();
}

TEST_P(PickTest, MaterialEmissionEditsRefreshTheSamplingDistribution) {
    LoadEmitterScene();
    const auto* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    u32 material = 0;
    bool found = false;
    for (u32 i = 0; i < scene->materials.size(); ++i) {
        if (scene->materials[i].name == "light") { material = i; found = true; }
    }
    ASSERT_TRUE(found);
    auto changed = scene->materials[material];
    changed.emissiveFactor *= 2.0f;
    {
        quantiloom::support::ScopedLogCapture capture;
        context->UpdateMaterial(material, changed);
        EXPECT_EQ(capture.Count(Log::Level::Info, "Emissive geometry:"), 1) << capture.Dump();
    }
    changed.roughnessFactor = 0.2f;
    {
        quantiloom::support::ScopedLogCapture capture;
        context->UpdateMaterial(material, changed);
        EXPECT_EQ(capture.Count(Log::Level::Info, "Emissive geometry:"), 0) << capture.Dump();
    }
}


TEST_P(PickTest, DeferredRefitsUseTheLatestPoseInTheRecordedFrameAndExactPick) {
    LoadEmitterScene();
    const auto* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    for (u32 material = 0; material < scene->materials.size(); ++material) {
        auto dark = scene->materials[material];
        dark.emissiveFactor = glm::vec3(0.0f);
        context->UpdateMaterial(material, dark);
    }
    context->SetCameraLookAt({0.25f, 0.25f, 5.0f}, {0.25f, 0.25f, 0.0f}, {0.0f, 1.0f, 0.0f});
    context->SetDebugMode(DebugVisualizationMode::GeometricNormal);
    std::vector<glm::mat4> rest;
    for (const auto& node : scene->nodes) rest.push_back(node.transform);
    RenderRawFrame();
    const auto baseline = context->CaptureScreenshot();
    ASSERT_TRUE(baseline.has_value());
    ASSERT_TRUE(context->Pick(kSize / 2, kSize / 2).value().hit);
    const auto away = glm::translate(glm::mat4(1.0f), {1000.0f, 1000.0f, 1000.0f});
    const auto move = [&](bool offscreen) {
        for (u32 node = 0; node < rest.size(); ++node)
            context->SetNodeTransform(node, offscreen ? away * rest[node] : rest[node]);
        context->RefitAccelerationStructure();
    };

    // Multiple host edits before the next frame coalesce to the final pose.
    move(true);
    move(false);
    context->ResetAccumulation();
    RenderRawFrame();
    auto returned = context->CaptureScreenshot();
    ASSERT_TRUE(returned.has_value());
    EXPECT_EQ(returned.value().data, baseline.value().data);

    // No synchronous pick/solve in between: RenderFrame itself must record UPDATE.
    move(true);
    context->ResetAccumulation();
    RenderRawFrame();
    const auto shifted = context->CaptureScreenshot();
    ASSERT_TRUE(shifted.has_value());
    EXPECT_NE(shifted.value().data, baseline.value().data);
    EXPECT_FALSE(context->Pick(kSize / 2, kSize / 2).value().hit);

    // Exact picking flushes a pending pose even before another frame is drawn.
    move(false);
    const auto restored = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(restored.has_value());
    EXPECT_TRUE(restored.value().hit);
}


TEST_P(PickTest, ViewportSampleBatchesPreserveTheRawSeededAccumulation) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf unavailable";
    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    constexpr u32 samples = 35;
    context->SetSPP(samples);
    context->SetSamplingSeed(424242u);
    context->SetViewportSampleBatch(1);
    for (u32 sample = 0; sample < samples; ++sample) RenderRawFrame();
    const auto expected = context->CaptureScreenshot();
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(context->GetAccumulatedSamples(), samples);

    context->SetSamplingSeed(424242u);
    context->SetViewportSampleBatch(16);
    RenderRawFrame();
    EXPECT_EQ(context->GetAccumulatedSamples(), 16u);
    RenderRawFrame();
    EXPECT_EQ(context->GetAccumulatedSamples(), 32u);
    context->SetViewportSampleBatch(3);
    RenderRawFrame();
    EXPECT_EQ(context->GetAccumulatedSamples(), samples);
    const auto actual = context->CaptureScreenshot();
    ASSERT_TRUE(actual.has_value()) << actual.error();
    EXPECT_EQ(actual.value().data, expected.value().data);
}

TEST_P(PickTest, ViewportSampleBatchLimitsDoNotResetTheSequence) {
    LoadEmitterScene();
    context->SetViewportSampleBatch(0);
    RenderRawFrame();
    EXPECT_EQ(context->GetAccumulatedSamples(), 1u);
    context->SetViewportSampleBatch(1000);
    RenderRawFrame();
    EXPECT_EQ(context->GetAccumulatedSamples(), 17u);
}


TEST_P(PickTest, OrdinaryMaterialUploadsDeferAroundSubmittedReaders) {
    LoadEmitterScene();
    auto edited = context->GetScene()->materials[1];
    edited.roughnessFactor = 0.3f;
    ExpectUploadDoesNotWaitForPriorReader([&] { context->UpdateMaterial(1, edited); });
}

TEST_P(PickTest, CachedThermalUploadsWaitEvenWithoutAPendingRefit) {
    LoadEmitterScene();
    ThermalMaterialParams material;
    material.conductivity_W_mK = 80.0f;
    context->SetThermalMaterial("light", material);
    context->SetThermalMaterial("dark", material);
    ThermalSolveParams params;
    params.initial = ThermalInitialCondition::Uniform;
    params.initialTemperature_K = 293.15;
    params.exchangeRays = 16;
    context->SetThermalSolveParams(params);
    context->SetThermalSolveEnabled(true);
    ASSERT_TRUE(context->SetThermalTime(0.0).has_value());
    context->SetSpectralMode(SpectralMode::LWIR_Fused);
    ExpectUploadCompletesPriorReader([&] {
        const auto uploaded = context->SetThermalTime(0.0);
        EXPECT_TRUE(uploaded.has_value()) << uploaded.error();
    });
}


TEST_P(PickTest, LightingUploadsDeferAroundSubmittedReaders) {
    LoadEmitterScene();
    auto lighting = context->GetLightingParams();
    lighting.skyRadiance_rgb *= 0.5f;
    ExpectUploadDoesNotWaitForPriorReader([&] { context->SetLightingParams(lighting); });
}

// Appending into spare capacity writes only slots past the live entry count
// through the deferred-upload queue, so a submitted reader of the old table
// sees unchanged data and the setter need not wait for it. Growing past the
// capacity replaces the buffer, and that path still drains the queue and the
// device before the old allocation goes (see the next test).
TEST_P(PickTest, AppendingSpectralTablesIntoSpareCapacityDoesNotWaitForReaders) {
    LoadEmitterScene();
    SpectralCurve curve;
    curve.samples = {{400.0f, 0.25f}, {780.0f, 0.5f}};
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {400.0f, 780.0f};
    cri.n = {1.4f, 1.5f};
    cri.k = {0.0f, 0.0f};
    const auto curveIndex = context->AddSpectralCurve(curve);
    const auto criIndex = context->AddComplexRefractiveIndex(cri);
    ASSERT_GE(curveIndex, 0);
    ASSERT_GE(criIndex, 0);
    auto material = context->GetScene()->materials[0];
    material.spectralReflectanceCurveIndex = curveIndex;
    material.complexRefractiveIndexIndex = criIndex;
    context->UpdateMaterial(0, material);
    ExpectUploadDoesNotWaitForPriorReader([&] { EXPECT_GE(context->AddSpectralCurve(curve), 0); });
    ExpectUploadDoesNotWaitForPriorReader([&] { EXPECT_GE(context->AddComplexRefractiveIndex(cri), 0); });
}

TEST_P(PickTest, GrowingSpectralTablesWaitsForSubmittedReaders) {
    LoadEmitterScene();
    SpectralCurve curve;
    curve.samples = {{400.0f, 0.25f}, {780.0f, 0.5f}};
    ASSERT_GE(context->AddSpectralCurve(curve), 0);
    // Tables start at eight entries of capacity and double, so sixty-four
    // appends from a handful of entries must replace the buffer at least
    // once; that replacement has to outwait the submitted reader.
    ExpectUploadCompletesPriorReader([&] {
        for (int i = 0; i < 64; ++i) EXPECT_GE(context->AddSpectralCurve(curve), 0);
    });
    // Binding an emission curve appends too, and then rewrites a material
    // through the deferred queue; it must simply succeed.
    const auto bound = context->SetMaterialEmissionSpectrum(0, "d65", "match_luminance", "");
    EXPECT_TRUE(bound.has_value());
}

TEST_P(PickTest, ApplyingConfigWaitsBeforeReplacingLiveSpectralTables) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf unavailable";
    const std::string cameraKeys = "position = [278.0, 273.0, -800.0]\n"
                                   "look_at = [278.0, 273.0, 0.0]\n";
    ApplyScene(cameraKeys);
    SpectralCurve curve;
    curve.samples = {{400.0f, 0.25f}, {780.0f, 0.5f}};
    ASSERT_GE(context->AddSpectralCurve(curve), 0);
    auto loaded = Config::Load((testDir / "scene.toml").string());
    ASSERT_TRUE(loaded.has_value());
    ExpectUploadCompletesPriorReader([&] {
        const auto applied = context->ApplyConfig(loaded.value());
        EXPECT_TRUE(applied.ok()) << applied.FirstError();
    });
}

TEST_P(PickTest, InvalidSdkMaterialReferencesUseLoggedFallbacks) {
    LoadEmitterScene();
    auto material = context->GetScene()->materials[0];
    material.baseColorTextureIndex = 1000000;
    material.normalTextureIndex = -2;
    material.spectralReflectanceCurveIndex = 1000000;
    material.complexRefractiveIndexIndex = 1000000;
    material.emissiveRadianceCurveIndex = 1000000;
    material.fluorescenceExcitationCurveIndex = 1000000;
    material.fluorescenceEmissionCurveIndex = 1000000;
    quantiloom::support::ScopedLogCapture capture;
    context->UpdateMaterial(0, material);
    EXPECT_GE(capture.Count(Log::Level::Warn, "using fallback"), 7) << capture.Dump();
    RenderRawFrame();
}
