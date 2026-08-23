// Wall-clock latency for the three interactive claims of the paper's Section
// VIII-G: a geometry edit through TLAS refit to the first progressive frame, a
// backward seek on the thermal timeline, and the deferred cost of a lazy
// invalidation. Section VIII-B's convergence curve is measured separately, by
// scripts/render-tests/measure_convergence.py against the CLI, because it wants
// a reference render rather than a latency.
//
// Every case is DISABLED_ on purpose. These are measurements, not assertions:
// they take minutes, they depend on which GPU is in the machine and what else
// it is doing, and a build gate that fails because a driver got slower would be
// noise. Run them deliberately:
//
//   ./build/tests/Release/libquantiloom_tests.exe \
//       --gtest_also_run_disabled_tests --gtest_filter='InteractiveBench*'
//
// The numbers print as "  [BENCH] <name> ..." lines that
// scripts/experiments/e3_timings.py parses into the paper's table.

#include <gtest/gtest.h>

#include "renderer/ExternalRenderContext.hpp"
#include "renderer/VulkanContext.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuImage.hpp"
#include "core/Config.hpp"
#include "support/VulkanTestDevice.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifndef QUANTILOOM_SOURCE_ROOT
#define QUANTILOOM_SOURCE_ROOT "."
#endif

using namespace quantiloom;

namespace {

constexpr u32 kBenchSize = 512;

// Warm-up runs are discarded rather than averaged in: the first refit allocates
// its scratch buffer, the first frame compiles nothing but does populate caches,
// and the first thermal seek builds the exchange matrix. None of those recur
// during the interaction the target describes.
constexpr int kWarmup = 3;
constexpr int kRuns = 30;

// Derived from the path-tracer seed, so a rerun perturbs the geometry the same
// way. Only the sequence of offsets needs to be reproducible, not their values.
constexpr u32 kBenchSeed = 0x547CU + 3U;

struct Summary {
    f64 median = 0.0;
    f64 p95 = 0.0;
    f64 min = 0.0;
    f64 max = 0.0;
    usize count = 0;
};

Summary Summarise(std::vector<f64> samples) {
    Summary s;
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    s.count = samples.size();
    s.min = samples.front();
    s.max = samples.back();
    s.median = samples[samples.size() / 2];
    // Nearest-rank p95, which for 30 samples is the 29th -- honest about being
    // the second worst of thirty rather than dressed up as a percentile fit.
    const usize rank = static_cast<usize>(std::ceil(0.95 * samples.size()));
    s.p95 = samples[std::min(rank, samples.size()) - 1];
    return s;
}

void Report(const std::string& name, const Summary& s, const std::string& note = "") {
    std::printf("  [BENCH] %-34s median %8.2f ms   p95 %8.2f ms   "
                "min %8.2f   max %8.2f   n=%zu%s%s\n",
                name.c_str(), s.median, s.p95, s.min, s.max, s.count,
                note.empty() ? "" : "   ", note.c_str());
    std::fflush(stdout);
}

std::filesystem::path SourceRoot() { return std::filesystem::path(QUANTILOOM_SOURCE_ROOT); }

// The desert config names its scene relative to the repository root and its
// forcing CSV relative to itself, and ResolveConfigPath only accepts a joined
// path that exists -- so no single baseDir resolves both. Rewriting the two
// keys as absolute paths is what the other GPU tests do, and it also lets a
// case swap in a different tessellation or checkpoint stride.
struct ConfigEdit {
    u32 divisions = 201;
    f64 checkpointStride_h = 1.0;
    f64 startTime_h = 0.0;
    // The geometry cases turn the solver off. ApplyConfig runs the trajectory
    // to `time_h` before it returns -- on the 201^2 scene that is 87 simulated
    // hours over 82,208 elements, minutes of setup that has nothing to do with
    // what a refit costs. The thermal cases obviously leave it on.
    bool thermal = true;
    // ApplyConfig solves all the way to `time_h` before it returns. The seek
    // benchmark only needs the exchange matrix built, so it lands the initial
    // solve at t=0 and pays for the trajectory in the seeks it is timing.
    f64 time_h = -1.0;  // negative: leave the config's own value alone
};

std::optional<Config> LoadDesertConfig(const std::filesystem::path& tempDir,
                                       const ConfigEdit& edit) {
    const auto configDir = SourceRoot() / "assets" / "configs" / "desert";
    const auto source = configDir / "desert_thermal_lwir.toml";
    const auto mesh = SourceRoot() / "assets" / "models" / "desert" /
                      ("desert_" + std::to_string(edit.divisions) + ".gltf");
    if (!std::filesystem::exists(source) || !std::filesystem::exists(mesh)) return std::nullopt;

    std::ifstream in(source);
    std::string line, out, section;
    while (std::getline(in, line)) {
        const auto starts = [&line](const char* key) {
            return line.rfind(key, 0) == 0;
        };
        // [thermal], [thermography] and [sensor] each have an `enabled`; only
        // the first one is ours to switch off.
        if (starts("[")) section = line;

        if (starts("enabled") && section == "[thermal]" && !edit.thermal) {
            out += "enabled = false\n";
        } else if (starts("gltf")) {
            out += "gltf = \"" + mesh.generic_string() + "\"\n";
        } else if (starts("forcing_file")) {
            out += "forcing_file = \"" +
                   (configDir / "desert_day.csv").generic_string() + "\"\n";
        } else if (starts("checkpoint_stride_h")) {
            out += "checkpoint_stride_h = " + std::to_string(edit.checkpointStride_h) + "\n";
        } else if (starts("time_h =") && section == "[thermal]" && edit.time_h >= 0.0) {
            out += "time_h = " + std::to_string(edit.time_h) + "\n";
        } else if (starts("start_time_h")) {
            out += "start_time_h = " + std::to_string(edit.startTime_h) + "\n";
        } else if (starts("dump_elements")) {
            out += "dump_elements = \"" +
                   (tempDir / "bench_elements.csv").generic_string() + "\"\n";
        } else {
            out += line + "\n";
        }
    }

    auto parsed = Config::Parse(out);
    if (!parsed.has_value()) return std::nullopt;
    return parsed.value();
}

class InteractiveBench : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        tempDir = std::filesystem::temp_directory_path() / "quantiloom_bench";
        std::filesystem::create_directories(tempDir);

        auto* shared = quantiloom::testing::SharedVulkanDevice();
        ASSERT_NE(shared, nullptr);

        ExternalRenderContext::InitParams params{};
        params.instance = shared->GetInstance();
        params.physicalDevice = shared->GetPhysicalDevice();
        params.device = shared->GetDevice();
        params.graphicsQueue = shared->GetGraphicsQueue();
        params.graphicsQueueFamily = shared->GetGraphicsQueueFamily();
        params.targetColorFormat = VK_FORMAT_B8G8R8A8_SRGB;
        params.width = kBenchSize;
        params.height = kBenchSize;
        params.pipelineCacheDir = tempDir.string();

        auto created = ExternalRenderContext::Create(params);
        ASSERT_TRUE(created.has_value()) << created.error();
        context = std::move(created.value());

        target = std::make_unique<GpuImage>(
            shared->GetAllocator(), shared->GetDevice(), kBenchSize, kBenchSize,
            VK_FORMAT_B8G8R8A8_SRGB,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    }

    void TearDown() override {
        target.reset();
        context.reset();
        std::error_code ignored;
        std::filesystem::remove_all(tempDir, ignored);
    }

    // One progressive frame, recorded and waited on. ExecuteImmediate submits
    // and blocks, so the clock around this call measures a frame that finished
    // rather than one that was queued.
    void DrawOneFrame() {
        CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
            context->RenderFrame(cmd, target->GetImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                                 kBenchSize, kBenchSize);
        });
    }

    bool LoadDesert(const ConfigEdit& edit) {
        auto config = LoadDesertConfig(tempDir, edit);
        if (!config.has_value()) return false;
        ConfigApplyOptions options;
        options.baseDir = SourceRoot().string();
        const auto report = context->ApplyConfig(config.value(), options);
        return report.ok() && context->HasScene();
    }

    // The sphere is the only thing a drag would move; the ground is the scene.
    [[nodiscard]] std::optional<u32> SphereNode() const {
        const Scene* scene = context->GetScene();
        if (scene == nullptr) return std::nullopt;
        for (u32 i = 0; i < scene->nodes.size(); ++i) {
            if (scene->nodes[i].name.find("Sphere") != String::npos) return i;
        }
        // Fall back to the smaller mesh rather than guessing an index.
        return scene->nodes.empty() ? std::nullopt
                                    : std::optional<u32>(static_cast<u32>(scene->nodes.size() - 1));
    }

    std::filesystem::path tempDir;
    std::unique_ptr<ExternalRenderContext> context;
    std::unique_ptr<GpuImage> target;
};

}  // namespace

// T1. A gizmo drag: move a node, refit the TLAS in place, and draw the first
// progressive frame. The target is 0.5 s. Reported alongside the full rebuild,
// which is what a drag release costs and is the number that scales with the
// instance count rather than with the transform.
TEST_F(InteractiveBench, DISABLED_GeometryEditToFirstFrame) {
    ASSERT_TRUE(LoadDesert({201, 1.0, 0.0, false}))
        << "desert scene or config missing";
    const auto node = SphereNode();
    ASSERT_TRUE(node.has_value());

    const Scene* scene = context->GetScene();
    const glm::mat4 rest = scene->nodes[node.value()].transform;

    std::mt19937 rng(kBenchSeed);
    std::uniform_real_distribution<f32> offset(-0.5f, 0.5f);

    // Split into its parts, because the combined number alone cannot tell a
    // fast refit from a refit that quietly did nothing.
    std::vector<f64> refitOnly, frameOnly, refitToFrame, rebuildToFrame;
    std::vector<f64> gpuTraceMs;
    for (int run = 0; run < kWarmup + kRuns; ++run) {
        const glm::mat4 moved = glm::translate(
            rest, glm::vec3(offset(rng), offset(rng), offset(rng)));

        auto start = std::chrono::steady_clock::now();
        context->SetNodeTransform(node.value(), moved);
        context->RefitAccelerationStructure();
        const auto afterRefit = std::chrono::steady_clock::now();
        context->ResetAccumulation();
        DrawOneFrame();
        const auto afterFrame = std::chrono::steady_clock::now();

        using ms = std::chrono::duration<f64, std::milli>;
        if (run >= kWarmup) {
            refitOnly.push_back(ms(afterRefit - start).count());
            frameOnly.push_back(ms(afterFrame - afterRefit).count());
            refitToFrame.push_back(ms(afterFrame - start).count());
            // The renderer's own timestamp query around the trace dispatch,
            // as an independent witness that the frame did tracing work.
            gpuTraceMs.push_back(static_cast<f64>(context->GetLastFrameTimeMs()));
        }

        start = std::chrono::steady_clock::now();
        context->RebuildAccelerationStructure();
        context->ResetAccumulation();
        DrawOneFrame();
        if (run >= kWarmup) {
            rebuildToFrame.push_back(ms(std::chrono::steady_clock::now() - start).count());
        }
    }

    EXPECT_GT(context->GetAccumulatedSamples(), 0u)
        << "no frame was traced; the timings below would be meaningless";

    Report("T1 refit alone", Summarise(refitOnly), "TLAS update only");
    Report("T1 first frame alone", Summarise(frameOnly), "submit and wait");
    Report("T1 GPU trace dispatch", Summarise(gpuTraceMs), "renderer timestamp");
    Report("T1 refit -> first frame", Summarise(refitToFrame), "during a drag");
    Report("T1 rebuild -> first frame", Summarise(rebuildToFrame), "on release");
}

// T1 again, across the four tessellations, so the refit cost can be read
// against instance and triangle count rather than asserted to be O(1).
TEST_F(InteractiveBench, DISABLED_GeometryEditAcrossTessellations) {
    for (const u32 divisions : {51u, 101u, 201u, 401u}) {
        if (!LoadDesert({divisions, 1.0, 0.0, false})) {
            std::printf("  [BENCH] desert_%u missing, skipped\n", divisions);
            continue;
        }
        const auto node = SphereNode();
        ASSERT_TRUE(node.has_value());
        const glm::mat4 rest = context->GetScene()->nodes[node.value()].transform;

        std::mt19937 rng(kBenchSeed);
        std::uniform_real_distribution<f32> offset(-0.5f, 0.5f);

        std::vector<f64> samples;
        for (int run = 0; run < kWarmup + kRuns; ++run) {
            const glm::mat4 moved = glm::translate(
                rest, glm::vec3(offset(rng), offset(rng), offset(rng)));
            const auto start = std::chrono::steady_clock::now();
            context->SetNodeTransform(node.value(), moved);
            context->RefitAccelerationStructure();
            context->ResetAccumulation();
            DrawOneFrame();
            const auto elapsed = std::chrono::duration<f64, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (run >= kWarmup) samples.push_back(elapsed);
        }
        Report("T1 refit @ " + std::to_string(divisions) + "^2", Summarise(samples));
    }
}

// T2. A backward seek on the thermal timeline, as a function of checkpoint
// stride. The target is 2 s. A backward seek is the expensive direction: it
// restores the nearest checkpoint and replays forward, so the worst case is one
// full stride of re-solve and the stride is the memory-versus-latency dial.
//
// ApplyConfig already calls SetThermalTime once, which is what builds the
// exchange matrix; every seek timed here is therefore a pure scrub.
TEST_F(InteractiveBench, DISABLED_ThermalTimelineBackwardSeek) {
    // The tessellation is a parameter of this case, not a constant, because the
    // first attempt at it ran the stride sweep on the 201^2 scene and did not
    // finish: 82,208 elements checkpointed every 0.25 h across 48 h is 192
    // snapshots of the full state, gigabytes of it, and the walk never got past
    // its first arm. The stride sweep therefore runs on a scene small enough to
    // sweep, and the reference scene gets its own single-stride measurement in
    // the case below. Both say which they are.
    constexpr u32 kSweepDivisions = 101;

    // Loaded once. Only the checkpoint stride changes between arms, and a
    // stride change dirties the timeline but not the ray-traced exchange
    // matrix -- so re-applying the whole config per arm would rebuild the view
    // factors four times over to measure a replay.
    const auto setupStart = std::chrono::steady_clock::now();
    if (!LoadDesert({kSweepDivisions, 1.0, 0.0, true, 0.0})) {
        std::printf("  [BENCH] desert config missing, skipped\n");
        return;
    }
    const f64 setupMs = std::chrono::duration<f64, std::milli>(
        std::chrono::steady_clock::now() - setupStart).count();
    // Reported because it is real and otherwise invisible: the ray-traced
    // exchange matrix and the steady initial condition are paid once when a
    // scene is opened, and no amount of checkpointing makes that cheaper.
    std::printf("  [BENCH] %-34s once   %8.1f ms   (%u elements, exchange + "
                "steady state)\n", "T2 setup, scene open", setupMs,
                context->GetThermalSolveStatus().elementCount);
    std::fflush(stdout);

    for (const f64 stride : {0.25, 0.5, 1.0, 2.0}) {
        // These mirror assets/configs/desert/desert_thermal_lwir.toml; only
        // checkpointStride_h varies.
        ThermalSolveParams params{};
        params.startTime_h = 0.0;
        params.timestep_s = 60.0;
        params.layerCount = 10;
        params.initial = ThermalInitialCondition::Steady;
        params.exchangeRays = 256;
        params.exchangeTopK = 32;
        params.forcingFile = (SourceRoot() / "assets" / "configs" / "desert" /
                              "desert_day.csv").generic_string();
        params.checkpointStride_h = stride;
        context->SetThermalSolveParams(params);

        const auto status = context->GetThermalSolveStatus();
        if (!status.solveValid && status.elementCount == 0) {
            std::printf("  [BENCH] stride %.2f h: solver inactive (%s)\n", stride,
                        status.error.c_str());
            continue;
        }

        // Walk monotonically backward from the far end, one slider step at a
        // time. The first version of this returned to the far end between
        // measurements so that every seek would be backward; that return is a
        // forward seek of up to the whole horizon, thousands of steps over
        // every element, and it cost orders of magnitude more than the thing
        // being timed. Walking backward keeps every seek a seek and bounds the
        // whole case by the walk.
        //
        // A step smaller than the stride is the honest worst case and the one
        // the target describes: the timeline restores the nearest checkpoint at
        // or before the destination and replays forward to it, so the work is
        // bounded by the stride however small the step is.
        constexpr f64 kFarEnd_h = 12.0;
        constexpr f64 kStep_h = 0.3;
        ASSERT_TRUE(context->SetThermalTime(kFarEnd_h).has_value());

        std::vector<f64> samples;
        // How many solver steps each seek actually replayed. The stride is
        // supposed to bound this, and the latency is supposed to follow it; if
        // the first varies across strides and the second does not, the cost is
        // not in the replay and the checkpoint dial is not buying what the
        // design says it buys.
        f64 stepSum = 0.0;
        f64 cursor = kFarEnd_h;
        for (int run = 0; run < kWarmup + kRuns; ++run) {
            cursor -= kStep_h;
            const auto start = std::chrono::steady_clock::now();
            const auto sought = context->SetThermalTime(cursor);
            const auto elapsed = std::chrono::duration<f64, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            ASSERT_TRUE(sought.has_value()) << sought.error();
            if (run >= kWarmup) {
                samples.push_back(elapsed);
                stepSum += context->GetThermalSolveStatus().lastStepCount;
            }
        }

        const auto after = context->GetThermalSolveStatus();
        char label[64];
        std::snprintf(label, sizeof(label), "T2 backward seek, stride %.2f h", stride);
        std::printf("  [BENCH] %-34s replayed %.1f steps per seek on average\n",
                    label, stepSum / static_cast<f64>(samples.size()));
        // Which stepper actually ran belongs in the record: the GPU path is
        // declined for a layer count above its thread-local Thomas solve, and a
        // latency measured on the CPU fallback is a different claim.
        char note[128];
        std::snprintf(note, sizeof(note), "%u elements, %u checkpoints, %s",
                      after.elementCount, after.checkpointCount,
                      after.stepperName.empty() ? "stepper unnamed"
                                                : after.stepperName.c_str());
        Report(label, Summarise(samples), note);
    }
}

// The reference scene at the default stride, which is the configuration the
// target in Section VIII-G is actually about. One stride rather than four,
// because the sweep above is what characterises the dial and this is what
// characterises the scene.
TEST_F(InteractiveBench, DISABLED_ThermalSeekOnReferenceScene) {
    for (const u32 divisions : {51u, 101u, 201u}) {
        const auto setupStart = std::chrono::steady_clock::now();
        if (!LoadDesert({divisions, 1.0, 0.0, true, 0.0})) {
            std::printf("  [BENCH] desert_%u missing, skipped\n", divisions);
            continue;
        }
        const f64 setupMs = std::chrono::duration<f64, std::milli>(
            std::chrono::steady_clock::now() - setupStart).count();

        const auto status = context->GetThermalSolveStatus();
        if (status.elementCount == 0) {
            std::printf("  [BENCH] desert_%u: solver inactive (%s)\n", divisions,
                        status.error.c_str());
            continue;
        }

        constexpr f64 kFarEnd_h = 12.0;
        constexpr f64 kStep_h = 0.3;
        ASSERT_TRUE(context->SetThermalTime(kFarEnd_h).has_value());

        std::vector<f64> samples;
        f64 cursor = kFarEnd_h;
        for (int run = 0; run < kWarmup + kRuns; ++run) {
            cursor -= kStep_h;
            const auto start = std::chrono::steady_clock::now();
            const auto sought = context->SetThermalTime(cursor);
            const auto elapsed = std::chrono::duration<f64, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            ASSERT_TRUE(sought.has_value()) << sought.error();
            if (run >= kWarmup) samples.push_back(elapsed);
        }

        const auto after = context->GetThermalSolveStatus();
        char label[64];
        std::snprintf(label, sizeof(label), "T2 seek @ %u^2, stride 1 h", divisions);
        char note[128];
        std::snprintf(note, sizeof(note), "%u elements, setup %.0f ms, %s",
                      after.elementCount, setupMs,
                      after.stepperName.empty() ? "stepper unnamed"
                                                : after.stepperName.c_str());
        Report(label, Summarise(samples), note);
    }
}

// T2's other half: the cost a lazy invalidation defers. Editing a thermal
// material does not re-solve; it sets a flag, and the whole timeline is rebuilt
// on the next seek. That first seek is the number a user actually waits for
// after letting go of a slider, and it is not the steady-state seek above.
TEST_F(InteractiveBench, DISABLED_ThermalLazyInvalidationRecovery) {
    // On the sweep scene, for the same reason the stride sweep is: each run
    // re-solves the trajectory from its steady initial condition, so this is
    // the most expensive measurement per sample in the file.
    if (!LoadDesert({101, 1.0, 0.0, true, 0.0})) {
        std::printf("  [BENCH] desert config missing, skipped\n");
        return;
    }
    if (context->GetThermalSolveStatus().elementCount == 0) {
        std::printf("  [BENCH] solver inactive, skipped\n");
        return;
    }

    ThermalMaterialParams sand{};
    sand.conductivity_W_mK = 0.30;
    sand.density_kg_m3 = 1600.0;
    sand.specificHeat_J_kgK = 800.0;
    sand.thickness_m = 0.15;
    sand.convection_W_m2K = 14.0;
    sand.shortwaveAbsorptivity = 0.72;

    // Fewer runs than the seek case: each one re-solves the trajectory from its
    // steady initial condition, so this is the expensive measurement rather
    // than the cheap one, and the destination is stated because the cost is
    // proportional to it.
    constexpr f64 kDestination_h = 6.0;
    constexpr int kEditRuns = 10;

    std::vector<f64> samples;
    for (int run = 0; run < kWarmup + kEditRuns; ++run) {
        // A property edit mid-timeline, which sets the dirty flag and defers
        // the work. Perturbing the conductivity each run keeps the rebuild from
        // being a no-op the implementation could legitimately skip.
        sand.conductivity_W_mK = 0.30 + 0.001 * run;
        context->SetThermalMaterial("DesertGround", sand);

        const auto start = std::chrono::steady_clock::now();
        const auto sought = context->SetThermalTime(kDestination_h);
        const auto elapsed = std::chrono::duration<f64, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        ASSERT_TRUE(sought.has_value()) << sought.error();
        if (run >= kWarmup) samples.push_back(elapsed);
    }

    Report("T2 first seek after a material edit", Summarise(samples),
           "timeline rebuilt to t=6 h");
}
