// Headless GPU timestamp producer for the 1080p camera-chain acceptance gate.
// Run from the repo root, after the shader/SDK build:
//   camera_gpu_bench assets/configs/camera_perf_1080p.toml \
//       --output renders/camera_gpu_perf.jsonl --warmup 30 --frames 300
// The Python checker owns the P95 limit. This tool records real Vulkan work.

#include "core/Log.hpp"
#include "renderer/ExternalRenderContext.hpp"
#include "renderer/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using namespace quantiloom;

namespace {

constexpr u32 kWidth = 1920;
constexpr u32 kHeight = 1080;

struct Options {
    std::filesystem::path config;
    std::filesystem::path output = "renders/camera_gpu_perf.jsonl";
    u32 warmup = 30;
    u32 frames = 300;
};

std::string Ascii(std::string_view input) {
    std::string result;
    result.reserve(input.size());
    constexpr char hex[] = "0123456789ABCDEF";
    for (unsigned char byte : input) {
        if (byte >= 32 && byte <= 126) result.push_back(static_cast<char>(byte));
        else {
            result += "\\x";
            result.push_back(hex[byte >> 4]);
            result.push_back(hex[byte & 15]);
        }
    }
    return result;
}

std::string JsonString(std::string_view input) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char byte : input) {
        if (byte == '"' || byte == '\\') {
            result.push_back('\\');
            result.push_back(static_cast<char>(byte));
        } else if (byte < 32) {
            result += "\\u00";
            result.push_back(hex[byte >> 4]);
            result.push_back(hex[byte & 15]);
        } else {
            result.push_back(static_cast<char>(byte));
        }
    }
    result.push_back('"');
    return result;
}

u32 PositiveNumber(const char* text, const char* label) {
    try {
        size_t parsed = 0;
        const unsigned long long number = std::stoull(text, &parsed);
        if (parsed != std::string_view(text).size())
            throw std::invalid_argument(label);
        if (number == 0 || number > std::numeric_limits<u32>::max())
            throw std::out_of_range(label);
        return static_cast<u32>(number);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(label) + " must be a positive integer");
    }
}

Options Parse(int argc, char** argv) {
    if (argc < 2)
        throw std::runtime_error(
            "usage: camera_gpu_bench <config.toml> [--output metrics.jsonl] "
            "[--warmup N] [--frames N]");
    Options options;
    options.config = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (i + 1 >= argc)
            throw std::runtime_error("missing value for benchmark option");
        const char* value = argv[++i];
        if (flag == "--output") options.output = value;
        else if (flag == "--warmup") options.warmup = PositiveNumber(value, "warmup");
        else if (flag == "--frames") options.frames = PositiveNumber(value, "frames");
        else throw std::runtime_error("unknown benchmark option: " + std::string(flag));
    }
    if (!std::filesystem::is_regular_file(options.config))
        throw std::runtime_error("benchmark config not found: " +
                                 options.config.string());
    if (static_cast<u64>(options.warmup) + options.frames >
        std::numeric_limits<u32>::max())
        throw std::runtime_error("benchmark acquisition count exceeds u32");
    return options;
}

// Record, submit, and wait for one real GPU interval. The command buffer is
// never submitted if facade recording fails. All Vulkan objects die before the
// owned VulkanContext so no query result can outlive the device.
template<typename Record>
void SubmitAndWait(const VulkanContext& context, Record&& record) {
    const VkDevice device = context.GetDevice();
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    struct Cleanup {
        VkDevice device;
        VkCommandPool& pool;
        VkFence& fence;
        ~Cleanup() {
            if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
            if (pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, pool, nullptr);
        }
    } cleanup{device, pool, fence};

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = context.GetGraphicsQueueFamily();
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS)
        throw std::runtime_error("cannot create benchmark command pool");
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &allocate, &cmd) != VK_SUCCESS)
        throw std::runtime_error("cannot allocate benchmark command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
        throw std::runtime_error("cannot begin benchmark command buffer");
    auto recorded = record(cmd);
    if (!recorded) throw std::runtime_error(recorded.error());
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("cannot end benchmark command buffer");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
        throw std::runtime_error("cannot create benchmark fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) != VK_SUCCESS)
        throw std::runtime_error("benchmark GPU submission failed");
    if (vkWaitForFences(device, 1, &fence, VK_TRUE,
                        std::numeric_limits<u64>::max()) != VK_SUCCESS)
        throw std::runtime_error("benchmark GPU fence wait failed");
}

template<typename Timings>
void WritePostStages(std::ofstream& output, const Timings& timings) {
    // M3 has the raw trace/full queries; M4 extends this struct with separate
    // dynamic, detector, ISP and HSV Vulkan query intervals. Omitted fields
    // intentionally fail the final Python gate until M4 is complete.
    if constexpr (requires { timings.psfMs; })
        output << ",\"psf_ms\":" << timings.psfMs;
    if constexpr (requires { timings.dynamicMs; })
        output << ",\"dynamic_ms\":" << timings.dynamicMs;
    if constexpr (requires { timings.detectorMs; })
        output << ",\"detector_ms\":" << timings.detectorMs;
    if constexpr (requires { timings.ispMs; })
        output << ",\"isp_ms\":" << timings.ispMs;
    if constexpr (requires { timings.hsvMs; })
        output << ",\"hsv_ms\":" << timings.hsvMs;
}

int Benchmark(const Options& options) {
    Log::Init(nullptr, Log::Level::Off);
    const auto coldStart = std::chrono::steady_clock::now();
    VulkanContext device;
    if (!device.IsRayTracingSupported())
        throw std::runtime_error("GPU lacks Vulkan ray tracing");
    ExternalRenderContext::InitParams params;
    params.instance = device.GetInstance();
    params.physicalDevice = device.GetPhysicalDevice();
    params.device = device.GetDevice();
    params.graphicsQueue = device.GetGraphicsQueue();
    params.graphicsQueueFamily = device.GetGraphicsQueueFamily();
    params.externalAllocator = device.GetAllocator();
    params.targetColorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    params.width = kWidth;
    params.height = kHeight;
    auto created = ExternalRenderContext::Create(params);
    if (!created) throw std::runtime_error(created.error());
    auto renderer = std::move(created.value());
    auto document = Config::Load(options.config.string());
    if (!document) throw std::runtime_error(document.error());
    ConfigApplyOptions applyOptions;
    applyOptions.missingRequired =
        ConfigApplyOptions::MissingKeyPolicy::Error;
    const auto applied = renderer->ApplyConfig(document.value(), applyOptions);
    if (!applied.ok()) throw std::runtime_error(applied.FirstError());
    const auto& cameraConfig = renderer->GetCameraConfig();
    if (!cameraConfig.enabled || cameraConfig.optics.sensorWidthPx != kWidth ||
        cameraConfig.optics.sensorHeightPx != kHeight ||
        cameraConfig.quality.backend != camera::ProcessingBackend::GpuPreview)
        throw std::runtime_error(
            "benchmark needs enabled 1920x1080 GPU preview camera; got enabled=" +
            std::to_string(cameraConfig.enabled) + ", extent=" +
            std::to_string(cameraConfig.optics.sensorWidthPx) + "x" +
            std::to_string(cameraConfig.optics.sensorHeightPx) + ", backend=" +
            std::to_string(static_cast<u32>(cameraConfig.quality.backend)));
    const f64 period = cameraConfig.readout.framePeriodSeconds;
    if (!std::isfinite(period) || period <= 0)
        throw std::runtime_error("benchmark camera needs a positive frame period");
    const auto sceneReady = std::chrono::steady_clock::now();
    const f64 sceneSetupWallMs =
        std::chrono::duration<f64, std::milli>(sceneReady - coldStart).count();

    struct Frame {
        f64 baselineMs;
        CameraGpuTimings cameraTimings;
    };
    const auto capture = [&](u32 index) -> Frame {
        auto queued = renderer->QueueCameraAcquisition(index, index * period);
        if (!queued) throw std::runtime_error(queued.error());
        SubmitAndWait(device, [&](VkCommandBuffer cmd) {
            return renderer->RecordCameraBaseline(cmd);
        });
        const f64 baseline = renderer->GetLastFrameTimeMs();
        if (!std::isfinite(baseline) || baseline <= 0)
            throw std::runtime_error("baseline GPU timestamp unavailable");
        SubmitAndWait(device, [&](VkCommandBuffer cmd) {
            return renderer->RecordQueuedCameraAcquisition(cmd);
        });
        auto completed = renderer->CompleteQueuedCameraAcquisition();
        if (!completed) throw std::runtime_error(completed.error());
        const CameraGpuTimings timings = renderer->GetLastCameraGpuTimings();
        if (!timings.valid || !std::isfinite(timings.traceMs) ||
            !std::isfinite(timings.fullMs) || timings.traceMs <= 0 ||
            timings.fullMs + 0.01f < timings.traceMs)
            throw std::runtime_error("camera GPU timestamps unavailable or invalid");
        return {baseline, timings};
    };
    // The first queued acquisition allocates camera images, compiles its GPU
    // compute pipelines and measures the first real trace. Cold start includes
    // that work; the separate scene-setup number explains its composition.
    const Frame firstFrame = capture(0);
    const auto firstReady = std::chrono::steady_clock::now();
    const f64 coldStartWallMs =
        std::chrono::duration<f64, std::milli>(firstReady - coldStart).count();

    std::error_code ec;
    if (!options.output.parent_path().empty())
        std::filesystem::create_directories(options.output.parent_path(), ec);
    if (ec) throw std::runtime_error("cannot create benchmark output directory");
    std::ofstream output(options.output, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open benchmark JSONL output");
    output.imbue(std::locale::classic());
    output << std::setprecision(9);
    const auto& props = device.GetDeviceProperties();
    VkPhysicalDeviceDriverProperties driverProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 deviceProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    deviceProps.pNext = &driverProps;
    vkGetPhysicalDeviceProperties2(device.GetPhysicalDevice(), &deviceProps);
    const std::string driver = driverProps.driverInfo[0] != '\0'
        ? std::string(driverProps.driverInfo)
        : std::to_string(props.driverVersion);
    output << "{\"type\":\"meta\",\"schema\":2,"
              "\"timing_source\":\"vulkan_timestamp\","
              "\"baseline_spectral_mode\":\"single_mid_response\","
              "\"resolution\":[" << kWidth << ',' << kHeight << "],"
           << "\"gpu_name\":" << JsonString(props.deviceName) << ','
           << "\"driver_version\":" << JsonString(driver) << ','
           << "\"driver_version_raw\":" << props.driverVersion << ','
           << "\"warmup_frames\":" << options.warmup << ','
           << "\"scene_setup_wall_ms\":" << sceneSetupWallMs << ','
           << "\"cold_start_wall_ms\":" << coldStartWallMs << "}\n";

    const u32 total = options.warmup + options.frames;
    for (u32 index = 0; index < total; ++index) {
        const Frame frame = index == 0 ? firstFrame : capture(index);
        output << "{\"type\":\"frame\",\"acquisition_index\":" << index << ','
               << "\"baseline_trace_ms\":" << frame.baselineMs << ','
               << "\"camera_trace_ms\":" << frame.cameraTimings.traceMs << ','
               << "\"full_camera_ms\":" << frame.cameraTimings.fullMs;
        WritePostStages(output, frame.cameraTimings);
        output << "}\n";
        if (!output) throw std::runtime_error("benchmark JSONL write failed");
        if ((index + 1) % 30 == 0)
            std::cout << "Recorded " << (index + 1) << '/' << total
                      << " Vulkan-timed acquisitions\n";
    }
    output.flush();
    if (!output) throw std::runtime_error("benchmark JSONL flush failed");
    std::cout << "GPU camera benchmark written to " << Ascii(options.output.string()) << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = Parse(argc, argv);
        const int status = Benchmark(options);
        Log::Shutdown();
        return status;
    } catch (const std::exception& error) {
        std::cerr << "camera_gpu_bench: " << Ascii(error.what()) << '\n';
        Log::Shutdown();
        return 1;
    }
}
