#include "renderer/GpuCameraPipeline.hpp"

#include "core/Log.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/VulkanContext.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <climits>
#include <unistd.h>
#endif

namespace quantiloom::rendercore {
namespace {

template<class T>
Result<T, String> Fail(String message) {
    return typename Result<T, String>::Err(std::move(message));
}

struct CameraPush {
    u32 width = 0, height = 0, physicalWidth = 0, physicalHeight = 0;
    u32 channelCount = 1, cfa = 0, detector = 0, flags = 0;
    u32 seed = 0, acquisitionLo = 0, acquisitionHi = 0, adcBits = 0;
    f32 frameTimeSeconds = 0.0f, deltaSeconds = 0.0f;
    u32 firstThermal = 1, direction = 0;
    f32 sigmaR = 0.0f, sigmaG = 0.0f, sigmaB = 0.0f;
    f32 sigmaRy = 0.0f, sigmaGy = 0.0f, sigmaBy = 0.0f;
    u32 rngNoiseClass = 0, rngCounter = 0;
    // Dynamic compositor scalars; per-stratum camera frames ride in the
    // binding-10 buffer. Must match CameraPush in camera_common.hlsli.
    u32 timeStratumCount = 1, rollingShutter = 0;
    f32 exposureSeconds = 0.0f, rowDelaySeconds = 0.0f;
    f32 firstRowMidSeconds = 0.0f;
    // Multi-phase camera passes (camera_stats) select their stage here.
    u32 ispPhase = 0;
};
static_assert(sizeof(CameraPush) == 120);

// DisplayToneMode / DisplayPalette orders must match DisplayControl.hpp and
// the shader constants.
namespace isp {
inline constexpr u32 kFlagDenoise = 1u;
inline constexpr u32 kFlagSharpen = 2u;
inline constexpr u32 kFlagClip = 4u;
inline constexpr u32 kFlagClahePersistent = 8u;
inline constexpr u32 kFlagDefects = 16u;
inline constexpr u32 kFlagEmpiricalNoise = 32u;  // camera_hsv.comp.hlsl bit 5
inline constexpr u32 kFlagTemporalDrift = 64u;   // camera_hsv.comp.hlsl bit 6
inline constexpr u32 kFlagHsv = 128u;  // any HSV parameter off its default

// ispStats buffer layout (uint slots), shared with camera_stats.comp.hlsl /
// camera_display.comp.hlsl. 256 histogram bins, then the named slots; the
// per-channel AWB rows are written by the stats reduce phase.
inline constexpr u32 kStatsSlots = 272;
inline constexpr u32 kStatsMin = 256;
inline constexpr u32 kStatsMax = 257;
inline constexpr u32 kStatsMean = 258;
inline constexpr u32 kStatsSat = 259;
inline constexpr u32 kStatsUnsat = 260;
inline constexpr u32 kStatsWindowLo = 261;
inline constexpr u32 kStatsWindowHi = 262;
inline constexpr u32 kStatsChSumR = 263;
inline constexpr u32 kStatsChCntR = 266;
} // namespace isp

enum CameraFlags : u32 {
    NoiseFree = 1u,
    Shot = 2u,
    DarkCurrent = 4u,
    DarkShot = 8u,
    Read = 16u,
    Fpn = 32u,
    Nuc = 64u,
    NucGainMap = 128u,
    NucOffsetMap = 256u,
    Vignette = 512u
};

using ConfigRows = std::array<std::array<f32, 4>, 9>;

std::filesystem::path ExecutableDirectory() {
#if defined(_WIN32)
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    return std::filesystem::path(buffer).parent_path();
#elif defined(__linux__)
    char buffer[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length > 0) {
        buffer[length] = '\0';
        return std::filesystem::path(buffer).parent_path();
    }
    return std::filesystem::current_path();
#else
    return std::filesystem::current_path();
#endif
}

Vector<u32> LoadSpirv(StringView name) {
    const auto exeDir = ExecutableDirectory();
    const std::filesystem::path candidates[] = {
        std::filesystem::path(name),
        exeDir / name,
        std::filesystem::path("shaders") / name,
        exeDir / "shaders" / name,
        std::filesystem::path("..") / "shaders" / name,
        std::filesystem::path("src") / "shaders" / name
    };
    for (const auto& path : candidates) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) continue;
        const auto bytes = static_cast<size_t>(file.tellg());
        if (bytes == 0 || bytes % sizeof(u32) != 0) continue;
        Vector<u32> words(bytes / sizeof(u32));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(words.data()),
                  static_cast<std::streamsize>(bytes));
        if (file) return words;
    }
    return {};
}

void StorageBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &barrier, 0, nullptr, 0, nullptr);
}

// Copies one thermal ping-pong pair on `cmd`, bracketing the transfers with
// layout-preserving barriers: the sources move their last writer's access to
// TRANSFER_READ, the destinations hand TRANSFER_WRITE back to the consumer
// stage. All four images live in GENERAL throughout the pipeline's life.
void CopyThermalPair(VkCommandBuffer cmd,
                     const GpuImage& src0, const GpuImage& src1,
                     const GpuImage& dst0, const GpuImage& dst1,
                     VkAccessFlags sourceAccess,
                     VkPipelineStageFlags sourceStage) {
    std::array<VkImageMemoryBarrier, 2> ready{};
    const GpuImage* sources[2] = {&src0, &src1};
    for (u32 i = 0; i < 2; ++i) {
        ready[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ready[i].srcAccessMask = sourceAccess;
        ready[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        ready[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        ready[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ready[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready[i].image = sources[i]->GetImage();
        ready[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ready[i].subresourceRange.levelCount = 1;
        ready[i].subresourceRange.layerCount = 1;
    }
    vkCmdPipelineBarrier(cmd, sourceStage, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr,
                         static_cast<u32>(ready.size()), ready.data());
    const VkImageCopy copy{
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {0, 0, 0},
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {0, 0, 0},
        {src0.GetExtent().width, src0.GetExtent().height, 1}};
    vkCmdCopyImage(cmd, src0.GetImage(), VK_IMAGE_LAYOUT_GENERAL,
                   dst0.GetImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    vkCmdCopyImage(cmd, src1.GetImage(), VK_IMAGE_LAYOUT_GENERAL,
                   dst1.GetImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    std::array<VkImageMemoryBarrier, 2> visible{};
    const GpuImage* destinations[2] = {&dst0, &dst1};
    for (u32 i = 0; i < 2; ++i) {
        visible[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        visible[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        visible[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT |
                                   VK_ACCESS_TRANSFER_READ_BIT;
        visible[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        visible[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        visible[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        visible[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        visible[i].image = destinations[i]->GetImage();
        visible[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        visible[i].subresourceRange.levelCount = 1;
        visible[i].subresourceRange.layerCount = 1;
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr,
                         static_cast<u32>(visible.size()), visible.data());
}

// Builds the ISP configuration rows (binding 12) and the shared ISP flag
// word from a capture config. Row 0 carries the white balance and the flag
// word; rows 1-3 the color correction matrix (row3.y doubles as the defect
// count); row 4 gamma/denoise/sharpen; row 5 percentiles/tone/palette; row 6
// the dynamic exposure/gain/well/ADC ceiling the statistics passes read;
// row 7 the HSV grading (hue offset degrees, saturation scale, value gamma,
// empirical-noise sigma); row 8.x the drift sigma. Configure and every M4-4
// effective-config re-upload share this so the two paths cannot drift apart.
void BuildIspRows(const camera::CameraConfig& config,
                  std::array<std::array<f32, 4>, 9>& rows, u32& flags) {
    const auto& isp = config.isp;
    rows = {};
    flags = 0;
    if (isp.denoise && isp.denoiseStrength > 0.0) flags |= isp::kFlagDenoise;
    if (isp.sharpen && isp.sharpenStrength > 0.0) flags |= isp::kFlagSharpen;
    if (isp.clipOutOfGamut) flags |= isp::kFlagClip;
    if (isp.infraredTone == DisplayToneMode::Clahe)
        flags |= isp::kFlagClahePersistent;
    if (!isp.defectPixels.empty()) flags |= isp::kFlagDefects;
    if (isp.hsv.empiricalNoise) flags |= isp::kFlagEmpiricalNoise;
    if (isp.hsv.temporalDrift) flags |= isp::kFlagTemporalDrift;
    // The HSV pass is dispatched at all only when a parameter is off its
    // default -- the CPU caller skips ApplyHsv the same way -- so an
    // untouched chain never reaches the pass and stays bit-identical.
    if (isp.hsv.hueOffsetDegrees != 0.0 || isp.hsv.saturationScale != 1.0 ||
        isp.hsv.valueGamma != 1.0 || isp.hsv.empiricalNoise ||
        isp.hsv.temporalDrift)
        flags |= isp::kFlagHsv;
    rows[0] = {{static_cast<f32>(isp.whiteBalance[0]),
                static_cast<f32>(isp.whiteBalance[1]),
                static_cast<f32>(isp.whiteBalance[2]),
                static_cast<f32>(flags)}};
    for (u32 i = 0; i < 9; ++i)
        rows[1 + i / 4][i % 4] = static_cast<f32>(isp.deviceToLinearSrgb[i]);
    // Row3.y doubles as the defect count (row3.x carries ccm[8]).
    rows[3][1] = static_cast<f32>(isp.defectPixels.size());
    rows[4] = {{static_cast<f32>(isp.toneGamma),
                static_cast<f32>(std::clamp(isp.denoiseStrength, 0.0, 1.0)),
                static_cast<f32>(std::max(isp.sharpenStrength, 0.0)), 0.0f}};
    rows[5] = {{static_cast<f32>(isp.contrastLowPercentile),
                static_cast<f32>(isp.contrastHighPercentile),
                static_cast<f32>(isp.infraredTone),
                static_cast<f32>(isp.infraredPalette)}};
    const f32 adcMax =
        std::exp2(static_cast<f32>(config.readout.adcBits)) - 1.0f;
    rows[6] = {{static_cast<f32>(config.readout.exposureSeconds),
                static_cast<f32>(config.readout.analogGain),
                static_cast<f32>(config.photon.fullWellElectrons), adcMax}};
    rows[7] = {{static_cast<f32>(isp.hsv.hueOffsetDegrees),
                static_cast<f32>(isp.hsv.saturationScale),
                static_cast<f32>(isp.hsv.valueGamma),
                static_cast<f32>(isp.hsv.empiricalNoiseSigma)}};
    rows[8] = {{static_cast<f32>(isp.hsv.temporalDriftSigma), 0.0f, 0.0f,
                0.0f}};
}

// Copies the statistics buffer (previous-frame AGC window + histogram) in
// the direction the M4-2 checkpoint needs, with the same access hand-off as
// the image pair above.
void CopyStatsBuffer(VkCommandBuffer cmd, const GpuBuffer& source,
                     const GpuBuffer& destination,
                     VkAccessFlags sourceAccess,
                     VkPipelineStageFlags sourceStage) {
    VkBufferMemoryBarrier ready{};
    ready.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    ready.srcAccessMask = sourceAccess;
    ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ready.buffer = source.GetHandle();
    ready.offset = 0;
    ready.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, sourceStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 1, &ready, 0, nullptr);
    VkBufferCopy copy{};
    copy.size = source.GetSize();
    vkCmdCopyBuffer(cmd, source.GetHandle(), destination.GetHandle(), 1, &copy);
    VkBufferMemoryBarrier visible{};
    visible.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    visible.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    visible.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT;
    visible.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    visible.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    visible.buffer = destination.GetHandle();
    visible.offset = 0;
    visible.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 1, &visible, 0, nullptr);
}

f64 EffectiveWavelengthNm(const camera::ResponseStack& response,
                          camera::DetectorKind detector) {
    // Preview-effective wavelength assumes flat incident spectral radiance
    // inside this channel. The physical capture integrates the actual scene
    // spectrum; this one number only sizes the approximate GPU Gaussian PSF.
    const camera::ResponseCurve* base = response.systemResponse
        ? &*response.systemResponse
        : detector == camera::DetectorKind::Photon
              ? &*response.quantumEfficiency : &*response.thermalAbsorptance;
    const auto at = [](const camera::ResponseCurve& curve, f64 lambdaNm) {
        if (lambdaNm < curve.MinNm() || lambdaNm > curve.MaxNm()) return 0.0;
        const auto upper = std::lower_bound(
            curve.wavelengthNm.begin(), curve.wavelengthNm.end(), lambdaNm);
        if (upper == curve.wavelengthNm.begin())
            return curve.amplitude * curve.value.front();
        if (upper == curve.wavelengthNm.end())
            return curve.amplitude * curve.value.back();
        const size_t hi = static_cast<size_t>(upper - curve.wavelengthNm.begin());
        const f64 t = (lambdaNm - curve.wavelengthNm[hi - 1]) /
                      (curve.wavelengthNm[hi] - curve.wavelengthNm[hi - 1]);
        return curve.amplitude *
            (curve.value[hi - 1] + t * (curve.value[hi] - curve.value[hi - 1]));
    };
    Vector<f64> knots = base->wavelengthNm;
    for (const auto* extra : {
             response.lensTransmission ? &*response.lensTransmission : nullptr,
             response.filterTransmission ? &*response.filterTransmission : nullptr}) {
        if (!extra) continue;
        for (const f64 lambda : extra->wavelengthNm)
            if (lambda > base->MinNm() && lambda < base->MaxNm())
                knots.push_back(lambda);
    }
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    const auto density = [&](f64 lambda) {
        f64 value = at(*base, lambda);
        if (response.lensTransmission)
            value *= at(*response.lensTransmission, lambda);
        if (response.filterTransmission)
            value *= at(*response.filterTransmission, lambda);
        // A photon detector's electron response includes photon energy;
        // a thermal absorber is weighted directly by power.
        if (detector == camera::DetectorKind::Photon) value *= lambda;
        return value;
    };
    // Three-point Gauss-Legendre is exact through degree 5 on each knot
    // interval. The stack is at most three linear factors, multiplied by up
    // to lambda^2 for the first moment.
    constexpr f64 node = 0.77459666924148337704;
    const std::array<std::pair<f64, f64>, 3> gauss{{
        {-node, 5.0 / 9.0}, {0.0, 8.0 / 9.0}, {node, 5.0 / 9.0}
    }};
    f64 weight = 0.0, moment = 0.0;
    for (size_t i = 1; i < knots.size(); ++i) {
        const f64 middle = 0.5 * (knots[i - 1] + knots[i]);
        const f64 half = 0.5 * (knots[i] - knots[i - 1]);
        for (const auto& [x, w] : gauss) {
            const f64 lambda = middle + half * x;
            const f64 contribution = half * w * density(lambda);
            weight += contribution;
            moment += lambda * contribution;
        }
    }
    return weight > 0.0 ? moment / weight :
           0.5 * (base->MinNm() + base->MaxNm());
}

VkImageMemoryBarrier TransitionToGeneral(const GpuImage& image) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.GetImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    return barrier;
}

} // namespace

struct GpuCameraPipeline::Impl {
    VulkanContext& context;
    VkDevice device = VK_NULL_HANDLE;
    u32 width = 0, height = 0;
    camera::CameraConfig config;
    bool configured = false;
    bool imagesInGeneral = false;
    bool hasCapture = false;
    bool firstCaptureWasFirst = false;
    u64 lastAcquisitionIndex = 0;
    f64 lastFrameTimeSeconds = 0.0;
    f64 beforeFrameTimeSeconds = 0.0;
    f64 currentDeltaSeconds = 0.0;
    u32 stateBefore = 0, stateCurrent = 1;
    u32 ispFlags = 0;
    std::array<f32, 3> sigmaX{}, sigmaY{};
    ConfigRows configRows{};

    std::unique_ptr<GpuImage> psfTemp, blurred, rawDn, corrected, display;
    std::unique_ptr<GpuImage> fastRgbRate;
    std::unique_ptr<GpuImage> compositedRate;  // dynamic compositor output, PSF input
    std::unique_ptr<GpuImage> expectedElectrons, preAdcElectrons;
    std::unique_ptr<GpuImage> linearRgb, colorRgb, agcSource;  // ISP stage images
    std::array<std::unique_ptr<GpuImage>, 2> thermalState;
    std::array<std::unique_ptr<GpuImage>, 2> thermalCheckpoint; // M4-2 history snapshot
    std::unique_ptr<GpuImage> randomVectors;
    std::unique_ptr<GpuBuffer> configBuffer, nucGainBuffer, nucOffsetBuffer;
    std::unique_ptr<GpuBuffer> layerCameraBuffer;   // binding 10, per-stratum frames
    std::unique_ptr<GpuBuffer> dynamicCounterBuffer; // binding 11, 4 x u32
    std::unique_ptr<GpuBuffer> ispConfigBuffer;     // binding 12, 8 x float4
    std::unique_ptr<GpuBuffer> defectPixelsBuffer;  // binding 13, uint2 array
    std::unique_ptr<GpuBuffer> ispStatsBuffer;      // binding 14, 264 x u32
    std::unique_ptr<GpuBuffer> cdfBuffer;           // binding 15, 256 x f32
    std::unique_ptr<GpuBuffer> tileStatsBuffer;     // binding 16, per-tile float4
    std::unique_ptr<GpuBuffer> ispStatsCheckpoint;  // M4-2 previous-frame AGC window snapshot

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline psfPipeline = VK_NULL_HANDLE;
    VkPipeline readoutPipeline = VK_NULL_HANDLE;
    VkPipeline statsTilesPipeline = VK_NULL_HANDLE;
    VkPipeline statsReducePipeline = VK_NULL_HANDLE;
    VkPipeline statsHistPipeline = VK_NULL_HANDLE;
    VkPipeline cdfPipeline = VK_NULL_HANDLE;
    VkPipeline demosaicPipeline = VK_NULL_HANDLE;
    VkPipeline colorPipeline = VK_NULL_HANDLE;
    VkPipeline displayPipeline = VK_NULL_HANDLE;
    VkPipeline hsvPipeline = VK_NULL_HANDLE;
    VkPipeline rngPipeline = VK_NULL_HANDLE;
    VkPipeline fastRgbPipeline = VK_NULL_HANDLE;
    VkPipeline dynamicPipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    // 0/1: PSF H/V reading the caller's measured rate directly (unstratified
    // path). 2/3: readout ping/pong. 4: display. 5: RNG audit. 6: fast RGB.
    // 7: dynamic compositor (strata array inputs, composited output).
    // 8/9: PSF H/V reading the compositor output (stratified path).
    // 10: acquisition statistics. 11: demosaic. 12: color. 13: CDF.
    // 14: HSV (display image bound as both input and in-place output).
    // Immutable while frames may be in flight; Configure waits idle first.
    std::array<VkDescriptorSet, 15> sets{};
    VkImageView measurementView = VK_NULL_HANDLE;
    VkImageView fastRgbInputView = VK_NULL_HANDLE;
    VkImageView strataRateView = VK_NULL_HANDLE;
    VkImageView strataDepthView = VK_NULL_HANDLE;

    Impl(VulkanContext& c, u32 w, u32 h)
        : context(c), device(c.GetDevice()), width(w), height(h) {}

    ~Impl() {
        if (device == VK_NULL_HANDLE) return;
        for (auto pipeline : {psfPipeline, readoutPipeline, statsTilesPipeline,
                              statsReducePipeline, statsHistPipeline,
                              cdfPipeline, demosaicPipeline, colorPipeline,
                              displayPipeline, hsvPipeline, rngPipeline,
                              fastRgbPipeline, dynamicPipeline})
            if (pipeline != VK_NULL_HANDLE)
                vkDestroyPipeline(device, pipeline, nullptr);
        if (descriptorPool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (pipelineLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (setLayout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    }

    bool MakeImage(std::unique_ptr<GpuImage>& target, VkFormat format) {
        target = std::make_unique<GpuImage>(
            context.GetAllocator(), device, width, height, format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        return target && target->IsValid();
    }

    Result<void, String> Initialize();
    Result<void, String> CreatePipeline(StringView spirvName, VkPipeline& output);
    void TransitionImages(VkCommandBuffer cmd);
    void UpdateSet(VkDescriptorSet set, const std::array<const GpuImage*, 7>& images);
    void BindAndDispatch(VkCommandBuffer cmd, VkPipeline pipeline,
                         VkDescriptorSet set, const CameraPush& push);
    void BindAndDispatchGrid(VkCommandBuffer cmd, VkPipeline pipeline,
                             VkDescriptorSet set, const CameraPush& push,
                             u32 groupsX, u32 groupsY);
    CameraPush MakePush(u64 acquisitionIndex, f64 frameTimeSeconds) const;
};

Result<void, String> GpuCameraPipeline::Impl::CreatePipeline(
    StringView spirvName, VkPipeline& output) {
    const auto spirv = LoadSpirv(spirvName);
    if (spirv.empty())
        return Result<void, String>::Err(
            "camera shader not found: " + String(spirvName));
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = spirv.size() * sizeof(u32);
    moduleInfo.pCode = spirv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &module) != VK_SUCCESS)
        return Result<void, String>::Err(
            "cannot create camera shader module: " + String(spirvName));
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = stage;
    info.layout = pipelineLayout;
    const VkResult created = vkCreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &info, nullptr, &output);
    vkDestroyShaderModule(device, module, nullptr);
    if (created != VK_SUCCESS)
        return Result<void, String>::Err(
            "cannot create camera compute pipeline: " + String(spirvName));
    return Result<void, String>::Ok();
}

Result<void, String> GpuCameraPipeline::Impl::Initialize() {
    if (width == 0 || height == 0)
        return Result<void, String>::Err("GPU camera extent must be positive");
    if (!MakeImage(psfTemp, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(blurred, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(rawDn, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(corrected, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(display, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(fastRgbRate, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(compositedRate, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(expectedElectrons, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(preAdcElectrons, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(linearRgb, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(colorRgb, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(agcSource, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(thermalState[0], VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(thermalState[1], VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(thermalCheckpoint[0], VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(thermalCheckpoint[1], VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(randomVectors, VK_FORMAT_R32G32B32A32_UINT))
        return Result<void, String>::Err("cannot allocate GPU camera images");

    // Bindings 0-6 are storage images, 7-16 storage buffers. The camera
    // config (binding 9), per-stratum frames (10) and the ISP set (12-16)
    // share one layout; every set binds all of them, used or not.
    std::array<VkDescriptorSetLayoutBinding, 17> bindings{};
    for (u32 i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = i < 7
            ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // Bindings 0/1 may hold 2D-array views (the dynamic compositor's strata)
    // that other sets of this shared layout never use; the compositor is the
    // only consumer and only touches them when a stratified measurement is
    // actually recorded.
    std::array<VkDescriptorBindingFlags, 17> bindingFlags{};
    bindingFlags[0] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    bindingFlags[1] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bindingFlagsInfo.bindingCount = static_cast<u32>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();
    VkDescriptorSetLayoutCreateInfo setInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setInfo.pNext = &bindingFlagsInfo;
    setInfo.bindingCount = static_cast<u32>(bindings.size());
    setInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &setInfo, nullptr, &setLayout) != VK_SUCCESS)
        return Result<void, String>::Err("cannot create GPU camera descriptor layout");

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(CameraPush);
    VkPipelineLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        return Result<void, String>::Err("cannot create GPU camera pipeline layout");

    const std::array<std::pair<StringView, VkPipeline*>, 13> shaders{{
        {"camera_psf.comp.spv", &psfPipeline},
        {"camera_readout.comp.spv", &readoutPipeline},
        {"camera_stats_tiles.comp.spv", &statsTilesPipeline},
        {"camera_stats_reduce.comp.spv", &statsReducePipeline},
        {"camera_stats_hist.comp.spv", &statsHistPipeline},
        {"camera_cdf.comp.spv", &cdfPipeline},
        {"camera_demosaic.comp.spv", &demosaicPipeline},
        {"camera_color.comp.spv", &colorPipeline},
        {"camera_display.comp.spv", &displayPipeline},
        {"camera_hsv.comp.spv", &hsvPipeline},
        {"camera_rng_vectors.comp.spv", &rngPipeline},
        {"camera_fast_rgb.comp.spv", &fastRgbPipeline},
        {"camera_dynamic.comp.spv", &dynamicPipeline}
    }};
    for (const auto& [name, pipeline] : shaders) {
        auto created = CreatePipeline(name, *pipeline);
        if (!created) return created;
    }

    const std::array<VkDescriptorPoolSize, 2> sizes{{
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7u * static_cast<u32>(sets.size())},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10u * static_cast<u32>(sets.size())}
    }};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = static_cast<u32>(sets.size());
    poolInfo.poolSizeCount = static_cast<u32>(sizes.size());
    poolInfo.pPoolSizes = sizes.data();
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        return Result<void, String>::Err("cannot create GPU camera descriptor pool");
    std::array<VkDescriptorSetLayout, 15> layouts{};
    layouts.fill(setLayout);
    VkDescriptorSetAllocateInfo allocInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<u32>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &allocInfo, sets.data()) != VK_SUCCESS)
        return Result<void, String>::Err("cannot allocate GPU camera descriptor sets");

    // Per-stratum camera frames (binding 10) and the compositor's disocclusion
    // counters (binding 11). The frame buffer is re-uploaded per measurement;
    // both are bound into every set, used or not.
    layerCameraBuffer = std::make_unique<GpuBuffer>(
        context.GetAllocator(),
        8u * sizeof(DynamicLayerCamera),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    dynamicCounterBuffer = std::make_unique<GpuBuffer>(
        context.GetAllocator(), 4u * sizeof(u32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    // ISP statistics (binding 14), Equalize CDF (15) and per-tile partials
    // (16). The statistics buffer is zeroed so the first acquisition's
    // previous-frame window is degenerate and the display pass falls back to
    // the current frame's range.
    ispStatsBuffer = std::make_unique<GpuBuffer>(
        context.GetAllocator(), isp::kStatsSlots * sizeof(u32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    cdfBuffer = std::make_unique<GpuBuffer>(
        context.GetAllocator(), 256u * sizeof(f32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    const VkDeviceSize tileCount =
        static_cast<VkDeviceSize>((width + 15u) / 16u) * ((height + 15u) / 16u);
    // Three float4 rows per tile: min/max/sum/sat, per-channel unsaturated
    // sums, per-channel unsaturated counts (camera_stats phases 0/1).
    tileStatsBuffer = std::make_unique<GpuBuffer>(
        context.GetAllocator(), tileCount * 12u * sizeof(f32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    if (!layerCameraBuffer->IsValid() || !dynamicCounterBuffer->IsValid() ||
        !ispStatsBuffer->IsValid() || !cdfBuffer->IsValid() ||
        !tileStatsBuffer->IsValid())
        return Result<void, String>::Err(
            "cannot allocate GPU camera dynamic buffers");
    // M4-2: the checkpoint twin of the statistics buffer. The previous
    // frame's Linear-AGC window rides in the stats slots; without a snapshot
    // a replayed tick's display tone would read the discarded run's window
    // and the replay would not be bit-identical.
    ispStatsCheckpoint = std::make_unique<GpuBuffer>(
        context.GetAllocator(), isp::kStatsSlots * sizeof(u32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    if (!ispStatsCheckpoint->IsValid())
        return Result<void, String>::Err(
            "cannot allocate GPU camera checkpoint buffer");
    const std::array<DynamicLayerCamera, 8> zeroLayers{};
    layerCameraBuffer->Upload(zeroLayers.data(), sizeof(zeroLayers));
    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, dynamicCounterBuffer->GetHandle(), 0,
                        VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, ispStatsBuffer->GetHandle(), 0, VK_WHOLE_SIZE, 0);
    });
    return Result<void, String>::Ok();
}

void GpuCameraPipeline::Impl::TransitionImages(VkCommandBuffer cmd) {
    if (imagesInGeneral) return;
    const std::array<const GpuImage*, 17> images{
        psfTemp.get(), blurred.get(), rawDn.get(), corrected.get(),
        display.get(), fastRgbRate.get(), expectedElectrons.get(), preAdcElectrons.get(),
        thermalState[0].get(), thermalState[1].get(), randomVectors.get(),
        compositedRate.get(), linearRgb.get(), colorRgb.get(), agcSource.get(),
        thermalCheckpoint[0].get(), thermalCheckpoint[1].get()};
    std::array<VkImageMemoryBarrier, 17> barriers{};
    for (size_t i = 0; i < images.size(); ++i)
        barriers[i] = TransitionToGeneral(*images[i]);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, static_cast<u32>(barriers.size()),
                         barriers.data());
    imagesInGeneral = true;
}

void GpuCameraPipeline::Impl::UpdateSet(
    VkDescriptorSet set, const std::array<const GpuImage*, 7>& images) {
    std::array<VkDescriptorImageInfo, 7> imageInfo{};
    std::array<VkDescriptorBufferInfo, 10> bufferInfo{};
    std::array<VkWriteDescriptorSet, 17> writes{};
    for (u32 i = 0; i < imageInfo.size(); ++i) {
        imageInfo[i].imageView = images[i]->GetView();
        imageInfo[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &imageInfo[i];
    }
    const GpuBuffer* buffers[] = {
        nucGainBuffer.get(), nucOffsetBuffer.get(), configBuffer.get(),
        layerCameraBuffer.get(), dynamicCounterBuffer.get(),
        ispConfigBuffer.get(), defectPixelsBuffer.get(),
        ispStatsBuffer.get(), cdfBuffer.get(), tileStatsBuffer.get()};
    for (u32 i = 0; i < bufferInfo.size(); ++i) {
        bufferInfo[i].buffer = buffers[i]->GetHandle();
        bufferInfo[i].offset = 0;
        bufferInfo[i].range = buffers[i]->GetSize();
        writes[7 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[7 + i].dstSet = set;
        writes[7 + i].dstBinding = 7 + i;
        writes[7 + i].descriptorCount = 1;
        writes[7 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[7 + i].pBufferInfo = &bufferInfo[i];
    }
    vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()),
                           writes.data(), 0, nullptr);
}

void GpuCameraPipeline::Impl::BindAndDispatch(
    VkCommandBuffer cmd, VkPipeline pipeline, VkDescriptorSet set,
    const CameraPush& push) {
    BindAndDispatchGrid(cmd, pipeline, set, push,
                        (width + 15) / 16, (height + 15) / 16);
}

void GpuCameraPipeline::Impl::BindAndDispatchGrid(
    VkCommandBuffer cmd, VkPipeline pipeline, VkDescriptorSet set,
    const CameraPush& push, u32 groupsX, u32 groupsY) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

CameraPush GpuCameraPipeline::Impl::MakePush(
    u64 acquisitionIndex, f64 frameTimeSeconds) const {
    CameraPush push;
    push.width = width;
    push.height = height;
    push.physicalWidth = config.optics.sensorWidthPx;
    push.physicalHeight = config.optics.sensorHeightPx;
    push.channelCount = static_cast<u32>(config.device.channels.size());
    push.cfa = static_cast<u32>(config.device.cfa);
    push.detector = static_cast<u32>(config.device.detector);
    push.seed = camera::DeviceRandomSeed(config);
    push.acquisitionLo = static_cast<u32>(acquisitionIndex);
    push.acquisitionHi = static_cast<u32>(acquisitionIndex >> 32);
    push.adcBits = config.readout.adcBits;
    push.frameTimeSeconds = static_cast<f32>(frameTimeSeconds);
    push.deltaSeconds = static_cast<f32>(currentDeltaSeconds);
    push.firstThermal = firstCaptureWasFirst ? 1u : 0u;
    push.sigmaR = sigmaX[0];
    push.sigmaG = sigmaX[1];
    push.sigmaB = sigmaX[2];
    push.sigmaRy = sigmaY[0];
    push.sigmaGy = sigmaY[1];
    push.sigmaBy = sigmaY[2];
    if (config.optics.cosFourthVignetting) push.flags |= Vignette;
    if (config.quality.noiseFree) push.flags |= NoiseFree;
    if (config.device.detector == camera::DetectorKind::Photon) {
        if (config.photon.enableShotNoise) push.flags |= Shot;
        if (config.photon.enableDarkCurrent) push.flags |= DarkCurrent;
        if (config.photon.enableDarkShotNoise) push.flags |= DarkShot;
        if (config.photon.enableReadNoise) push.flags |= Read;
        if (config.photon.enableFpn) push.flags |= Fpn;
        if (config.photon.applyNuc) push.flags |= Nuc;
        if (!config.photon.nucGainMap.empty()) push.flags |= NucGainMap;
        if (!config.photon.nucOffsetElectronsMap.empty()) push.flags |= NucOffsetMap;
    } else {
        if (!config.thermal.nucGainMap.empty()) push.flags |= NucGainMap;
        if (!config.thermal.nucOffsetDnMap.empty()) push.flags |= NucOffsetMap;
    }
    return push;
}

GpuCameraPipeline::GpuCameraPipeline(std::unique_ptr<Impl> impl)
    : m_impl(std::move(impl)) {}

GpuCameraPipeline::~GpuCameraPipeline() = default;

Result<std::unique_ptr<GpuCameraPipeline>, String>
GpuCameraPipeline::Create(VulkanContext& context, u32 width, u32 height) {
    auto impl = std::make_unique<Impl>(context, width, height);
    auto initialized = impl->Initialize();
    if (!initialized)
        return Fail<std::unique_ptr<GpuCameraPipeline>>(initialized.error());
    return std::unique_ptr<GpuCameraPipeline>(
        new GpuCameraPipeline(std::move(impl)));
}

Result<void, String>
GpuCameraPipeline::Configure(const camera::CameraConfig& config) {
    const auto valid = camera::ValidateCameraConfig(config);
    if (!valid) return Result<void, String>::Err(valid.error());
    if (config.device.channels.size() > 3)
        return Result<void, String>::Err(
            "GPU camera supports at most three response channels");
    if ((m_impl->width != config.optics.sensorWidthPx ||
         m_impl->height != config.optics.sensorHeightPx) &&
        config.quality.backend != camera::ProcessingBackend::GpuPreview)
        return Result<void, String>::Err(
            "physical CPU/reference camera requires full sensor extent; "
            "set gpu_preview for an explicitly scaled viewport");
    // Descriptors and static parameter buffers are immutable during a run.
    // A Qt window may have older command buffers in flight when a panel
    // changes the camera; wait before replacing anything they still read.
    if (vkDeviceWaitIdle(m_impl->device) != VK_SUCCESS)
        return Result<void, String>::Err(
            "device lost while reconfiguring GPU camera");
    m_impl->config = config;
    const auto& optics = config.optics;
    for (size_t i = 0; i < config.device.channels.size(); ++i) {
        f64 sigma = optics.psfSigmaPixelsOverride;
        if (sigma < 0.0) {
            const f64 wavelength = EffectiveWavelengthNm(
                config.device.channels[i].response, config.device.detector);
            sigma = 0.437 * wavelength * 1e-3 * optics.fNumber /
                    optics.pixelPitchUm;
        }
        const f64 sigmaX = sigma * m_impl->width / optics.sensorWidthPx;
        const f64 sigmaY = sigma * m_impl->height / optics.sensorHeightPx;
        if (std::max(sigmaX, sigmaY) > 4.0)
            QL_LOG_WARN("GPU camera effective PSF sigma ({}, {}) preview px is "
                        "capped at 4 px", sigmaX, sigmaY);
        m_impl->sigmaX[i] = static_cast<f32>(std::clamp(sigmaX, 0.0, 4.0));
        m_impl->sigmaY[i] = static_cast<f32>(std::clamp(sigmaY, 0.0, 4.0));
    }
    if (config.device.channels.size() == 1) {
        m_impl->sigmaX[1] = m_impl->sigmaX[2] = m_impl->sigmaX[0];
        m_impl->sigmaY[1] = m_impl->sigmaY[2] = m_impl->sigmaY[0];
    }

    std::array<f32, 3> thermalNoise{
        static_cast<f32>(config.thermal.readNoiseDnRms),
        static_cast<f32>(config.thermal.readNoiseDnRms),
        static_cast<f32>(config.thermal.readNoiseDnRms)};
    if (config.device.detector == camera::DetectorKind::Thermal &&
        config.thermal.netdKelvin > 0.0) {
        const auto area = camera::PixelCollectionAreaM2(optics);
        if (!area) return Result<void, String>::Err(area.error());
        for (size_t i = 0; i < config.device.channels.size(); ++i) {
            const auto slope = camera::BlackbodyThermalDerivativeWPerK(
                config.thermal.netdReferenceTemperatureK,
                config.device.channels[i].response, optics.fNumber, *area);
            if (!slope) return Result<void, String>::Err(slope.error());
            f64 noise = config.thermal.netdKelvin * *slope *
                        config.thermal.responsivityDnPerWatt;
            if (config.thermal.readoutWindowSeconds > 0.0)
                noise *= std::sqrt(
                    (1.0 / (2.0 * config.thermal.readoutWindowSeconds)) /
                    config.thermal.netdNoiseBandwidthHz);
            thermalNoise[i] = static_cast<f32>(noise);
        }
    }
    const auto& photon = config.photon;
    const auto& readout = config.readout;
    const auto& thermal = config.thermal;
    m_impl->configRows = {{
        {{static_cast<f32>(readout.exposureSeconds),
          static_cast<f32>(photon.fullWellElectrons),
          static_cast<f32>(photon.darkCurrentElectronsPerSecond),
          static_cast<f32>(photon.readNoiseElectronsRms)}},
        {{static_cast<f32>(photon.prnuSigma),
          static_cast<f32>(photon.dsnuElectronsRms),
          static_cast<f32>(photon.dsnuReferenceExposureSeconds),
          static_cast<f32>(photon.biasDnRms)}},
        {{static_cast<f32>(readout.analogGain),
          static_cast<f32>(readout.electronsPerDn),
          static_cast<f32>(readout.blackLevelDn),
          static_cast<f32>(photon.nucResidualFraction)}},
        {{static_cast<f32>(thermal.timeConstantSeconds),
          static_cast<f32>(thermal.responsivityDnPerWatt),
          static_cast<f32>(thermal.driftDnPerSecond),
          thermalNoise[0]}},
        {{thermalNoise[1], thermalNoise[2], 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f, 0.0f}}
    }};
    // The RGB approximation starts with linear scene RGB. Each row maps it
    // to one latent device channel, then applies the same absolute radiance,
    // aperture, pixel-area and response integral as the CPU adapter.
    const auto aperture = camera::ApertureSolidAngleSr(optics.fNumber);
    const auto pixelArea = camera::PixelCollectionAreaM2(optics);
    if (!aperture || !pixelArea)
        return Result<void, String>::Err(
            aperture ? pixelArea.error() : aperture.error());
    const f64 radianceScale = config.fastRgbRadianceScale > 0.0
        ? config.fastRgbRadianceScale : 1.0;
    for (size_t channel = 0; channel < config.device.channels.size(); ++channel) {
        const auto& response = config.device.channels[channel].response;
        const auto& base = response.systemResponse ? *response.systemResponse :
            config.device.detector == camera::DetectorKind::Photon
                ? *response.quantumEfficiency : *response.thermalAbsorptance;
        Vector<camera::SpectralIrradianceSample> flat;
        flat.reserve(base.wavelengthNm.size());
        for (const f64 wavelength : base.wavelengthNm)
            flat.push_back({wavelength, 1.0});
        f64 responseIntegral = 0.0;
        if (config.device.detector == camera::DetectorKind::Photon) {
            const auto measured = camera::IntegratePhoton(
                flat, response, pixelArea.value());
            if (!measured) return Result<void, String>::Err(measured.error());
            responseIntegral = measured.value().electronRatePerSecond;
        } else {
            const auto measured = camera::IntegrateThermal(
                flat, response, pixelArea.value());
            if (!measured) return Result<void, String>::Err(measured.error());
            responseIntegral = measured.value().absorbedPowerW;
        }
        const f64 factor = radianceScale * aperture.value() * responseIntegral;
        for (size_t rgb = 0; rgb < 3; ++rgb)
            m_impl->configRows[5 + channel][rgb] = static_cast<f32>(
                config.fastRgbToDevice[channel * 3 + rgb] * factor);
    }
    m_impl->configRows[8][0] = static_cast<f32>(optics.focalLengthMm * 1e-3);
    m_impl->configRows[8][1] = static_cast<f32>(optics.pixelPitchUm * 1e-6);
    const auto makeBuffer = [&](const void* data, VkDeviceSize size)
        -> std::unique_ptr<GpuBuffer> {
        auto buffer = std::make_unique<GpuBuffer>(
            m_impl->context.GetAllocator(), size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        if (!buffer->IsValid()) return {};
        buffer->Upload(data, size);
        return buffer;
    };
    m_impl->configBuffer = makeBuffer(
        m_impl->configRows.data(), sizeof(m_impl->configRows));
    if (!m_impl->configBuffer)
        return Result<void, String>::Err("cannot upload GPU camera parameters");
    const auto& gainMap = config.device.detector == camera::DetectorKind::Photon
        ? config.photon.nucGainMap : config.thermal.nucGainMap;
    const auto& offsetMap = config.device.detector == camera::DetectorKind::Photon
        ? config.photon.nucOffsetElectronsMap : config.thermal.nucOffsetDnMap;
    const auto uploadMap = [&](const std::vector<f64>& values, f32 fallback)
        -> std::unique_ptr<GpuBuffer> {
        std::vector<f32> upload;
        upload.reserve(std::max<size_t>(values.size(), 1));
        if (values.empty()) upload.push_back(fallback);
        else for (const f64 value : values) upload.push_back(static_cast<f32>(value));
        return makeBuffer(upload.data(), upload.size() * sizeof(f32));
    };
    m_impl->nucGainBuffer = uploadMap(gainMap, 1.0f);
    m_impl->nucOffsetBuffer = uploadMap(offsetMap, 0.0f);
    if (!m_impl->nucGainBuffer || !m_impl->nucOffsetBuffer)
        return Result<void, String>::Err("cannot upload GPU camera NUC maps");

    // The ISP configuration (binding 12) is one row of float4 per parameter
    // group; see BuildIspRows for the layout. The buffer is CPU_TO_GPU so the
    // M4-4 AE/AWB loop can re-upload the dynamic rows (white balance, row 6
    // exposure/gain, HSV) per acquisition through ApplyEffectiveConfig. The
    // defect list (binding 13) is a uint2 array with a one-element
    // placeholder when empty, matching the NUC map convention.
    std::array<std::array<f32, 4>, 9> ispRows{};
    BuildIspRows(config, ispRows, m_impl->ispFlags);
    m_impl->ispConfigBuffer = makeBuffer(ispRows.data(), sizeof(ispRows));
    std::vector<std::array<u32, 2>> defects = config.isp.defectPixels;
    if (defects.empty()) defects.push_back({0u, 0u});
    if (!m_impl->ispConfigBuffer)
        return Result<void, String>::Err("cannot upload GPU camera ISP parameters");
    m_impl->defectPixelsBuffer = std::make_unique<GpuBuffer>(
        m_impl->context.GetAllocator(),
        std::max<size_t>(defects.size(), 1u) * 2u * sizeof(u32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    if (!m_impl->defectPixelsBuffer)
        return Result<void, String>::Err("cannot allocate GPU camera defect list");
    m_impl->defectPixelsBuffer->Upload(defects.data(),
                                       defects.size() * 2u * sizeof(u32));

    const auto dummy = m_impl->psfTemp.get();
    m_impl->UpdateSet(m_impl->sets[0], {
        dummy, m_impl->psfTemp.get(),
        dummy, dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[1], {
        m_impl->psfTemp.get(), m_impl->blurred.get(),
        dummy, dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[2], {
        m_impl->blurred.get(), m_impl->thermalState[0].get(),
        m_impl->rawDn.get(), m_impl->corrected.get(),
        m_impl->expectedElectrons.get(), m_impl->preAdcElectrons.get(),
        m_impl->thermalState[1].get()});
    m_impl->UpdateSet(m_impl->sets[3], {
        m_impl->blurred.get(), m_impl->thermalState[1].get(),
        m_impl->rawDn.get(), m_impl->corrected.get(),
        m_impl->expectedElectrons.get(), m_impl->preAdcElectrons.get(),
        m_impl->thermalState[0].get()});
    // Display: the color-pass product feeds the visible branch, the corrected
    // scalar feeds the infrared branch; both write the display and agcSource.
    m_impl->UpdateSet(m_impl->sets[4], {
        m_impl->corrected.get(), m_impl->linearRgb.get(),
        m_impl->colorRgb.get(), m_impl->display.get(),
        m_impl->agcSource.get(), dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[5], {
        dummy, dummy, m_impl->randomVectors.get(),
        dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[6], {
        dummy, m_impl->fastRgbRate.get(),
        dummy, dummy, dummy, dummy, dummy});
    // Dynamic compositor set: bindings 0/1 re-pointed at the caller's strata
    // array views on the first stratified measurement (their PARTIALLY_BOUND
    // flag covers every other state of this set).
    m_impl->UpdateSet(m_impl->sets[7], {
        m_impl->compositedRate.get(), m_impl->compositedRate.get(),
        m_impl->compositedRate.get(), dummy, dummy, dummy, dummy});
    // PSF pair reading the compositor output (stratified path).
    m_impl->UpdateSet(m_impl->sets[8], {
        m_impl->compositedRate.get(), m_impl->psfTemp.get(),
        dummy, dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[9], {
        m_impl->psfTemp.get(), m_impl->blurred.get(),
        dummy, dummy, dummy, dummy, dummy});
    // Acquisition statistics read the corrected product; demosaic, color and
    // the CDF pass chain the ISP stage images.
    m_impl->UpdateSet(m_impl->sets[10], {
        m_impl->corrected.get(), dummy, dummy, dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[11], {
        m_impl->corrected.get(), m_impl->linearRgb.get(),
        dummy, dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[12], {
        dummy, m_impl->linearRgb.get(), m_impl->colorRgb.get(),
        dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[13], {
        dummy, dummy, dummy, dummy, dummy, dummy, dummy});
    // HSV: the display image is both input (binding 3) and in-place output
    // (binding 4); the pass runs after the display pass has finished and
    // every thread touches only its own texel.
    m_impl->UpdateSet(m_impl->sets[14], {
        dummy, dummy, dummy,
        m_impl->display.get(), m_impl->display.get(),
        dummy, dummy});
    m_impl->measurementView = VK_NULL_HANDLE;
    m_impl->fastRgbInputView = VK_NULL_HANDLE;
    m_impl->strataRateView = VK_NULL_HANDLE;
    m_impl->strataDepthView = VK_NULL_HANDLE;
    m_impl->configured = true;
    ResetState();
    return Result<void, String>::Ok();
}

Result<void, String>
GpuCameraPipeline::ApplyEffectiveConfig(const camera::CameraConfig& config) {
    if (!m_impl->configured)
        return Result<void, String>::Err("GPU camera must be configured first");
    // The effective capture config differs from the authored one only where
    // the AE/AWB loop moved exposure, gain or white balance; adopt it as the
    // pipeline's config so MakePush and the display/stats shading see one
    // consistent state, and refresh the two buffers the shaders actually
    // read. The PSF sigmas and response integrals do not depend on the
    // dynamic rows, so the static configBuffer cells stay valid apart from
    // the exposure/gain pair.
    m_impl->config = config;
    m_impl->configRows[0][0] = static_cast<f32>(config.readout.exposureSeconds);
    m_impl->configRows[2][0] = static_cast<f32>(config.readout.analogGain);
    if (!m_impl->configBuffer || !m_impl->ispConfigBuffer)
        return Result<void, String>::Err(
            "GPU camera parameter buffers are not allocated");
    m_impl->configBuffer->Upload(m_impl->configRows.data(),
                                 sizeof(m_impl->configRows));
    std::array<std::array<f32, 4>, 9> ispRows{};
    BuildIspRows(config, ispRows, m_impl->ispFlags);
    m_impl->ispConfigBuffer->Upload(ispRows.data(), sizeof(ispRows));
    return Result<void, String>::Ok();
}

Result<void, String>
GpuCameraPipeline::RecordDisplayReprocess(
    VkCommandBuffer cmd, const camera::CameraConfig& config) {
    if (!m_impl->configured || cmd == VK_NULL_HANDLE)
        return Result<void, String>::Err(
            "GPU camera display reprocess needs a configured pipeline and a "
            "command buffer");
    auto applied = ApplyEffectiveConfig(config);
    if (!applied) return applied;
    m_impl->TransitionImages(cmd);
    // The same acquisition index keys the HSV noise streams, so a reprocess
    // of unchanged parameters is bit-identical to the original display.
    auto push = m_impl->MakePush(m_impl->lastAcquisitionIndex,
                                 m_impl->lastFrameTimeSeconds);
    // demosaic -> color -> display -> HSV. No statistics passes: the
    // Linear-AGC window keeps riding in the statistics buffer, which is what
    // makes a reprocess stateless. corrected/linearRgb/colorRgb still hold
    // the last acquisition and serve as the inputs.
    m_impl->BindAndDispatch(cmd, m_impl->demosaicPipeline, m_impl->sets[11],
                            push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, m_impl->colorPipeline, m_impl->sets[12],
                            push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, m_impl->displayPipeline, m_impl->sets[4],
                            push);
    StorageBarrier(cmd);
    if (m_impl->ispFlags & isp::kFlagHsv) {
        m_impl->BindAndDispatch(cmd, m_impl->hsvPipeline, m_impl->sets[14],
                                push);
        StorageBarrier(cmd);
    }
    return Result<void, String>::Ok();
}

void GpuCameraPipeline::ResetState() {
    m_impl->hasCapture = false;
    m_impl->firstCaptureWasFirst = false;
    m_impl->lastAcquisitionIndex = 0;
    m_impl->lastFrameTimeSeconds = 0.0;
    m_impl->beforeFrameTimeSeconds = 0.0;
    m_impl->currentDeltaSeconds = 0.0;
    m_impl->stateBefore = 0;
    m_impl->stateCurrent = 1;
    // The statistics buffer persists the previous frame's window and
    // histogram; resetting history makes the next acquisition the first one
    // again, whose window must fall back to its own range.
    if (m_impl->ispStatsBuffer && m_impl->ispStatsBuffer->IsValid()) {
        CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, m_impl->ispStatsBuffer->GetHandle(), 0,
                            VK_WHOLE_SIZE, 0);
        });
    }
}

Result<GpuCameraPipeline::GpuCheckpoint, String>
GpuCameraPipeline::RecordStateCheckpoint(VkCommandBuffer cmd) {
    if (!m_impl->configured || cmd == VK_NULL_HANDLE)
        return Fail<GpuCheckpoint>(
            "GPU camera checkpoint needs a configured pipeline and a command "
            "buffer");
    m_impl->TransitionImages(cmd);
    CopyThermalPair(cmd, *m_impl->thermalState[0], *m_impl->thermalState[1],
                    *m_impl->thermalCheckpoint[0], *m_impl->thermalCheckpoint[1],
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    CopyStatsBuffer(cmd, *m_impl->ispStatsBuffer, *m_impl->ispStatsCheckpoint,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    GpuCheckpoint checkpoint;
    checkpoint.acquisitionIndex = m_impl->lastAcquisitionIndex;
    checkpoint.frameTimeSeconds = m_impl->lastFrameTimeSeconds;
    checkpoint.hasCapture = m_impl->hasCapture;
    checkpoint.firstCaptureWasFirst = m_impl->firstCaptureWasFirst;
    checkpoint.beforeFrameTimeSeconds = m_impl->beforeFrameTimeSeconds;
    checkpoint.currentDeltaSeconds = m_impl->currentDeltaSeconds;
    checkpoint.stateBefore = m_impl->stateBefore;
    checkpoint.stateCurrent = m_impl->stateCurrent;
    return checkpoint;
}

Result<void, String>
GpuCameraPipeline::RestoreStateCheckpoint(const GpuCheckpoint& checkpoint,
                                          VkCommandBuffer cmd) {
    if (!m_impl->configured || cmd == VK_NULL_HANDLE)
        return Result<void, String>::Err(
            "GPU camera checkpoint restore needs a configured pipeline and a "
            "command buffer");
    m_impl->TransitionImages(cmd);
    CopyThermalPair(cmd, *m_impl->thermalCheckpoint[0],
                    *m_impl->thermalCheckpoint[1], *m_impl->thermalState[0],
                    *m_impl->thermalState[1],
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_TRANSFER_BIT);
    CopyStatsBuffer(cmd, *m_impl->ispStatsCheckpoint, *m_impl->ispStatsBuffer,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_TRANSFER_BIT);
    m_impl->hasCapture = checkpoint.hasCapture;
    m_impl->firstCaptureWasFirst = checkpoint.firstCaptureWasFirst;
    m_impl->lastAcquisitionIndex = checkpoint.acquisitionIndex;
    m_impl->lastFrameTimeSeconds = checkpoint.frameTimeSeconds;
    m_impl->beforeFrameTimeSeconds = checkpoint.beforeFrameTimeSeconds;
    m_impl->currentDeltaSeconds = checkpoint.currentDeltaSeconds;
    m_impl->stateBefore = checkpoint.stateBefore;
    m_impl->stateCurrent = checkpoint.stateCurrent;
    return Result<void, String>::Ok();
}

GpuCameraPipeline::Images GpuCameraPipeline::GetOutputs() const {
    if (!m_impl->configured) return {};
    return {m_impl->rawDn.get(), m_impl->corrected.get(),
            m_impl->display.get(), m_impl->expectedElectrons.get(),
            m_impl->preAdcElectrons.get(),
            m_impl->strataRateView != VK_NULL_HANDLE
                ? m_impl->compositedRate.get() : nullptr,
            m_impl->linearRgb.get(), m_impl->colorRgb.get(),
            m_impl->agcSource.get()};
}

const GpuImage* GpuCameraPipeline::GetRandomVectors() const {
    return m_impl->randomVectors.get();
}

Result<std::array<u32, 4>, String>
GpuCameraPipeline::ReadDynamicCounters() const {
    if (!m_impl->dynamicCounterBuffer || !m_impl->dynamicCounterBuffer->IsValid())
        return Result<std::array<u32, 4>, String>::Err(
            "GPU camera dynamic counters are not allocated");
    std::array<u32, 4> counts{};
    GpuBuffer staging(m_impl->context.GetAllocator(), sizeof(counts),
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VMA_MEMORY_USAGE_GPU_TO_CPU);
    if (!staging.IsValid())
        return Result<std::array<u32, 4>, String>::Err(
            "cannot allocate GPU camera counter staging");
    CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
        VkBufferMemoryBarrier ready{};
        ready.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.buffer = m_impl->dynamicCounterBuffer->GetHandle();
        ready.offset = 0;
        ready.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 1, &ready, 0, nullptr);
        VkBufferCopy copy{};
        copy.size = sizeof(counts);
        vkCmdCopyBuffer(cmd, m_impl->dynamicCounterBuffer->GetHandle(),
                        staging.GetHandle(), 1, &copy);
    });
    const void* data = staging.Map();
    if (!data)
        return Result<std::array<u32, 4>, String>::Err(
            "cannot map GPU camera counter staging");
    std::memcpy(counts.data(), data, sizeof(counts));
    staging.Unmap();
    return counts;
}

Result<GpuCameraPipeline::IspStats, String>
GpuCameraPipeline::ReadIspStats() const {
    if (!m_impl->ispStatsBuffer || !m_impl->ispStatsBuffer->IsValid())
        return Result<IspStats, String>::Err(
            "GPU camera statistics are not allocated");
    std::array<u32, isp::kStatsSlots> slots{};
    GpuBuffer staging(m_impl->context.GetAllocator(), sizeof(slots),
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VMA_MEMORY_USAGE_GPU_TO_CPU);
    if (!staging.IsValid())
        return Result<IspStats, String>::Err(
            "cannot allocate GPU camera statistics staging");
    CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
        VkBufferMemoryBarrier ready{};
        ready.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ready.buffer = m_impl->ispStatsBuffer->GetHandle();
        ready.offset = 0;
        ready.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 1, &ready, 0, nullptr);
        VkBufferCopy copy{};
        copy.size = sizeof(slots);
        vkCmdCopyBuffer(cmd, m_impl->ispStatsBuffer->GetHandle(),
                        staging.GetHandle(), 1, &copy);
    });
    const void* data = staging.Map();
    if (!data)
        return Result<IspStats, String>::Err(
            "cannot map GPU camera statistics staging");
    std::memcpy(slots.data(), data, sizeof(slots));
    staging.Unmap();
    IspStats stats;
    std::memcpy(&stats.minValue, &slots[isp::kStatsMin], sizeof(f32));
    std::memcpy(&stats.maxValue, &slots[isp::kStatsMax], sizeof(f32));
    std::memcpy(&stats.meanValue, &slots[isp::kStatsMean], sizeof(f32));
    stats.saturatedCount = slots[isp::kStatsSat];
    stats.unsaturatedCount = slots[isp::kStatsUnsat];
    std::copy_n(slots.data(), stats.histogram.size(), stats.histogram.data());
    // Per-channel unsaturated sums (float bits) and counts (exact float bits
    // converted back), written by the stats reduce phase for the AWB loop.
    std::memcpy(stats.channelSums.data(), &slots[isp::kStatsChSumR],
                3u * sizeof(f32));
    f32 channelCounts[3]{};
    std::memcpy(channelCounts, &slots[isp::kStatsChCntR], 3u * sizeof(f32));
    for (u32 c = 0; c < 3u; ++c)
        stats.channelCounts[c] = static_cast<u32>(channelCounts[c] + 0.5f);
    return stats;
}

Result<void, String> GpuCameraPipeline::RecordMeasurement(
    VkCommandBuffer cmd, const GpuImage& measuredRate, u64 acquisitionIndex,
    f64 frameTimeSeconds, TimingQueries timing,
    const DynamicExposureInput& dynamic) {
    if (!m_impl->configured)
        return Result<void, String>::Err("GPU camera must be configured first");
    if (cmd == VK_NULL_HANDLE || !measuredRate.IsValid() ||
        measuredRate.GetFormat() != VK_FORMAT_R32G32B32A32_SFLOAT ||
        measuredRate.GetExtent().width != m_impl->width ||
        measuredRate.GetExtent().height != m_impl->height)
        return Result<void, String>::Err(
            "GPU camera needs a same-extent RGBA32F measured-rate image");
    if (!std::isfinite(frameTimeSeconds))
        return Result<void, String>::Err("GPU camera frame time must be finite");

    // The dynamic pass runs whenever the caller supplies strata images, even
    // for a single stratum: the PSF pair it feeds expects the compositor's
    // output image, and the caller's strata may be array views a 2D-typed PSF
    // set cannot bind. Without strata the unstratified path reads the caller's
    // measured rate directly, exactly as before M4-1.
    const bool stratified = dynamic.strataRate != nullptr &&
                            dynamic.strataDepth != nullptr &&
                            dynamic.layerCameras != nullptr;
    if (stratified &&
        (dynamic.strataRate->GetFormat() != VK_FORMAT_R32G32B32A32_SFLOAT ||
         dynamic.strataDepth->GetFormat() != VK_FORMAT_R32_SFLOAT ||
         dynamic.strataRate->GetExtent().width != m_impl->width ||
         dynamic.strataRate->GetExtent().height != m_impl->height ||
         dynamic.strataDepth->GetExtent().width != m_impl->width ||
         dynamic.strataDepth->GetExtent().height != m_impl->height))
        return Result<void, String>::Err(
            "GPU camera dynamic inputs must match the sensor extent "
            "(strata rate RGBA32F array, strata depth R32F array)");
    if (stratified &&
        ((m_impl->strataRateView != VK_NULL_HANDLE &&
          m_impl->strataRateView != dynamic.strataRate->GetView()) ||
         (m_impl->strataDepthView != VK_NULL_HANDLE &&
          m_impl->strataDepthView != dynamic.strataDepth->GetView())))
        return Result<void, String>::Err(
            "GPU camera strata images changed; reconfigure after "
            "synchronizing in-flight frames");

    if (m_impl->measurementView != VK_NULL_HANDLE &&
        m_impl->measurementView != measuredRate.GetView() && !stratified)
        return Result<void, String>::Err(
            "GPU camera measured-rate image changed; reconfigure after "
            "synchronizing in-flight frames");

    if (!m_impl->hasCapture) {
        m_impl->firstCaptureWasFirst = true;
        m_impl->currentDeltaSeconds = 0.0;
    } else if (acquisitionIndex == m_impl->lastAcquisitionIndex) {
        if (std::abs(frameTimeSeconds - m_impl->lastFrameTimeSeconds) > 1e-9)
            return Result<void, String>::Err(
                "same acquisition index must keep the same frame time");
        // Keep the pre-capture state and elapsed interval, then overwrite
        // this tick's current state. SPP convergence changes the measurement,
        // never the detector's count of captures or its elapsed history.
    } else if (acquisitionIndex == m_impl->lastAcquisitionIndex + 1) {
        if (frameTimeSeconds < m_impl->lastFrameTimeSeconds)
            return Result<void, String>::Err(
                "GPU camera acquisitions must advance scene time");
        m_impl->stateBefore = m_impl->stateCurrent;
        m_impl->stateCurrent = 1u - m_impl->stateCurrent;
        m_impl->beforeFrameTimeSeconds = m_impl->lastFrameTimeSeconds;
        m_impl->currentDeltaSeconds =
            frameTimeSeconds - m_impl->beforeFrameTimeSeconds;
        m_impl->firstCaptureWasFirst = false;
    } else {
        return Result<void, String>::Err(
            "GPU camera acquisition index skipped or moved backward; process "
            "intermediate captures or explicitly ResetState after a scrub");
    }
    m_impl->lastAcquisitionIndex = acquisitionIndex;
    m_impl->lastFrameTimeSeconds = frameTimeSeconds;
    m_impl->hasCapture = true;

    auto push = m_impl->MakePush(acquisitionIndex, frameTimeSeconds);
    m_impl->TransitionImages(cmd);
    if (!stratified && m_impl->measurementView == VK_NULL_HANDLE) {
        const auto dummy = m_impl->psfTemp.get();
        m_impl->UpdateSet(m_impl->sets[0], {
            &measuredRate, m_impl->psfTemp.get(),
            dummy, dummy, dummy, dummy, dummy});
        m_impl->measurementView = measuredRate.GetView();
    }

    if (stratified) {
        // The per-stratum frames change per acquisition. Uploading at record
        // time is safe for the interactive host, which serializes frames; the
        // explicit Queue/Record/Complete contract makes the same promise.
        std::array<DynamicLayerCamera, 8> frames{};
        const u32 count = std::min<u32>(dynamic.strataCount, 8u);
        std::copy_n(dynamic.layerCameras, count, frames.data());
        m_impl->layerCameraBuffer->Upload(frames.data(), sizeof(frames));
        if (m_impl->strataRateView == VK_NULL_HANDLE) {
            const auto dummy = m_impl->psfTemp.get();
            m_impl->UpdateSet(m_impl->sets[7], {
                dynamic.strataRate, dynamic.strataDepth,
                m_impl->compositedRate.get(), dummy, dummy, dummy, dummy});
            m_impl->strataRateView = dynamic.strataRate->GetView();
            m_impl->strataDepthView = dynamic.strataDepth->GetView();
        }
        push.timeStratumCount = count;
        push.rollingShutter = dynamic.rowDelaySeconds > 0.0 ? 1u : 0u;
        push.exposureSeconds = static_cast<f32>(dynamic.exposureSeconds);
        push.rowDelaySeconds = static_cast<f32>(dynamic.rowDelaySeconds);
        push.firstRowMidSeconds = static_cast<f32>(dynamic.firstRowMidSeconds);
        vkCmdFillBuffer(cmd, m_impl->dynamicCounterBuffer->GetHandle(), 0,
                        VK_WHOLE_SIZE, 0);
        VkBufferMemoryBarrier zeroed{};
        zeroed.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        zeroed.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        zeroed.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                               VK_ACCESS_SHADER_WRITE_BIT;
        zeroed.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        zeroed.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        zeroed.buffer = m_impl->dynamicCounterBuffer->GetHandle();
        zeroed.offset = 0;
        zeroed.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 1, &zeroed, 0, nullptr);
    }

    const auto stamp = [&](u32 offset) {
        if (timing.pool != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                timing.pool, timing.firstQuery + offset);
    };
    if (timing.pool != VK_NULL_HANDLE)
        vkCmdResetQueryPool(cmd, timing.pool, timing.firstQuery,
                            kTimingQueryCount);
    stamp(0);
    if (stratified) {
        m_impl->BindAndDispatch(cmd, m_impl->dynamicPipeline,
                                m_impl->sets[7], push);
        StorageBarrier(cmd);
    }
    stamp(1);
    push.direction = 0;
    const VkDescriptorSet psfHorizontal = stratified ? m_impl->sets[8]
                                                     : m_impl->sets[0];
    const VkDescriptorSet psfVertical = stratified ? m_impl->sets[9]
                                                   : m_impl->sets[1];
    m_impl->BindAndDispatch(cmd, m_impl->psfPipeline, psfHorizontal, push);
    StorageBarrier(cmd);
    push.direction = 1;
    m_impl->BindAndDispatch(cmd, m_impl->psfPipeline, psfVertical, push);
    StorageBarrier(cmd);
    stamp(2);
    const VkDescriptorSet readoutSet =
        m_impl->stateBefore == 0 ? m_impl->sets[2] : m_impl->sets[3];
    m_impl->BindAndDispatch(cmd, m_impl->readoutPipeline, readoutSet, push);
    StorageBarrier(cmd);
    stamp(3);
    // M4-3: the classic ISP. Statistics first: per-tile partials, a global
    // reduce that also derives the next Linear-AGC window from the previous
    // frame's histogram and then zeroes it, and the histogram itself (three
    // compile-time phases of camera_stats.comp.hlsl), then the Equalize CDF,
    // demosaic, color and the display/AGC pass. The reduce and CDF passes are
    // single workgroups.
    m_impl->BindAndDispatch(cmd, m_impl->statsTilesPipeline, m_impl->sets[10],
                            push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatchGrid(cmd, m_impl->statsReducePipeline,
                                m_impl->sets[10], push, 1, 1);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, m_impl->statsHistPipeline, m_impl->sets[10],
                            push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatchGrid(cmd, m_impl->cdfPipeline, m_impl->sets[13],
                                push, 1, 1);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, m_impl->demosaicPipeline, m_impl->sets[11],
                            push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, m_impl->colorPipeline, m_impl->sets[12],
                            push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, m_impl->displayPipeline, m_impl->sets[4],
                            push);
    StorageBarrier(cmd);
    stamp(4);
    // M4-4: display HSV grading and the empirical display effects, in place
    // on the display image. Skipped entirely when every HSV parameter sits at
    // its default (the kFlagHsv bit), so an untouched chain is bit-identical
    // to the pre-HSV pipeline.
    if (m_impl->ispFlags & isp::kFlagHsv) {
        m_impl->BindAndDispatch(cmd, m_impl->hsvPipeline, m_impl->sets[14],
                                push);
        StorageBarrier(cmd);
    }
    stamp(5);
    stamp(6);
    return Result<void, String>::Ok();
}

Result<void, String> GpuCameraPipeline::RecordFastRgbMeasurement(
    VkCommandBuffer cmd, const GpuImage& linearRgb, u64 acquisitionIndex,
    f64 frameTimeSeconds, TimingQueries timing) {
    if (!m_impl->configured)
        return Result<void, String>::Err("GPU camera must be configured first");
    if (m_impl->config.inputKind != camera::CameraInputKind::FastRgbApproximation)
        return Result<void, String>::Err(
            "fast RGB measurement requires input_kind=fast_rgb");
    if (cmd == VK_NULL_HANDLE || !linearRgb.IsValid() ||
        linearRgb.GetFormat() != VK_FORMAT_R32G32B32A32_SFLOAT)
        return Result<void, String>::Err(
            "GPU fast RGB camera needs an RGBA32F linear-RGB image");
    if (m_impl->fastRgbInputView != VK_NULL_HANDLE &&
        m_impl->fastRgbInputView != linearRgb.GetView())
        return Result<void, String>::Err(
            "GPU fast RGB input image changed; reconfigure after synchronizing "
            "in-flight frames");

    m_impl->TransitionImages(cmd);
    if (m_impl->fastRgbInputView == VK_NULL_HANDLE) {
        const auto dummy = m_impl->psfTemp.get();
        m_impl->UpdateSet(m_impl->sets[6], {
            &linearRgb, m_impl->fastRgbRate.get(),
            dummy, dummy, dummy, dummy, dummy});
        m_impl->fastRgbInputView = linearRgb.GetView();
    }
    auto push = m_impl->MakePush(acquisitionIndex, frameTimeSeconds);
    m_impl->BindAndDispatch(
        cmd, m_impl->fastRgbPipeline, m_impl->sets[6], push);
    StorageBarrier(cmd);
    return RecordMeasurement(
        cmd, *m_impl->fastRgbRate, acquisitionIndex, frameTimeSeconds, timing);
}

Result<void, String> GpuCameraPipeline::RecordRandomVectors(
    VkCommandBuffer cmd, u32 seed, u64 acquisitionIndex,
    camera::NoiseClass noiseClass, u32 counter) {
    if (!m_impl->configured || cmd == VK_NULL_HANDLE)
        return Result<void, String>::Err(
            "GPU camera RNG vectors need a configured pipeline and command buffer");
    m_impl->TransitionImages(cmd);
    auto push = m_impl->MakePush(acquisitionIndex, 0.0);
    push.seed = seed;
    push.rngNoiseClass = static_cast<u32>(noiseClass);
    push.rngCounter = counter;
    m_impl->BindAndDispatch(cmd, m_impl->rngPipeline, m_impl->sets[5], push);
    return Result<void, String>::Ok();
}

} // namespace quantiloom::rendercore
