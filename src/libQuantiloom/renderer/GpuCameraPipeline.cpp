#include "renderer/GpuCameraPipeline.hpp"

#include "core/Log.hpp"
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
};
static_assert(sizeof(CameraPush) == 96);

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
    std::array<f32, 3> sigmaX{}, sigmaY{};
    ConfigRows configRows{};

    std::unique_ptr<GpuImage> psfTemp, blurred, rawDn, corrected, display;
    std::unique_ptr<GpuImage> fastRgbRate;
    std::unique_ptr<GpuImage> expectedElectrons, preAdcElectrons;
    std::array<std::unique_ptr<GpuImage>, 2> thermalState;
    std::unique_ptr<GpuImage> randomVectors;
    std::unique_ptr<GpuBuffer> configBuffer, nucGainBuffer, nucOffsetBuffer;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline psfPipeline = VK_NULL_HANDLE;
    VkPipeline readoutPipeline = VK_NULL_HANDLE;
    VkPipeline previewPipeline = VK_NULL_HANDLE;
    VkPipeline rngPipeline = VK_NULL_HANDLE;
    VkPipeline fastRgbPipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    // H, V, readout before=0, readout before=1, preview, RNG, fast RGB.
    // Immutable while frames may be in flight; Configure waits idle first.
    std::array<VkDescriptorSet, 7> sets{};
    VkImageView measurementView = VK_NULL_HANDLE;
    VkImageView fastRgbInputView = VK_NULL_HANDLE;

    Impl(VulkanContext& c, u32 w, u32 h)
        : context(c), device(c.GetDevice()), width(w), height(h) {}

    ~Impl() {
        if (device == VK_NULL_HANDLE) return;
        for (auto pipeline : {psfPipeline, readoutPipeline, previewPipeline,
                              rngPipeline, fastRgbPipeline})
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
        !MakeImage(expectedElectrons, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(preAdcElectrons, VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(thermalState[0], VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(thermalState[1], VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !MakeImage(randomVectors, VK_FORMAT_R32G32B32A32_UINT))
        return Result<void, String>::Err("cannot allocate GPU camera images");

    std::array<VkDescriptorSetLayoutBinding, 10> bindings{};
    for (u32 i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = i < 7
            ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo setInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
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

    const std::array<std::pair<StringView, VkPipeline*>, 5> shaders{{
        {"camera_psf.comp.spv", &psfPipeline},
        {"camera_readout.comp.spv", &readoutPipeline},
        {"camera_preview.comp.spv", &previewPipeline},
        {"camera_rng_vectors.comp.spv", &rngPipeline},
        {"camera_fast_rgb.comp.spv", &fastRgbPipeline}
    }};
    for (const auto& [name, pipeline] : shaders) {
        auto created = CreatePipeline(name, *pipeline);
        if (!created) return created;
    }

    const std::array<VkDescriptorPoolSize, 2> sizes{{
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7u * static_cast<u32>(sets.size())},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3u * static_cast<u32>(sets.size())}
    }};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = static_cast<u32>(sets.size());
    poolInfo.poolSizeCount = static_cast<u32>(sizes.size());
    poolInfo.pPoolSizes = sizes.data();
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        return Result<void, String>::Err("cannot create GPU camera descriptor pool");
    std::array<VkDescriptorSetLayout, 7> layouts{};
    layouts.fill(setLayout);
    VkDescriptorSetAllocateInfo allocInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<u32>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &allocInfo, sets.data()) != VK_SUCCESS)
        return Result<void, String>::Err("cannot allocate GPU camera descriptor sets");
    return Result<void, String>::Ok();
}

void GpuCameraPipeline::Impl::TransitionImages(VkCommandBuffer cmd) {
    if (imagesInGeneral) return;
    const std::array<const GpuImage*, 11> images{
        psfTemp.get(), blurred.get(), rawDn.get(), corrected.get(),
        display.get(), fastRgbRate.get(), expectedElectrons.get(), preAdcElectrons.get(),
        thermalState[0].get(), thermalState[1].get(), randomVectors.get()};
    std::array<VkImageMemoryBarrier, 11> barriers{};
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
    std::array<VkDescriptorBufferInfo, 3> bufferInfo{};
    std::array<VkWriteDescriptorSet, 10> writes{};
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
        nucGainBuffer.get(), nucOffsetBuffer.get(), configBuffer.get()};
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
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
    const auto dummy = m_impl->psfTemp.get();
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
    m_impl->UpdateSet(m_impl->sets[4], {
        m_impl->corrected.get(), m_impl->display.get(), m_impl->rawDn.get(),
        dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[5], {
        dummy, dummy, m_impl->randomVectors.get(),
        dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->sets[6], {
        dummy, m_impl->fastRgbRate.get(),
        dummy, dummy, dummy, dummy, dummy});
    m_impl->measurementView = VK_NULL_HANDLE;
    m_impl->fastRgbInputView = VK_NULL_HANDLE;
    m_impl->configured = true;
    ResetState();
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
}

GpuCameraPipeline::Images GpuCameraPipeline::GetOutputs() const {
    if (!m_impl->configured) return {};
    return {m_impl->rawDn.get(), m_impl->corrected.get(),
            m_impl->display.get(), m_impl->expectedElectrons.get(),
            m_impl->preAdcElectrons.get()};
}

const GpuImage* GpuCameraPipeline::GetRandomVectors() const {
    return m_impl->randomVectors.get();
}

Result<void, String> GpuCameraPipeline::RecordMeasurement(
    VkCommandBuffer cmd, const GpuImage& measuredRate, u64 acquisitionIndex,
    f64 frameTimeSeconds, TimingQueries timing) {
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
    if (m_impl->measurementView != VK_NULL_HANDLE &&
        m_impl->measurementView != measuredRate.GetView())
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
    if (m_impl->measurementView == VK_NULL_HANDLE) {
        const auto dummy = m_impl->psfTemp.get();
        m_impl->UpdateSet(m_impl->sets[0], {
            &measuredRate, m_impl->psfTemp.get(),
            dummy, dummy, dummy, dummy, dummy});
        m_impl->measurementView = measuredRate.GetView();
    }

    const auto stamp = [&](u32 offset) {
        if (timing.pool != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                timing.pool, timing.firstQuery + offset);
    };
    stamp(0);
    push.direction = 0;
    m_impl->BindAndDispatch(cmd, m_impl->psfPipeline, m_impl->sets[0], push);
    StorageBarrier(cmd);
    push.direction = 1;
    m_impl->BindAndDispatch(cmd, m_impl->psfPipeline, m_impl->sets[1], push);
    StorageBarrier(cmd);
    stamp(1);
    const VkDescriptorSet readoutSet =
        m_impl->stateBefore == 0 ? m_impl->sets[2] : m_impl->sets[3];
    m_impl->BindAndDispatch(cmd, m_impl->readoutPipeline, readoutSet, push);
    StorageBarrier(cmd);
    stamp(2);
    m_impl->BindAndDispatch(cmd, m_impl->previewPipeline, m_impl->sets[4], push);
    stamp(3);
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
