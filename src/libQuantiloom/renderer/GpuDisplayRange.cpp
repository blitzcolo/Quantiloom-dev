#include "renderer/GpuDisplayRange.hpp"

#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/ShaderBinary.hpp"
#include "renderer/VulkanContext.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>


namespace quantiloom::rendercore {
namespace {

constexpr u32 kScratchWords = 4;

u32 FloatToOrdered(f32 value) {
    const u32 bits = std::bit_cast<u32>(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

f32 OrderedToFloat(u32 ordered) {
    const u32 bits = (ordered & 0x80000000u)
        ? (ordered ^ 0x80000000u) : ~ordered;
    return std::bit_cast<f32>(bits);
}

VkShaderModule CreateShaderModule(VkDevice device, const Vector<u32>& code) {
    if (code.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size() * sizeof(u32);
    info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    return vkCreateShaderModule(device, &info, nullptr, &module) == VK_SUCCESS
        ? module : VK_NULL_HANDLE;
}

}  // namespace

struct GpuDisplayRange::Impl {
    VulkanContext& context;
    VkDevice device = VK_NULL_HANDLE;

    VkShaderModule extentsShader = VK_NULL_HANDLE;
    VkShaderModule histogramShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline extentsPipeline = VK_NULL_HANDLE;
    VkPipeline histogramPipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    std::unique_ptr<GpuBuffer> scratch;
    std::unique_ptr<GpuBuffer> histogram;
    std::unique_ptr<GpuBuffer> scratchReadback;
    std::unique_ptr<GpuBuffer> histogramReadback;

    explicit Impl(VulkanContext& ctx) : context(ctx), device(ctx.GetDevice()) {}

    ~Impl() {
        if (extentsPipeline) vkDestroyPipeline(device, extentsPipeline, nullptr);
        if (histogramPipeline) vkDestroyPipeline(device, histogramPipeline, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (descriptorSetLayout)
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        if (extentsShader) vkDestroyShaderModule(device, extentsShader, nullptr);
        if (histogramShader) vkDestroyShaderModule(device, histogramShader, nullptr);
    }

    Result<void, String> Initialize() {
        extentsShader = CreateShaderModule(
            device, LoadSpirv("display_range_extents.comp.spv"));
        histogramShader = CreateShaderModule(
            device, LoadSpirv("display_range_histogram.comp.spv"));
        if (!extentsShader || !histogramShader)
            return Result<void, String>::Err(
                "display-range compute shaders are missing or invalid");

        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo setInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        setInfo.bindingCount = static_cast<u32>(bindings.size());
        setInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &setInfo, nullptr,
                                        &descriptorSetLayout) != VK_SUCCESS)
            return Result<void, String>::Err(
                "cannot create display-range descriptor layout");

        struct Push { u32 width; u32 height; f32 absMin; f32 scale; };
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = sizeof(Push);
        VkPipelineLayoutCreateInfo layoutInfo{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr,
                                   &pipelineLayout) != VK_SUCCESS)
            return Result<void, String>::Err(
                "cannot create display-range pipeline layout");

        const auto createPipeline = [&](VkShaderModule shader,
                                        VkPipeline& pipeline) {
            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            VkComputePipelineCreateInfo info{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = stage;
            info.layout = pipelineLayout;
            return vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info,
                                            nullptr, &pipeline) == VK_SUCCESS;
        };
        if (!createPipeline(extentsShader, extentsPipeline) ||
            !createPipeline(histogramShader, histogramPipeline))
            return Result<void, String>::Err(
                "cannot create display-range compute pipelines");

        const std::array<VkDescriptorPoolSize, 2> poolSizes{{
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
        }};
        VkDescriptorPoolCreateInfo poolInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr,
                                   &descriptorPool) != VK_SUCCESS)
            return Result<void, String>::Err(
                "cannot create display-range descriptor pool");
        VkDescriptorSetAllocateInfo allocInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo,
                                     &descriptorSet) != VK_SUCCESS)
            return Result<void, String>::Err(
                "cannot allocate display-range descriptor set");

        const VkDeviceSize scratchBytes = kScratchWords * sizeof(u32);
        const VkDeviceSize histogramBytes =
            static_cast<VkDeviceSize>(GpuDisplayRange::kHistogramBins) * sizeof(u32);
        scratch = std::make_unique<GpuBuffer>(
            context.GetAllocator(), scratchBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        histogram = std::make_unique<GpuBuffer>(
            context.GetAllocator(), histogramBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        scratchReadback = std::make_unique<GpuBuffer>(
            context.GetAllocator(), scratchBytes,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        histogramReadback = std::make_unique<GpuBuffer>(
            context.GetAllocator(), histogramBytes,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

        const VkDescriptorBufferInfo scratchInfo{
            scratch->GetHandle(), 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo histogramInfo{
            histogram->GetHandle(), 0, VK_WHOLE_SIZE};
        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 1;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &scratchInfo;
        writes[1] = writes[0];
        writes[1].dstBinding = 2;
        writes[1].pBufferInfo = &histogramInfo;
        vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()),
                               writes.data(), 0, nullptr);
        return Result<void, String>::Ok();
    }
};

GpuDisplayRange::GpuDisplayRange(std::unique_ptr<Impl> impl)
    : m_impl(std::move(impl)) {}

GpuDisplayRange::~GpuDisplayRange() = default;

Result<std::unique_ptr<GpuDisplayRange>, String>
GpuDisplayRange::Create(VulkanContext& context) {
    auto impl = std::make_unique<Impl>(context);
    auto initialized = impl->Initialize();
    if (!initialized)
        return Result<std::unique_ptr<GpuDisplayRange>, String>::Err(
            initialized.error());
    return std::unique_ptr<GpuDisplayRange>(
        new GpuDisplayRange(std::move(impl)));
}

Result<DisplayRange, String> GpuDisplayRange::Compute(
    const GpuImage& image, u32 width, u32 height,
    f32 percentileLow, f32 percentileHigh) {
    if (!image.IsValid() || image.GetFormat() != VK_FORMAT_R32G32B32A32_SFLOAT)
        return Result<DisplayRange, String>::Err(
            "display range requires an RGBA32F image");
    const VkExtent2D extent = image.GetExtent();
    if (width == 0 || height == 0 || width > extent.width || height > extent.height)
        return Result<DisplayRange, String>::Err(
            "display-range extent is empty or exceeds its image");

    const VkDescriptorImageInfo inputInfo{
        VK_NULL_HANDLE, image.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet inputWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    inputWrite.dstSet = m_impl->descriptorSet;
    inputWrite.dstBinding = 0;
    inputWrite.descriptorCount = 1;
    inputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    inputWrite.pImageInfo = &inputInfo;
    vkUpdateDescriptorSets(m_impl->device, 1, &inputWrite, 0, nullptr);

    const std::array<u32, kScratchWords> initialScratch{
        FloatToOrdered(std::numeric_limits<f32>::max()),
        FloatToOrdered(std::numeric_limits<f32>::lowest()), 0u, 0u};
    struct Push { u32 width; u32 height; f32 absMin; f32 scale; };
    const Push extentsPush{width, height, 0.0f, 0.0f};
    const u32 groupsX = (width + 15u) / 16u;
    const u32 groupsY = (height + 15u) / 16u;

    CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier inputReady{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        inputReady.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        inputReady.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        inputReady.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        inputReady.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        inputReady.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        inputReady.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        inputReady.image = image.GetImage();
        inputReady.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        inputReady.subresourceRange.levelCount = 1;
        inputReady.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &inputReady);

        vkCmdUpdateBuffer(cmd, m_impl->scratch->GetHandle(), 0,
                          sizeof(initialScratch), initialScratch.data());
        VkBufferMemoryBarrier initialized{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        initialized.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        initialized.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_SHADER_WRITE_BIT;
        initialized.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initialized.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initialized.buffer = m_impl->scratch->GetHandle();
        initialized.offset = 0;
        initialized.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 1, &initialized, 0, nullptr);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_impl->pipelineLayout, 0, 1,
                                &m_impl->descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_impl->pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(extentsPush), &extentsPush);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          m_impl->extentsPipeline);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        VkBufferMemoryBarrier rangeReady{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        rangeReady.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rangeReady.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        rangeReady.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rangeReady.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rangeReady.buffer = m_impl->scratch->GetHandle();
        rangeReady.offset = 0;
        rangeReady.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 1, &rangeReady, 0, nullptr);

        VkBufferCopy scratchCopy{};
        scratchCopy.size = m_impl->scratch->GetSize();
        vkCmdCopyBuffer(cmd, m_impl->scratch->GetHandle(),
                        m_impl->scratchReadback->GetHandle(), 1, &scratchCopy);
    });

    std::array<u32, kScratchWords> scratch{};
    const void* scratchData = m_impl->scratchReadback->MapRead();
    if (!scratchData)
        return Result<DisplayRange, String>::Err(
            "cannot map display-range scalar readback");
    std::memcpy(scratch.data(), scratchData, sizeof(scratch));
    m_impl->scratchReadback->Unmap();

    if (scratch[3] != 0u)
        return Result<DisplayRange, String>::Err(
            "display range input needs CPU small-value arithmetic");
    const u32 validCount = scratch[2];
    const f32 absMin = OrderedToFloat(scratch[0]);
    const f32 absMax = OrderedToFloat(scratch[1]);
    if (validCount == 0 || absMin >= absMax) return DisplayRange{};

    // This is deliberately host arithmetic: Vulkan permits a less accurate
    // floating-point division than MSVC's f32 division, enough to move an
    // endpoint across a 65,536-bin boundary. Reading these three words first
    // preserves the existing bin definition without reading back the image.
    const f32 span = absMax - absMin;
    const f32 scale = static_cast<f32>(kHistogramBins - 1) / span;
    if (!std::isfinite(span) || !(span > 0.0f) || !std::isfinite(scale))
        return Result<DisplayRange, String>::Err(
            "display range input needs CPU scale arithmetic");
    const Push histogramPush{width, height, absMin, scale};
    CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, m_impl->histogram->GetHandle(), 0,
                        VK_WHOLE_SIZE, 0u);
        VkBufferMemoryBarrier initialized{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        initialized.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        initialized.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_SHADER_WRITE_BIT;
        initialized.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initialized.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initialized.buffer = m_impl->histogram->GetHandle();
        initialized.offset = 0;
        initialized.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 1, &initialized, 0, nullptr);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_impl->pipelineLayout, 0, 1,
                                &m_impl->descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_impl->pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(histogramPush), &histogramPush);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          m_impl->histogramPipeline);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        VkBufferMemoryBarrier copyReady{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        copyReady.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        copyReady.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        copyReady.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyReady.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyReady.buffer = m_impl->histogram->GetHandle();
        copyReady.offset = 0;
        copyReady.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 1, &copyReady, 0, nullptr);
        VkBufferCopy histogramCopy{};
        histogramCopy.size = m_impl->histogram->GetSize();
        vkCmdCopyBuffer(cmd, m_impl->histogram->GetHandle(),
                        m_impl->histogramReadback->GetHandle(), 1,
                        &histogramCopy);
    });

    std::array<u32, kHistogramBins> bins{};
    const void* histogramData = m_impl->histogramReadback->MapRead();
    if (!histogramData)
        return Result<DisplayRange, String>::Err(
            "cannot map display-range histogram readback");
    std::memcpy(bins.data(), histogramData, sizeof(bins));
    m_impl->histogramReadback->Unmap();

    const f64 lowFraction =
        std::clamp(static_cast<f64>(percentileLow), 0.0, 100.0) / 100.0;
    const f64 highFraction =
        std::clamp(static_cast<f64>(percentileHigh), 0.0, 100.0) / 100.0;
    const usize targetLow =
        static_cast<usize>(static_cast<f64>(validCount) * lowFraction);
    const usize targetHigh =
        static_cast<usize>(static_cast<f64>(validCount) * highFraction);

    usize cumulative = 0;
    usize binLow = 0;
    usize binHigh = kHistogramBins - 1;
    bool haveLow = false;
    for (usize i = 0; i < bins.size(); ++i) {
        cumulative += bins[i];
        if (!haveLow && cumulative >= targetLow) {
            binLow = i;
            haveLow = true;
        }
        if (cumulative >= targetHigh) {
            binHigh = i;
            break;
        }
    }

    const f32 invScale =
        (absMax - absMin) / static_cast<f32>(kHistogramBins - 1);
    DisplayRange result{
        absMin + static_cast<f32>(binLow) * invScale,
        absMin + static_cast<f32>(binHigh) * invScale};
    if (!(result.max - result.min > (absMax - absMin) * 1e-3f))
        result = {absMin, absMax};
    return result;
}

}  // namespace quantiloom::rendercore
