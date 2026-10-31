/**
 * @file GpuClahe.cpp
 * @brief The CLAHE display-enhancement passes, extracted from ExternalRenderContext
 */

#include "renderer/GpuClahe.hpp"

#include "core/Log.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/ShaderBinary.hpp"
#include "renderer/VulkanContext.hpp"

#include <algorithm>
#include <fstream>
#include <vector>


namespace quantiloom::rendercore {

namespace {

/// Push constants (must match CLAHEPushConstants in clahe.comp.hlsl). The
/// pipeline-layout range and the record-time payload share this one 48-byte
/// block; the words past passIndex are padding in the layout range and the
/// tone/palette selectors at record time.
struct CLAHEPushConstants {
    u32 imageWidth;
    u32 imageHeight;
    u32 tileCountX;
    u32 tileCountY;
    f32 clipLimit;
    u32 luminanceOnly;
    f32 inputMin;
    f32 inputMax;
    u32 passIndex;
    u32 toneMode;
    u32 palette;
    u32 padding;
};
static_assert(sizeof(CLAHEPushConstants) == 48);

}  // namespace

struct GpuClahe::Impl {
    VulkanContext& context;
    VkDevice device = VK_NULL_HANDLE;

    VkShaderModule claheHistogramShader = VK_NULL_HANDLE;
    VkShaderModule claheCdfShader = VK_NULL_HANDLE;
    VkShaderModule claheApplyShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout claheDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout clahePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool claheDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet claheDescriptorSet = VK_NULL_HANDLE;
    VkPipeline claheHistogramPipeline = VK_NULL_HANDLE;
    VkPipeline claheCdfPipeline = VK_NULL_HANDLE;
    VkPipeline claheApplyPipeline = VK_NULL_HANDLE;

    std::unique_ptr<GpuBuffer> claheHistogramBuffer;  // Per-tile histograms
    std::unique_ptr<GpuBuffer> claheCdfBuffer;        // Per-tile CDFs
    std::unique_ptr<GpuBuffer> claheMinMaxBuffer;     // Per-tile min/max for normalization

    // Last input view written to claheDescriptorSet binding 0. The view only
    // changes when the render target is recreated or the camera chain starts
    // feeding CLAHE -- moments that drain the device first -- so the write is
    // skipped while it is current: vkUpdateDescriptorSets on a set a submitted
    // frame may still be executing is a VUID-03047 violation, and every frame
    // used to take it for a view that had not moved.
    VkImageView claheBoundInputView = VK_NULL_HANDLE;

    explicit Impl(VulkanContext& ctx) : context(ctx), device(ctx.GetDevice()) {}

    ~Impl() {
        claheHistogramBuffer.reset();
        claheCdfBuffer.reset();
        claheMinMaxBuffer.reset();
        if (claheHistogramPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, claheHistogramPipeline, nullptr);
            claheHistogramPipeline = VK_NULL_HANDLE;
        }
        if (claheCdfPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, claheCdfPipeline, nullptr);
            claheCdfPipeline = VK_NULL_HANDLE;
        }
        if (claheApplyPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, claheApplyPipeline, nullptr);
            claheApplyPipeline = VK_NULL_HANDLE;
        }
        if (claheHistogramShader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, claheHistogramShader, nullptr);
            claheHistogramShader = VK_NULL_HANDLE;
        }
        if (claheCdfShader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, claheCdfShader, nullptr);
            claheCdfShader = VK_NULL_HANDLE;
        }
        if (claheApplyShader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, claheApplyShader, nullptr);
            claheApplyShader = VK_NULL_HANDLE;
        }
        if (claheDescriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, claheDescriptorPool, nullptr);
            claheDescriptorPool = VK_NULL_HANDLE;
        }
        if (clahePipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, clahePipelineLayout, nullptr);
            clahePipelineLayout = VK_NULL_HANDLE;
        }
        if (claheDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, claheDescriptorSetLayout, nullptr);
            claheDescriptorSetLayout = VK_NULL_HANDLE;
        }
    }

    Result<void, String> Initialize() {
        QL_LOG_INFO("Creating CLAHE compute pipeline...");

        auto allocator = context.GetAllocator();

        // Helper function to load shader file
        auto loadShaderFile = [](const String& path) -> std::vector<u32> {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                return {};
            }
            size_t fileSize = static_cast<size_t>(file.tellg());
            if (fileSize == 0 || fileSize % 4 != 0) {
                return {};
            }
            std::vector<u32> code(fileSize / 4);
            file.seekg(0);
            file.read(reinterpret_cast<char*>(code.data()), fileSize);
            return code;
        };

        // The three shaders must come from one directory: a histogram from the
        // SDK install beside a cdf from the source tree may not be the same
        // build. Search by candidate base, not by file.
        std::vector<u32> histogramCode, cdfCode, applyCode;
        for (const auto& basePath : SpirvSearchPaths("clahe_histogram.spv")) {
            // Convert path to string and replace "histogram" with cdf/apply
            String histPath = basePath.string();
            String cdfPath = histPath;
            String applyPath = histPath;
            size_t pos = histPath.find("histogram");
            if (pos != String::npos) {
                cdfPath.replace(pos, 9, "cdf");
                applyPath.replace(pos, 9, "apply");
            }

            histogramCode = loadShaderFile(histPath);
            if (!histogramCode.empty()) {
                cdfCode = loadShaderFile(cdfPath);
                applyCode = loadShaderFile(applyPath);
                if (!cdfCode.empty() && !applyCode.empty()) {
                    QL_LOG_DEBUG("CLAHE: Loaded shaders from {}", histPath);
                    break;
                }
            }
        }

        if (histogramCode.empty() || cdfCode.empty() || applyCode.empty()) {
            return Result<void, String>::Err(
                "CLAHE: Could not load shader files, CLAHE disabled");
        }

        // Create shader modules
        auto createShaderModule = [this](const std::vector<u32>& code) -> VkShaderModule {
            VkShaderModuleCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size() * sizeof(u32);
            createInfo.pCode = code.data();
            VkShaderModule module;
            if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            return module;
        };

        claheHistogramShader = createShaderModule(histogramCode);
        claheCdfShader = createShaderModule(cdfCode);
        claheApplyShader = createShaderModule(applyCode);

        if (claheHistogramShader == VK_NULL_HANDLE ||
            claheCdfShader == VK_NULL_HANDLE ||
            claheApplyShader == VK_NULL_HANDLE) {
            return Result<void, String>::Err("CLAHE: Failed to create shader modules");
        }

        // Create descriptor set layout
        // binding 0: inputImage (storage image)
        // binding 1: outputImage (storage image)
        // binding 2: histogramBuffer (storage buffer)
        // binding 3: cdfBuffer (storage buffer)
        // binding 4: minMaxBuffer (storage buffer)
        std::vector<VkDescriptorSetLayoutBinding> bindings(5);

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &claheDescriptorSetLayout) != VK_SUCCESS) {
            return Result<void, String>::Err("CLAHE: Failed to create descriptor set layout");
        }

        // Create pipeline layout with push constants
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(CLAHEPushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &claheDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &clahePipelineLayout) != VK_SUCCESS) {
            return Result<void, String>::Err("CLAHE: Failed to create pipeline layout");
        }

        // Create compute pipelines for each pass
        auto createComputePipeline = [this](VkShaderModule shader) -> VkPipeline {
            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = shader;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = clahePipelineLayout;

            VkPipeline pipeline;
            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            return pipeline;
        };

        claheHistogramPipeline = createComputePipeline(claheHistogramShader);
        claheCdfPipeline = createComputePipeline(claheCdfShader);
        claheApplyPipeline = createComputePipeline(claheApplyShader);

        if (claheHistogramPipeline == VK_NULL_HANDLE ||
            claheCdfPipeline == VK_NULL_HANDLE ||
            claheApplyPipeline == VK_NULL_HANDLE) {
            return Result<void, String>::Err("CLAHE: Failed to create compute pipelines");
        }

        // Create descriptor pool
        std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}
        };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 1;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &claheDescriptorPool) != VK_SUCCESS) {
            return Result<void, String>::Err("CLAHE: Failed to create descriptor pool");
        }

        // Create buffers for CLAHE
        // Max 64x64 tiles, 256 bins per tile
        constexpr u32 maxTiles = 64 * 64;
        constexpr u32 histogramBins = 256;

        claheHistogramBuffer = std::make_unique<GpuBuffer>(
            allocator,
            maxTiles * histogramBins * sizeof(u32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        claheCdfBuffer = std::make_unique<GpuBuffer>(
            allocator,
            maxTiles * histogramBins * sizeof(f32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        claheMinMaxBuffer = std::make_unique<GpuBuffer>(
            allocator,
            maxTiles * 2 * sizeof(f32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = claheDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &claheDescriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &claheDescriptorSet) != VK_SUCCESS) {
            return Result<void, String>::Err("CLAHE: Failed to allocate descriptor set");
        }
        return Result<void, String>::Ok();
    }

    void BindAll(const GpuImage& input, const GpuImage& output) {
        VkDescriptorImageInfo inputImageInfo{};
        inputImageInfo.imageView = input.GetView();
        inputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputImageInfo{};
        outputImageInfo.imageView = output.GetView();
        outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo histogramBufferInfo{};
        histogramBufferInfo.buffer = claheHistogramBuffer->GetHandle();
        histogramBufferInfo.offset = 0;
        histogramBufferInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo cdfBufferInfo{};
        cdfBufferInfo.buffer = claheCdfBuffer->GetHandle();
        cdfBufferInfo.offset = 0;
        cdfBufferInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo minMaxBufferInfo{};
        minMaxBufferInfo.buffer = claheMinMaxBuffer->GetHandle();
        minMaxBufferInfo.offset = 0;
        minMaxBufferInfo.range = VK_WHOLE_SIZE;

        std::vector<VkWriteDescriptorSet> writes(5);

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = claheDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &inputImageInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = claheDescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputImageInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = claheDescriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &histogramBufferInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = claheDescriptorSet;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &cdfBufferInfo;

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = claheDescriptorSet;
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &minMaxBufferInfo;

        vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
    }

    void SetOutput(const GpuImage& output) {
        VkDescriptorImageInfo outputImageInfo{};
        outputImageInfo.imageView = output.GetView();
        outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = claheDescriptorSet;
        write.dstBinding = 1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &outputImageInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    void Record(VkCommandBuffer cmd, const GpuImage& input, const ClaheRun& run) {
        // The input view is rewritten only when it actually moved; see
        // claheBoundInputView above.
        VkImageView inputView = input.GetView();
        if (inputView != claheBoundInputView) {
            VkDescriptorImageInfo inputImageInfo{};
            inputImageInfo.imageView = inputView;
            inputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            inputImageInfo.sampler = VK_NULL_HANDLE;

            VkWriteDescriptorSet inputWrite{};
            inputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            inputWrite.dstSet = claheDescriptorSet;
            inputWrite.dstBinding = 0;
            inputWrite.dstArrayElement = 0;
            inputWrite.descriptorCount = 1;
            inputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            inputWrite.pImageInfo = &inputImageInfo;

            vkUpdateDescriptorSets(device, 1, &inputWrite, 0, nullptr);
            claheBoundInputView = inputView;
        }

        // tileSize from UI represents tile grid dimension (e.g., 8 = 8x8 grid)
        // NOT pixels per tile
        u32 tileCountX = static_cast<u32>(run.params.tileSize);
        u32 tileCountY = static_cast<u32>(run.params.tileSize);

        // Clamp to max tiles (matching buffer allocation)
        tileCountX = std::min(tileCountX, 64u);
        tileCountY = std::min(tileCountY, 64u);

        // Ensure at least 1 tile
        tileCountX = std::max(tileCountX, 1u);
        tileCountY = std::max(tileCountY, 1u);

        CLAHEPushConstants pushConstants{};
        pushConstants.imageWidth = run.width;
        pushConstants.imageHeight = run.height;
        pushConstants.tileCountX = tileCountX;
        pushConstants.tileCountY = tileCountY;
        pushConstants.clipLimit = run.params.clipLimit;
        pushConstants.luminanceOnly = run.params.luminanceOnly ? 1 : 0;
        pushConstants.toneMode = static_cast<u32>(run.params.toneMode);
        pushConstants.palette = static_cast<u32>(run.params.palette);
        pushConstants.inputMin = run.inputMin;
        pushConstants.inputMax = run.inputMax;

        // A linear stretch is the window and nothing else, so the two passes that
        // exist to build a CDF have nothing to contribute -- and skipping them
        // makes the cheapest tone operator also the cheapest to run.
        const bool needsHistogram = run.params.toneMode != DisplayToneMode::Linear;

        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        // Memory barrier: wait for the input's last writer and make the write
        // visible to the compute passes that read the image -- pass 1's histogram
        // is a compute read in both tone modes, so the destination is the compute
        // stage with shader read|write even when a buffer fill (transfer) is the
        // next command: the fill is ordered by its own buffer barrier below.
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd,
            run.inputWrittenByCompute ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                      : VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &memBarrier,
            0, nullptr,
            0, nullptr
        );

        if (needsHistogram) {

        // Clear histogram buffer to zero before Pass 1
        // Without this, garbage data from uninitialized GPU memory causes
        // corrupted histograms and eventual TDR timeout (GPU crash)
        vkCmdFillBuffer(cmd, claheHistogramBuffer->GetHandle(), 0, VK_WHOLE_SIZE, 0);

        // Buffer barrier: fill write → compute shader read/write
        VkBufferMemoryBarrier fillBarrier{};
        fillBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        fillBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fillBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fillBarrier.buffer = claheHistogramBuffer->GetHandle();
        fillBarrier.offset = 0;
        fillBarrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            1, &fillBarrier,
            0, nullptr
        );

        // Pass 1: Build histograms
        // Dispatch one workgroup per tile
        pushConstants.passIndex = 0;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, claheHistogramPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                clahePipelineLayout, 0, 1,
                                &claheDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, clahePipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, tileCountX, tileCountY, 1);

        // Barrier between passes
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &memBarrier,
            0, nullptr,
            0, nullptr
        );

        // Pass 2: Clip, redistribute, compute CDF
        pushConstants.passIndex = 1;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, claheCdfPipeline);
        vkCmdPushConstants(cmd, clahePipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, tileCountX, tileCountY, 1);

        // Barrier between passes
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &memBarrier,
            0, nullptr,
            0, nullptr
        );

        }  // needsHistogram

        // Pass 3: Apply the mapping, then the palette
        pushConstants.passIndex = 2;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, claheApplyPipeline);
        // Bound here as well as in pass 1, because pass 1 does not always run.
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                clahePipelineLayout, 0, 1,
                                &claheDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, clahePipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(pushConstants), &pushConstants);
        // Dispatch one thread per pixel
        u32 groupCountX = (run.width + 15) / 16;
        u32 groupCountY = (run.height + 15) / 16;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // Final barrier: CLAHE output ready for blit
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            1, &memBarrier,
            0, nullptr,
            0, nullptr
        );
    }
};

GpuClahe::GpuClahe(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) {}

GpuClahe::~GpuClahe() = default;

Result<std::unique_ptr<GpuClahe>, String> GpuClahe::Create(VulkanContext& context) {
    auto impl = std::make_unique<Impl>(context);
    auto initialized = impl->Initialize();
    if (!initialized) {
        // The caller logs the text, as the facade's warning path did.
        return Result<std::unique_ptr<GpuClahe>, String>::Err(initialized.error());
    }
    QL_LOG_INFO("CLAHE: Compute pipeline created successfully");
    return std::unique_ptr<GpuClahe>(new GpuClahe(std::move(impl)));
}

void GpuClahe::BindAll(const GpuImage& input, const GpuImage& output) {
    m_impl->BindAll(input, output);
}

void GpuClahe::SetOutput(const GpuImage& output) {
    m_impl->SetOutput(output);
}

void GpuClahe::Record(VkCommandBuffer cmd, const GpuImage& input, const ClaheRun& run) {
    m_impl->Record(cmd, input, run);
}

}  // namespace quantiloom::rendercore
