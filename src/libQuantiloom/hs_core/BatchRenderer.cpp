/**
 * @file BatchRenderer.cpp
 * @brief Implementation of batch wavelength rendering
 *
 * @author blitzcolo
 */

#include "hs_core/BatchRenderer.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/VulkanContext.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/CommandHelper.hpp"
#include "scene/Scene.hpp"
#include "scene/Camera.hpp"
#include "core/Log.hpp"

#include <chrono>
#include <algorithm>
#include <cstring>

namespace quantiloom {

// ============================================================================
// Status String Conversion
// ============================================================================

const char* BatchRenderStatusToString(BatchRenderStatus status) {
    switch (status) {
        case BatchRenderStatus::Success:       return "Success";
        case BatchRenderStatus::InvalidInput:  return "Invalid input";
        case BatchRenderStatus::PipelineError: return "Pipeline error";
        case BatchRenderStatus::RenderFailed:  return "Render failed";
        case BatchRenderStatus::Cancelled:     return "Cancelled";
        case BatchRenderStatus::OutOfMemory:   return "Out of memory";
        case BatchRenderStatus::GPUError:      return "GPU error";
        default:                               return "Unknown status";
    }
}

// ============================================================================
// BatchRenderer::Impl - PIMPL Implementation
// ============================================================================

struct BatchRenderer::Impl {
    VulkanContext& context;
    RayTracingPipeline& pipeline;
    Scene& scene;

    std::atomic<bool> cancelled{false};

    // Statistics
    f64 lastRenderTime = 0.0;
    f64 avgTimePerBand = 0.0;
    u32 lastBandCount = 0;

    // GPU resources for single-wavelength rendering
    std::unique_ptr<GpuImage> outputImage;
    u32 imageWidth = 0;
    u32 imageHeight = 0;

    Impl(VulkanContext& ctx, RayTracingPipeline& pipe, Scene& sc)
        : context(ctx), pipeline(pipe), scene(sc) {}

    /**
     * @brief Initialize GPU output image if needed
     */
    bool EnsureOutputImage(u32 width, u32 height) {
        if (outputImage && imageWidth == width && imageHeight == height) {
            return true;  // Already allocated with correct size
        }

        try {
            // Create RGBA32F storage image for ray tracing output
            outputImage = std::make_unique<GpuImage>(
                context.GetAllocator(),
                context.GetDevice(),
                width,
                height,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            );

            // Transition to GENERAL layout for storage image usage
            CommandHelper::TransitionImageLayoutImmediate(
                context,
                outputImage->GetImage(),
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL
            );

            imageWidth = width;
            imageHeight = height;

            LOG_DEBUG("BatchRenderer: Created output image {}x{}", width, height);
            return true;
        }
        catch (const std::exception& e) {
            LOG_ERROR("BatchRenderer: Failed to create output image: {}", e.what());
            return false;
        }
    }

    /**
     * @brief Render single wavelength and read back to CPU
     */
    bool RenderWavelength(f32 wavelength_nm, const BatchRenderParams& params,
                          Image& outImage) {
        if (!outputImage) {
            LOG_ERROR("BatchRenderer: Output image not initialized");
            return false;
        }

        // Get camera from scene
        const Camera& camera = scene.camera;
        CameraData cameraData = camera.GetCameraData();

        // Set wavelength and spectral mode (single wavelength)
        cameraData.wavelength_nm = wavelength_nm;
        cameraData.spectral_mode = 0;  // SPECTRAL_MODE_SINGLE

        // Bind output image
        pipeline.BindOutputImage(*outputImage);

        // Render with accumulation (multiple samples)
        u32 randomSeed = static_cast<u32>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch().count()
        );

        pipeline.SetCameraData(cameraData);

        // The shader branches on the SPEC_SPECTRAL_MODE specialization
        // constant, not the push-constant copy above -- select the SINGLE
        // wavelength variant (cached after first creation)
        pipeline.SetSpecConstants(0 /* SPECTRAL_MODE_SINGLE */, false);

        // Keep each submit well under the ~2s Windows TDR limit
        constexpr u32 BATCH_SIZE = 2;

        VkCommandPoolCreateInfo cmdPoolInfo{};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cmdPoolInfo.queueFamilyIndex = context.GetGraphicsQueueFamily();

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(context.GetDevice(), &cmdPoolInfo, nullptr, &cmdPool) != VK_SUCCESS) {
            LOG_ERROR("BatchRenderer: Failed to create command pool");
            return false;
        }

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = cmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(context.GetDevice(), &allocInfo, &cmd);

        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vkCreateFence(context.GetDevice(), &fenceCreateInfo, nullptr, &fence);

        for (u32 batchStart = 0; batchStart < params.spp; batchStart += BATCH_SIZE) {
            u32 batchEnd = std::min(batchStart + BATCH_SIZE, params.spp);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            for (u32 sample = batchStart; sample < batchEnd; ++sample) {
                // randomSeed advances per sample; the sequence seed does not --
                // it fixes the Owen scrambles for the whole accumulation, which
                // is what makes the samples stratified against each other.
                pipeline.SetSamplingParams(0, sample, params.spp, randomSeed + sample, randomSeed);
                pipeline.TraceRays(cmd, imageWidth, imageHeight, sample == params.spp - 1);
            }

            vkEndCommandBuffer(cmd);

            vkResetFences(context.GetDevice(), 1, &fence);
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            if (vkQueueSubmit(context.GetGraphicsQueue(), 1, &submitInfo, fence) != VK_SUCCESS ||
                vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
                LOG_ERROR("BatchRenderer: GPU submit/wait failed (possible device lost)");
                vkDestroyFence(context.GetDevice(), fence, nullptr);
                vkDestroyCommandPool(context.GetDevice(), cmdPool, nullptr);
                return false;
            }
        }

        vkDestroyFence(context.GetDevice(), fence, nullptr);
        vkDestroyCommandPool(context.GetDevice(), cmdPool, nullptr);

        // Read back result to CPU
        VkDeviceSize bufferSize = imageWidth * imageHeight * 4 * sizeof(f32);
        GpuBuffer stagingBuffer(
            context.GetAllocator(),
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY
        );

        // Copy image to staging buffer
        CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
            // Transition image for transfer
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.image = outputImage->GetImage();
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            // Copy to buffer
            VkBufferImageCopy region{};
            region.imageExtent = {imageWidth, imageHeight, 1};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;

            vkCmdCopyImageToBuffer(cmd, outputImage->GetImage(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                stagingBuffer.GetHandle(), 1, &region);

            // Transition back to general layout
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        });

        // Map and copy data
        f32* mappedData = static_cast<f32*>(stagingBuffer.Map());
        if (!mappedData) {
            LOG_ERROR("BatchRenderer: Failed to map staging buffer");
            return false;
        }

        // Extract R channel (radiance) to grayscale output
        outImage.Resize(imageWidth, imageHeight, 1);
        for (u32 y = 0; y < imageHeight; ++y) {
            for (u32 x = 0; x < imageWidth; ++x) {
                u32 idx = (y * imageWidth + x) * 4;
                // For single wavelength mode, RGB channels should be equal
                // Use R channel as radiance value
                outImage(x, y, 0) = mappedData[idx];
            }
        }

        stagingBuffer.Unmap();
        return true;
    }
};

// ============================================================================
// BatchRenderer - Public Interface
// ============================================================================

BatchRenderer::BatchRenderer(
    VulkanContext& context,
    RayTracingPipeline& pipeline,
    Scene& scene
) : m_impl(std::make_unique<Impl>(context, pipeline, scene)) {
    LOG_DEBUG("BatchRenderer initialized");
}

BatchRenderer::~BatchRenderer() = default;

BatchRenderer::BatchRenderer(BatchRenderer&&) noexcept = default;
BatchRenderer& BatchRenderer::operator=(BatchRenderer&&) noexcept = default;

std::pair<BatchRenderStatus, SpectralCube> BatchRenderer::RenderBatch(
    const Vector<f32>& wavelengths,
    const BatchRenderParams& params,
    BatchRenderProgressCallback progressCallback,
    void* userData
) {
    using Clock = std::chrono::high_resolution_clock;

    // Reset cancellation flag
    m_impl->cancelled.store(false);

    // Validate input
    if (wavelengths.empty()) {
        LOG_ERROR("BatchRenderer: Empty wavelength list");
        return {BatchRenderStatus::InvalidInput, SpectralCube{}};
    }

    if (!ValidateWavelengthList(wavelengths)) {
        LOG_ERROR("BatchRenderer: Invalid wavelength list");
        return {BatchRenderStatus::InvalidInput, SpectralCube{}};
    }

    // Get image dimensions
    u32 width = m_impl->scene.width;
    u32 height = m_impl->scene.height;

    if (width == 0 || height == 0) {
        LOG_ERROR("BatchRenderer: Invalid image dimensions {}x{}", width, height);
        return {BatchRenderStatus::InvalidInput, SpectralCube{}};
    }

    // Ensure GPU resources
    if (!m_impl->EnsureOutputImage(width, height)) {
        return {BatchRenderStatus::GPUError, SpectralCube{}};
    }

    u32 numBands = static_cast<u32>(wavelengths.size());

    LOG_INFO("BatchRenderer: Starting batch render of {} bands, {}x{} px, {} spp",
             numBands, width, height, params.spp);

    // Allocate result cube
    SpectralCube result;
    try {
        result = SpectralCube(
            width, height, numBands,
            wavelengths.front(), wavelengths.back()
        );
        // Set actual wavelengths (may not be uniform)
        result.wavelengths = wavelengths;
    }
    catch (const std::exception& e) {
        LOG_ERROR("BatchRenderer: Failed to allocate SpectralCube: {}", e.what());
        return {BatchRenderStatus::OutOfMemory, SpectralCube{}};
    }

    // Add metadata
    result.metadata["renderer"] = "Quantiloom BatchRenderer";
    result.metadata["spp"] = std::to_string(params.spp);
    result.metadata["bands"] = std::to_string(numBands);

    auto startTime = Clock::now();
    Image bandImage;

    // ========================================================================
    // Main Rendering Loop
    // ========================================================================

    for (u32 bandIdx = 0; bandIdx < numBands; ++bandIdx) {
        // Check for cancellation
        if (m_impl->cancelled.load()) {
            LOG_INFO("BatchRenderer: Cancelled at band {}/{}", bandIdx, numBands);
            m_impl->lastBandCount = bandIdx;
            return {BatchRenderStatus::Cancelled, std::move(result)};
        }

        f32 wavelength_nm = wavelengths[bandIdx];

        if (params.verbose) {
            LOG_DEBUG("BatchRenderer: Rendering band {}/{}: {} nm",
                      bandIdx + 1, numBands, wavelength_nm);
        }

        // Render this wavelength
        if (!m_impl->RenderWavelength(wavelength_nm, params, bandImage)) {
            LOG_ERROR("BatchRenderer: Failed to render wavelength {} nm",
                      wavelength_nm);
            m_impl->lastBandCount = bandIdx;
            return {BatchRenderStatus::RenderFailed, std::move(result)};
        }

        // Copy band data to result cube
        f32* bandPtr = result.BandPtr(bandIdx);
        std::memcpy(bandPtr, bandImage.data.data(), width * height * sizeof(f32));

        // Update progress
        auto currentTime = Clock::now();
        f64 elapsed = std::chrono::duration<f64>(currentTime - startTime).count();

        if (progressCallback) {
            BatchRenderProgress progress;
            progress.currentBand = bandIdx + 1;
            progress.totalBands = numBands;
            progress.currentWavelength_nm = wavelength_nm;
            progress.elapsedSeconds = static_cast<f32>(elapsed);

            if (bandIdx > 0) {
                f64 avgTime = elapsed / (bandIdx + 1);
                progress.estimatedTotalSeconds = static_cast<f32>(avgTime * numBands);
            } else {
                progress.estimatedTotalSeconds = 0.0f;
            }

            progressCallback(progress, userData);
        }
    }

    // Calculate statistics
    auto endTime = Clock::now();
    m_impl->lastRenderTime = std::chrono::duration<f64>(endTime - startTime).count();
    m_impl->avgTimePerBand = m_impl->lastRenderTime / numBands;
    m_impl->lastBandCount = numBands;

    LOG_INFO("BatchRenderer: Complete - {} bands in {:.2f}s ({:.3f}s/band)",
             numBands, m_impl->lastRenderTime, m_impl->avgTimePerBand);

    return {BatchRenderStatus::Success, std::move(result)};
}

bool BatchRenderer::RenderSingleBand(
    f32 wavelength_nm,
    const BatchRenderParams& params,
    Image& outImage
) {
    // Ensure GPU resources
    u32 width = m_impl->scene.width;
    u32 height = m_impl->scene.height;

    if (!m_impl->EnsureOutputImage(width, height)) {
        return false;
    }

    return m_impl->RenderWavelength(wavelength_nm, params, outImage);
}

void BatchRenderer::Cancel() {
    m_impl->cancelled.store(true);
    LOG_DEBUG("BatchRenderer: Cancellation requested");
}

bool BatchRenderer::IsCancelled() const {
    return m_impl->cancelled.load();
}

void BatchRenderer::ResetCancellation() {
    m_impl->cancelled.store(false);
}

f64 BatchRenderer::GetLastRenderTime() const {
    return m_impl->lastRenderTime;
}

f64 BatchRenderer::GetAverageTimePerBand() const {
    return m_impl->avgTimePerBand;
}

u32 BatchRenderer::GetLastBandCount() const {
    return m_impl->lastBandCount;
}

std::pair<u32, u32> BatchRenderer::GetImageDimensions() const {
    return {m_impl->scene.width, m_impl->scene.height};
}

bool BatchRenderer::IsReady() const {
    return m_impl->outputImage != nullptr;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool ValidateWavelengthList(const Vector<f32>& wavelengths) {
    if (wavelengths.empty()) {
        return false;
    }

    // Check all values are positive
    for (f32 wl : wavelengths) {
        if (wl <= 0.0f) {
            return false;
        }
    }

    // Check sorted (ascending)
    for (usize i = 1; i < wavelengths.size(); ++i) {
        if (wavelengths[i] < wavelengths[i - 1]) {
            return false;
        }
    }

    return true;
}

void SortWavelengths(Vector<f32>& wavelengths) {
    std::sort(wavelengths.begin(), wavelengths.end());
}

f64 EstimateBatchTime(u32 numBands, f64 singleBandTime) {
    return static_cast<f64>(numBands) * singleBandTime;
}

} // namespace quantiloom
