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
#include "renderer/TextureTransitions.hpp"
#include "gpu/Device.hpp"
#include "gpu/vulkan/VulkanBridge.hpp"
#include "scene/Scene.hpp"
#include "scene/Camera.hpp"
#include "core/Log.hpp"

#include <chrono>
#include <algorithm>
#include <cstring>
#include <span>

namespace quantiloom {

namespace {
u32 StableMix(u32 value) {
    value = (value ^ 61u) ^ (value >> 16u);
    value *= 9u;
    value ^= value >> 4u;
    value *= 0x27d4eb2du;
    return value ^ (value >> 15u);
}
} // namespace

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
    /// The compute RHI's view of `context`, adopted and non-owning. Declared
    /// right after the context so every gpu:: object dies before it does.
    std::unique_ptr<gpu::Device> gpuDevice;
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
        : context(ctx), gpuDevice(gpu::vulkan::AdoptContext(ctx)), pipeline(pipe), scene(sc) {}

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
            rendercore::LegacyTransitionImmediate(
                *gpuDevice, gpu::vulkan::Ref(*outputImage),
                rendercore::Layout::Undefined, rendercore::Layout::General
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
                          const CameraData* cameraOverride, Image& outImage) {
        if (!outputImage) {
            LOG_ERROR("BatchRenderer: Output image not initialized");
            return false;
        }

        // The scene camera is the historical batch default. Camera capture
        // supplies its resolved pose explicitly, so a timeline preview cannot
        // accidentally snap back to the static asset camera.
        CameraData cameraData = cameraOverride != nullptr
            ? *cameraOverride : scene.camera.GetCameraData();

        // Set wavelength and spectral mode (single wavelength)
        cameraData.wavelength_nm = wavelength_nm;
        cameraData.spectral_mode = 0;  // SPECTRAL_MODE_SINGLE

        // Bind output image
        pipeline.BindOutputImage(gpu::vulkan::Ref(*outputImage));

        // Render with accumulation (multiple samples)
        // Common random numbers across wavelengths reduce the variance of a
        // response integral. Neither wall time nor wavelength belongs in the
        // ray seed; the acquisition/frame and sample identify this draw.
        const u32 sequenceSeed = StableMix(params.renderSeed ^
                                           StableMix(params.frameIndex) ^
                                           StableMix(params.frameIndexHigh ^ 0x9e3779b9u));

        pipeline.SetCameraData(cameraData);

        // The shader branches on the SPEC_SPECTRAL_MODE specialization
        // constant, not the push-constant copy above -- select the SINGLE
        // wavelength variant (cached after first creation)
        pipeline.SetSpecConstants(0 /* SPECTRAL_MODE_SINGLE */, false);

        // Keep each submit well under the ~2s Windows TDR limit
        constexpr u32 BATCH_SIZE = 2;

        {
            gpu::CommandPoolDesc poolDesc;
            poolDesc.transient = true;
            poolDesc.resettable = true;
            auto createdPool = gpuDevice->CreateCommandPool(poolDesc);
            if (!createdPool) {
                LOG_ERROR("BatchRenderer: Failed to create command pool");
                return false;
            }
            // Reverse declaration order at every exit: fence, then the command
            // buffer, then the pool. The scope ends before the readback, where
            // the hand-written destroys were.
            std::unique_ptr<gpu::CommandPool> cmdPool = std::move(createdPool).value();

            auto allocated = cmdPool->Allocate();
            auto createdFence = gpuDevice->CreateFence();
            if (!allocated || !createdFence) {
                // Neither call was checked before; the batch failed at its submit.
                LOG_ERROR("BatchRenderer: GPU submit/wait failed (possible device lost)");
                return false;
            }
            std::unique_ptr<gpu::CommandBuffer> cmd = std::move(allocated).value();
            std::unique_ptr<gpu::Fence> fence = std::move(createdFence).value();

            for (u32 batchStart = 0; batchStart < params.spp; batchStart += BATCH_SIZE) {
                u32 batchEnd = std::min(batchStart + BATCH_SIZE, params.spp);

                // A primary command buffer must return to INITIAL before it is
                // begun again. The pool permits individual resets; omitting this
                // was invalid Vulkan and made repeated camera wavelength renders
                // fail nondeterministically after the first two-sample batch.
                if (batchStart != 0 && !cmd->Reset()) {
                    LOG_ERROR("BatchRenderer: Failed to reset command buffer");
                    return false;
                }

                auto begun = cmd->Begin();
                if (!begun) {
                    // Unchecked before, as were end and the fence reset below;
                    // the batch failed at its submit, whose text this keeps.
                    LOG_ERROR("BatchRenderer: GPU submit/wait failed (possible device lost)");
                    return false;
                }
                gpu::CommandList& list = *begun.value();

                for (u32 sample = batchStart; sample < batchEnd; ++sample) {
                    // The per-sample seed changes; the sequence seed does not --
                    // it fixes the Owen scrambles for the whole accumulation, which
                    // is what makes the samples stratified against each other.
                    const u32 sampleSeed = StableMix(sequenceSeed ^ StableMix(sample));
                    pipeline.SetSamplingParams(params.frameIndex, sample, params.spp,
                                               sampleSeed, sequenceSeed);
                    pipeline.TraceRays(list, imageWidth, imageHeight, sample == params.spp - 1);
                }

                static_cast<void>(cmd->End());

                static_cast<void>(fence->Reset());
                if (!gpuDevice->Submit(*cmd, fence.get()) || !fence->Wait()) {
                    LOG_ERROR("BatchRenderer: GPU submit/wait failed (possible device lost)");
                    return false;
                }
            }
        }

        // Read back result to CPU
        VkDeviceSize bufferSize = imageWidth * imageHeight * 4 * sizeof(f32);
        GpuBuffer stagingBuffer(
            context.GetAllocator(),
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY
        );

        // Copy image to staging buffer
        const gpu::TextureRef source =
            gpu::vulkan::Ref(*outputImage).WithLayout(gpu::TextureLayoutPolicy::Optimal);
        gpuDevice->ExecuteImmediate([&](gpu::CommandList& list) {
            // Transition image for transfer
            const gpu::TextureBarrier toTransfer{source, gpu::Access::RayTraceWrite,
                                                 gpu::Access::TransferRead};
            gpu::BarrierBatch before;
            before.textures = std::span<const gpu::TextureBarrier>(&toTransfer, 1);
            list.Barrier(before);

            // Copy to buffer
            gpu::BufferTextureCopy region;
            region.width = imageWidth;
            region.height = imageHeight;
            list.CopyTextureToBuffer(source, gpu::vulkan::Ref(stagingBuffer), region);

            // Transition back to general layout
            const gpu::TextureBarrier toGeneral{source, gpu::Access::TransferRead,
                                                gpu::Access::RayTraceWrite};
            gpu::BarrierBatch after;
            after.textures = std::span<const gpu::TextureBarrier>(&toGeneral, 1);
            list.Barrier(after);
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
    u32 width = params.outputWidth != 0 ? params.outputWidth : m_impl->scene.width;
    u32 height = params.outputHeight != 0 ? params.outputHeight : m_impl->scene.height;

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
        if (!m_impl->RenderWavelength(wavelength_nm, params, nullptr, bandImage)) {
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
    u32 width = params.outputWidth != 0 ? params.outputWidth : m_impl->scene.width;
    u32 height = params.outputHeight != 0 ? params.outputHeight : m_impl->scene.height;

    if (!m_impl->EnsureOutputImage(width, height)) {
        return false;
    }

    return m_impl->RenderWavelength(wavelength_nm, params, nullptr, outImage);
}

bool BatchRenderer::RenderSingleBand(
    f32 wavelength_nm,
    const BatchRenderParams& params,
    const CameraData& cameraData,
    Image& outImage
) {
    const u32 width = params.outputWidth != 0 ? params.outputWidth : m_impl->scene.width;
    const u32 height = params.outputHeight != 0 ? params.outputHeight : m_impl->scene.height;
    if (!m_impl->EnsureOutputImage(width, height)) {
        return false;
    }
    return m_impl->RenderWavelength(wavelength_nm, params, &cameraData, outImage);
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
