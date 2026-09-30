#include "TextureManager.hpp"
#include "TextureCompressor.hpp"
#include "GpuBuffer.hpp"
#include "CommandHelper.hpp"
#include "core/Log.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace quantiloom {

struct TextureManager::PendingUpload {
    std::unique_ptr<GpuBuffer> staging;
    std::unique_ptr<GpuImage> image;
    VkFormat format = VK_FORMAT_UNDEFINED;
    u32 width = 0;
    u32 height = 0;
    u32 mipLevels = 1;
    u32 bufferRowLength = 0;
    u32 bufferImageHeight = 0;
    bool compressed = false;
};

// ============================================================================
// Construction / Destruction
// ============================================================================

TextureManager::TextureManager(VulkanContext& context, VkDeviceSize uploadBatchBytes)
    : m_context(context)
    , m_uploadBatchBytes(std::max<VkDeviceSize>(uploadBatchBytes, 1)) {
    // Resources are allocated lazily in UploadTextures
}

TextureManager::~TextureManager() {
    // Destroy all VkSampler objects
    VkDevice device = m_context.GetDevice();
    for (VkSampler sampler : m_samplers) {
        if (sampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, sampler, nullptr);
        }
    }

    // GpuImage objects will destroy VkImage and VkImageView automatically (RAII)
}

// ============================================================================
// Texture Upload
// ============================================================================

void TextureManager::UploadTextures(std::vector<Texture>& textures) {
    // Clear previous state
    m_images.clear();
    for (VkSampler sampler : m_samplers) {
        vkDestroySampler(m_context.GetDevice(), sampler, nullptr);
    }
    m_samplers.clear();
    m_imageViews.clear();
    m_lastUploadBatchCount = 0;

    // Handle empty texture list: create dummy 1x1 white texture
    if (textures.empty()) {
        QL_LOG_INFO("No textures to upload, creating dummy 1x1 white texture");
        Texture dummyTex = CreateDummyTexture();

        auto pending = PrepareTexture(dummyTex);
        CommandHelper::ExecuteImmediate(m_context, [&](VkCommandBuffer cmd) {
            RecordTextureUpload(cmd, *pending);
        });
        ++m_lastUploadBatchCount;
        m_images.push_back(std::move(pending->image));
        m_samplers.push_back(CreateSampler(dummyTex.sampler));
        m_imageViews.push_back(m_images.back()->GetView());

        return;
    }

    // Upload all textures
    QL_LOG_INFO("Uploading {} textures to GPU", textures.size());

    struct BatchEntry {
        Texture* source = nullptr;
        std::unique_ptr<PendingUpload> upload;
    };
    std::vector<BatchEntry> batch;
    VkDeviceSize batchBytes = 0;

    const auto flush = [&]() {
        if (batch.empty()) return;
        CommandHelper::ExecuteImmediate(m_context, [&](VkCommandBuffer cmd) {
            for (const BatchEntry& entry : batch) {
                RecordTextureUpload(cmd, *entry.upload);
            }
        });
        ++m_lastUploadBatchCount;

        for (BatchEntry& entry : batch) {
            VkSampler sampler = CreateSampler(entry.source->sampler);
            m_imageViews.push_back(entry.upload->image->GetView());
            m_samplers.push_back(sampler);
            m_images.push_back(std::move(entry.upload->image));

            // Release only after the submission that consumed the staging
            // bytes completed. Retained pixels feed interactive re-unmixing.
            if (!entry.source->retainCpuPixels) {
                entry.source->ReleaseCPUMemory();
            }
        }
        batch.clear();
        batchBytes = 0;
    };

    for (Texture& texture : textures) {
        const VkDeviceSize rawBytes =
            static_cast<VkDeviceSize>(texture.width) * texture.height * 4;
        const VkDeviceSize estimate = texture.HasBC7Data()
            ? static_cast<VkDeviceSize>(texture.bc7Data->data.size()) : rawBytes;
        if (!batch.empty() && batchBytes + estimate > m_uploadBatchBytes) {
            flush();
        }

        auto pending = PrepareTexture(texture);
        const VkDeviceSize uploadBytes = pending->staging->GetSize();
        if (!batch.empty() && batchBytes + uploadBytes > m_uploadBatchBytes) {
            flush();
        }
        batchBytes += uploadBytes;
        batch.push_back({&texture, std::move(pending)});

        // A texture at or above the cap owns one batch. The cap bounds staging
        // memory for ordinary textures; a single larger texture is unavoidable.
        if (batchBytes >= m_uploadBatchBytes) {
            flush();
        }
    }
    flush();

    QL_LOG_INFO("  Texture upload complete: {} textures, {} samplers, {} batch(es) "
                "(CPU memory released)",
                m_images.size(), m_samplers.size(), m_lastUploadBatchCount);
}

i32 TextureManager::AppendTexture(const Texture& texture) {
    if (!texture.IsValid()) {
        QL_LOG_WARN("AppendTexture: invalid texture '{}'", texture.name);
        return -1;
    }

    auto gpuImage = UploadTexture(texture);
    if (!gpuImage) {
        return -1;
    }
    VkSampler sampler = CreateSampler(texture.sampler);

    const i32 index = static_cast<i32>(m_images.size());
    m_imageViews.push_back(gpuImage->GetView());
    m_samplers.push_back(sampler);
    m_images.push_back(std::move(gpuImage));

    QL_LOG_INFO("  Appended texture '{}' at index {} ({} total)", texture.name, index,
                m_images.size());
    return index;
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

std::unique_ptr<GpuImage> TextureManager::UploadTexture(const Texture& texture) const {
    auto pending = PrepareTexture(texture);
    CommandHelper::ExecuteImmediate(m_context, [&](VkCommandBuffer cmd) {
        RecordTextureUpload(cmd, *pending);
    });
    return std::move(pending->image);
}

std::unique_ptr<TextureManager::PendingUpload> TextureManager::PrepareTexture(
    const Texture& texture) const {
    // Validate texture data
    if (texture.pixels.empty()) {
        QL_LOG_ERROR("Texture '{}' has no pixel data", texture.name);
        throw std::runtime_error("Cannot upload empty texture");
    }

    if (texture.channels != 4) {
        QL_LOG_ERROR("Texture '{}' has {} channels (expected 4 for RGBA8)",
                     texture.name, texture.channels);
        throw std::runtime_error("Only RGBA8 textures are supported");
    }

    VkDeviceSize uncompressedSize = static_cast<VkDeviceSize>(texture.width) * texture.height * 4;

    if (texture.pixels.size() != uncompressedSize) {
        QL_LOG_ERROR("Texture '{}' pixel data size mismatch: expected {} bytes, got {}",
                     texture.name, uncompressedSize, texture.pixels.size());
        throw std::runtime_error("Texture pixel data size mismatch");
    }

    // ========================================================================
    // Try BC7 Compression (4:1 VRAM savings)
    // ========================================================================

    // Check for pre-compressed BC7 data first (from parallel compression during loading)
    if (texture.HasBC7Data()) {
        // Use pre-compressed data directly
        BC7CompressedData compressed;
        compressed.width = texture.width;
        compressed.height = texture.height;
        compressed.blockCountX = texture.bc7Data->blockCountX;
        compressed.blockCountY = texture.bc7Data->blockCountY;
        compressed.isSRGB = texture.isSRGB;
        compressed.data = texture.bc7Data->data;  // Copy to avoid modifying original

        QL_LOG_DEBUG("  Using pre-compressed BC7 data for '{}'", texture.name);
        return PrepareBC7Texture(texture, compressed);
    }

    // Fall back to runtime compression if no pre-compressed data
    if (TextureCompressor::IsAvailable() && TextureCompressor::CanCompress(texture)) {
        auto compressed = TextureCompressor::CompressBC7(texture, false /* fast mode */);
        if (compressed.has_value()) {
            return PrepareBC7Texture(texture, compressed.value());
        }
        // Fall through to uncompressed upload if compression failed
        QL_LOG_WARN("BC7 compression failed for '{}', using uncompressed", texture.name);
    }

    // ========================================================================
    // Uncompressed Upload Path (original code)
    // ========================================================================
    VkFormat format = texture.isSRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    const char* formatName = texture.isSRGB ? "RGBA8_SRGB" : "RGBA8_UNORM";

    QL_LOG_INFO("  Uploading texture '{}': {}x{} {} ({} bytes)",
                texture.name, texture.width, texture.height, formatName, uncompressedSize);

    // Step 1: Create staging buffer (CPU-accessible)
    auto pending = std::make_unique<PendingUpload>();
    pending->staging = std::make_unique<GpuBuffer>(
        m_context.GetAllocator(),
        uncompressedSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    // Step 2: Upload CPU pixel data to staging buffer
    pending->staging->Upload(texture.pixels.data(), uncompressedSize);

    // Step 3: Create device-local GPU image with correct format and full mipmap chain
    u32 mipLevels = static_cast<u32>(std::floor(std::log2(std::max(texture.width, texture.height)))) + 1;
    pending->image = std::make_unique<GpuImage>(
        m_context.GetAllocator(),
        m_context.GetDevice(),
        texture.width,
        texture.height,
        format,  // sRGB or UNORM based on texture usage
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        mipLevels  // Full mipmap chain
    );

    pending->format = format;
    pending->width = texture.width;
    pending->height = texture.height;
    pending->mipLevels = mipLevels;
    return pending;
}

VkSampler TextureManager::CreateSampler(const TextureSampler& samplerInfo) const {
    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

    // Filter modes
    samplerCreateInfo.magFilter = ToVkFilter(samplerInfo.magFilter);
    samplerCreateInfo.minFilter = ToVkFilter(samplerInfo.minFilter);

    // Wrap modes
    samplerCreateInfo.addressModeU = ToVkAddressMode(samplerInfo.wrapS);
    samplerCreateInfo.addressModeV = ToVkAddressMode(samplerInfo.wrapT);
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;  // Default for W

    // Anisotropy (disabled for M1 - can be enabled in M2+ if needed)
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.maxAnisotropy = 1.0f;

    // Border color (for clamp-to-border mode, not used in glTF)
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

    // Coordinate system (normalized [0, 1] for glTF)
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

    // Comparison (for shadow mapping, not used here)
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    // Mipmapping (trilinear filtering)
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = VK_LOD_CLAMP_NONE;  // Use all mip levels

    VkSampler sampler;

    if (const VkResult result = vkCreateSampler(m_context.GetDevice(), &samplerCreateInfo, nullptr, &sampler);
        result != VK_SUCCESS) {
        QL_LOG_ERROR("Failed to create VkSampler: error code {}", static_cast<int>(result));
        throw std::runtime_error("VkSampler creation failed");
    }

    return sampler;
}

VkFilter TextureManager::ToVkFilter(TextureSampler::Filter filter) {
    switch (filter) {
        case TextureSampler::Filter::Nearest:
            return VK_FILTER_NEAREST;
        case TextureSampler::Filter::Linear:
            return VK_FILTER_LINEAR;
        default:
            QL_LOG_WARN("Unknown TextureSampler::Filter, defaulting to LINEAR");
            return VK_FILTER_LINEAR;
    }
}

VkSamplerAddressMode TextureManager::ToVkAddressMode(TextureSampler::WrapMode wrapMode) {
    switch (wrapMode) {
        case TextureSampler::WrapMode::Repeat:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case TextureSampler::WrapMode::ClampToEdge:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case TextureSampler::WrapMode::MirroredRepeat:
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        default:
            QL_LOG_WARN("Unknown TextureSampler::WrapMode, defaulting to REPEAT");
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

Texture TextureManager::CreateDummyTexture() {
    Texture dummy;
    dummy.name = "DummyWhiteTexture";
    dummy.sourceUri = "";
    dummy.width = 1;
    dummy.height = 1;
    dummy.channels = 4;

    // 1x1 white pixel (RGBA8: 255, 255, 255, 255)
    dummy.pixels = {255, 255, 255, 255};

    // Default sampler: linear filtering, repeat wrapping
    dummy.sampler.minFilter = TextureSampler::Filter::Linear;
    dummy.sampler.magFilter = TextureSampler::Filter::Linear;
    dummy.sampler.wrapS = TextureSampler::WrapMode::Repeat;
    dummy.sampler.wrapT = TextureSampler::WrapMode::Repeat;

    return dummy;
}

// ============================================================================
// BC7 Compressed Texture Upload
// ============================================================================

std::unique_ptr<TextureManager::PendingUpload> TextureManager::PrepareBC7Texture(
    const Texture& texture,
    const BC7CompressedData& compressed) const {

    // Choose BC7 format based on color space
    VkFormat format = compressed.isSRGB ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
    const char* formatName = compressed.isSRGB ? "BC7_SRGB" : "BC7_UNORM";

    QL_LOG_INFO("  Uploading BC7 texture '{}': {}x{} {} ({} bytes, ratio: {:.1f}x)",
                texture.name, texture.width, texture.height, formatName,
                compressed.GetCompressedSize(), compressed.GetCompressionRatio());

    // Get aligned dimensions for BC7 (must be multiple of 4)
    u32 alignedWidth = compressed.blockCountX * 4;
    u32 alignedHeight = compressed.blockCountY * 4;

    // Calculate mip levels for BC7 (limited because each level must be at least 4x4)
    // For BC7, we can't generate mipmaps via blit because it's a block-compressed format
    // Option 1: Single mip level (simpler, used here)
    // Option 2: Pre-compress all mip levels on CPU (more complex, future enhancement)
    u32 mipLevels = 1;

    // Step 1: Create staging buffer for compressed data
    VkDeviceSize compressedSize = static_cast<VkDeviceSize>(compressed.GetCompressedSize());
    auto pending = std::make_unique<PendingUpload>();
    pending->staging = std::make_unique<GpuBuffer>(
        m_context.GetAllocator(),
        compressedSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    // Step 2: Copy compressed data to staging buffer
    pending->staging->Upload(compressed.data.data(), compressedSize);

    // Step 3: Create device-local GPU image with BC7 format
    // Note: For BC7, image dimensions should be the original texture dimensions,
    // not the aligned dimensions. Vulkan handles the block alignment internally.
    pending->image = std::make_unique<GpuImage>(
        m_context.GetAllocator(),
        m_context.GetDevice(),
        texture.width,  // Original width (Vulkan handles alignment)
        texture.height, // Original height
        format,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        mipLevels
    );

    pending->format = format;
    pending->width = texture.width;
    pending->height = texture.height;
    pending->mipLevels = mipLevels;
    pending->bufferRowLength = alignedWidth;
    pending->bufferImageHeight = alignedHeight;
    pending->compressed = true;
    return pending;
}

void TextureManager::RecordTextureUpload(VkCommandBuffer cmd,
                                         const PendingUpload& upload) const {
    CommandHelper::TransitionImageLayout(
        cmd, upload.image->GetImage(), upload.format,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        upload.compressed ? upload.mipLevels : 1);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = upload.bufferRowLength;
    region.bufferImageHeight = upload.bufferImageHeight;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {upload.width, upload.height, 1};
    vkCmdCopyBufferToImage(cmd, upload.staging->GetHandle(), upload.image->GetImage(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    if (!upload.compressed) {
        GenerateMipmaps(cmd, upload.image.get(), upload.format,
                        upload.width, upload.height, upload.mipLevels);
        return;
    }

    VkImageMemoryBarrier ready{};
    ready.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ready.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ready.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ready.image = upload.image->GetImage();
    ready.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ready.subresourceRange.baseMipLevel = 0;
    ready.subresourceRange.levelCount = upload.mipLevels;
    ready.subresourceRange.baseArrayLayer = 0;
    ready.subresourceRange.layerCount = 1;
    ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &ready);
}

// ============================================================================
// Mipmap Generation Helper
// ============================================================================

void TextureManager::GenerateMipmaps(VkCommandBuffer cmd, GpuImage* gpuImage,
                                      VkFormat format, u32 texWidth, u32 texHeight,
                                      u32 mipLevels) const {
    // Generate mipmaps using vkCmdBlitImage
    // Transition base level to TRANSFER_SRC for blit source
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = gpuImage->GetImage();
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    u32 mipWidth = texWidth;
    u32 mipHeight = texHeight;

    for (u32 i = 1; i < mipLevels; ++i) {
        // Transition previous level to TRANSFER_SRC_OPTIMAL
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        // Blit from level i-1 to level i
        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {static_cast<i32>(mipWidth), static_cast<i32>(mipHeight), 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;

        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {static_cast<i32>(mipWidth), static_cast<i32>(mipHeight), 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        // Transition current level to TRANSFER_DST before blit
        barrier.subresourceRange.baseMipLevel = i;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        vkCmdBlitImage(cmd,
            gpuImage->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            gpuImage->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit,
            VK_FILTER_LINEAR);

        // Transition previous level to SHADER_READ_ONLY
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);
    }

    // Transition last mip level to SHADER_READ_ONLY
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

} // namespace quantiloom
