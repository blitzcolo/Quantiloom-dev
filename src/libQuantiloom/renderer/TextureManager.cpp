#include "TextureManager.hpp"
#include "GpuBuffer.hpp"
#include "CommandHelper.hpp"
#include "core/Log.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace quantiloom {

// ============================================================================
// Construction / Destruction
// ============================================================================

TextureManager::TextureManager(VulkanContext& context)
    : m_context(context) {
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

void TextureManager::UploadTextures(const std::vector<Texture>& textures) {
    // Clear previous state
    m_images.clear();
    for (VkSampler sampler : m_samplers) {
        vkDestroySampler(m_context.GetDevice(), sampler, nullptr);
    }
    m_samplers.clear();
    m_imageViews.clear();

    // Handle empty texture list: create dummy 1x1 white texture
    if (textures.empty()) {
        QL_LOG_INFO("No textures to upload, creating dummy 1x1 white texture");
        const Texture dummyTex = CreateDummyTexture();

        m_images.push_back(UploadTexture(dummyTex));
        m_samplers.push_back(CreateSampler(dummyTex.sampler));
        m_imageViews.push_back(m_images.back()->GetView());

        return;
    }

    // Upload all textures
    QL_LOG_INFO("Uploading {} textures to GPU", textures.size());

    for (const Texture& texture : textures) {
        // Upload texture to GPU
        auto gpuImage = UploadTexture(texture);

        // Create sampler for this texture
        VkSampler sampler = CreateSampler(texture.sampler);

        // Store resources
        m_imageViews.push_back(gpuImage->GetView());
        m_samplers.push_back(sampler);
        m_images.push_back(std::move(gpuImage));
    }

    QL_LOG_INFO("  Texture upload complete: {} textures, {} samplers",
                m_images.size(), m_samplers.size());
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

std::unique_ptr<GpuImage> TextureManager::UploadTexture(const Texture& texture) const {
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

    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(texture.width) * texture.height * 4;

    if (texture.pixels.size() != bufferSize) {
        QL_LOG_ERROR("Texture '{}' pixel data size mismatch: expected {} bytes, got {}",
                     texture.name, bufferSize, texture.pixels.size());
        throw std::runtime_error("Texture pixel data size mismatch");
    }

    // Choose format based on color space (glTF 2.0 spec)
    // sRGB textures (baseColor, emissive) need gamma-correct sampling
    VkFormat format = texture.isSRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    const char* formatName = texture.isSRGB ? "RGBA8_SRGB" : "RGBA8_UNORM";

    QL_LOG_INFO("  Uploading texture '{}': {}x{} {} ({} bytes)",
                texture.name, texture.width, texture.height, formatName, bufferSize);

    // Step 1: Create staging buffer (CPU-accessible)
    GpuBuffer stagingBuffer(
        m_context.GetAllocator(),
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    // Step 2: Upload CPU pixel data to staging buffer
    void* data = stagingBuffer.Map();
    std::memcpy(data, texture.pixels.data(), bufferSize);
    stagingBuffer.Unmap();

    // Step 3: Create device-local GPU image with correct format and full mipmap chain
    u32 mipLevels = static_cast<u32>(std::floor(std::log2(std::max(texture.width, texture.height)))) + 1;
    auto gpuImage = std::make_unique<GpuImage>(
        m_context.GetAllocator(),
        m_context.GetDevice(),
        texture.width,
        texture.height,
        format,  // sRGB or UNORM based on texture usage
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        mipLevels  // Full mipmap chain
    );

    // Step 4: Execute upload via command buffer
    CommandHelper::ExecuteImmediate(m_context, [&](VkCommandBuffer cmd) {
        // Transition: UNDEFINED -> TRANSFER_DST_OPTIMAL (base level only, for upload)
        CommandHelper::TransitionImageLayout(
            cmd,
            gpuImage->GetImage(),
            format,  // Use correct format (SRGB or UNORM)
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1  // Only base level for initial upload
        );

        // Define copy region (buffer -> image)
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;   // Tightly packed
        region.bufferImageHeight = 0; // Tightly packed
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {texture.width, texture.height, 1};

        // Copy staging buffer to GPU image (base level only)
        vkCmdCopyBufferToImage(
            cmd,
            stagingBuffer.GetHandle(),
            gpuImage->GetImage(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

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

        u32 mipWidth = texture.width;
        u32 mipHeight = texture.height;

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
    });

    return gpuImage;
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

} // namespace quantiloom
