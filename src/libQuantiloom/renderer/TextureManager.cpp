#include "TextureManager.hpp"
#include "GpuBuffer.hpp"
#include "CommandHelper.hpp"
#include "core/Log.hpp"
#include <stdexcept>
#include <cstring>

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
        Texture dummyTex = CreateDummyTexture();

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

std::unique_ptr<GpuImage> TextureManager::UploadTexture(const Texture& texture) {
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

    QL_LOG_INFO("  Uploading texture '{}': {}x{} RGBA8 ({} bytes)",
                texture.name, texture.width, texture.height, bufferSize);

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

    // Step 3: Create device-local GPU image (RGBA8_UNORM)
    auto gpuImage = std::make_unique<GpuImage>(
        m_context.GetAllocator(),
        m_context.GetDevice(),
        texture.width,
        texture.height,
        VK_FORMAT_R8G8B8A8_UNORM,  // Standard 8-bit RGBA format
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        1  // mipLevels (no mipmapping for M1)
    );

    // Step 4: Execute upload via command buffer
    CommandHelper::ExecuteImmediate(m_context, [&](VkCommandBuffer cmd) {
        // Transition: UNDEFINED -> TRANSFER_DST_OPTIMAL
        CommandHelper::TransitionImageLayout(
            cmd,
            gpuImage->GetImage(),
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1  // mipLevels
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

        // Copy staging buffer to GPU image
        vkCmdCopyBufferToImage(
            cmd,
            stagingBuffer.GetHandle(),
            gpuImage->GetImage(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

        // Transition: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
        CommandHelper::TransitionImageLayout(
            cmd,
            gpuImage->GetImage(),
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1  // mipLevels
        );
    });

    return gpuImage;
}

VkSampler TextureManager::CreateSampler(const TextureSampler& samplerInfo) {
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

    // Mipmapping (disabled for M1)
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;  // No mipmaps

    VkSampler sampler;
    VkResult result = vkCreateSampler(m_context.GetDevice(), &samplerCreateInfo, nullptr, &sampler);

    if (result != VK_SUCCESS) {
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
