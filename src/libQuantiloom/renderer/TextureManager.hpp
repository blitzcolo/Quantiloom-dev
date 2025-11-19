#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "scene/Texture.hpp"
#include "VulkanContext.hpp"
#include "GpuImage.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

// ============================================================================
// TextureManager - Manages GPU texture upload and binding
// ============================================================================
// Responsibilities:
// - Upload CPU Texture objects to GPU VkImage resources
// - Create VkSampler for each texture based on TextureSampler settings
// - Manage lifetime of all texture resources (RAII)
// - Provide arrays of VkImageView and VkSampler for bindless descriptor sets
//
// Architecture:
// - One GpuImage per texture (VkImage + VkImageView)
// - One VkSampler per texture (filter + wrap mode)
// - All textures uploaded to RGBA8_UNORM format
// - Image layout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
//
// Usage:
//   TextureManager texMgr(context);
//   texMgr.UploadTextures(scene.textures);
//
//   // Bind to descriptor set
//   const auto& views = texMgr.GetImageViews();
//   const auto& samplers = texMgr.GetSamplers();
// ============================================================================

namespace quantiloom {

class QL_API TextureManager {
public:
    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    // Create texture manager
    // Note: Does not upload any textures yet (call UploadTextures)
    explicit TextureManager(VulkanContext& context);

    // Destructor: automatically destroys all VkSampler objects
    // (GpuImage handles VkImage/VkImageView destruction via RAII)
    ~TextureManager();

    // ========================================================================
    // Texture Upload
    // ========================================================================

    // Upload all textures from CPU to GPU
    // - Creates VkImage + VkImageView for each texture (RGBA8_UNORM)
    // - Creates VkSampler based on TextureSampler settings
    // - Uploads pixel data via staging buffer
    // - Transitions layout to SHADER_READ_ONLY_OPTIMAL
    //
    // Note: If textures vector is empty, creates a single 1x1 white dummy texture
    // (This allows shader code to always sample without null checks)
    void UploadTextures(const std::vector<Texture>& textures);

    // ========================================================================
    // Accessors
    // ========================================================================

    // Get number of textures
    u32 GetTextureCount() const { return static_cast<u32>(m_images.size()); }

    // Get array of VkImageView handles (for descriptor set binding)
    // Index matches original texture index from scene
    const std::vector<VkImageView>& GetImageViews() const { return m_imageViews; }

    // Get array of VkSampler handles (for descriptor set binding)
    // Index matches original texture index from scene
    const std::vector<VkSampler>& GetSamplers() const { return m_samplers; }

    // Check if textures have been uploaded
    bool IsEmpty() const { return m_images.empty(); }

private:
    // ========================================================================
    // Internal Helper Functions
    // ========================================================================

    // Upload a single texture to GPU
    // Returns GpuImage containing VkImage + VkImageView
    std::unique_ptr<GpuImage> UploadTexture(const Texture& texture);

    // Create VkSampler based on TextureSampler settings
    VkSampler CreateSampler(const TextureSampler& samplerInfo);

    // Convert TextureSampler::Filter to VkFilter
    static VkFilter ToVkFilter(TextureSampler::Filter filter);

    // Convert TextureSampler::WrapMode to VkSamplerAddressMode
    static VkSamplerAddressMode ToVkAddressMode(TextureSampler::WrapMode wrapMode);

    // Create a 1x1 white dummy texture (fallback for empty texture list)
    Texture CreateDummyTexture();

    // ========================================================================
    // Member Variables
    // ========================================================================

    VulkanContext& m_context;

    // GPU image resources (VkImage + VkImageView managed by GpuImage)
    std::vector<std::unique_ptr<GpuImage>> m_images;

    // VkSampler objects (created manually, must be destroyed in destructor)
    std::vector<VkSampler> m_samplers;

    // Cached arrays for descriptor binding (updated after upload)
    std::vector<VkImageView> m_imageViews;  // Extracted from m_images
};

} // namespace quantiloom
