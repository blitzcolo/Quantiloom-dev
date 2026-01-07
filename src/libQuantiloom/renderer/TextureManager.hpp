/**
 * @file TextureManager.hpp
 * @brief GPU texture upload and bindless descriptor array management
 *
 * Provides TextureManager class for batch texture upload to GPU:
 * - Uploads multiple CPU Texture objects to GPU VkImage resources
 * - Creates VkSampler for each texture based on filter/wrap settings
 * - Manages lifetime of all texture images and samplers (RAII)
 * - Provides VkImageView and VkSampler arrays for bindless descriptor sets
 *
 * Texture format conversion:
 * - All textures uploaded as RGBA8_UNORM (32 bpp)
 * - Source formats (1-4 channels) expanded to RGBA automatically
 * - Image layout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (after upload)
 *
 * Bindless descriptor indexing:
 * - Each texture gets a unique index (matches Material texture indices)
 * - Shader access via: texture2DArray[textureIndex]
 * - Requires VK_EXT_descriptor_indexing extension
 *
 * Fallback behavior:
 * - If no textures provided, creates 1x1 white dummy texture
 * - Prevents null descriptor access in shaders
 *
 * @note All textures uploaded during UploadTextures() call (batch operation)
 * @note Samplers destroyed manually in destructor (GpuImage handles images via RAII)
 * @note Texture indices are immutable after upload
 *
 * @author wtflmao
 */

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

namespace quantiloom {
    
/**
 * @class TextureManager
 * @brief Manages batch upload of textures to GPU with bindless descriptor arrays
 *
 * Central texture management system for Quantiloom renderer.
 * Handles conversion from CPU Texture data to GPU VkImage resources.
 *
 * Upload workflow:
 * 1. Constructor creates TextureManager (no textures uploaded yet)
 * 2. UploadTextures() converts all CPU textures to GPU images
 * 3. For each texture:
 *    - Create VkImage (RGBA8_UNORM format)
 *    - Upload pixel data via staging buffer
 *    - Transition layout to SHADER_READ_ONLY_OPTIMAL
 *    - Create VkSampler with specified filter/wrap modes
 * 4. GetImageViews()/GetSamplers() return arrays for descriptor binding
 *
 * Usage example:
 * @code
 * // Create manager
 * TextureManager texMgr(context);
 *
 * // Upload all scene textures
 * texMgr.UploadTextures(scene.textures);
 *
 * // Bind to pipeline descriptor set
 * const auto& views = texMgr.GetImageViews();
 * const auto& samplers = texMgr.GetSamplers();
 * pipeline.BindTextures(views, samplers);  // Binding 6-7
 *
 * // Shader access (HLSL):
 * // [[vk::binding(6, 0)]] Texture2D textures[];
 * // [[vk::binding(7, 0)]] SamplerState samplers[];
 * // float4 color = textures[matData.baseColorTextureIndex].Sample(
 * //     samplers[matData.baseColorTextureIndex], uv);
 * @endcode
 *
 * @note Empty texture list triggers creation of 1x1 white dummy texture
 * @note All textures converted to RGBA8_UNORM regardless of source format
 * @note Texture indices must match Material texture index references
 *
 * @see Texture for CPU-side texture data structure
 * @see GpuImage for GPU image resource management
 * @see Material for texture index references
 */
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

    // Non-copyable (contains unique_ptr members)
    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    // Movable
    TextureManager(TextureManager&&) noexcept = default;
    TextureManager& operator=(TextureManager&&) noexcept = default;

    // ========================================================================
    // Texture Upload
    // ========================================================================

    // Upload all textures from CPU to GPU
    // - Creates VkImage + VkImageView for each texture (RGBA8_UNORM)
    // - Creates VkSampler based on TextureSampler settings
    // - Uploads pixel data via staging buffer
    // - Transitions layout to SHADER_READ_ONLY_OPTIMAL
    // - RELEASES CPU MEMORY: texture.pixels is cleared after upload to free ~8GB RAM
    //
    // Note: If textures vector is empty, creates a single 1x1 white dummy texture
    // (This allows shader code to always sample without null checks)
    void UploadTextures(std::vector<Texture>& textures);

    // ========================================================================
    // Accessors
    // ========================================================================

    // Get number of textures
    [[nodiscard]] u32 GetTextureCount() const { return static_cast<u32>(m_images.size()); }

    // Get array of VkImageView handles (for descriptor set binding)
    // Index matches original texture index from scene
    [[nodiscard]] const std::vector<VkImageView>& GetImageViews() const { return m_imageViews; }

    // Get array of VkSampler handles (for descriptor set binding)
    // Index matches original texture index from scene
    [[nodiscard]] const std::vector<VkSampler>& GetSamplers() const { return m_samplers; }

    // Check if textures have been uploaded
    [[nodiscard]] bool IsEmpty() const { return m_images.empty(); }

private:
    // ========================================================================
    // Internal Helper Functions
    // ========================================================================

    // Upload a single texture to GPU
    // Returns GpuImage containing VkImage + VkImageView
    [[nodiscard]] std::unique_ptr<GpuImage> UploadTexture(const Texture& texture) const;

    // Create VkSampler based on TextureSampler settings
    [[nodiscard]] VkSampler CreateSampler(const TextureSampler& samplerInfo) const;

    // Convert TextureSampler::Filter to VkFilter
    static VkFilter ToVkFilter(TextureSampler::Filter filter);

    // Convert TextureSampler::WrapMode to VkSamplerAddressMode
    static VkSamplerAddressMode ToVkAddressMode(TextureSampler::WrapMode wrapMode);

    // Create a 1x1 white dummy texture (fallback for empty texture list)
    static Texture CreateDummyTexture();

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
