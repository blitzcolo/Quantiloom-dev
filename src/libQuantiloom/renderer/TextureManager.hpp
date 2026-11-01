/**
 * @file TextureManager.hpp
 * @brief GPU texture upload and bindless descriptor array management
 *
 * Provides TextureManager class for batch texture upload to GPU:
 * - Uploads multiple CPU Texture objects to gpu::Texture resources
 * - Creates a gpu::Sampler for each texture based on filter/wrap settings
 * - Manages lifetime of all texture images and samplers (RAII)
 * - Provides TextureRef and SamplerRef arrays for bindless descriptor sets
 *
 * Texture format conversion:
 * - All textures uploaded as RGBA8 (32 bpp), sRGB or UNORM, or as BC7
 * - Source formats (1-4 channels) expanded to RGBA automatically
 * - Ready for shader reads after upload (shader-read-only layout on Vulkan)
 *
 * Bindless descriptor indexing:
 * - Each texture gets a unique index (matches Material texture indices)
 * - Shader access via: texture2DArray[textureIndex]
 * - Up to DeviceInfo::maxBindlessTextures entries (descriptor indexing on Vulkan)
 *
 * Fallback behavior:
 * - If no textures provided, creates 1x1 white dummy texture
 * - Prevents null descriptor access in shaders
 *
 * @note All textures uploaded during UploadTextures() call (batch operation)
 * @note Samplers are owned gpu::Sampler objects, destroyed before the images (RAII)
 * @note Texture indices are immutable after upload
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "scene/Texture.hpp"
#include "TextureCompressor.hpp"
#include "gpu/Device.hpp"
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
 * Handles conversion from CPU Texture data to gpu::Texture resources.
 *
 * Upload workflow:
 * 1. Constructor creates TextureManager (no textures uploaded yet)
 * 2. UploadTextures() converts all CPU textures to GPU images
 * 3. For each texture:
 *    - Create a gpu::Texture (RGBA8 format)
 *    - Upload pixel data via staging buffer
 *    - Make it readable by the shaders
 *    - Create a gpu::Sampler with specified filter/wrap modes
 * 4. TextureRefs()/SamplerRefs() return arrays for descriptor binding
 *
 * Usage example:
 * @code
 * // Create manager
 * TextureManager texMgr(device);
 *
 * // Upload all scene textures
 * texMgr.UploadTextures(scene.textures);
 *
 * // Bind to pipeline descriptor set
 * const auto& textures = texMgr.TextureRefs();
 * const auto& samplers = texMgr.SamplerRefs();
 * pipeline.BindTextures(textures, samplers);  // Binding 6-7
 *
 * // Shader access (HLSL):
 * // [[vk::binding(6, 0)]] Texture2D textures[];
 * // [[vk::binding(7, 0)]] SamplerState samplers[];
 * // float4 color = textures[matData.baseColorTextureIndex].Sample(
 * //     samplers[matData.baseColorTextureIndex], uv);
 * @endcode
 *
 * @note Empty texture list triggers creation of 1x1 white dummy texture
 * @note All textures converted to RGBA8 regardless of source format
 * @note Texture indices must match Material texture index references
 *
 * @see Texture for CPU-side texture data structure
 * @see gpu::Texture for GPU image resource management
 * @see Material for texture index references
 */
class TextureManager {
public:
    static constexpr u64 kDefaultUploadBatchBytes = 64ull * 1024ull * 1024ull;
    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    // Create texture manager
    // Note: Does not upload any textures yet (call UploadTextures)
    explicit TextureManager(
        gpu::Device& device,
        u64 uploadBatchBytes = kDefaultUploadBatchBytes);

    // Destructor: the gpu::Sampler owners destroy every sampler, then every
    // gpu::Texture destroys its image and view (RAII)
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
    // - Creates a gpu::Texture for each texture (RGBA8, or BC7)
    // - Creates a gpu::Sampler based on TextureSampler settings
    // - Uploads pixel data via staging buffer
    // - Leaves it in the shader-read layout
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

    // The textures, for descriptor set binding (sampled, shader-read layout)
    // Index matches original texture index from scene
    [[nodiscard]] const std::vector<gpu::TextureRef>& TextureRefs() const { return m_textureRefs; }

    // Their samplers, for descriptor set binding
    // Index matches original texture index from scene
    [[nodiscard]] const std::vector<gpu::SamplerRef>& SamplerRefs() const { return m_samplerRefs; }

    /// Number of queue submissions used by the most recent UploadTextures().
    /// Exposed for load-path regression tests and diagnostics.
    [[nodiscard]] u32 GetLastUploadBatchCount() const { return m_lastUploadBatchCount; }

    /// Internal diagnostic access for byte/mip/format GPU upload tests.
    [[nodiscard]] const gpu::Texture& GetImageForDiagnostics(u32 index) const {
        return *m_images.at(index);
    }

    // Upload one more texture and return its index, leaving the ones already
    // uploaded alone.
    //
    // UploadTextures above is a full rebuild -- it destroys every image and
    // sampler first -- which is right when a scene is replaced and wrong when
    // a single texture is added to a scene already on the device. The caller
    // must rebind the descriptor array afterwards, and must have waited for
    // the device to be idle: this changes what the shader's texture array
    // means while a frame could be reading it.
    //
    // CPU pixels are NOT released here. An appended texture is typically one
    // the caller just generated and may regenerate.
    [[nodiscard]] i32 AppendTexture(const Texture& texture);

    // Check if textures have been uploaded
    [[nodiscard]] bool IsEmpty() const { return m_images.empty(); }

private:
    // ========================================================================
    // Internal Helper Functions
    // ========================================================================

    struct PendingUpload;

    [[nodiscard]] std::unique_ptr<PendingUpload> PrepareTexture(
        const Texture& texture) const;
    [[nodiscard]] std::unique_ptr<PendingUpload> PrepareBC7Texture(
        const Texture& texture, const BC7CompressedData& compressed) const;
    void RecordTextureUpload(gpu::CommandList& list, const PendingUpload& upload) const;

    // Upload a single texture to GPU
    // Returns the gpu::Texture holding its image and view
    [[nodiscard]] std::unique_ptr<gpu::Texture> UploadTexture(const Texture& texture) const;

    // Generate mipmaps for an uncompressed texture by blitting each level from the last
    void GenerateMipmaps(gpu::CommandList& list, const gpu::Texture& image,
                          gpu::Format format, u32 texWidth, u32 texHeight,
                          u32 mipLevels) const;

    // Create a sampler based on TextureSampler settings
    [[nodiscard]] std::unique_ptr<gpu::Sampler> CreateSampler(const TextureSampler& samplerInfo) const;

    // Convert TextureSampler::Filter to gpu::Filter
    static gpu::Filter ToFilter(TextureSampler::Filter filter);

    // Convert TextureSampler::WrapMode to gpu::AddressMode
    static gpu::AddressMode ToAddressMode(TextureSampler::WrapMode wrapMode);

    // Create a 1x1 white dummy texture (fallback for empty texture list)
    static Texture CreateDummyTexture();

    // ========================================================================
    // Member Variables
    // ========================================================================

    // The caller's device; it outlives this manager
    gpu::Device& m_device;
    u64 m_uploadBatchBytes = kDefaultUploadBatchBytes;
    u32 m_lastUploadBatchCount = 0;

    // GPU image resources (image + view owned by each gpu::Texture)
    std::vector<std::unique_ptr<gpu::Texture>> m_images;

    // One sampler per image. Declared after m_images, so the samplers are destroyed first,
    // as the hand-written destructor did
    std::vector<std::unique_ptr<gpu::Sampler>> m_samplerObjects;

    // Their refs, for descriptor binding
    std::vector<gpu::SamplerRef> m_samplerRefs;

    // Cached arrays for descriptor binding (updated after upload)
    std::vector<gpu::TextureRef> m_textureRefs;  // Extracted from m_images
};

} // namespace quantiloom
