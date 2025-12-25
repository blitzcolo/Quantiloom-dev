/**
 * @file Texture.hpp
 * @brief CPU-side texture metadata and pixel data for GPU upload
 *
 * Provides texture data structures:
 * - TextureSampler: Sampler parameters (filter modes, wrap modes)
 * - Texture: CPU-side 2D texture image with metadata
 *
 * Texture workflow:
 * 1. Created during scene loading (GltfLoader extracts from glTF, or procedural)
 * 2. Stored in Scene::textures vector (CPU memory)
 * 3. Referenced by Material via texture index (-1 = no texture)
 * 4. Uploaded to GPU by TextureManager (converts to VkImage + VkSampler)
 *
 * Pixel format:
 * - CPU: u8 RGBA8 (4 bytes per pixel, row-major)
 * - GPU: Converted to RGBA8_UNORM or RGBA8_SRGB based on isSRGB flag
 *
 * Color space:
 * - Base color textures: sRGB (isSRGB = true, requires gamma correction)
 * - Metallic/roughness/normal: Linear (isSRGB = false, no gamma)
 * - Emissive textures: sRGB (glTF 2.0 spec)
 *
 * @note Texture data remains in CPU memory until GPU upload
 * @note TextureManager handles GPU resource creation
 * @note Material texture indices must reference valid Scene::textures entries
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include <string>

// ============================================================================
// Texture - GPU texture resource metadata
// ============================================================================

namespace quantiloom {

/**
 * @struct TextureSampler
 * @brief Texture sampler parameters (maps to glTF 2.0 sampler specification)
 *
 * Defines how textures are filtered and wrapped during shader sampling.
 * Parameters directly map to VkSamplerCreateInfo fields.
 *
 * Filter modes:
 * - Nearest: No filtering (blocky pixels, sharp edges)
 * - Linear: Bilinear filtering (smooth pixels, blurred edges)
 *
 * Wrap modes:
 * - Repeat: Tiling (UV wraps at 1.0 → 0.0)
 * - ClampToEdge: Edge pixels stretched beyond [0,1]
 * - MirroredRepeat: Mirrored tiling (UV mirrors at boundaries)
 *
 * @see Texture for texture image data
 * @see TextureManager for GPU upload and VkSampler creation
 */
struct TextureSampler {
    enum class Filter {
        Nearest = 0,
        Linear = 1
    };

    enum class WrapMode {
        Repeat = 0,
        ClampToEdge = 1,
        MirroredRepeat = 2
    };

    Filter minFilter = Filter::Linear;
    Filter magFilter = Filter::Linear;
    WrapMode wrapS = WrapMode::Repeat;
    WrapMode wrapT = WrapMode::Repeat;
};

// Texture image data (CPU-side)
struct Texture {
    // Image metadata
    u32 width = 0;
    u32 height = 0;
    u32 channels = 4;  // RGBA (PNG/JPEG decoded to RGBA8)

    // Pixel data (CPU memory, row-major, RGBA8 format)
    std::vector<u8> pixels;

    // Color space (glTF 2.0 spec requires baseColor/emissive in sRGB, others in linear)
    bool isSRGB = false;  // Default to linear

    // Sampler parameters
    TextureSampler sampler;

    // Metadata
    String name;  // Texture name (for debugging)
    String sourceUri;  // Original file path (if from external file)

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if texture is valid
    bool IsValid() const {
        // Must have non-zero dimensions
        if (width == 0 || height == 0) {
            return false;
        }

        // Channels must be 1, 2, 3, or 4
        if (channels < 1 || channels > 4) {
            return false;
        }

        // Pixel data size must match dimensions
        if (const size_t expectedSize = static_cast<size_t>(width) * height * channels; pixels.size() != expectedSize) {
            return false;
        }

        return true;
    }

    // Get size in bytes
    size_t GetSizeInBytes() const {
        return pixels.size();
    }

    // Get pixel data pointer (for GPU upload)
    const u8* GetData() const {
        return pixels.data();
    }
};

} // namespace quantiloom
