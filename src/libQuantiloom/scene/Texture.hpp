#pragma once

#include "core/Types.hpp"
#include <string>

// ============================================================================
// Texture - GPU texture resource metadata
// ============================================================================
// Represents a 2D texture image with metadata needed for:
// - Vulkan resource creation (VkImage, VkImageView, VkSampler)
// - glTF material binding
// - Spectral mode fallback
//
// Lifecycle:
// - Created during scene loading (GltfLoader or procedural)
// - Uploaded to GPU by TextureManager
// - Referenced by Material via textureIndex
//
// Ownership:
// - CPU-side pixel data owned by Scene::textures
// - GPU-side resources owned by TextureManager
// ============================================================================

namespace quantiloom {

// Texture sampler parameters (maps to glTF sampler spec)
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
        size_t expectedSize = static_cast<size_t>(width) * height * channels;
        if (pixels.size() != expectedSize) {
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
