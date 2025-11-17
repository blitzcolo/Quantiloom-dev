#pragma once

#include "GpuBuffer.hpp"
#include "GpuImage.hpp"
#include "VulkanContext.hpp"
#include "core/Image.hpp"
#include "scene/Mesh.hpp"
#include <memory>

// ============================================================================
// Upload - Helper functions for CPU -> GPU data transfer
// ============================================================================
// Responsibilities:
// - Upload CPU data (Image, Mesh) to GPU buffers/images
// - Handle staging buffers for efficient transfer
// - Manage command buffer submission and synchronization
//
// M1 Implementation:
// - Synchronous upload (blocking)
// - Simple staging buffer approach
//
// Future (M2+):
// - Asynchronous upload with fences
// - Command buffer pooling
// - Transfer queue optimization
// ============================================================================

namespace quantiloom {

// ============================================================================
// Vertex Buffer Upload
// ============================================================================

// Upload mesh vertex positions to GPU buffer
// Returns GPU buffer containing vertex data (vec3 positions)
std::unique_ptr<GpuBuffer> UploadVertexBuffer(
    const VulkanContext& ctx,
    const Mesh& mesh
);

// Upload mesh indices to GPU buffer
// Returns GPU buffer containing index data (uint32 indices)
std::unique_ptr<GpuBuffer> UploadIndexBuffer(
    const VulkanContext& ctx,
    const Mesh& mesh
);

// ============================================================================
// Image Upload (TODO: M1+)
// ============================================================================

// Upload CPU image to GPU image
// Note: Requires command buffer recording and layout transitions
// Not implemented in M0, placeholder for M1
std::unique_ptr<GpuImage> UploadImage(
    const VulkanContext& ctx,
    const Image& cpuImage
);

} // namespace quantiloom
