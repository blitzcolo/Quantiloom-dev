#pragma once

#include "core/Types.hpp"
#include "VulkanContext.hpp"
#include "GpuBuffer.hpp"
#include "scene/Mesh.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

// ============================================================================
// AccelerationStructure - Manages BLAS and TLAS for ray tracing
// ============================================================================
// Responsibilities:
// - Build Bottom-Level Acceleration Structures (BLAS) from geometry
// - Build Top-Level Acceleration Structures (TLAS) from BLAS instances
// - Manage buffers and memory for acceleration structures
// - Handle synchronization via command buffers
//
// Lifetime:
// - Must be created AFTER VulkanContext
// - Must be destroyed BEFORE VulkanContext
// - Non-copyable, movable
//
// Usage (M2 updated):
//   1. Create BLAS for each GeometryPrimitive (in each Mesh)
//   2. Create TLAS referencing all BLAS instances with transforms
//   3. Bind TLAS to ray tracing pipeline
// ============================================================================

namespace quantiloom {

// ============================================================================
// BLAS (Bottom-Level Acceleration Structure)
// ============================================================================

class QL_API BLAS {
public:
    BLAS(VulkanContext& context, const GeometryPrimitive& primitive);
    ~BLAS();

    // Non-copyable, movable
    BLAS(const BLAS&) = delete;
    BLAS& operator=(const BLAS&) = delete;
    BLAS(BLAS&& other) noexcept;
    BLAS& operator=(BLAS&& other) noexcept;

    // Build BLAS from mesh data (records commands into cmd buffer)
    void Build(VkCommandBuffer cmd);

    // Accessors
    [[nodiscard]] VkAccelerationStructureKHR GetHandle() const { return m_as; }
    [[nodiscard]] VkDeviceAddress GetDeviceAddress() const { return m_deviceAddress; }
    [[nodiscard]] bool IsBuilt() const { return m_built; }

    // Geometry buffer accessors (for shader binding)
    [[nodiscard]] const GpuBuffer& GetVertexBuffer() const { return *m_vertexBuffer; }
    [[nodiscard]] const GpuBuffer& GetIndexBuffer() const { return *m_indexBuffer; }
    [[nodiscard]] const GpuBuffer& GetUVBuffer() const { return *m_uvBuffer; }  // UV coordinates
    [[nodiscard]] bool HasUVs() const { return m_uvBuffer != nullptr; }  // Check if UVs are available
    [[nodiscard]] const GpuBuffer& GetTangentBuffer() const { return *m_tangentBuffer; }  // Tangent vectors
    [[nodiscard]] bool HasTangents() const { return m_tangentBuffer != nullptr; }  // Check if tangents are available

private:
    // Helper: Upload vertex and index data to GPU buffers
    void UploadGeometryBuffers();

    VulkanContext& m_context;

    // Acceleration structure handle
    VkAccelerationStructureKHR m_as = VK_NULL_HANDLE;

    // Buffers (backing memory for AS)
    std::unique_ptr<GpuBuffer> m_asBuffer;       // AS storage
    std::unique_ptr<GpuBuffer> m_vertexBuffer;   // Vertex data (device-local)
    std::unique_ptr<GpuBuffer> m_indexBuffer;    // Index data (device-local)
    std::unique_ptr<GpuBuffer> m_uvBuffer;       // UV coordinates (device-local, optional)
    std::unique_ptr<GpuBuffer> m_tangentBuffer;  // Tangent vectors (device-local, optional)
    std::unique_ptr<GpuBuffer> m_scratchBuffer;  // Scratch space for build

    // Device address
    VkDeviceAddress m_deviceAddress = 0;

    // Build state
    bool m_built = false;

    // Cached geometry info
    const GeometryPrimitive& m_primitive;
};

// ============================================================================
// TLAS (Top-Level Acceleration Structure)
// ============================================================================

class QL_API TLAS {
public:
    TLAS(VulkanContext& context);
    ~TLAS();

    // Non-copyable, movable
    TLAS(const TLAS&) = delete;
    TLAS& operator=(const TLAS&) = delete;
    TLAS(TLAS&& other) noexcept;
    TLAS& operator=(TLAS&& other) noexcept;

    // Add BLAS instance to TLAS (must call before Build)
    // materialId: Index into Scene::materials, passed to shader via instanceCustomIndex
    void AddInstance(const BLAS& blas, u32 materialId, const glm::mat4& transform = glm::mat4(1.0f));

    // Build TLAS from instances (records commands into cmd buffer)
    void Build(VkCommandBuffer cmd);

    // Accessors
    [[nodiscard]] VkAccelerationStructureKHR GetHandle() const { return m_as; }
    [[nodiscard]] bool IsBuilt() const { return m_built; }

private:
    VulkanContext& m_context;

    // Acceleration structure handle
    VkAccelerationStructureKHR m_as = VK_NULL_HANDLE;

    // Buffers
    std::unique_ptr<GpuBuffer> m_asBuffer;         // AS storage
    std::unique_ptr<GpuBuffer> m_instanceBuffer;   // Instance data
    std::unique_ptr<GpuBuffer> m_scratchBuffer;    // Scratch space for build

    // Build state
    bool m_built = false;

    // Instance data (accumulated before Build)
    std::vector<VkAccelerationStructureInstanceKHR> m_instances;
};

} // namespace quantiloom
