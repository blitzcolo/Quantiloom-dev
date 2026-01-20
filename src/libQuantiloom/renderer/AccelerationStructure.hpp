/**
 * @file AccelerationStructure.hpp
 * @brief Vulkan ray tracing acceleration structure management (BLAS and TLAS)
 *
 * Provides BLAS and TLAS classes for building and managing ray tracing acceleration structures:
 * - BLAS (Bottom-Level AS): Per-geometry acceleration structures for triangle meshes
 * - TLAS (Top-Level AS): Scene-level acceleration structure for instancing BLAS with transforms
 *
 * Acceleration structures enable fast ray-triangle intersection queries in hardware.
 * Quantiloom uses the Vulkan Ray Tracing Pipeline extension (VK_KHR_ray_tracing_pipeline).
 *
 * Build workflow:
 * 1. Create BLAS for each GeometryPrimitive in the scene
 * 2. Build all BLAS on GPU (command buffer submission)
 * 3. Create TLAS and add BLAS instances with world transforms
 * 4. Build TLAS on GPU (references BLAS device addresses)
 * 5. Bind TLAS to ray tracing pipeline
 *
 * Memory management:
 * - BLAS/TLAS buffers managed via VMA (automatic cleanup)
 * - Scratch buffers created temporarily for build operations
 * - Device addresses stored for shader access
 *
 * @note Requires VK_KHR_ray_tracing_pipeline and VK_KHR_acceleration_structure extensions
 * @note Build operations must be recorded into command buffers and submitted to GPU
 * @note BLAS must outlive TLAS (TLAS references BLAS device addresses)
 *
 * @author wtflmao
 */

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

namespace quantiloom {

// ============================================================================
// BLAS (Bottom-Level Acceleration Structure)
// ============================================================================
/**
 * @class BLAS
 * @brief Bottom-Level Acceleration Structure for a single geometry primitive
 *
 * BLAS represents a single triangle mesh accelerated for ray tracing.
 * Each GeometryPrimitive (subset of a Mesh with one material) gets its own BLAS.
 *
 * Build process:
 * 1. Constructor uploads vertex/index/UV/tangent/normal data to GPU buffers
 * 2. Build() records VkAccelerationStructureBuildGeometryInfoKHR into command buffer
 * 3. GPU executes build asynchronously (requires synchronization before TLAS build)
 *
 * Memory layout:
 * - Vertex buffer: Device-local, contains positions (vec3)
 * - Index buffer: Device-local, contains triangle indices (u32)
 * - UV buffer: Device-local, optional texture coordinates (vec2)
 * - Tangent buffer: Device-local, optional tangent vectors (vec4)
 * - Normal buffer: Device-local, required smooth normals (vec3)
 * - AS buffer: Device-local, contains acceleration structure data
 * - Scratch buffer: Device-local, temporary storage during build (destroyed after)
 *
 * Usage example:
 * @code
 * // Create BLAS for each primitive in scene
 * std::vector<BLAS> blasList;
 * for (const auto& mesh : scene.meshes) {
 *     for (const auto& primitive : mesh.primitives) {
 *         blasList.emplace_back(context, primitive);
 *     }
 * }
 *
 * // Build all BLAS on GPU
 * CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
 *     for (auto& blas : blasList) {
 *         blas.Build(cmd);
 *     }
 * });
 * @endcode
 *
 * @note Non-copyable, movable (transfer ownership)
 * @note Build() must be called before BLAS can be used in TLAS
 * @note Geometry buffers are uploaded automatically in constructor
 *
 * @see TLAS for scene-level acceleration structure
 * @see GeometryPrimitive for input geometry data
 * @see GpuBuffer for buffer management
 */
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
    [[nodiscard]] const GpuBuffer& GetNormalBuffer() const { return *m_normalBuffer; }  // Normal vectors
    [[nodiscard]] bool HasNormals() const { return m_normalBuffer != nullptr; }  // Check if normals are available

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
    std::unique_ptr<GpuBuffer> m_normalBuffer;   // Normal vectors (device-local, required for smooth shading)
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
    explicit TLAS(VulkanContext& context);
    ~TLAS();

    // Non-copyable, movable
    TLAS(const TLAS&) = delete;
    TLAS& operator=(const TLAS&) = delete;
    TLAS(TLAS&& other) noexcept;
    TLAS& operator=(TLAS&& other) noexcept;

    // Add BLAS instance to TLAS (must call before Build)
    // materialId: Index into Scene::materials, passed to shader via instanceCustomIndex
    // doubleSided: If true, disable backface culling; if false, enable hardware culling
    void AddInstance(const BLAS& blas, u32 materialId, const glm::mat4& transform = glm::mat4(1.0f), bool doubleSided = true);

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
