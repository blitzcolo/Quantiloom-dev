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
 * @author blitzcolo
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
 * 1. Constructor receives a slice of the scene's merged vertex/index buffers
 * 2. Build() records VkAccelerationStructureBuildGeometryInfoKHR into command buffer
 * 3. GPU executes build asynchronously (requires synchronization before TLAS build)
 *
 * Memory layout:
 * - Vertex/index data: non-owning slices of SceneGeometry's merged buffers
 * - AS buffer: Device-local, contains acceleration structure data
 * - Scratch buffer: Device-local, temporary storage during build (destroyed after)
 *
 * Usage example:
 * @code
 * // Create BLAS for each primitive in scene
 * std::vector<BLAS> blasList;
 * for (const auto& mesh : scene.meshes) {
 *     // SceneGeometry constructs each BLAS with the primitive's merged slices.
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
 * @note The referenced geometry buffers must outlive the BLAS
 *
 * @see TLAS for scene-level acceleration structure
 * @see GeometryPrimitive for input geometry data
 * @see GpuBuffer for buffer management
 */
class BLAS {
public:
    struct GeometrySlice {
        const GpuBuffer* vertices = nullptr;
        const GpuBuffer* indices = nullptr;
        u32 vertexOffset = 0;  ///< In glm::vec3 elements
        u32 vertexCount = 0;
        u32 indexOffset = 0;   ///< In u32 elements
        u32 indexCount = 0;
    };

    /// @param opaque  Whether the traversal may skip the any-hit shader for this
    ///                geometry. False for a material with alphaMode MASK or BLEND,
    ///                whose coverage is decided per texel. Frozen at build time --
    ///                see IsOpaque() and SceneGeometry::RefreshMaterialOpacity.
    BLAS(VulkanContext& context, GeometrySlice geometry, bool opaque = true);
    ~BLAS();

    // Non-copyable, movable
    BLAS(const BLAS&) = delete;
    BLAS& operator=(const BLAS&) = delete;
    BLAS(BLAS&& other) noexcept;
    BLAS& operator=(BLAS&& other) noexcept;

    // Build BLAS from mesh data (records commands into cmd buffer)
    void Build(VkCommandBuffer cmd);

    // Release the temporary build workspace after the submission containing
    // Build() has completed. Static BLAS are never updated in place, so keeping
    // this device-local allocation for their entire lifetime only wastes VRAM.
    void ReleaseBuildScratch();

    // Accessors
    [[nodiscard]] VkAccelerationStructureKHR GetHandle() const { return m_as; }
    [[nodiscard]] VkDeviceAddress GetDeviceAddress() const { return m_deviceAddress; }
    [[nodiscard]] bool IsBuilt() const { return m_built; }
    [[nodiscard]] bool HasBuildScratch() const { return m_scratchBuffer != nullptr; }

    [[nodiscard]] bool UsesGeometryBuffers(const GpuBuffer& vertices,
                                           const GpuBuffer& indices) const {
        return m_geometry.vertices == &vertices && m_geometry.indices == &indices;
    }

private:
    VulkanContext& m_context;

    // Acceleration structure handle
    VkAccelerationStructureKHR m_as = VK_NULL_HANDLE;

    // Buffers (backing memory for AS)
    std::unique_ptr<GpuBuffer> m_asBuffer;       // AS storage
    std::unique_ptr<GpuBuffer> m_scratchBuffer;  // Scratch space for build

    // Device address
    VkDeviceAddress m_deviceAddress = 0;

    // Build state
    bool m_built = false;

    // What VK_GEOMETRY_OPAQUE_BIT_KHR was set to when this was built. Opacity
    // is baked into the acceleration structure, so changing a material's
    // alphaMode means rebuilding, and this is what says whether we have to.
    bool m_opaque = true;

    // Non-owning slice of SceneGeometry's merged buffers. SceneGeometry owns
    // the buffers and destroys its BLAS before releasing them.
    GeometrySlice m_geometry;

public:
    /// Whether this was built opaque. Compare against the material's current
    /// classification to find out whether the structure is stale.
    [[nodiscard]] bool IsOpaque() const { return m_opaque; }
};

// ============================================================================
// TLAS (Top-Level Acceleration Structure)
// ============================================================================

class TLAS {
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

    // Overwrite one instance's transform in the CPU-side list (before Update)
    void SetInstanceTransform(size_t index, const glm::mat4& transform);

    // Refit in place: record the instance-list upload and an UPDATE-mode build
    // (src == dst == this AS). Transform-only edits; the instance count, BLAS
    // references and flags must be unchanged since Build. The upload is part
    // of @p cmd, so the caller can place it between old and new traces without
    // a host wait or a host write racing an earlier update.
    // Orders of magnitude cheaper than a rebuild -- no allocation, no
    // teardown -- which is what makes interactive dragging possible.
    void Update(VkCommandBuffer cmd);

    // Accessors
    [[nodiscard]] VkAccelerationStructureKHR GetHandle() const { return m_as; }
    [[nodiscard]] bool IsBuilt() const { return m_built; }
    [[nodiscard]] size_t InstanceCount() const { return m_instances.size(); }
    /// Exposed for synchronization regression tests that snapshot the exact
    /// GPU input between two updates recorded into one command buffer.
    [[nodiscard]] const GpuBuffer& InstanceBufferForDiagnostics() const {
        return *m_instanceBuffer;
    }

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
