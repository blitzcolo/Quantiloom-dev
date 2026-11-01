/**
 * @file AccelerationStructure.hpp
 * @brief Ray tracing acceleration structure management (BLAS and TLAS)
 *
 * Provides BLAS and TLAS classes for building and managing ray tracing acceleration structures:
 * - BLAS (Bottom-Level AS): Per-geometry acceleration structures for triangle meshes
 * - TLAS (Top-Level AS): Scene-level acceleration structure for instancing BLAS with transforms
 *
 * Acceleration structures enable fast ray-triangle intersection queries in hardware.
 * Both levels are gpu::RtAccelerationStructure objects; which barriers surround a build
 * and when a refit replaces a rebuild is decided here (gpu/AGENTS.md).
 *
 * Build workflow:
 * 1. Create BLAS for each GeometryPrimitive in the scene
 * 2. Build all BLAS on GPU (command buffer submission)
 * 3. Create TLAS and add BLAS instances with world transforms
 * 4. Build TLAS on GPU (references BLAS device addresses)
 * 5. Bind TLAS to ray tracing pipeline
 *
 * Memory management:
 * - Every structure, scratch and instance buffer is a gpu:: owner (automatic cleanup)
 * - Scratch buffers created temporarily for build operations
 * - Device addresses stored for shader access
 *
 * @note Requires a device with hardware ray tracing (DeviceInfo::hardwareRayTracing)
 * @note Build operations are recorded into a gpu::CommandList that a gpu::Device handed out
 * @note BLAS must outlive TLAS (TLAS references BLAS device addresses)
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "gpu/GpuTypes.hpp"
#include "gpu/Resources.hpp"
#include "gpu/RtScene.hpp"
#include "scene/Mesh.hpp"
#include <cstddef>
#include <vector>
#include <memory>

namespace quantiloom::gpu {
class CommandList;
class Device;
class RtAccelerationStructure;
}  // namespace quantiloom::gpu

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
 * 2. Build() records one acceleration structure build into a command list
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
 * device.ExecuteImmediate([&](gpu::CommandList& list) {
 *     for (auto& blas : blasList) {
 *         blas.Build(list);
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
 * @see gpu::Buffer for buffer management
 */
class BLAS {
public:
    struct GeometrySlice {
        gpu::BufferRef vertices;  ///< SceneGeometry's merged vertex buffer
        gpu::BufferRef indices;   ///< SceneGeometry's merged index buffer
        u32 vertexOffset = 0;  ///< In glm::vec3 elements
        u32 vertexCount = 0;
        u32 indexOffset = 0;   ///< In u32 elements
        u32 indexCount = 0;
    };

    /// @param opaque  Whether the traversal may skip the any-hit shader for this
    ///                geometry. False for a material with alphaMode MASK or BLEND,
    ///                whose coverage is decided per texel. Frozen at build time --
    ///                see IsOpaque() and SceneGeometry::RefreshMaterialOpacity.
    BLAS(gpu::Device& device, GeometrySlice geometry, bool opaque = true);
    ~BLAS();

    // Non-copyable, movable
    BLAS(const BLAS&) = delete;
    BLAS& operator=(const BLAS&) = delete;
    BLAS(BLAS&& other) noexcept;
    BLAS& operator=(BLAS&& other) noexcept;

    // Build BLAS from mesh data. The list must be one a gpu::Device handed out
    // (ExecuteImmediate, WrapRecording, a CommandPool's buffer): it carries the
    // ray tracing entry points the build needs.
    void Build(gpu::CommandList& list);

    // Release the temporary build workspace after the submission containing
    // Build() has completed. Static BLAS are never updated in place, so keeping
    // this device-local allocation for their entire lifetime only wastes VRAM.
    void ReleaseBuildScratch();

    // Accessors
    /// The structure and the device address a TLAS instance names it by; empty
    /// before Build and after a move.
    [[nodiscard]] gpu::RtSceneRef Ref() const;
    [[nodiscard]] bool IsBuilt() const { return m_built; }
    [[nodiscard]] bool HasBuildScratch() const { return m_scratchBuffer != nullptr; }

    [[nodiscard]] bool UsesGeometryBuffers(const gpu::Buffer& vertices,
                                           const gpu::Buffer& indices) const {
        return m_geometry.vertices == vertices.Ref() && m_geometry.indices == indices.Ref();
    }

private:
    // The caller's device; it outlives the BLAS (SceneGeometry and its owner).
    gpu::Device& m_device;

    std::unique_ptr<gpu::Buffer> m_scratchBuffer;  // Scratch space for build
    // The acceleration structure with its storage. Declared last, so it is
    // destroyed first: the structure, then its storage, then the scratch.
    std::unique_ptr<gpu::RtAccelerationStructure> m_structure;

    // Build state
    bool m_built = false;

    // What the geometry's opaque flag was when this was built. Opacity
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
    explicit TLAS(gpu::Device& device);
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

    // Build TLAS from instances (records into a list from a gpu::Device, as for BLAS)
    void Build(gpu::CommandList& list);

    // Overwrite one instance's transform in the CPU-side list (before Update)
    void SetInstanceTransform(size_t index, const glm::mat4& transform);

    // Refit in place: record the instance-list upload and an UPDATE-mode build
    // (src == dst == this AS). Transform-only edits; the instance count, BLAS
    // references and flags must be unchanged since Build. The upload is part
    // of @p list, so the caller can place it between old and new traces without
    // a host wait or a host write racing an earlier update.
    // Orders of magnitude cheaper than a rebuild -- no allocation, no
    // teardown -- which is what makes interactive dragging possible.
    void Update(gpu::CommandList& list);

    // Accessors
    /// The structure a descriptor binds; empty before Build and after a move.
    [[nodiscard]] gpu::RtSceneRef Ref() const;
    [[nodiscard]] bool IsBuilt() const { return m_built; }
    [[nodiscard]] size_t InstanceCount() const { return m_instances.size(); }
    /// Exposed for synchronization regression tests that snapshot the exact
    /// GPU input between two updates recorded into one command buffer.
    [[nodiscard]] const gpu::Buffer& InstanceBufferForDiagnostics() const {
        return *m_instanceBuffer;
    }

private:
    /// The instance list as the native records the build reads (Device::PackRtInstances).
    [[nodiscard]] std::vector<std::byte> PackedInstances() const;

    // The caller's device; it outlives the TLAS, as for BLAS.
    gpu::Device& m_device;

    // Buffers
    std::unique_ptr<gpu::Buffer> m_instanceBuffer;   // Instance data
    std::unique_ptr<gpu::Buffer> m_scratchBuffer;    // Scratch space for build and refits
    // The acceleration structure with its storage. Declared last, so it is
    // destroyed first: the structure, then its storage, then the buffers above.
    std::unique_ptr<gpu::RtAccelerationStructure> m_structure;

    // Build state
    bool m_built = false;

    // Instance data (accumulated before Build), packed before every upload
    std::vector<gpu::RtInstance> m_instances;
};

} // namespace quantiloom
