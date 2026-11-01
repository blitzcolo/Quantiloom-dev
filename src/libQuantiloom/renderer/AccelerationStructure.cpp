#include "AccelerationStructure.hpp"
#include "core/Log.hpp"
#include "gpu/Device.hpp"
#include "renderer/GpuAlloc.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <memory>
#include <utility>

namespace quantiloom {

namespace {

// The Vulkan spec's scratch-address rule (build geometry info, scratchData-03710):
// the scratch device address must be a multiple of
// minAccelerationStructureScratchOffsetAlignment (128 on NVIDIA). Nothing in
// the scratch buffer's usage flags implies that, so the allocator is free to
// suballocate at a 16-byte boundary and the build reads and writes off its own
// scratch -- which surfaces as a GPU hang and a lost device, not as an error.
u64 ScratchAlignment(const gpu::DeviceInfo& info) {
    const u64 reported = info.rtScratchAlignment;
    // A zero here would mean the property was never queried; 128 covers every
    // desktop driver we target and costs nothing when the real value is lower.
    return reported > 0 ? reported : 128;
}

// The instance data address rule (instances data, arrayOfPointers-03779).
constexpr u64 kInstanceDataAlignment = 16;
constexpr u64 kMaxCmdUpdateBytes = 65536;

// The failure mode this guards against has no error code -- an unaligned build
// input hangs the GPU and the device-lost surfaces later, at whatever the next
// queue wait happens to be. Say it out loud here instead.
void WarnIfMisaligned(const char* what, u64 address, u64 alignment) {
    if (alignment > 0 && (address % alignment) != 0) {
        QL_LOG_ERROR("  {} device address 0x{:x} is not {}-byte aligned -- "
                     "the acceleration structure build will corrupt memory",
                     what, address, alignment);
    }
}

// Device-local storage with a device address, at the scratch offset alignment.
// Allocated through CreateBufferOrThrow, whose exception carries the backend's
// allocation text unprefixed, as the scratch allocation always threw it.
gpu::BufferDesc ScratchDesc(u64 bytes, u64 alignment) {
    gpu::BufferDesc desc;
    desc.size = bytes;
    desc.usage = gpu::BufferUsage::Storage | gpu::BufferUsage::DeviceAddress;
    desc.memory = gpu::MemoryClass::DeviceLocal;
    desc.minAlignment = alignment;
    return desc;
}

// The one geometry of a Top level: `count` packed instance records from the
// start of `instances`. Its own geometry flag stays 0 (see the backend).
gpu::RtGeometry TopLevelGeometry(const gpu::Buffer& instances, u32 count) {
    gpu::RtGeometry geometry;
    geometry.level = gpu::RtLevel::Top;
    // ALLOW_UPDATE so Update() can refit transforms in place during drags
    geometry.flags = gpu::RtBuildFlags::PreferFastTrace | gpu::RtBuildFlags::AllowUpdate;
    geometry.instances.buffer = instances.Ref();
    geometry.instances.offset = 0;
    geometry.instances.count = count;
    return geometry;
}

// An instance transform is the top three rows of the object-to-world matrix,
// row-major; glm stores the matrix column-major, hence the transpose.
void ToInstanceTransform(const glm::mat4& transform, f32 (&out)[3][4]) {
    const f32* mat = glm::value_ptr(transform);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            out[row][col] = mat[col * 4 + row];  // transpose
        }
    }
}

}  // namespace

// ============================================================================
// BLAS Implementation
// ============================================================================

BLAS::BLAS(gpu::Device& device, GeometrySlice geometry, bool opaque)
    : m_device(device)
    , m_opaque(opaque)
    , m_geometry(geometry)
{
    if (!geometry.vertices.IsValid() || !geometry.indices.IsValid() || geometry.vertexCount == 0 ||
        geometry.indexCount == 0) {
        throw std::runtime_error("Cannot create BLAS from an empty geometry slice");
    }

    QL_LOG_INFO("Creating BLAS for primitive with {} vertices, {} triangles",
                geometry.vertexCount, geometry.indexCount / 3);
}

// The structure is destroyed first, then its storage, then the scratch.
BLAS::~BLAS() = default;

BLAS::BLAS(BLAS&& other) noexcept
    : m_device(other.m_device)
    , m_scratchBuffer(std::move(other.m_scratchBuffer))
    , m_structure(std::move(other.m_structure))
    , m_built(other.m_built)
    , m_opaque(other.m_opaque)
    , m_geometry(other.m_geometry)
{
    other.m_built = false;
}

BLAS& BLAS::operator=(BLAS&& other) noexcept {
    if (this != &other) {
        // Destroys the current structure, then its storage, then the old
        // scratch. m_device stays: a move assignment does not change the
        // device either.
        m_structure = std::move(other.m_structure);
        m_scratchBuffer = std::move(other.m_scratchBuffer);
        m_built = other.m_built;
        m_opaque = other.m_opaque;
        m_geometry = other.m_geometry;

        // Nullify source
        other.m_built = false;
    }
    return *this;
}

gpu::RtSceneRef BLAS::Ref() const {
    return m_structure ? m_structure->Ref() : gpu::RtSceneRef{};
}

void BLAS::Build(gpu::CommandList& list) {
    if (!m_geometry.vertices.IsValid() || !m_geometry.indices.IsValid()) {
        throw std::runtime_error("BLAS geometry slice is invalid");
    }

    // Define geometry (triangles): RGB32Float positions with a 12-byte stride,
    // u32 indices, offsets in bytes into the merged buffers.
    gpu::RtGeometry geometry;
    geometry.level = gpu::RtLevel::Bottom;
    geometry.flags = gpu::RtBuildFlags::PreferFastTrace;  // M1: static geometry
    geometry.triangles.vertices = m_geometry.vertices;
    geometry.triangles.vertexOffset =
        static_cast<u64>(m_geometry.vertexOffset) * sizeof(glm::vec3);
    geometry.triangles.vertexCount = m_geometry.vertexCount;
    geometry.triangles.indices = m_geometry.indices;
    geometry.triangles.indexOffset =
        static_cast<u64>(m_geometry.indexOffset) * sizeof(u32);
    geometry.triangles.triangleCount = m_geometry.indexCount / 3;
    // The one flag that decides whether an any-hit shader runs at all: set, the
    // traversal is free to skip the stage entirely -- which is what we want for
    // the overwhelming majority of geometry and what would silently disable
    // alpha coverage on the rest.
    geometry.triangles.opaque = m_opaque;

    // Query build sizes
    const gpu::RtBuildSizes sizes = m_device.GetRtBuildSizes(geometry);

    QL_LOG_INFO("  BLAS build sizes: AS={} bytes, scratch={} bytes",
                sizes.structure, sizes.buildScratch);

    // Create the AS buffer, the acceleration structure on it, and its device address
    m_structure = ValueOrThrow(
        m_device.CreateRtAccelerationStructure(gpu::RtLevel::Bottom, sizes.structure));

    // Create scratch buffer
    const u64 scratchAlignment = ScratchAlignment(m_device.Info());
    m_scratchBuffer = CreateBufferOrThrow(m_device, ScratchDesc(sizes.buildScratch, scratchAlignment));
    WarnIfMisaligned("BLAS scratch", m_device.BufferDeviceAddress(m_scratchBuffer->Ref()),
                     scratchAlignment);

    // Build acceleration structure
    gpu::RtBuild build;
    build.geometry = geometry;
    build.mode = gpu::RtBuildMode::Build;
    build.dst = m_structure->Ref();
    build.scratch = m_scratchBuffer->Ref();

    list.BuildRtAccelerationStructure(build);

    // CRITICAL: Insert memory barrier to ensure BLAS build completes before TLAS reads it
    // Without this barrier, TLAS may reference incomplete BLAS data
    list.Barrier(gpu::GlobalBarrier{gpu::Access::AccelerationStructureWrite,
                                    gpu::Access::AccelerationStructureRead});

    m_built = true;
    QL_LOG_INFO("  BLAS built successfully (device address: 0x{:x})", m_structure->Ref().address);
}

void BLAS::ReleaseBuildScratch() {
    m_scratchBuffer.reset();
}

// ============================================================================
// TLAS Implementation
// ============================================================================

TLAS::TLAS(gpu::Device& device)
    : m_device(device)
{
    QL_LOG_INFO("Creating TLAS...");
}

// The structure is destroyed first, then its storage, then scratch and instances.
TLAS::~TLAS() = default;

TLAS::TLAS(TLAS&& other) noexcept
    : m_device(other.m_device)
    , m_instanceBuffer(std::move(other.m_instanceBuffer))
    , m_scratchBuffer(std::move(other.m_scratchBuffer))
    , m_structure(std::move(other.m_structure))
    , m_built(other.m_built)
    , m_instances(std::move(other.m_instances))
{
    other.m_built = false;
}

TLAS& TLAS::operator=(TLAS&& other) noexcept {
    if (this != &other) {
        // Destroys the current structure, then its storage, then the old
        // instance and scratch buffers. m_device stays, as for the BLAS.
        m_structure = std::move(other.m_structure);
        m_instanceBuffer = std::move(other.m_instanceBuffer);
        m_scratchBuffer = std::move(other.m_scratchBuffer);
        m_built = other.m_built;
        m_instances = std::move(other.m_instances);

        // Nullify source
        other.m_built = false;
    }
    return *this;
}

gpu::RtSceneRef TLAS::Ref() const {
    return m_structure ? m_structure->Ref() : gpu::RtSceneRef{};
}

std::vector<std::byte> TLAS::PackedInstances() const {
    std::vector<std::byte> packed(m_instances.size() * m_device.RtInstanceStride());
    m_device.PackRtInstances(m_instances, packed);
    return packed;
}

void TLAS::AddInstance(const BLAS& blas, u32 materialId, const glm::mat4& transform, bool doubleSided) {
    if (m_built) {
        throw std::runtime_error("Cannot add instance to already-built TLAS");
    }

    gpu::RtInstance instance;
    ToInstanceTransform(transform, instance.transform);  // row-major 3x4
    instance.userId = materialId;  // Material ID accessible in shader via InstanceID()
    instance.mask = 0xFF;  // Visible to all rays; the one hit group is implied

    // Set backface culling based on material's doubleSided property
    // - doubleSided=true: Disable culling, backfaces will be shaded (normal flipped in shader)
    // - doubleSided=false: Enable culling, rays pass through backfaces (no hit)
    instance.cullDisable = doubleSided;

    instance.bottomLevel = blas.Ref();

    m_instances.push_back(instance);

    QL_LOG_INFO("  Added instance {} to TLAS (material {}, doubleSided={}, BLAS addr: 0x{:x})",
                m_instances.size() - 1, materialId, doubleSided, blas.Ref().address);
}

void TLAS::Build(gpu::CommandList& list) {
    if (m_instances.empty()) {
        throw std::runtime_error("Cannot build TLAS with no instances");
    }

    // Upload instances to GPU, packed into the native records
    const std::vector<std::byte> packed = PackedInstances();
    const u64 instanceBufferSize = packed.size();

    gpu::BufferDesc instanceDesc;
    instanceDesc.size = instanceBufferSize;
    instanceDesc.usage = gpu::BufferUsage::AccelerationStructureInput |
                         gpu::BufferUsage::DeviceAddress |
                         gpu::BufferUsage::TransferDst |
                         gpu::BufferUsage::TransferSrc;
    instanceDesc.memory = gpu::MemoryClass::Upload;
    instanceDesc.minAlignment = kInstanceDataAlignment;
    m_instanceBuffer = CreateBufferOrThrow(m_device, instanceDesc);

    m_instanceBuffer->Upload(packed.data(), instanceBufferSize);

    WarnIfMisaligned("TLAS instance data", m_device.BufferDeviceAddress(m_instanceBuffer->Ref()),
                     kInstanceDataAlignment);

    // Define geometry (instances)
    const u32 instanceCount = static_cast<u32>(m_instances.size());
    const gpu::RtGeometry geometry = TopLevelGeometry(*m_instanceBuffer, instanceCount);

    // Query build sizes
    const gpu::RtBuildSizes sizes = m_device.GetRtBuildSizes(geometry);

    QL_LOG_INFO("  TLAS build sizes: AS={} bytes, scratch={} bytes",
                sizes.structure, sizes.buildScratch);

    // Create the AS buffer and the acceleration structure on it
    m_structure = ValueOrThrow(
        m_device.CreateRtAccelerationStructure(gpu::RtLevel::Top, sizes.structure));

    // Create scratch buffer, sized for both the build and later refits
    const u64 scratchAlignment = ScratchAlignment(m_device.Info());
    m_scratchBuffer = CreateBufferOrThrow(
        m_device, ScratchDesc(std::max(sizes.buildScratch, sizes.updateScratch), scratchAlignment));
    WarnIfMisaligned("TLAS scratch", m_device.BufferDeviceAddress(m_scratchBuffer->Ref()),
                     scratchAlignment);

    // Build acceleration structure
    gpu::RtBuild build;
    build.geometry = geometry;
    build.mode = gpu::RtBuildMode::Build;
    build.dst = m_structure->Ref();
    build.scratch = m_scratchBuffer->Ref();

    list.BuildRtAccelerationStructure(build);

    // CRITICAL: Insert memory barrier to ensure TLAS build completes before ray tracing shaders use it
    // Without this barrier, a trace may read incomplete TLAS data.
    // TLAS is read by ray tracing shaders.
    list.Barrier(gpu::GlobalBarrier{gpu::Access::AccelerationStructureWrite,
                                    gpu::Access::RayTraceSceneRead});

    m_built = true;
    QL_LOG_INFO("  TLAS built successfully with {} instance(s)", m_instances.size());
}

void TLAS::SetInstanceTransform(size_t index, const glm::mat4& transform) {
    if (index >= m_instances.size()) {
        QL_LOG_WARN("TLAS::SetInstanceTransform: index {} out of range ({})",
                    index, m_instances.size());
        return;
    }

    ToInstanceTransform(transform, m_instances[index].transform);
}

void TLAS::Update(gpu::CommandList& list) {
    if (!m_built) {
        throw std::runtime_error("TLAS::Update called before Build");
    }

    // The list comes from a device: the refit below needs its ray tracing entry points.
    const gpu::BufferRef instanceInput = m_instanceBuffer->Ref();

    // Reuse one device-address-stable input buffer. Recording the bytes into
    // the command buffer avoids a host write racing a previous frame's AS
    // update. vkCmdUpdateBuffer is limited to 64 KiB per call, so large scenes
    // are split on the 64-byte instance boundary.
    list.Barrier(gpu::BufferBarrier{instanceInput,
                                    gpu::Access::AccelerationStructureRead | gpu::Access::TransferRead,
                                    gpu::Access::TransferWrite});

    // The update copies the bytes into the command when it is recorded, so the
    // packed records need only live until the loop ends.
    const std::vector<std::byte> packed = PackedInstances();
    const u64 instanceBytes = packed.size();
    for (u64 offset = 0; offset < instanceBytes; offset += kMaxCmdUpdateBytes) {
        const u64 bytes = std::min(kMaxCmdUpdateBytes, instanceBytes - offset);
        list.UpdateBuffer(instanceInput, offset,
                          std::span<const std::byte>(packed.data() + offset, static_cast<std::size_t>(bytes)));
    }

    list.Barrier(gpu::BufferBarrier{instanceInput, gpu::Access::TransferWrite,
                                    gpu::Access::AccelerationStructureRead});

    // Refit in place: src == dst == this AS, with the scratch sized for it at Build.
    gpu::RtBuild build;
    build.geometry = TopLevelGeometry(*m_instanceBuffer, static_cast<u32>(m_instances.size()));
    build.mode = gpu::RtBuildMode::Update;
    build.src = m_structure->Ref();
    build.dst = m_structure->Ref();
    build.scratch = m_scratchBuffer->Ref();

    // In-flight frames may still be reading this AS: order the refit after them
    constexpr gpu::Access kSceneReaders = gpu::Access::RayTraceSceneRead | gpu::Access::ComputeSceneRead;
    list.Barrier(gpu::GlobalBarrier{kSceneReaders, gpu::Access::AccelerationStructureWrite});

    list.BuildRtAccelerationStructure(build);

    list.Barrier(gpu::GlobalBarrier{gpu::Access::AccelerationStructureWrite, kSceneReaders});
}

} // namespace quantiloom
