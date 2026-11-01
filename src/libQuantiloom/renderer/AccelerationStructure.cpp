#include "AccelerationStructure.hpp"
#include "core/Log.hpp"
#include "gpu/Device.hpp"
#include "gpu/vulkan/VulkanBridge.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <memory>
#include <utility>

namespace quantiloom {

namespace {

// VUID-VkAccelerationStructureBuildGeometryInfoKHR-scratchData-03710: the
// scratch device address must be a multiple of
// minAccelerationStructureScratchOffsetAlignment (128 on NVIDIA). Nothing in
// the scratch buffer's usage flags implies that, so VMA is free to suballocate
// at a 16-byte boundary and the build reads and writes off its own scratch --
// which surfaces as a GPU hang and VK_ERROR_DEVICE_LOST, not as an error.
VkDeviceSize ScratchAlignment(const VulkanContext& context) {
    const VkDeviceSize reported =
        context.GetAccelerationStructureProperties().minAccelerationStructureScratchOffsetAlignment;
    // A zero here would mean the property was never queried; 128 covers every
    // desktop driver we target and costs nothing when the real value is lower.
    return reported > 0 ? reported : 128;
}

// VUID-VkAccelerationStructureGeometryInstancesDataKHR-arrayOfPointers-03779.
constexpr VkDeviceSize kInstanceDataAlignment = 16;
constexpr VkDeviceSize kMaxCmdUpdateBytes = 65536;

// The failure mode this guards against has no error code -- an unaligned build
// input hangs the GPU and the device-lost surfaces later, at whatever the next
// queue wait happens to be. Say it out loud here instead.
void WarnIfMisaligned(const char* what, VkDeviceAddress address, VkDeviceSize alignment) {
    if (alignment > 0 && (address % alignment) != 0) {
        QL_LOG_ERROR("  {} device address 0x{:x} is not {}-byte aligned -- "
                     "the acceleration structure build will corrupt memory",
                     what, address, alignment);
    }
}

// The RHI's Err here is the text this file threw before it used the RHI
// (gpu/AGENTS.md), so a failure surfaces as the same std::runtime_error.
template <class T>
T TakeOrThrow(Result<T, String>&& result) {
    if (!result) throw std::runtime_error(result.error());
    return std::move(result).value();
}

// Device-local storage with a device address, at the scratch offset alignment.
std::unique_ptr<gpu::Buffer> CreateScratch(gpu::Device& device, VkDeviceSize bytes, VkDeviceSize alignment) {
    gpu::BufferDesc desc;
    desc.size = bytes;
    desc.usage = gpu::BufferUsage::Storage | gpu::BufferUsage::DeviceAddress;
    desc.memory = gpu::MemoryClass::DeviceLocal;
    desc.minAlignment = alignment;
    return TakeOrThrow(device.CreateBuffer(desc));
}

// The one geometry of a Top level: `count` packed instance records from the
// start of `instances`. Its own geometry flag stays 0 (see the backend).
gpu::RtGeometry TopLevelGeometry(const GpuBuffer& instances, u32 count) {
    gpu::RtGeometry geometry;
    geometry.level = gpu::RtLevel::Top;
    // ALLOW_UPDATE so Update() can refit transforms in place during drags
    geometry.flags = gpu::RtBuildFlags::PreferFastTrace | gpu::RtBuildFlags::AllowUpdate;
    geometry.instances.buffer = gpu::vulkan::Ref(instances);
    geometry.instances.offset = 0;
    geometry.instances.count = count;
    return geometry;
}

// A list the device hands out carries the ray tracing entry points; a
// vulkan::Wrap list has none, and an acceleration structure build on it throws.
std::unique_ptr<gpu::CommandList> Record(gpu::Device& device, VkCommandBuffer cmd) {
    return device.WrapRecording(gpu::vulkan::ToHandle(cmd));
}

}  // namespace

// ============================================================================
// BLAS Implementation
// ============================================================================

BLAS::BLAS(VulkanContext& context, GeometrySlice geometry, bool opaque)
    : m_context(context)
    , m_opaque(opaque)
    , m_geometry(geometry)
{
    if (!geometry.vertices || !geometry.indices || geometry.vertexCount == 0 ||
        geometry.indexCount == 0) {
        throw std::runtime_error("Cannot create BLAS from an empty geometry slice");
    }

    QL_LOG_INFO("Creating BLAS for primitive with {} vertices, {} triangles",
                geometry.vertexCount, geometry.indexCount / 3);
}

// The structure is destroyed first, then its storage, then the scratch.
BLAS::~BLAS() = default;

BLAS::BLAS(BLAS&& other) noexcept
    : m_context(other.m_context)
    , m_gpu(std::move(other.m_gpu))
    , m_scratchBuffer(std::move(other.m_scratchBuffer))
    , m_structure(std::move(other.m_structure))
    , m_deviceAddress(other.m_deviceAddress)
    , m_built(other.m_built)
    , m_opaque(other.m_opaque)
    , m_geometry(other.m_geometry)
{
    other.m_deviceAddress = 0;
    other.m_built = false;
}

BLAS& BLAS::operator=(BLAS&& other) noexcept {
    if (this != &other) {
        // Destroys the current structure, then its storage, then the old
        // scratch. m_gpu stays: it is the adoption of m_context, which a move
        // assignment does not change either.
        m_structure = std::move(other.m_structure);
        m_scratchBuffer = std::move(other.m_scratchBuffer);
        m_deviceAddress = other.m_deviceAddress;
        m_built = other.m_built;
        m_opaque = other.m_opaque;
        m_geometry = other.m_geometry;

        // Nullify source
        other.m_deviceAddress = 0;
        other.m_built = false;
    }
    return *this;
}

// Adopted on first use rather than in the constructor: constructing or moving
// a BLAS makes no native call, and a moved-from one can still build.
gpu::Device& BLAS::Gpu() {
    if (!m_gpu) m_gpu = gpu::vulkan::AdoptContext(m_context);
    return *m_gpu;
}

VkAccelerationStructureKHR BLAS::GetHandle() const {
    return m_structure ? gpu::vulkan::NativeScene(m_structure->Ref()) : VK_NULL_HANDLE;
}

void BLAS::Build(VkCommandBuffer cmd) {
    VkDevice device = m_context.GetDevice();

    if (!m_geometry.vertices || !m_geometry.indices) {
        throw std::runtime_error("BLAS geometry slice is invalid");
    }

    // The ray tracing entry points come with the device, fetched once there.
    gpu::Device& gpuDevice = Gpu();

    // Define geometry (triangles): RGB32Float positions with a 12-byte stride,
    // u32 indices, offsets in bytes into the merged buffers.
    gpu::RtGeometry geometry;
    geometry.level = gpu::RtLevel::Bottom;
    geometry.flags = gpu::RtBuildFlags::PreferFastTrace;  // M1: static geometry
    geometry.triangles.vertices = gpu::vulkan::Ref(*m_geometry.vertices);
    geometry.triangles.vertexOffset =
        static_cast<VkDeviceSize>(m_geometry.vertexOffset) * sizeof(glm::vec3);
    geometry.triangles.vertexCount = m_geometry.vertexCount;
    geometry.triangles.indices = gpu::vulkan::Ref(*m_geometry.indices);
    geometry.triangles.indexOffset =
        static_cast<VkDeviceSize>(m_geometry.indexOffset) * sizeof(u32);
    geometry.triangles.triangleCount = m_geometry.indexCount / 3;
    // The one flag that decides whether an any-hit shader runs at all: set, the
    // traversal is free to skip the stage entirely -- which is what we want for
    // the overwhelming majority of geometry and what would silently disable
    // alpha coverage on the rest.
    geometry.triangles.opaque = m_opaque;

    // Query build sizes
    const gpu::RtBuildSizes sizes = gpuDevice.GetRtBuildSizes(geometry);

    QL_LOG_INFO("  BLAS build sizes: AS={} bytes, scratch={} bytes",
                sizes.structure, sizes.buildScratch);

    // Create the AS buffer, the acceleration structure on it, and its device address
    m_structure = TakeOrThrow(
        gpuDevice.CreateRtAccelerationStructure(gpu::RtLevel::Bottom, sizes.structure));
    m_deviceAddress = m_structure->Ref().address;

    // Create scratch buffer
    m_scratchBuffer = CreateScratch(gpuDevice, sizes.buildScratch, ScratchAlignment(m_context));
    WarnIfMisaligned("BLAS scratch",
                     gpu::vulkan::AsGpuBuffer(*m_scratchBuffer).GetDeviceAddress(device),
                     ScratchAlignment(m_context));

    // Build acceleration structure
    gpu::RtBuild build;
    build.geometry = geometry;
    build.mode = gpu::RtBuildMode::Build;
    build.dst = m_structure->Ref();
    build.scratch = m_scratchBuffer->Ref();

    const std::unique_ptr<gpu::CommandList> list = Record(gpuDevice, cmd);
    list->BuildRtAccelerationStructure(build);

    // CRITICAL: Insert memory barrier to ensure BLAS build completes before TLAS reads it
    // Without this barrier, TLAS may reference incomplete BLAS data
    list->Barrier(gpu::GlobalBarrier{gpu::Access::AccelerationStructureWrite,
                                     gpu::Access::AccelerationStructureRead});

    m_built = true;
    QL_LOG_INFO("  BLAS built successfully (device address: 0x{:x})", m_deviceAddress);
}

void BLAS::ReleaseBuildScratch() {
    m_scratchBuffer.reset();
}

// ============================================================================
// TLAS Implementation
// ============================================================================

TLAS::TLAS(VulkanContext& context)
    : m_context(context)
{
    QL_LOG_INFO("Creating TLAS...");
}

// The structure is destroyed first, then its storage, then scratch and instances.
TLAS::~TLAS() = default;

TLAS::TLAS(TLAS&& other) noexcept
    : m_context(other.m_context)
    , m_gpu(std::move(other.m_gpu))
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
        // instance and scratch buffers. m_gpu stays: it is the adoption of
        // m_context, which a move assignment does not change either.
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

// Adopted on first use, as for the BLAS.
gpu::Device& TLAS::Gpu() {
    if (!m_gpu) m_gpu = gpu::vulkan::AdoptContext(m_context);
    return *m_gpu;
}

VkAccelerationStructureKHR TLAS::GetHandle() const {
    return m_structure ? gpu::vulkan::NativeScene(m_structure->Ref()) : VK_NULL_HANDLE;
}

void TLAS::AddInstance(const BLAS& blas, u32 materialId, const glm::mat4& transform, bool doubleSided) {
    if (m_built) {
        throw std::runtime_error("Cannot add instance to already-built TLAS");
    }

    // Convert glm::mat4 to VkTransformMatrixKHR (row-major 3x4)
    VkTransformMatrixKHR vkTransform{};
    const f32* mat = glm::value_ptr(transform);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            vkTransform.matrix[row][col] = mat[col * 4 + row];  // transpose
        }
    }

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform = vkTransform;
    instance.instanceCustomIndex = materialId;  // Material ID accessible in shader via InstanceID()
    instance.mask = 0xFF;  // Visible to all rays
    instance.instanceShaderBindingTableRecordOffset = 0;  // Single hit group

    // Set backface culling based on material's doubleSided property
    // - doubleSided=true: Disable culling, backfaces will be shaded (normal flipped in shader)
    // - doubleSided=false: Enable culling, rays pass through backfaces (no hit)
    if (doubleSided) {
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    } else {
        instance.flags = 0;  // Enable backface culling (default Vulkan behavior)
    }

    instance.accelerationStructureReference = blas.GetDeviceAddress();

    m_instances.push_back(instance);

    QL_LOG_INFO("  Added instance {} to TLAS (material {}, doubleSided={}, BLAS addr: 0x{:x})",
                m_instances.size() - 1, materialId, doubleSided, blas.GetDeviceAddress());
}

void TLAS::Build(VkCommandBuffer cmd) {
    if (m_instances.empty()) {
        throw std::runtime_error("Cannot build TLAS with no instances");
    }

    VkDevice device = m_context.GetDevice();
    VmaAllocator allocator = m_context.GetAllocator();

    // The ray tracing entry points come with the device, fetched once there.
    gpu::Device& gpuDevice = Gpu();

    // Upload instances to GPU
    const VkDeviceSize instanceBufferSize = m_instances.size() * sizeof(VkAccelerationStructureInstanceKHR);

    m_instanceBuffer = std::make_unique<GpuBuffer>(
        allocator,
        instanceBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU,
        kInstanceDataAlignment
    );

    m_instanceBuffer->Upload(m_instances.data(), instanceBufferSize);

    WarnIfMisaligned("TLAS instance data", m_instanceBuffer->GetDeviceAddress(device),
                     kInstanceDataAlignment);

    // Define geometry (instances)
    const u32 instanceCount = static_cast<u32>(m_instances.size());
    const gpu::RtGeometry geometry = TopLevelGeometry(*m_instanceBuffer, instanceCount);

    // Query build sizes
    const gpu::RtBuildSizes sizes = gpuDevice.GetRtBuildSizes(geometry);

    QL_LOG_INFO("  TLAS build sizes: AS={} bytes, scratch={} bytes",
                sizes.structure, sizes.buildScratch);

    // Create the AS buffer and the acceleration structure on it
    m_structure = TakeOrThrow(
        gpuDevice.CreateRtAccelerationStructure(gpu::RtLevel::Top, sizes.structure));

    // Create scratch buffer, sized for both the build and later refits
    m_scratchBuffer = CreateScratch(gpuDevice, std::max(sizes.buildScratch, sizes.updateScratch),
                                    ScratchAlignment(m_context));
    WarnIfMisaligned("TLAS scratch",
                     gpu::vulkan::AsGpuBuffer(*m_scratchBuffer).GetDeviceAddress(device),
                     ScratchAlignment(m_context));

    // Build acceleration structure
    gpu::RtBuild build;
    build.geometry = geometry;
    build.mode = gpu::RtBuildMode::Build;
    build.dst = m_structure->Ref();
    build.scratch = m_scratchBuffer->Ref();

    const std::unique_ptr<gpu::CommandList> list = Record(gpuDevice, cmd);
    list->BuildRtAccelerationStructure(build);

    // CRITICAL: Insert memory barrier to ensure TLAS build completes before ray tracing shaders use it
    // Without this barrier, vkCmdTraceRaysKHR may read incomplete TLAS data.
    // TLAS is read by ray tracing shaders.
    list->Barrier(gpu::GlobalBarrier{gpu::Access::AccelerationStructureWrite,
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

    VkTransformMatrixKHR vkTransform{};
    const f32* mat = glm::value_ptr(transform);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            vkTransform.matrix[row][col] = mat[col * 4 + row];  // transpose
        }
    }
    m_instances[index].transform = vkTransform;
}

void TLAS::Update(VkCommandBuffer cmd) {
    if (!m_built) {
        throw std::runtime_error("TLAS::Update called before Build");
    }

    // A list from the device: the refit below needs its ray tracing entry points.
    const std::unique_ptr<gpu::CommandList> list = Record(Gpu(), cmd);
    const gpu::BufferRef instanceInput = gpu::vulkan::Ref(*m_instanceBuffer);

    // Reuse one device-address-stable input buffer. Recording the bytes into
    // the command buffer avoids a host write racing a previous frame's AS
    // update. vkCmdUpdateBuffer is limited to 64 KiB per call, so large scenes
    // are split on the 64-byte instance boundary.
    list->Barrier(gpu::BufferBarrier{instanceInput,
                                     gpu::Access::AccelerationStructureRead | gpu::Access::TransferRead,
                                     gpu::Access::TransferWrite});

    const VkDeviceSize instanceBytes =
        m_instances.size() * sizeof(VkAccelerationStructureInstanceKHR);
    const auto* source = reinterpret_cast<const std::byte*>(m_instances.data());
    for (VkDeviceSize offset = 0; offset < instanceBytes; offset += kMaxCmdUpdateBytes) {
        const VkDeviceSize bytes = std::min(kMaxCmdUpdateBytes, instanceBytes - offset);
        list->UpdateBuffer(instanceInput, offset,
                           std::span<const std::byte>(source + offset, static_cast<std::size_t>(bytes)));
    }

    list->Barrier(gpu::BufferBarrier{instanceInput, gpu::Access::TransferWrite,
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
    list->Barrier(gpu::GlobalBarrier{kSceneReaders, gpu::Access::AccelerationStructureWrite});

    list->BuildRtAccelerationStructure(build);

    list->Barrier(gpu::GlobalBarrier{gpu::Access::AccelerationStructureWrite, kSceneReaders});
}

} // namespace quantiloom
