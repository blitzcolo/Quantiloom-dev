#include "AccelerationStructure.hpp"
#include "core/Log.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <stdexcept>
#include <memory>

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

BLAS::~BLAS() {
    if (m_as != VK_NULL_HANDLE) {
        const auto vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(
            m_context.GetDevice(), "vkDestroyAccelerationStructureKHR"));

        if (vkDestroyAccelerationStructureKHR) {
            vkDestroyAccelerationStructureKHR(m_context.GetDevice(), m_as, nullptr);
        }
    }
}

BLAS::BLAS(BLAS&& other) noexcept
    : m_context(other.m_context)
    , m_as(other.m_as)
    , m_asBuffer(std::move(other.m_asBuffer))
    , m_scratchBuffer(std::move(other.m_scratchBuffer))
    , m_deviceAddress(other.m_deviceAddress)
    , m_built(other.m_built)
    , m_opaque(other.m_opaque)
    , m_geometry(other.m_geometry)
{
    other.m_as = VK_NULL_HANDLE;
    other.m_deviceAddress = 0;
    other.m_built = false;
}

BLAS& BLAS::operator=(BLAS&& other) noexcept {
    if (this != &other) {
        // Destroy current resources
        if (m_as != VK_NULL_HANDLE) {
            const auto vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(
                m_context.GetDevice(), "vkDestroyAccelerationStructureKHR"));

            if (vkDestroyAccelerationStructureKHR) {
                vkDestroyAccelerationStructureKHR(m_context.GetDevice(), m_as, nullptr);
            }
        }

        // Move from other
        m_as = other.m_as;
        m_asBuffer = std::move(other.m_asBuffer);
        m_scratchBuffer = std::move(other.m_scratchBuffer);
        m_deviceAddress = other.m_deviceAddress;
        m_built = other.m_built;
        m_opaque = other.m_opaque;
        m_geometry = other.m_geometry;

        // Nullify source
        other.m_as = VK_NULL_HANDLE;
        other.m_deviceAddress = 0;
        other.m_built = false;
    }
    return *this;
}

void BLAS::Build(VkCommandBuffer cmd) {
    VkDevice device = m_context.GetDevice();
    VmaAllocator allocator = m_context.GetAllocator();

    if (!m_geometry.vertices || !m_geometry.indices) {
        throw std::runtime_error("BLAS geometry slice is invalid");
    }

    // Get function pointers
    auto vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    auto vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    auto vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
    auto vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));

    if (!vkGetAccelerationStructureBuildSizesKHR || !vkCreateAccelerationStructureKHR ||
        !vkGetAccelerationStructureDeviceAddressKHR || !vkCmdBuildAccelerationStructuresKHR) {
        throw std::runtime_error("Failed to load acceleration structure functions");
    }

    // Define geometry (triangles)
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    // The one flag that decides whether an any-hit shader runs at all: set, the
    // traversal is free to skip the stage entirely -- which is what we want for
    // the overwhelming majority of geometry and what would silently disable
    // alpha coverage on the rest.
    geometry.flags = m_opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;

    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.vertexData.deviceAddress =
        m_geometry.vertices->GetDeviceAddress(device) +
        static_cast<VkDeviceSize>(m_geometry.vertexOffset) * sizeof(glm::vec3);
    geometry.geometry.triangles.vertexStride = sizeof(glm::vec3);
    geometry.geometry.triangles.maxVertex = m_geometry.vertexCount - 1;
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.indexData.deviceAddress =
        m_geometry.indices->GetDeviceAddress(device) +
        static_cast<VkDeviceSize>(m_geometry.indexOffset) * sizeof(u32);

    // Build info
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;  // M1: static geometry
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // Query build sizes
    u32 primitiveCount = m_geometry.indexCount / 3;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetAccelerationStructureBuildSizesKHR(
        device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &primitiveCount,
        &sizeInfo
    );

    QL_LOG_INFO("  BLAS build sizes: AS={} bytes, scratch={} bytes",
                sizeInfo.accelerationStructureSize, sizeInfo.buildScratchSize);

    // Create AS buffer
    m_asBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Create acceleration structure
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = m_asBuffer->GetHandle();
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    if (VkResult result = vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &m_as);
        result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create BLAS");
    }

    // Get device address
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = m_as;
    m_deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);

    // Create scratch buffer
    m_scratchBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        ScratchAlignment(m_context)
    );

    // Build acceleration structure
    buildInfo.dstAccelerationStructure = m_as;
    buildInfo.scratchData.deviceAddress = m_scratchBuffer->GetDeviceAddress(device);
    WarnIfMisaligned("BLAS scratch", buildInfo.scratchData.deviceAddress, ScratchAlignment(m_context));

    VkAccelerationStructureBuildRangeInfoKHR buildRange{};
    buildRange.primitiveCount = primitiveCount;
    buildRange.primitiveOffset = 0;
    buildRange.firstVertex = 0;
    buildRange.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* pBuildRange = &buildRange;

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pBuildRange);

    // CRITICAL: Insert memory barrier to ensure BLAS build completes before TLAS reads it
    // Without this barrier, TLAS may reference incomplete BLAS data
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );

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

TLAS::~TLAS() {
    if (m_as != VK_NULL_HANDLE) {
        const auto vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(m_context.GetDevice(), "vkDestroyAccelerationStructureKHR"));

        if (vkDestroyAccelerationStructureKHR) {
            vkDestroyAccelerationStructureKHR(m_context.GetDevice(), m_as, nullptr);
        }
    }
}

TLAS::TLAS(TLAS&& other) noexcept
    : m_context(other.m_context)
    , m_as(other.m_as)
    , m_asBuffer(std::move(other.m_asBuffer))
    , m_instanceBuffer(std::move(other.m_instanceBuffer))
    , m_scratchBuffer(std::move(other.m_scratchBuffer))
    , m_built(other.m_built)
    , m_instances(std::move(other.m_instances))
{
    other.m_as = VK_NULL_HANDLE;
    other.m_built = false;
}

TLAS& TLAS::operator=(TLAS&& other) noexcept {
    if (this != &other) {
        // Destroy current resources
        if (m_as != VK_NULL_HANDLE) {
            const auto vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
                vkGetDeviceProcAddr(m_context.GetDevice(), "vkDestroyAccelerationStructureKHR"));

            if (vkDestroyAccelerationStructureKHR) {
                vkDestroyAccelerationStructureKHR(m_context.GetDevice(), m_as, nullptr);
            }
        }

        // Move from other
        m_as = other.m_as;
        m_asBuffer = std::move(other.m_asBuffer);
        m_instanceBuffer = std::move(other.m_instanceBuffer);
        m_scratchBuffer = std::move(other.m_scratchBuffer);
        m_built = other.m_built;
        m_instances = std::move(other.m_instances);

        // Nullify source
        other.m_as = VK_NULL_HANDLE;
        other.m_built = false;
    }
    return *this;
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

    // Get function pointers
    auto vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    auto vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    auto vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));

    if (!vkGetAccelerationStructureBuildSizesKHR || !vkCreateAccelerationStructureKHR ||
        !vkCmdBuildAccelerationStructuresKHR) {
        throw std::runtime_error("Failed to load acceleration structure functions");
    }

    // Upload instances to GPU
    const VkDeviceSize instanceBufferSize = m_instances.size() * sizeof(VkAccelerationStructureInstanceKHR);

    m_instanceBuffer = std::make_unique<GpuBuffer>(
        allocator,
        instanceBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU,
        kInstanceDataAlignment
    );

    m_instanceBuffer->Upload(m_instances.data(), instanceBufferSize);

    // Define geometry (instances)
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    // Deliberately not set. Opacity is decided by the BLAS triangle geometry,
    // then the instance flags, then the ray flags; a top-level geometry of type
    // INSTANCES is not a thing that can be hit, so its own flag plays no part.
    // It was set here inertly, and is cleared rather than left because the risk
    // is asymmetric: clearing it cannot make opaque geometry non-opaque, while
    // leaving it could silently disable alpha coverage on a driver that reads
    // it.
    geometry.flags = 0;

    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = m_instanceBuffer->GetDeviceAddress(device);
    WarnIfMisaligned("TLAS instance data", geometry.geometry.instances.data.deviceAddress,
                     kInstanceDataAlignment);

    // Build info
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    // ALLOW_UPDATE so Update() can refit transforms in place during drags
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // Query build sizes
    u32 instanceCount = static_cast<u32>(m_instances.size());
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetAccelerationStructureBuildSizesKHR(
        device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &instanceCount,
        &sizeInfo
    );

    QL_LOG_INFO("  TLAS build sizes: AS={} bytes, scratch={} bytes",
                sizeInfo.accelerationStructureSize, sizeInfo.buildScratchSize);

    // Create AS buffer
    m_asBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Create acceleration structure
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = m_asBuffer->GetHandle();
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    if (VkResult result = vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &m_as); result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS");
    }

    // Create scratch buffer, sized for both the build and later refits
    m_scratchBuffer = std::make_unique<GpuBuffer>(
        allocator,
        std::max(sizeInfo.buildScratchSize, sizeInfo.updateScratchSize),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        ScratchAlignment(m_context)
    );

    // Build acceleration structure
    buildInfo.dstAccelerationStructure = m_as;
    buildInfo.scratchData.deviceAddress = m_scratchBuffer->GetDeviceAddress(device);
    WarnIfMisaligned("TLAS scratch", buildInfo.scratchData.deviceAddress, ScratchAlignment(m_context));

    VkAccelerationStructureBuildRangeInfoKHR buildRange{};
    buildRange.primitiveCount = instanceCount;
    buildRange.primitiveOffset = 0;
    buildRange.firstVertex = 0;
    buildRange.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* pBuildRange = &buildRange;

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pBuildRange);

    // CRITICAL: Insert memory barrier to ensure TLAS build completes before ray tracing shaders use it
    // Without this barrier, vkCmdTraceRaysKHR may read incomplete TLAS data
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,  // TLAS is read by ray tracing shaders
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );

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

    VkDevice device = m_context.GetDevice();

    auto vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
    if (!vkCmdBuildAccelerationStructuresKHR) {
        throw std::runtime_error("Failed to load acceleration structure functions");
    }

    // Same size, same buffer: only the transforms changed
    m_instanceBuffer->Upload(m_instances.data(),
                             m_instances.size() * sizeof(VkAccelerationStructureInstanceKHR));

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    // Deliberately not set. Opacity is decided by the BLAS triangle geometry,
    // then the instance flags, then the ray flags; a top-level geometry of type
    // INSTANCES is not a thing that can be hit, so its own flag plays no part.
    // It was set here inertly, and is cleared rather than left because the risk
    // is asymmetric: clearing it cannot make opaque geometry non-opaque, while
    // leaving it could silently disable alpha coverage on a driver that reads
    // it.
    geometry.flags = 0;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = m_instanceBuffer->GetDeviceAddress(device);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = m_as;
    buildInfo.dstAccelerationStructure = m_as;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.scratchData.deviceAddress = m_scratchBuffer->GetDeviceAddress(device);

    // In-flight frames may still be reading this AS: order the refit after them
    VkMemoryBarrier preBarrier{};
    preBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    preBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    preBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &preBarrier, 0, nullptr, 0, nullptr);

    VkAccelerationStructureBuildRangeInfoKHR buildRange{};
    buildRange.primitiveCount = static_cast<u32>(m_instances.size());
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildRange = &buildRange;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pBuildRange);

    VkMemoryBarrier postBarrier{};
    postBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    postBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    postBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 1, &postBarrier, 0, nullptr, 0, nullptr);
}

} // namespace quantiloom
