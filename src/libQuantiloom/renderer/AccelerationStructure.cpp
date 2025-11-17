#include "AccelerationStructure.hpp"
#include "core/Log.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>
#include <cstring>

namespace quantiloom {

// ============================================================================
// BLAS Implementation
// ============================================================================

BLAS::BLAS(VulkanContext& context, const Mesh& mesh)
    : m_context(context)
    , m_mesh(mesh)
{
    if (mesh.positions.empty()) {
        throw std::runtime_error("Cannot create BLAS from empty mesh");
    }

    QL_LOG_INFO("Creating BLAS for mesh with {} vertices, {} triangles",
                mesh.positions.size(), mesh.indices.size() / 3);
}

BLAS::~BLAS() {
    if (m_as != VK_NULL_HANDLE) {
        auto vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)
            vkGetDeviceProcAddr(m_context.GetDevice(), "vkDestroyAccelerationStructureKHR");

        if (vkDestroyAccelerationStructureKHR) {
            vkDestroyAccelerationStructureKHR(m_context.GetDevice(), m_as, nullptr);
        }
    }
}

BLAS::BLAS(BLAS&& other) noexcept
    : m_context(other.m_context)
    , m_as(other.m_as)
    , m_asBuffer(std::move(other.m_asBuffer))
    , m_vertexBuffer(std::move(other.m_vertexBuffer))
    , m_indexBuffer(std::move(other.m_indexBuffer))
    , m_scratchBuffer(std::move(other.m_scratchBuffer))
    , m_deviceAddress(other.m_deviceAddress)
    , m_built(other.m_built)
    , m_mesh(other.m_mesh)
{
    other.m_as = VK_NULL_HANDLE;
    other.m_deviceAddress = 0;
    other.m_built = false;
}

BLAS& BLAS::operator=(BLAS&& other) noexcept {
    if (this != &other) {
        // Destroy current resources
        if (m_as != VK_NULL_HANDLE) {
            auto vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)
                vkGetDeviceProcAddr(m_context.GetDevice(), "vkDestroyAccelerationStructureKHR");

            if (vkDestroyAccelerationStructureKHR) {
                vkDestroyAccelerationStructureKHR(m_context.GetDevice(), m_as, nullptr);
            }
        }

        // Move from other
        m_as = other.m_as;
        m_asBuffer = std::move(other.m_asBuffer);
        m_vertexBuffer = std::move(other.m_vertexBuffer);
        m_indexBuffer = std::move(other.m_indexBuffer);
        m_scratchBuffer = std::move(other.m_scratchBuffer);
        m_deviceAddress = other.m_deviceAddress;
        m_built = other.m_built;

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

    // Get function pointers
    auto vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
    auto vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
    auto vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
    auto vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");

    if (!vkGetAccelerationStructureBuildSizesKHR || !vkCreateAccelerationStructureKHR ||
        !vkGetAccelerationStructureDeviceAddressKHR || !vkCmdBuildAccelerationStructuresKHR) {
        throw std::runtime_error("Failed to load acceleration structure functions");
    }

    // Upload vertex and index data to GPU
    const VkDeviceSize vertexBufferSize = m_mesh.positions.size() * sizeof(glm::vec3);
    const VkDeviceSize indexBufferSize = m_mesh.indices.size() * sizeof(u32);

    m_vertexBuffer = std::make_unique<GpuBuffer>(
        allocator,
        vertexBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );

    m_indexBuffer = std::make_unique<GpuBuffer>(
        allocator,
        indexBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );

    m_vertexBuffer->Upload(m_mesh.positions.data(), vertexBufferSize);
    m_indexBuffer->Upload(m_mesh.indices.data(), indexBufferSize);

    // Define geometry (triangles)
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;  // M1: all geometry is opaque

    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.vertexData.deviceAddress = m_vertexBuffer->GetDeviceAddress(device);
    geometry.geometry.triangles.vertexStride = sizeof(glm::vec3);
    geometry.geometry.triangles.maxVertex = static_cast<u32>(m_mesh.positions.size() - 1);
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.indexData.deviceAddress = m_indexBuffer->GetDeviceAddress(device);

    // Build info
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;  // M1: static geometry
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // Query build sizes
    u32 primitiveCount = static_cast<u32>(m_mesh.indices.size() / 3);
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

    VkResult result = vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &m_as);
    if (result != VK_SUCCESS) {
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
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Build acceleration structure
    buildInfo.dstAccelerationStructure = m_as;
    buildInfo.scratchData.deviceAddress = m_scratchBuffer->GetDeviceAddress(device);

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
        auto vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)
            vkGetDeviceProcAddr(m_context.GetDevice(), "vkDestroyAccelerationStructureKHR");

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
            auto vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)
                vkGetDeviceProcAddr(m_context.GetDevice(), "vkDestroyAccelerationStructureKHR");

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

void TLAS::AddInstance(const BLAS& blas, const glm::mat4& transform) {
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
    instance.instanceCustomIndex = static_cast<u32>(m_instances.size());
    instance.mask = 0xFF;  // Visible to all rays
    instance.instanceShaderBindingTableRecordOffset = 0;  // M1: single hit group
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blas.GetDeviceAddress();

    m_instances.push_back(instance);

    QL_LOG_INFO("  Added instance {} to TLAS (BLAS addr: 0x{:x})",
                m_instances.size() - 1, blas.GetDeviceAddress());
}

void TLAS::Build(VkCommandBuffer cmd) {
    if (m_instances.empty()) {
        throw std::runtime_error("Cannot build TLAS with no instances");
    }

    VkDevice device = m_context.GetDevice();
    VmaAllocator allocator = m_context.GetAllocator();

    // Get function pointers
    auto vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
    auto vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
    auto vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");

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
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );

    m_instanceBuffer->Upload(m_instances.data(), instanceBufferSize);

    // Define geometry (instances)
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = m_instanceBuffer->GetDeviceAddress(device);

    // Build info
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
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

    VkResult result = vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &m_as);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS");
    }

    // Create scratch buffer
    m_scratchBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Build acceleration structure
    buildInfo.dstAccelerationStructure = m_as;
    buildInfo.scratchData.deviceAddress = m_scratchBuffer->GetDeviceAddress(device);

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

} // namespace quantiloom
