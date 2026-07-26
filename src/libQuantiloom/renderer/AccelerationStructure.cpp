#include "AccelerationStructure.hpp"
#include "CommandHelper.hpp"
#include "core/Log.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>
#include <cstring>
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

BLAS::BLAS(VulkanContext& context, const GeometryPrimitive& primitive)
    : m_context(context)
    , m_primitive(primitive)
{
    if (primitive.positions.empty()) {
        throw std::runtime_error("Cannot create BLAS from empty primitive");
    }

    QL_LOG_INFO("Creating BLAS for primitive with {} vertices, {} triangles",
                primitive.positions.size(), primitive.indices.size() / 3);

    // Upload vertex and index data to GPU immediately (using ExecuteImmediate)
    // This ensures staging buffers are not destroyed before GPU upload completes
    UploadGeometryBuffers();
}

void BLAS::UploadGeometryBuffers() {
    VmaAllocator allocator = m_context.GetAllocator();

    const VkDeviceSize vertexBufferSize = m_primitive.positions.size() * sizeof(glm::vec3);
    const VkDeviceSize indexBufferSize = m_primitive.indices.size() * sizeof(u32);

    // Create device-local buffers (GPU-only, fastest for AS build and shader access)
    // CRITICAL: Add VK_BUFFER_USAGE_STORAGE_BUFFER_BIT for shader StructuredBuffer access
    m_vertexBuffer = std::make_unique<GpuBuffer>(
        allocator,
        vertexBufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  // Required for StructuredBuffer in shaders
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    m_indexBuffer = std::make_unique<GpuBuffer>(
        allocator,
        indexBufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  // Required for StructuredBuffer in shaders
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Create UV buffer if UVs are present (optional)
    const bool hasUVs = !m_primitive.uvs.empty();
    if (hasUVs) {
        const VkDeviceSize uvBufferSize = m_primitive.uvs.size() * sizeof(glm::vec2);
        m_uvBuffer = std::make_unique<GpuBuffer>(
            allocator,
            uvBufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  // For StructuredBuffer access in shaders
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        QL_LOG_DEBUG("  [DEBUG] Created UV buffer: {} UVs ({} bytes)", m_primitive.uvs.size(), uvBufferSize);
    } else {
        QL_LOG_DEBUG("  [DEBUG] No UVs to upload (primitive.uvs is empty)");
    }

    // Create tangent buffer (always create, use fallback if not present)
    // CRITICAL: Always bind tangent buffer to prevent GPU crash when shader accesses it
    const bool hasTangents = !m_primitive.tangents.empty();
    const size_t tangentCount = hasTangents ? m_primitive.tangents.size() : m_primitive.positions.size();
    const VkDeviceSize tangentBufferSize = tangentCount * sizeof(glm::vec4);
    m_tangentBuffer = std::make_unique<GpuBuffer>(
        allocator,
        tangentBufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  // For StructuredBuffer access in shaders
        VMA_MEMORY_USAGE_GPU_ONLY
    );
    if (hasTangents) {
        QL_LOG_DEBUG("  BLAS: Created tangent buffer: {} tangents ({} bytes)", tangentCount, tangentBufferSize);
    } else {
        QL_LOG_DEBUG("  BLAS: Created fallback tangent buffer: {} vertices ({} bytes)", tangentCount, tangentBufferSize);
    }

    // Create normal buffer (always create, use fallback if not present)
    // CRITICAL: Always bind normal buffer for smooth shading interpolation
    const bool hasNormals = !m_primitive.normals.empty();
    const size_t normalCount = hasNormals ? m_primitive.normals.size() : m_primitive.positions.size();
    const VkDeviceSize normalBufferSize = normalCount * sizeof(glm::vec3);
    m_normalBuffer = std::make_unique<GpuBuffer>(
        allocator,
        normalBufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  // For StructuredBuffer access in shaders
        VMA_MEMORY_USAGE_GPU_ONLY
    );
    if (hasNormals) {
        QL_LOG_DEBUG("  BLAS: Created normal buffer: {} normals ({} bytes)", normalCount, normalBufferSize);
    } else {
        QL_LOG_WARN("  BLAS: No normals provided, creating fallback flat normal buffer: {} vertices ({} bytes)", normalCount, normalBufferSize);
    }

    // Upload data using ExecuteImmediate (ensures staging buffers live until upload completes)
    // CRITICAL: Staging buffers MUST be created OUTSIDE the lambda to ensure they
    // remain valid until the command buffer is submitted and GPU operations complete.
    // If created inside the lambda, they would be destroyed before vkEndCommandBuffer,
    // causing validation errors (VkBuffer destroyed while command buffer still recording).

    // Create staging buffers OUTSIDE the lambda (CPU-accessible)
    GpuBuffer vertexStaging(
        allocator,
        vertexBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    GpuBuffer indexStaging(
        allocator,
        indexBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    // Upload data to staging buffers
    vertexStaging.Upload(m_primitive.positions.data(), vertexBufferSize);
    indexStaging.Upload(m_primitive.indices.data(), indexBufferSize);

    // Create UV staging buffer if needed
    std::unique_ptr<GpuBuffer> uvStaging;
    if (hasUVs) {
        const VkDeviceSize uvBufferSize = m_primitive.uvs.size() * sizeof(glm::vec2);
        uvStaging = std::make_unique<GpuBuffer>(
            allocator,
            uvBufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY
        );
        uvStaging->Upload(m_primitive.uvs.data(), uvBufferSize);
    }

    // Create tangent staging buffer
    GpuBuffer tangentStaging(
        allocator,
        tangentBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    if (hasTangents) {
        tangentStaging.Upload(m_primitive.tangents.data(), tangentBufferSize);
        QL_LOG_DEBUG("  BLAS: Prepared {} real tangents for upload", tangentCount);
    } else {
        // Generate fallback tangents dynamically based on vertex normals
        // This ensures the tangent is always perpendicular to the normal,
        // avoiding TBN matrix degeneration when normal is parallel to X-axis
        std::vector<glm::vec4> fallbackTangents;
        fallbackTangents.reserve(tangentCount);

        // Use normals if available, otherwise generate from face
        const bool hasNormals = !m_primitive.normals.empty();

        for (size_t i = 0; i < tangentCount; ++i) {
            glm::vec3 normal;
            if (hasNormals && i < m_primitive.normals.size()) {
                normal = glm::normalize(m_primitive.normals[i]);
            } else {
                // Fallback to up vector if no normals
                normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            // Choose a reference vector that is not parallel to the normal
            // If normal is close to Y-axis (up/down), use X-axis as reference
            // Otherwise, use Y-axis as reference
            glm::vec3 refVector = (std::abs(normal.y) > 0.9f)
                ? glm::vec3(1.0f, 0.0f, 0.0f)
                : glm::vec3(0.0f, 1.0f, 0.0f);

            // Compute tangent as cross product of normal and reference vector
            glm::vec3 tangent = glm::normalize(glm::cross(normal, refVector));

            // Store tangent with handedness = +1 (right-handed)
            fallbackTangents.emplace_back(tangent, 1.0f);
        }

        tangentStaging.Upload(fallbackTangents.data(), tangentBufferSize);
        QL_LOG_DEBUG("  BLAS: Prepared {} dynamic fallback tangents for upload", tangentCount);
    }

    // Create normal staging buffer
    GpuBuffer normalStaging(
        allocator,
        normalBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    if (hasNormals) {
        normalStaging.Upload(m_primitive.normals.data(), normalBufferSize);
        QL_LOG_DEBUG("  BLAS: Prepared {} real normals for upload", normalCount);
    } else {
        // Loaders should have generated normals - this is a fallback for edge cases
        QL_LOG_WARN("  BLAS: Mesh has no normals after loading - using fallback up vector");
        std::vector<glm::vec3> fallbackNormals(normalCount, glm::vec3(0.0f, 1.0f, 0.0f));
        normalStaging.Upload(fallbackNormals.data(), normalBufferSize);
    }

    // Execute copy commands (staging buffers remain valid throughout)
    CommandHelper::ExecuteImmediate(m_context, [&](VkCommandBuffer cmd) {
        // Copy staging → device-local
        VkBufferCopy vertexCopyRegion{};
        vertexCopyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(cmd, vertexStaging.GetHandle(), m_vertexBuffer->GetHandle(), 1, &vertexCopyRegion);

        VkBufferCopy indexCopyRegion{};
        indexCopyRegion.size = indexBufferSize;
        vkCmdCopyBuffer(cmd, indexStaging.GetHandle(), m_indexBuffer->GetHandle(), 1, &indexCopyRegion);

        // Upload UV data if present
        if (hasUVs && uvStaging) {
            const VkDeviceSize uvBufferSize = m_primitive.uvs.size() * sizeof(glm::vec2);
            VkBufferCopy uvCopyRegion{};
            uvCopyRegion.size = uvBufferSize;
            vkCmdCopyBuffer(cmd, uvStaging->GetHandle(), m_uvBuffer->GetHandle(), 1, &uvCopyRegion);

            QL_LOG_DEBUG("  [DEBUG] Uploaded {} UV coordinates to GPU", m_primitive.uvs.size());
        }

        // Upload tangent data
        VkBufferCopy tangentCopyRegion{};
        tangentCopyRegion.size = tangentBufferSize;
        vkCmdCopyBuffer(cmd, tangentStaging.GetHandle(), m_tangentBuffer->GetHandle(), 1, &tangentCopyRegion);

        // Upload normal data
        VkBufferCopy normalCopyRegion{};
        normalCopyRegion.size = normalBufferSize;
        vkCmdCopyBuffer(cmd, normalStaging.GetHandle(), m_normalBuffer->GetHandle(), 1, &normalCopyRegion);

        // Insert barrier - transfer writes must complete before AS build reads
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0,
            1, &barrier,
            0, nullptr,
            0, nullptr
        );
    });
    // Staging buffers are destroyed here, AFTER ExecuteImmediate completes (GPU done)

    QL_LOG_INFO("  Uploaded geometry via staging buffers: {} vertices, {} indices",
                m_primitive.positions.size(), m_primitive.indices.size());
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
    , m_vertexBuffer(std::move(other.m_vertexBuffer))
    , m_indexBuffer(std::move(other.m_indexBuffer))
    , m_uvBuffer(std::move(other.m_uvBuffer))
    , m_tangentBuffer(std::move(other.m_tangentBuffer))
    , m_normalBuffer(std::move(other.m_normalBuffer))
    , m_scratchBuffer(std::move(other.m_scratchBuffer))
    , m_deviceAddress(other.m_deviceAddress)
    , m_built(other.m_built)
    , m_primitive(other.m_primitive)
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
        m_vertexBuffer = std::move(other.m_vertexBuffer);
        m_indexBuffer = std::move(other.m_indexBuffer);
        m_uvBuffer = std::move(other.m_uvBuffer);
        m_tangentBuffer = std::move(other.m_tangentBuffer);
        m_normalBuffer = std::move(other.m_normalBuffer);
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

    // Verify geometry buffers were uploaded in constructor
    if (!m_vertexBuffer || !m_indexBuffer) {
        throw std::runtime_error("Geometry buffers not uploaded. This should not happen.");
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
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;  // M1: all geometry is opaque

    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.vertexData.deviceAddress = m_vertexBuffer->GetDeviceAddress(device);
    geometry.geometry.triangles.vertexStride = sizeof(glm::vec3);
    geometry.geometry.triangles.maxVertex = static_cast<u32>(m_primitive.positions.size() - 1);
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
    u32 primitiveCount = static_cast<u32>(m_primitive.indices.size() / 3);
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
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = m_instanceBuffer->GetDeviceAddress(device);
    WarnIfMisaligned("TLAS instance data", geometry.geometry.instances.data.deviceAddress,
                     kInstanceDataAlignment);

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

    if (VkResult result = vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &m_as); result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS");
    }

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

} // namespace quantiloom
