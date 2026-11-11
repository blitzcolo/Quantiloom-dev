#include "gpu/vulkan/VulkanRt.hpp"

#include "core/Platform.hpp"
#include "core/ResultFail.hpp"

#include <cstring>
#include <exception>

namespace quantiloom::gpu::vulkan {

namespace {

template <class Fn>
Fn LoadDeviceFunction(VkDevice device, const char* name) {
    return reinterpret_cast<Fn>(vkGetDeviceProcAddr(device, name));
}

VkAccelerationStructureTypeKHR StructureType(RtLevel level) {
    return level == RtLevel::Top ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                                 : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
}

VkBuildAccelerationStructureFlagsKHR BuildFlags(RtBuildFlags flags) {
    VkBuildAccelerationStructureFlagsKHR native = 0;
    if (HasAny(flags, RtBuildFlags::PreferFastTrace)) native |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (HasAny(flags, RtBuildFlags::AllowUpdate)) native |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    return native;
}

}  // namespace

RtFunctions LoadRtFunctions(VkDevice device) {
    RtFunctions rt;
    rt.device = device;
    rt.getBuildSizes = LoadDeviceFunction<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        device, "vkGetAccelerationStructureBuildSizesKHR");
    rt.createStructure =
        LoadDeviceFunction<PFN_vkCreateAccelerationStructureKHR>(device, "vkCreateAccelerationStructureKHR");
    rt.getStructureAddress = LoadDeviceFunction<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        device, "vkGetAccelerationStructureDeviceAddressKHR");
    rt.destroyStructure =
        LoadDeviceFunction<PFN_vkDestroyAccelerationStructureKHR>(device, "vkDestroyAccelerationStructureKHR");
    rt.cmdBuildStructures =
        LoadDeviceFunction<PFN_vkCmdBuildAccelerationStructuresKHR>(device, "vkCmdBuildAccelerationStructuresKHR");
    rt.createPipelines =
        LoadDeviceFunction<PFN_vkCreateRayTracingPipelinesKHR>(device, "vkCreateRayTracingPipelinesKHR");
    rt.getGroupHandles = LoadDeviceFunction<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
        device, "vkGetRayTracingShaderGroupHandlesKHR");
    rt.getGroupStackSize = LoadDeviceFunction<PFN_vkGetRayTracingShaderGroupStackSizeKHR>(
        device, "vkGetRayTracingShaderGroupStackSizeKHR");
    rt.cmdTraceRays = LoadDeviceFunction<PFN_vkCmdTraceRaysKHR>(device, "vkCmdTraceRaysKHR");
    return rt;
}

VkDeviceAddress BufferAddress(VkDevice device, BufferRef buffer, u64 offset) {
    if (!buffer.IsValid()) return 0;
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = FromHandle<VkBuffer>(buffer.handle);
    return vkGetBufferDeviceAddress(device, &info) + offset;
}

void FillRtGeometry(VkDevice device, const RtGeometry& geometry, NativeRtGeometry& out) {
    out.geometry = {};
    out.geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    if (geometry.level == RtLevel::Bottom) {
        const RtTriangles& source = geometry.triangles;
        out.geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        // The one flag that decides whether an any-hit shader runs at all (AccelerationStructure.cpp).
        out.geometry.flags = 0;
        if (source.opaque) out.geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        VkAccelerationStructureGeometryTrianglesDataKHR& triangles = out.geometry.geometry.triangles;
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = BufferAddress(device, source.vertices, source.vertexOffset);
        triangles.vertexStride = 3 * sizeof(f32);
        triangles.maxVertex = source.vertexCount - 1;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = BufferAddress(device, source.indices, source.indexOffset);
        out.primitiveCount = source.triangleCount;
    } else {
        const RtInstances& source = geometry.instances;
        out.geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        // Deliberately 0: a geometry of instances cannot be hit, so its flag plays no part in
        // opacity, and setting it risks disabling alpha coverage on a driver that reads it.
        out.geometry.flags = 0;
        VkAccelerationStructureGeometryInstancesDataKHR& instances = out.geometry.geometry.instances;
        instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instances.arrayOfPointers = VK_FALSE;
        instances.data.deviceAddress = BufferAddress(device, source.buffer, source.offset);
        out.primitiveCount = source.count;
    }

    out.info = {};
    out.info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    out.info.type = StructureType(geometry.level);
    out.info.flags = BuildFlags(geometry.flags);
    out.info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    out.info.geometryCount = 1;
    out.info.pGeometries = &out.geometry;
}

RtBuildSizes QueryRtBuildSizes(const RtFunctions& rt, const RtGeometry& geometry) {
    if (rt.getBuildSizes == nullptr) return RtBuildSizes{};
    NativeRtGeometry native;
    FillRtGeometry(rt.device, geometry, native);
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    rt.getBuildSizes(rt.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &native.info,
                     &native.primitiveCount, &sizes);
    return RtBuildSizes{sizes.accelerationStructureSize, sizes.buildScratchSize, sizes.updateScratchSize};
}

void WriteRtInstances(std::span<const RtInstance> instances, std::span<std::byte> out) {
    QL_ASSERT(out.size() >= instances.size() * kRtInstanceStride, "instance output is smaller than count x stride");
    for (size_t i = 0; i < instances.size(); ++i) {
        const RtInstance& instance = instances[i];
        VkAccelerationStructureInstanceKHR native{};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                native.transform.matrix[row][col] = instance.transform[row][col];
            }
        }
        native.instanceCustomIndex = instance.userId;
        native.mask = instance.mask;
        native.instanceShaderBindingTableRecordOffset = 0;  // the one hit group
        if (instance.cullDisable) {
            native.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        } else {
            native.flags = 0;
        }
        native.accelerationStructureReference = instance.bottomLevel.address;
        std::memcpy(out.data() + i * kRtInstanceStride, &native, sizeof(native));
    }
}

// ----------------------------------------------------------------------------
// VulkanRtAccelerationStructure
// ----------------------------------------------------------------------------

Result<std::unique_ptr<RtAccelerationStructure>, String> VulkanRtAccelerationStructure::Create(
    VmaAllocator allocator, const RtFunctions& rt, RtLevel level, u64 bytes) {
    using Out = std::unique_ptr<RtAccelerationStructure>;
    if (rt.createStructure == nullptr || rt.destroyStructure == nullptr || rt.getStructureAddress == nullptr) {
        return Fail<Out>("Failed to load acceleration structure functions");
    }

    std::unique_ptr<VulkanRtAccelerationStructure> structure;
    try {
        structure.reset(new VulkanRtAccelerationStructure(
            GpuBuffer(allocator, bytes,
                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                      VMA_MEMORY_USAGE_GPU_ONLY),
            level));
    } catch (const std::exception& e) {
        return Fail<Out>(e.what());
    }

    VkAccelerationStructureCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    info.buffer = structure->storage_.GetHandle();
    info.size = bytes;
    info.type = StructureType(level);
    if (rt.createStructure(rt.device, &info, nullptr, &structure->structure_) != VK_SUCCESS) {
        structure->structure_ = VK_NULL_HANDLE;
        return Fail<Out>(level == RtLevel::Top ? "Failed to create TLAS" : "Failed to create BLAS");
    }
    structure->device_ = rt.device;
    structure->destroy_ = rt.destroyStructure;

    // Only a Bottom level's address is ever named (by an instance), so only it is queried.
    if (level == RtLevel::Bottom) {
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = structure->structure_;
        structure->address_ = rt.getStructureAddress(rt.device, &addressInfo);
    }
    return Out(std::move(structure));
}

VulkanRtAccelerationStructure::~VulkanRtAccelerationStructure() {
    if (structure_ != VK_NULL_HANDLE && destroy_ != nullptr) destroy_(device_, structure_, nullptr);
}

}  // namespace quantiloom::gpu::vulkan
