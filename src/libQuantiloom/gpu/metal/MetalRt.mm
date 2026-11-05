#include "gpu/metal/MetalBackend.hpp"

#include "core/Log.hpp"
#include "core/ResultFail.hpp"

#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

#include <cstring>

namespace quantiloom::gpu::metal {

MetalRtAccelerationStructure::MetalRtAccelerationStructure(MetalDevice& device, id<MTLAccelerationStructure> structure,
                                                           RtLevel level)
    : device_(device), structure_(structure), level_(level) {
    device_.RegisterAccelerationStructure(this);
    if (level_ == RtLevel::Top) PrepareHeader(1);
}

MetalRtAccelerationStructure::~MetalRtAccelerationStructure() {
    device_.UnregisterAccelerationStructure(this);
}

RtSceneRef MetalRtAccelerationStructure::Ref() const {
    return RtSceneRef{ToHandle(structure_), [structure_ gpuResourceID]._impl};
}

void MetalRtAccelerationStructure::PrepareHeader(u32 instances) {
    if (header_ != nil && instances <= headerCapacity_) return;
    // The header, then the instance contributions it points at. Inline ray queries never
    // read a contribution to a hit-group index, so they stay zero.
    const u32 capacity = std::max(instances, 1u);
    const u64 headerBytes = sizeof(IRRaytracingAccelerationStructureGPUHeader);
    const u64 bytes = headerBytes + sizeof(u32) * capacity;
    id<MTLBuffer> header = [device_.NativeDevice() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    std::memset([header contents], 0, bytes);
    const Vector<u32> contributions(capacity, 0);
    auto* base = static_cast<uint8_t*>([header contents]);
    IRRaytracingSetAccelerationStructure(base, [structure_ gpuResourceID], base + headerBytes,
                                         [header gpuAddress] + headerBytes, contributions.data(), capacity);
    header_ = header;
    headerCapacity_ = capacity;
}

MTLAccelerationStructureDescriptor* MetalDevice::DescribeGeometry(const RtGeometry& geometry) const {
    const bool refit = HasAny(geometry.flags, RtBuildFlags::AllowUpdate);
    if (geometry.level == RtLevel::Bottom) {
        const RtTriangles& t = geometry.triangles;
        MTLAccelerationStructureTriangleGeometryDescriptor* triangles =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        triangles.vertexBuffer = FromHandle<id<MTLBuffer>>(t.vertices.handle);
        triangles.vertexBufferOffset = t.vertexOffset;
        triangles.vertexStride = 12;
        triangles.vertexFormat = MTLAttributeFormatFloat3;
        triangles.indexBuffer = FromHandle<id<MTLBuffer>>(t.indices.handle);
        triangles.indexBufferOffset = t.indexOffset;
        triangles.indexType = MTLIndexTypeUInt32;
        triangles.triangleCount = t.triangleCount;
        triangles.opaque = t.opaque;
        MTLPrimitiveAccelerationStructureDescriptor* d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        d.geometryDescriptors = @[ triangles ];
        if (refit) d.usage = MTLAccelerationStructureUsageRefit;
        return d;
    }
    // Instances name their bottom level by resource ID (indirect descriptors), so the
    // descriptor needs no list of bottom levels: they are resident through the device's
    // residency set.
    const RtInstances& i = geometry.instances;
    MTLInstanceAccelerationStructureDescriptor* d = [MTLInstanceAccelerationStructureDescriptor descriptor];
    d.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeIndirect;
    d.instanceCount = i.count;
    if (i.buffer.IsValid()) {
        d.instanceDescriptorBuffer = FromHandle<id<MTLBuffer>>(i.buffer.handle);
        d.instanceDescriptorBufferOffset = i.offset;
    }
    d.instanceDescriptorStride = sizeof(MTLIndirectAccelerationStructureInstanceDescriptor);
    if (refit) d.usage = MTLAccelerationStructureUsageRefit;
    return d;
}

RtBuildSizes MetalDevice::GetRtBuildSizes(const RtGeometry& geometry) const {
    if (!info_.hardwareRayTracing) return {};
    const MTLAccelerationStructureSizes sizes = [device_ accelerationStructureSizesWithDescriptor:DescribeGeometry(geometry)];
    return RtBuildSizes{sizes.accelerationStructureSize, sizes.buildScratchBufferSize, sizes.refitScratchBufferSize};
}

Result<std::unique_ptr<RtAccelerationStructure>, String> MetalDevice::CreateRtAccelerationStructure(RtLevel level,
                                                                                                    u64 bytes) {
    id<MTLAccelerationStructure> structure = [device_ newAccelerationStructureWithSize:bytes];
    if (structure == nil) {
        return Fail<std::unique_ptr<RtAccelerationStructure>>("Failed to create acceleration structure of " +
                                                              std::to_string(bytes) + " bytes");
    }
    return std::unique_ptr<RtAccelerationStructure>(
        std::make_unique<MetalRtAccelerationStructure>(*this, structure, level));
}

u32 MetalDevice::RtInstanceStride() const {
    return static_cast<u32>(sizeof(MTLIndirectAccelerationStructureInstanceDescriptor));
}

void MetalDevice::PackRtInstances(std::span<const RtInstance> instances, std::span<std::byte> out) const {
    const u32 stride = RtInstanceStride();
    if (out.size() < instances.size() * stride) {
        QL_LOG_ERROR("Metal: instance output holds {} bytes, {} instances need {}", out.size(), instances.size(),
                     instances.size() * stride);
        return;
    }
    for (size_t n = 0; n < instances.size(); ++n) {
        const RtInstance& instance = instances[n];
        MTLIndirectAccelerationStructureInstanceDescriptor native{};
        // RtInstance is a row-major 3x4; MTLPackedFloat4x3 is four packed columns.
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 3; ++row) {
                native.transformationMatrix.columns[column].elements[row] = instance.transform[row][column];
            }
        }
        native.options = instance.cullDisable ? MTLAccelerationStructureInstanceOptionDisableTriangleCulling
                                              : MTLAccelerationStructureInstanceOptionNone;
        native.mask = instance.mask;
        native.intersectionFunctionTableOffset = 0;
        native.userID = instance.userId;
        native.accelerationStructureID = MTLResourceID{instance.bottomLevel.address};
        std::memcpy(out.data() + n * stride, &native, sizeof(native));
    }
}

}  // namespace quantiloom::gpu::metal
