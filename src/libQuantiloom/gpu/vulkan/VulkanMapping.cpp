#include "gpu/vulkan/VulkanMapping.hpp"

namespace quantiloom::gpu::vulkan {

namespace {

struct BitScope {
    Access bit;
    VkPipelineStageFlags stages;
    VkAccessFlags access;
};

constexpr BitScope kScopes[] = {
    {Access::ComputeRead, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT},
    {Access::ComputeWrite, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT},
    {Access::RayTraceRead, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_READ_BIT},
    {Access::RayTraceWrite, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_WRITE_BIT},
    {Access::TransferRead, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT},
    {Access::TransferWrite, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT},
    {Access::HostRead, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT},
    {Access::HostWrite, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT},
    {Access::AnyRead, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT},
    {Access::AnyWrite, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT},
    {Access::AccelerationStructureRead, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
     VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
    {Access::AccelerationStructureWrite, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
     VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR},
};

}  // namespace

AccessScope ScopeOf(Access bits, bool asSource) {
    if (bits == Access::None) {
        return {asSource ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0};
    }
    AccessScope scope;
    for (const BitScope& entry : kScopes) {
        if (HasAny(bits, entry.bit)) {
            scope.stages |= entry.stages;
            scope.access |= entry.access;
        }
    }
    return scope;
}

VkImageLayout LayoutOf(TextureLayoutPolicy policy, Access bits, bool discard) {
    if (discard) return VK_IMAGE_LAYOUT_UNDEFINED;
    if (policy == TextureLayoutPolicy::General) return VK_IMAGE_LAYOUT_GENERAL;
    constexpr Access kShader = Access::ComputeRead | Access::RayTraceRead;
    constexpr Access kXferRead = Access::TransferRead;
    constexpr Access kXferWrite = Access::TransferWrite;
    const auto only = [bits](Access mask) {
        return bits != Access::None && (bits & mask) == bits;
    };
    if (only(kShader)) return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (only(kXferRead)) return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    if (only(kXferWrite)) return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    return VK_IMAGE_LAYOUT_GENERAL;
}

VkPipelineStageFlagBits TimestampStage(Stage stage) {
    switch (stage) {
        case Stage::Compute: return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        case Stage::RayTracing: return VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        case Stage::Transfer: return VK_PIPELINE_STAGE_TRANSFER_BIT;
        case Stage::AllCommands: return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
}

VkDescriptorType ToDescriptorType(BindingKind kind) {
    switch (kind) {
        case BindingKind::ReadOnlyBuffer:
        case BindingKind::ReadWriteBuffer: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case BindingKind::StorageTexture: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case BindingKind::SampledTexture: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case BindingKind::Sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
        case BindingKind::ConstantBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case BindingKind::AccelerationStructure: return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    }
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

VkFormat ToVk(Format format) {
    switch (format) {
        case Format::Undefined: return VK_FORMAT_UNDEFINED;
        case Format::R32Float: return VK_FORMAT_R32_SFLOAT;
        case Format::RGBA32Float: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case Format::RGBA32Uint: return VK_FORMAT_R32G32B32A32_UINT;
        case Format::RGBA8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::RGBA8Srgb: return VK_FORMAT_R8G8B8A8_SRGB;
        case Format::BGRA8Srgb: return VK_FORMAT_B8G8R8A8_SRGB;
    }
    return VK_FORMAT_UNDEFINED;
}

Format FromVk(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R32_SFLOAT: return Format::R32Float;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return Format::RGBA32Float;
        case VK_FORMAT_R32G32B32A32_UINT: return Format::RGBA32Uint;
        case VK_FORMAT_R8G8B8A8_UNORM: return Format::RGBA8Unorm;
        case VK_FORMAT_R8G8B8A8_SRGB: return Format::RGBA8Srgb;
        case VK_FORMAT_B8G8R8A8_SRGB: return Format::BGRA8Srgb;
        default: return Format::Undefined;
    }
}

VkBufferUsageFlags ToVk(BufferUsage usage) {
    VkBufferUsageFlags flags = 0;
    if (HasAny(usage, BufferUsage::Storage)) flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (HasAny(usage, BufferUsage::Uniform)) flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (HasAny(usage, BufferUsage::TransferSrc)) flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (HasAny(usage, BufferUsage::TransferDst)) flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (HasAny(usage, BufferUsage::DeviceAddress)) flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (HasAny(usage, BufferUsage::AccelerationStructureInput)) {
        flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }
    return flags;
}

VkImageUsageFlags ToVk(TextureUsage usage) {
    VkImageUsageFlags flags = 0;
    if (HasAny(usage, TextureUsage::Storage)) flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (HasAny(usage, TextureUsage::Sampled)) flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (HasAny(usage, TextureUsage::TransferSrc)) flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (HasAny(usage, TextureUsage::TransferDst)) flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return flags;
}

VmaMemoryUsage ToVma(MemoryClass memory) {
    switch (memory) {
        case MemoryClass::DeviceLocal: return VMA_MEMORY_USAGE_GPU_ONLY;
        case MemoryClass::Upload: return VMA_MEMORY_USAGE_CPU_TO_GPU;
        case MemoryClass::Readback: return VMA_MEMORY_USAGE_GPU_TO_CPU;
        case MemoryClass::HostOnly: return VMA_MEMORY_USAGE_CPU_ONLY;
    }
    return VMA_MEMORY_USAGE_GPU_ONLY;
}

}  // namespace quantiloom::gpu::vulkan
