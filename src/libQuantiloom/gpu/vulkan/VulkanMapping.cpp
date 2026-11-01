#include "gpu/vulkan/VulkanMapping.hpp"

#include <string>

namespace quantiloom::gpu::vulkan {

static_assert(kRemaining == VK_REMAINING_MIP_LEVELS && kRemaining == VK_REMAINING_ARRAY_LAYERS);
static_assert(kLodClampNone == VK_LOD_CLAMP_NONE);

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
    {Access::RayTraceSceneRead, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
     VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
    {Access::ComputeSceneRead, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
    {Access::FragmentRead, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT},
    {Access::UniformRead, 0, VK_ACCESS_UNIFORM_READ_BIT},
};

struct StageBit {
    PipelineStage bit;
    VkPipelineStageFlags native;
};

constexpr StageBit kStages[] = {
    {PipelineStage::TopOfPipe, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
    {PipelineStage::Compute, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT},
    {PipelineStage::RayTracing, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR},
    {PipelineStage::Transfer, VK_PIPELINE_STAGE_TRANSFER_BIT},
    {PipelineStage::Host, VK_PIPELINE_STAGE_HOST_BIT},
    {PipelineStage::AccelerationStructureBuild, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR},
    {PipelineStage::Fragment, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
    {PipelineStage::BottomOfPipe, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT},
    {PipelineStage::AllCommands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
};

struct ShaderStageBit {
    ShaderStages bit;
    VkShaderStageFlags native;
};

constexpr ShaderStageBit kShaderStages[] = {
    {ShaderStages::Compute, VK_SHADER_STAGE_COMPUTE_BIT},
    {ShaderStages::RayGen, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
    {ShaderStages::ClosestHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
    {ShaderStages::AnyHit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR},
    {ShaderStages::Miss, VK_SHADER_STAGE_MISS_BIT_KHR},
};

VkSamplerAddressMode ToVkAddressMode(AddressMode mode) {
    switch (mode) {
        case AddressMode::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case AddressMode::MirroredRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case AddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case AddressMode::ClampToBorder: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    }
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkCompareOp ToVkCompareOp(CompareOp op) {
    switch (op) {
        case CompareOp::Never: return VK_COMPARE_OP_NEVER;
        case CompareOp::Less: return VK_COMPARE_OP_LESS;
        case CompareOp::Equal: return VK_COMPARE_OP_EQUAL;
        case CompareOp::LessOrEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
        case CompareOp::Greater: return VK_COMPARE_OP_GREATER;
        case CompareOp::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
        case CompareOp::GreaterOrEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case CompareOp::Always: return VK_COMPARE_OP_ALWAYS;
    }
    return VK_COMPARE_OP_NEVER;
}

VkBorderColor ToVkBorderColor(BorderColor color) {
    switch (color) {
        case BorderColor::FloatTransparentBlack: return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        case BorderColor::IntTransparentBlack: return VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
        case BorderColor::FloatOpaqueBlack: return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        case BorderColor::IntOpaqueBlack: return VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        case BorderColor::FloatOpaqueWhite: return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        case BorderColor::IntOpaqueWhite: return VK_BORDER_COLOR_INT_OPAQUE_WHITE;
    }
    return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
}

}  // namespace

AccessScope ScopeOf(Access bits, bool asSource) {
    if (bits == Access::None) {
        AccessScope none;
        none.stages = asSource ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        return none;
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
    constexpr Access kShader = Access::ComputeRead | Access::RayTraceRead | Access::FragmentRead;
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
        case Stage::TopOfPipe: return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        case Stage::BottomOfPipe: return VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }
    return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
}

VkPipelineStageFlags ToVk(PipelineStage stages) {
    VkPipelineStageFlags flags = 0;
    for (const StageBit& entry : kStages) {
        if (HasAny(stages, entry.bit)) flags |= entry.native;
    }
    return flags;
}

VkShaderStageFlags ToVk(ShaderStages stages) {
    VkShaderStageFlags flags = 0;
    for (const ShaderStageBit& entry : kShaderStages) {
        if (HasAny(stages, entry.bit)) flags |= entry.native;
    }
    return flags;
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
        case Format::RG32Float: return VK_FORMAT_R32G32_SFLOAT;
        case Format::RGBA16Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case Format::BC7Unorm: return VK_FORMAT_BC7_UNORM_BLOCK;
        case Format::BC7Srgb: return VK_FORMAT_BC7_SRGB_BLOCK;
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
        case VK_FORMAT_R32G32_SFLOAT: return Format::RG32Float;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return Format::RGBA16Float;
        case VK_FORMAT_BC7_UNORM_BLOCK: return Format::BC7Unorm;
        case VK_FORMAT_BC7_SRGB_BLOCK: return Format::BC7Srgb;
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

VkFilter ToVk(Filter filter) {
    return filter == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

VkSamplerCreateInfo ToVk(const SamplerDesc& desc) {
    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = ToVk(desc.magFilter);
    info.minFilter = ToVk(desc.minFilter);
    info.mipmapMode =
        desc.mipmapMode == Filter::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = ToVkAddressMode(desc.addressModeU);
    info.addressModeV = ToVkAddressMode(desc.addressModeV);
    info.addressModeW = ToVkAddressMode(desc.addressModeW);
    info.mipLodBias = desc.mipLodBias;
    info.anisotropyEnable = desc.anisotropyEnable ? VK_TRUE : VK_FALSE;
    info.maxAnisotropy = desc.maxAnisotropy;
    info.compareEnable = desc.compareEnable ? VK_TRUE : VK_FALSE;
    info.compareOp = ToVkCompareOp(desc.compareOp);
    info.minLod = desc.minLod;
    info.maxLod = desc.maxLod;
    info.borderColor = ToVkBorderColor(desc.borderColor);
    info.unnormalizedCoordinates = desc.unnormalizedCoordinates ? VK_TRUE : VK_FALSE;
    return info;
}

String ResultCode(VkResult result) {
    return std::to_string(static_cast<i32>(result));
}

}  // namespace quantiloom::gpu::vulkan
