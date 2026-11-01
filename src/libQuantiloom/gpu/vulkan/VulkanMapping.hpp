#pragma once

#include "gpu/Binding.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/GpuTypes.hpp"
#include "gpu/Resources.hpp"
#include "renderer/GpuBuffer.hpp"

#include <vulkan/vulkan.h>

#include <type_traits>

namespace quantiloom::gpu::vulkan {

struct AccessScope {
    VkPipelineStageFlags stages = 0;
    VkAccessFlags access = 0;
};

[[nodiscard]] AccessScope ScopeOf(Access bits, bool asSource);
[[nodiscard]] VkImageLayout LayoutOf(TextureLayoutPolicy policy, Access bits, bool discard);
/// The stage after which a timestamp is written or an event is set or reset.
[[nodiscard]] VkPipelineStageFlagBits TimestampStage(Stage stage);
[[nodiscard]] VkPipelineStageFlags ToVk(PipelineStage stages);
[[nodiscard]] VkShaderStageFlags ToVk(ShaderStages stages);
[[nodiscard]] VkDescriptorType ToDescriptorType(BindingKind kind);
[[nodiscard]] VkFormat ToVk(Format format);
[[nodiscard]] Format FromVk(VkFormat format);
[[nodiscard]] VkBufferUsageFlags ToVk(BufferUsage usage);
[[nodiscard]] VkImageUsageFlags ToVk(TextureUsage usage);
[[nodiscard]] VmaMemoryUsage ToVma(MemoryClass memory);
[[nodiscard]] VkFilter ToVk(Filter filter);
/// sType set; pNext, flags and every field the desc has no member for are zero.
[[nodiscard]] VkSamplerCreateInfo ToVk(const SamplerDesc& desc);
/// The Err text of a failed Vulkan call in gpu/Sync.hpp's convention: the VkResult as a decimal.
[[nodiscard]] String ResultCode(VkResult result);

template <class T>
[[nodiscard]] NativeHandle ToHandle(T handle) {
    if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<NativeHandle>(handle);
    } else {
        return static_cast<NativeHandle>(handle);
    }
}

template <class T>
[[nodiscard]] T FromHandle(NativeHandle handle) {
    if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<T>(handle);
    } else {
        return static_cast<T>(handle);
    }
}

}  // namespace quantiloom::gpu::vulkan
