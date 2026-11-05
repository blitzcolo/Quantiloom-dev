// The Vulkan side of the support files in a build without gpu/vulkan/. Nothing reaches
// these: the GPU suites are instantiated over gpu::CompiledBackends(), which has no
// Vulkan here, so every switch case that would call one is dead.

#include "support/VulkanTestDevice.hpp"

#include "gpu/GpuTypes.hpp"

namespace quantiloom::testing {

namespace {
const std::string kNoVulkan = "this build has no Vulkan backend";
}  // namespace

VulkanContext* SharedVulkanDevice() {
    return nullptr;
}

const std::string& VulkanDeviceUnavailableReason() {
    return kNoVulkan;
}

GpuDeviceHandles VulkanHostHandles(gpu::Device&) {
    ADD_FAILURE() << kNoVulkan;
    return {};
}

GpuCommandRecording VulkanHostRecording(gpu::CommandList&) {
    ADD_FAILURE() << kNoVulkan;
    return {};
}

u32 VulkanTextureMipLevels(const gpu::Texture&) {
    ADD_FAILURE() << kNoVulkan;
    return 0;
}

f32 VulkanPackedInstanceTranslationX(const void*, u32) {
    ADD_FAILURE() << kNoVulkan;
    return 0.0f;
}

}  // namespace quantiloom::testing
