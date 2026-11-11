// The Vulkan entry points of gpu/DeviceFactory.hpp in a build without gpu/vulkan/ (a Mac
// build, see QUANTILOOM_BACKEND_VULKAN): every request is refused with an Err.

#include "gpu/DeviceFactory.hpp"

#include "core/ResultFail.hpp"

namespace quantiloom::gpu::vulkan {

namespace {
constexpr const char* kNoVulkanBackend = "this SDK build has no Vulkan backend";
}  // namespace

bool IsCompiled() {
    return false;
}

Result<std::unique_ptr<Device>, String> CreateSystemDevice(const SystemDeviceOptions&) {
    return Fail<std::unique_ptr<Device>>(kNoVulkanBackend);
}

Result<std::unique_ptr<Device>, String> CreateDeviceFromHandles(const GpuDeviceHandles&) {
    return Fail<std::unique_ptr<Device>>(kNoVulkanBackend);
}

}  // namespace quantiloom::gpu::vulkan
