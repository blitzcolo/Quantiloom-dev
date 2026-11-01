#include "support/VulkanTestDevice.hpp"

#include "gpu/vulkan/VulkanDevice.hpp"
#include "gpu/vulkan/VulkanContext.hpp"

#include <exception>
#include <memory>

namespace quantiloom::testing {

namespace {

std::unique_ptr<VulkanContext> g_device;
std::unique_ptr<gpu::Device> g_gpuDevice;
std::string g_reason;
bool g_attempted = false;

} // namespace

VulkanContext* SharedVulkanDevice() {
    if (g_attempted) {
        return g_device.get();
    }
    g_attempted = true;

    // VulkanContext throws rather than aborting when it cannot find a loader, a
    // physical device, or the ray tracing extensions, so every one of those cases
    // turns into a skip with the reason attached.
    try {
        g_device = std::make_unique<VulkanContext>();
    } catch (const std::exception& e) {
        g_reason = std::string("no Vulkan device for tests: ") + e.what();
        g_device.reset();
        return nullptr;
    }

    // The constructor is documented to require ray tracing, but assert rather than
    // assume: a context without it would fail every case that used it, one confusing
    // failure at a time, instead of skipping once with a reason.
    if (!g_device->IsRayTracingSupported()) {
        g_reason = "Vulkan device has no ray tracing support";
        g_device.reset();
        return nullptr;
    }

    return g_device.get();
}

const std::string& VulkanDeviceUnavailableReason() {
    return g_reason;
}

gpu::Device* SharedGpuDevice() {
    if (!g_gpuDevice && SharedVulkanDevice() != nullptr) {
        g_gpuDevice = std::make_unique<gpu::vulkan::VulkanDevice>(*g_device);
    }
    return g_gpuDevice.get();
}

void ReleaseSharedVulkanDevice() {
    g_gpuDevice.reset();
    g_device.reset();
}

void VulkanDeviceTest::SetUp() {
    if (SharedVulkanDevice() == nullptr) {
        GTEST_SKIP() << VulkanDeviceUnavailableReason();
    }
}

VulkanContext& VulkanDeviceTest::Device() const {
    return *SharedVulkanDevice();
}

gpu::Device& VulkanDeviceTest::GpuDevice() const {
    return *SharedGpuDevice();
}

} // namespace quantiloom::testing
