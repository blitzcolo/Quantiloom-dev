#include "support/VulkanTestDevice.hpp"

#include "renderer/VulkanContext.hpp"

#include <exception>
#include <memory>

namespace quantiloom::testing {

namespace {

std::unique_ptr<VulkanContext> g_device;
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

void ReleaseSharedVulkanDevice() {
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

} // namespace quantiloom::testing
