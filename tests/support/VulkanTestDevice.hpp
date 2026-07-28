/**
 * @file VulkanTestDevice.hpp
 * @brief One headless Vulkan device, shared by every test that needs a GPU
 *
 * The suite was entirely CPU-side until now: not one test created a VkDevice. That
 * is why the render path had no cover, and why a bug like the FPN maps coming out
 * 0x0 could reach a running build -- the only thing that exercised it was a person
 * launching the GUI.
 *
 * Two rules keep the cost of changing that low:
 *
 *   - **Created once, on first use.** Device creation costs a few hundred
 *     milliseconds against a suite that runs in two seconds, and a filtered run that
 *     touches no GPU test never pays it.
 *   - **Absence skips, it never fails.** A machine without a ray-tracing GPU is not
 *     a broken build. Tests derive from VulkanDeviceTest and get that for free; the
 *     skip message says which stage of device creation was unavailable.
 *
 * Not thread-safe: GoogleTest runs cases serially here.
 */

#pragma once

#include <gtest/gtest.h>

#include <string>

namespace quantiloom {
class VulkanContext;
}

namespace quantiloom::testing {

/// The shared headless context, or nullptr when this machine cannot provide one.
/// Creates it on the first call; a failed attempt is not retried.
VulkanContext* SharedVulkanDevice();

/// Why SharedVulkanDevice() returned nullptr. Empty before the first attempt.
const std::string& VulkanDeviceUnavailableReason();

/// Destroy the shared context. Called from the test main's environment teardown,
/// which runs before logging shuts down -- the destructor logs.
void ReleaseSharedVulkanDevice();

/**
 * @brief Base for tests that need a GPU
 *
 * Skips the case when no device is available instead of failing it.
 *
 * @code
 * TEST_F(VulkanDeviceTest, BuildsABottomLevelStructure) {
 *     GpuBuffer buffer(Device().GetAllocator(), ...);
 * }
 * @endcode
 */
class VulkanDeviceTest : public ::testing::Test {
protected:
    void SetUp() override;

    /// Valid for the body of any case that reached it -- SetUp skipped otherwise.
    [[nodiscard]] VulkanContext& Device() const;
};

} // namespace quantiloom::testing
