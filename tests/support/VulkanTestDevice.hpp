/**
 * @file VulkanTestDevice.hpp
 * @brief The Vulkan view of the shared test device, for the Vulkan-only tests
 *
 * Every GPU test that can run on any backend derives from GpuDeviceTest
 * (GpuTestDevice.hpp). What remains here is for the few that are about Vulkan itself
 * -- VMA round trips, shader binding table alignment -- and for the Vulkan side of the
 * host-interop builders. There is still one device per process: SharedVulkanDevice()
 * is the context of SharedGpuDevice(gpu::Backend::Vulkan), not a second device.
 *
 * Not thread-safe: GoogleTest runs cases serially here.
 */

#pragma once

#include <gtest/gtest.h>

#include "renderer/GpuInterop.hpp"

#include <string>

namespace quantiloom {
class VulkanContext;
}

namespace quantiloom::gpu {
class CommandList;
class Device;
class Texture;
}  // namespace quantiloom::gpu

namespace quantiloom::testing {

/// The context of the shared Vulkan device, or nullptr when this machine cannot provide one.
VulkanContext* SharedVulkanDevice();

/// Why SharedVulkanDevice() returned nullptr. Empty before the first attempt.
const std::string& VulkanDeviceUnavailableReason();

/// GpuHandles and GpuRecording (GpuInteropTestUtils.hpp) for a Vulkan device or list.
GpuDeviceHandles VulkanHostHandles(gpu::Device& device);
GpuCommandRecording VulkanHostRecording(gpu::CommandList& list);

/// TextureMipLevels and PackedInstanceTranslationX (GpuTestHelpers.hpp) on Vulkan.
u32 VulkanTextureMipLevels(const gpu::Texture& texture);
f32 VulkanPackedInstanceTranslationX(const void* packed, u32 index);

/**
 * @brief Base for the Vulkan-only tests
 *
 * Skips the case when no device is available instead of failing it.
 *
 * @code
 * TEST_F(VulkanDeviceTest, RoundTripsHostVisibleMemory) {
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
