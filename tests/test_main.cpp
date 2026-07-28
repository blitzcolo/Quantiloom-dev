// ============================================================================
// Quantiloom - Test Main Entry Point
// ============================================================================
// This file provides the main() function for Google Test and initializes
// the Quantiloom logging system before running tests.
// ============================================================================

#include <gtest/gtest.h>
#include "core/Log.hpp"
#include "support/VulkanTestDevice.hpp"

using namespace quantiloom;

// ============================================================================
// Global Test Environment
// ============================================================================
// This environment is set up once before all tests and torn down after all tests.

class QuantiloomTestEnvironment final : public ::testing::Environment {
public:
    void SetUp() override {
        // Initialize logging system for tests
        // Use nullptr to disable file logging (console only)
        Log::Init(nullptr, Log::Level::Warn);  // Only show warnings and errors during tests
    }

    void TearDown() override {
        // Shutdown logging system
        Log::Shutdown();
    }
};

// Owns nothing at start-up: the shared Vulkan device is created on first use, so a
// run that touches no GPU test never pays for it. This exists only to destroy it
// while the logger is still alive -- VulkanContext's destructor logs, and gtest tears
// environments down in reverse registration order, so registering this one after
// QuantiloomTestEnvironment puts it ahead of Log::Shutdown.
class VulkanDeviceEnvironment final : public ::testing::Environment {
public:
    void TearDown() override {
        ::quantiloom::testing::ReleaseSharedVulkanDevice();
    }
};

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Register global test environments. Order matters: teardown runs in reverse,
    // so the Vulkan device is destroyed while the logger it writes to still exists.
    ::testing::AddGlobalTestEnvironment(new QuantiloomTestEnvironment());
    ::testing::AddGlobalTestEnvironment(new VulkanDeviceEnvironment());

    // Run all tests
    return RUN_ALL_TESTS();
}
