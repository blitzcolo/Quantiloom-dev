// ============================================================================
// Quantiloom - Test Main Entry Point
// ============================================================================
// This file provides the main() function for Google Test and initializes
// the Quantiloom logging system before running tests.
// ============================================================================

#include <gtest/gtest.h>
#include "core/Log.hpp"

using namespace quantiloom;

// ============================================================================
// Global Test Environment
// ============================================================================
// This environment is set up once before all tests and torn down after all tests.

class QuantiloomTestEnvironment : public ::testing::Environment {
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

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Register global test environment
    ::testing::AddGlobalTestEnvironment(new QuantiloomTestEnvironment());

    // Run all tests
    return RUN_ALL_TESTS();
}
