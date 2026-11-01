/**
 * @file GpuTestDevice.hpp
 * @brief One shared gpu::Device per backend, and the fixture every GPU test derives from
 *
 * A GPU test is written once against gpu::Device and runs on every backend this build
 * compiles: the fixture is value-parameterized over gpu::CompiledBackends(), so a case
 * is named `Gpu/<Fixture>.<Case>/Vulkan` today and gains a `/Metal` twin when that
 * backend is compiled in.
 *
 * The rules VulkanTestDevice.hpp kept still hold, per backend:
 *
 *   - **Created once, on first use.** Device creation costs a few hundred
 *     milliseconds against a suite that runs in seconds, and a filtered run that
 *     touches no GPU test never pays it.
 *   - **Absence skips, it never fails.** A machine without a ray-tracing GPU is not
 *     a broken build. Each case skips with the reason the device could not be made.
 *     The suites are instantiated over the *compiled* backends, not the available
 *     ones, so every case exists on every machine and the totals do not depend on
 *     the hardware.
 *
 * Not thread-safe: GoogleTest runs cases serially here.
 */

#pragma once

#include <gtest/gtest.h>

#include "gpu/DeviceFactory.hpp"
#include "gpu/GpuTypes.hpp"

#include <string>

namespace quantiloom::testing {

/// The shared device of @p backend, or nullptr when this machine cannot provide one.
/// Created on the first call; a failed attempt is not retried.
gpu::Device* SharedGpuDevice(gpu::Backend backend);

/// Why SharedGpuDevice(@p backend) returned nullptr. Empty before the first attempt.
const std::string& GpuDeviceUnavailableReason(gpu::Backend backend);

/// Destroy every shared device. Called from the test main's environment teardown,
/// which runs before logging shuts down -- the backends' destructors log.
void ReleaseSharedGpuDevices();

/**
 * @brief Base for tests that need a GPU, parameterized over the compiled backends
 *
 * Skips the case when the backend's device is unavailable instead of failing it.
 *
 * @code
 * TEST_P(GpuDeviceTest, BuildsABottomLevelStructure) {
 *     auto buffer = Gpu().CreateBuffer(...);
 * }
 * @endcode
 *
 * A derived fixture is instantiated once, with QL_INSTANTIATE_GPU_SUITE(Fixture), in
 * the file that defines it. GpuDeviceTest itself is instantiated in GpuTestDevice.cpp:
 * a suite may be instantiated in one translation unit only.
 */
class GpuDeviceTest : public ::testing::TestWithParam<gpu::Backend> {
protected:
    void SetUp() override;

    /// Valid for the body of any case that reached it -- SetUp skipped otherwise.
    [[nodiscard]] gpu::Device& Gpu() const;

    [[nodiscard]] gpu::Backend Backend() const { return GetParam(); }
};

/// The instantiation's name for a backend: "Vulkan" or "Metal".
std::string BackendParamName(const ::testing::TestParamInfo<gpu::Backend>& info);

}  // namespace quantiloom::testing

/// Instantiates a GpuDeviceTest-derived fixture over every compiled backend, as
/// `Gpu/<Fixture>.<Case>/<Backend>`. Use the fixture's unqualified name.
#define QL_INSTANTIATE_GPU_SUITE(Fixture)                                        \
    INSTANTIATE_TEST_SUITE_P(Gpu, Fixture,                                       \
                             ::testing::ValuesIn(::quantiloom::gpu::CompiledBackends()), \
                             ::quantiloom::testing::BackendParamName)
