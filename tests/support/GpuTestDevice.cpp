#include "support/GpuTestDevice.hpp"

#include "gpu/Device.hpp"
#include "renderer/GpuInterop.hpp"
#include "support/GpuInteropTestUtils.hpp"
#include "support/MetalTestDevice.hpp"
#include "support/VulkanTestDevice.hpp"

#include <array>
#include <memory>

namespace quantiloom::testing {

namespace {

struct SharedDevice {
    std::unique_ptr<gpu::Device> device;
    std::string reason;
    bool attempted = false;
};

// Indexed by the Backend value (Vulkan = 1, Metal = 2).
std::array<SharedDevice, 3> g_devices;

SharedDevice& Slot(gpu::Backend backend) {
    return g_devices[static_cast<size_t>(backend)];
}

static_assert(static_cast<u32>(gpu::Backend::Vulkan) == static_cast<u32>(GpuBackend::Vulkan) &&
              static_cast<u32>(gpu::Backend::Metal) == static_cast<u32>(GpuBackend::Metal));

}  // namespace

gpu::Device* SharedGpuDevice(gpu::Backend backend) {
    SharedDevice& shared = Slot(backend);
    if (shared.attempted) {
        return shared.device.get();
    }
    shared.attempted = true;
    const std::string name(gpu::BackendName(backend));

    // The backend reports a missing loader, physical device or ray tracing extension
    // as an Err rather than aborting, so every one of those cases turns into a skip
    // with the reason attached.
    auto created = gpu::CreateSystemDevice(backend);
    if (!created.has_value()) {
        shared.reason = "no " + name + " device for tests: " + created.error();
        return nullptr;
    }
    shared.device = std::move(created.value());

    // Assert rather than assume ray tracing: a device without it would fail every case
    // that used it, one confusing failure at a time, instead of skipping once with a reason.
    if (!shared.device->Info().hardwareRayTracing) {
        shared.reason = name + " device has no ray tracing support";
        shared.device.reset();
        return nullptr;
    }

    return shared.device.get();
}

const std::string& GpuDeviceUnavailableReason(gpu::Backend backend) {
    return Slot(backend).reason;
}

void ReleaseSharedGpuDevices() {
    for (SharedDevice& shared : g_devices) {
        shared.device.reset();
    }
}

void GpuDeviceTest::SetUp() {
    if (SharedGpuDevice(Backend()) == nullptr) {
        GTEST_SKIP() << GpuDeviceUnavailableReason(Backend());
    }
}

gpu::Device& GpuDeviceTest::Gpu() const {
    return *SharedGpuDevice(Backend());
}

std::string BackendParamName(const ::testing::TestParamInfo<gpu::Backend>& info) {
    return std::string(gpu::BackendName(info.param));
}

// The TEST_P cases on GpuDeviceTest itself live in several files; the suite is
// instantiated here, once.
QL_INSTANTIATE_GPU_SUITE(GpuDeviceTest);

GpuDeviceHandles GpuHandles(gpu::Device& device) {
    switch (device.Info().backend) {
        case gpu::Backend::Vulkan: return VulkanHostHandles(device);
        case gpu::Backend::Metal: return MetalHostHandles(device);
    }
    ADD_FAILURE() << "no host handles for the " << gpu::BackendName(device.Info().backend) << " test device";
    return {};
}

GpuCommandRecording GpuRecording(gpu::CommandList& list) {
    switch (list.GetBackend()) {
        case gpu::Backend::Vulkan: return VulkanHostRecording(list);
        case gpu::Backend::Metal: return MetalHostRecording(list);
    }
    ADD_FAILURE() << "no host recording for a " << gpu::BackendName(list.GetBackend()) << " command list";
    return {};
}

GpuImageTarget GpuTarget(gpu::Backend backend, gpu::TextureRef texture, GpuTargetFormat format,
                         u32 currentLayout, u32 finalLayout) {
    GpuImageTarget target;
    target.backend = static_cast<GpuBackend>(static_cast<u32>(backend));
    target.image = texture.image;
    target.format = format;
    target.width = texture.width;
    target.height = texture.height;
    target.currentLayout = currentLayout;
    target.finalLayout = finalLayout;
    return target;
}

}  // namespace quantiloom::testing
