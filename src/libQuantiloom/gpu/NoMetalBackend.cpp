// The Metal entry points of gpu/DeviceFactory.hpp in a build without gpu/metal/: every
// request is refused with the text interop/MetalInterop.hpp promises.

#include "gpu/DeviceFactory.hpp"

#include "core/ResultFail.hpp"

namespace quantiloom::gpu::metal {

namespace {
constexpr const char* kNoMetalBackend = "this SDK build has no Metal backend";
}  // namespace

bool IsCompiled() {
    return false;
}

Result<std::unique_ptr<Device>, String> CreateSystemDevice(const SystemDeviceOptions&) {
    return Fail<std::unique_ptr<Device>>(kNoMetalBackend);
}

Result<std::unique_ptr<Device>, String> CreateDeviceFromHandles(const GpuDeviceHandles&) {
    return Fail<std::unique_ptr<Device>>(kNoMetalBackend);
}

}  // namespace quantiloom::gpu::metal
