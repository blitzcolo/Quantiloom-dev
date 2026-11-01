#include "gpu/DeviceFactory.hpp"

#include "core/ResultFail.hpp"
#include "renderer/GpuInterop.hpp"

#include <string>

namespace quantiloom::gpu {

namespace {
String UnknownBackend(u32 value) {
    return "unknown GPU backend " + std::to_string(value);
}
}  // namespace

Result<std::unique_ptr<Device>, String> CreateSystemDevice(Backend backend, const SystemDeviceOptions& options) {
    switch (backend) {
        case Backend::Vulkan: return vulkan::CreateSystemDevice(options);
        case Backend::Metal: return metal::CreateSystemDevice(options);
    }
    return Fail<std::unique_ptr<Device>>(UnknownBackend(static_cast<u32>(backend)));
}

Result<std::unique_ptr<Device>, String> CreateDeviceFromHandles(const GpuDeviceHandles& handles) {
    switch (handles.backend) {
        case GpuBackend::Vulkan: return vulkan::CreateDeviceFromHandles(handles);
        case GpuBackend::Metal: return metal::CreateDeviceFromHandles(handles);
    }
    return Fail<std::unique_ptr<Device>>(UnknownBackend(static_cast<u32>(handles.backend)));
}

std::span<const Backend> CompiledBackends() {
    static const Vector<Backend> compiled = [] {
        Vector<Backend> backends;
        if (vulkan::IsCompiled()) backends.push_back(Backend::Vulkan);
        if (metal::IsCompiled()) backends.push_back(Backend::Metal);
        return backends;
    }();
    return compiled;
}

StringView BackendName(Backend backend) {
    switch (backend) {
        case Backend::Vulkan: return "Vulkan";
        case Backend::Metal: return "Metal";
    }
    return "unknown";
}

}  // namespace quantiloom::gpu
