// The Metal entry points of gpu/DeviceFactory.hpp.

#include "gpu/DeviceFactory.hpp"
#include "gpu/metal/MetalBackend.hpp"

#include "core/ResultFail.hpp"
#include "renderer/GpuInterop.hpp"

namespace quantiloom::gpu::metal {

bool IsCompiled() {
    return true;
}

Result<std::unique_ptr<Device>, String> CreateSystemDevice(const SystemDeviceOptions&) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return Fail<std::unique_ptr<Device>>("no Metal device on this machine");
    return MetalDevice::Create(device, nil);
}

Result<std::unique_ptr<Device>, String> CreateDeviceFromHandles(const GpuDeviceHandles& handles) {
    if (handles.device == 0) return Fail<std::unique_ptr<Device>>("GpuDeviceHandles.device (the MTLDevice) is null");
    if (handles.queue == 0) {
        return Fail<std::unique_ptr<Device>>("GpuDeviceHandles.queue (the MTLCommandQueue) is null");
    }
    return MetalDevice::Create(FromHandle<id<MTLDevice>>(handles.device),
                               FromHandle<id<MTLCommandQueue>>(handles.queue));
}

}  // namespace quantiloom::gpu::metal
