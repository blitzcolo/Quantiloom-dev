// The Metal side of the host-interop builders and native inspections
// (support/MetalTestDevice.hpp).

#include "support/MetalTestDevice.hpp"

#include "gpu/metal/MetalBackend.hpp"
#include "renderer/interop/MetalInterop.hpp"

namespace quantiloom::testing {

GpuDeviceHandles MetalHostHandles(gpu::Device& device) {
    auto& metal = static_cast<gpu::metal::MetalDevice&>(device);
    return mtl::Handles((__bridge void*)metal.NativeDevice(), (__bridge void*)metal.NativeQueue());
}

GpuCommandRecording MetalHostRecording(gpu::CommandList& list) {
    // A host's recording has no encoder open (MetalInterop.hpp): end the list's own first.
    auto& metal = static_cast<gpu::metal::MetalCommandList&>(list);
    metal.Finish();
    return mtl::Commands((__bridge void*)metal.Native());
}

u32 MetalTextureMipLevels(const gpu::Texture& texture) {
    return static_cast<u32>([static_cast<const gpu::metal::MetalTexture&>(texture).Native() mipmapLevelCount]);
}

f32 MetalPackedInstanceTranslationX(const void* packed, u32 index) {
    const auto* instances = static_cast<const MTLIndirectAccelerationStructureInstanceDescriptor*>(packed);
    return instances[index].transformationMatrix.columns[3].elements[0];
}

}  // namespace quantiloom::testing
