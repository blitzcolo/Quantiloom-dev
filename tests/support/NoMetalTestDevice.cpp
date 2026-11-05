// The Metal side of the support files in a build without gpu/metal/. Unreachable for
// the reason NoVulkanTestDevice.cpp gives.

#include "support/MetalTestDevice.hpp"

#include <gtest/gtest.h>

namespace quantiloom::testing {

namespace {
constexpr const char* kNoMetal = "this build has no Metal backend";
}  // namespace

GpuDeviceHandles MetalHostHandles(gpu::Device&) {
    ADD_FAILURE() << kNoMetal;
    return {};
}

GpuCommandRecording MetalHostRecording(gpu::CommandList&) {
    ADD_FAILURE() << kNoMetal;
    return {};
}

u32 MetalTextureMipLevels(const gpu::Texture&) {
    ADD_FAILURE() << kNoMetal;
    return 0;
}

f32 MetalPackedInstanceTranslationX(const void*, u32) {
    ADD_FAILURE() << kNoMetal;
    return 0.0f;
}

}  // namespace quantiloom::testing
