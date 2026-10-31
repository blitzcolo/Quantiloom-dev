/**
 * @file Device.hpp
 * @brief The factory and synchronous submission surface of the compute RHI (internal)
 */

#pragma once

#include "gpu/Binding.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/Kernel.hpp"
#include "gpu/Resources.hpp"
#include "gpu/ShaderLibrary.hpp"

#include <functional>
#include <memory>
#include <span>

namespace quantiloom::gpu {

class Device {
public:
    virtual ~Device() = default;
    [[nodiscard]] virtual const DeviceInfo& Info() const = 0;
    [[nodiscard]] virtual const ShaderLibrary& Shaders() const = 0;

    [[nodiscard]] virtual Result<std::unique_ptr<Buffer>, String> CreateBuffer(const BufferDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<Texture>, String> CreateTexture(const TextureDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<BindingLayout>, String> CreateBindingLayout(
        const BindingLayoutDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<KernelLayout>, String> CreateKernelLayout(
        const KernelLayoutDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<ComputeKernel>, String> CreateComputeKernel(
        const ComputeKernelDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<BindGroupPool>, String> CreateBindGroupPool(
        const BindingLayout&, u32 maxGroups) = 0;

    /// One native update, writes applied in order. Same rule as vkUpdateDescriptorSets:
    /// the group must not be in use by a pending submission.
    virtual void WriteBindGroup(BindGroup group, std::span<const BindingWrite> writes) = 0;

    /// Record, submit, and wait. Re-entrant: a nested call gets its own pool and fence.
    /// Throws std::runtime_error when the submit or the wait fails (CommandHelper's contract).
    virtual void ExecuteImmediate(const std::function<void(CommandList&)>& record) = 0;

    /// vkDeviceWaitIdle. Err on device loss.
    [[nodiscard]] virtual Result<void, String> WaitIdle() = 0;
};

}  // namespace quantiloom::gpu
