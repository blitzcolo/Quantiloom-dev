/**
 * @file DeviceFactory.hpp
 * @brief Where a gpu::Device comes from: the system's own, or a host's handles (internal)
 *
 * The neutral entry points dispatch on the backend. Each compiled backend implements its
 * namespace's three functions below in its own entry file (gpu/vulkan/VulkanDeviceFactory.cpp);
 * a backend this build does not have is a stub that answers with an Err
 * (gpu/NoMetalBackend.cpp). CMake picks the files, so no #ifdef selects a backend here.
 */

#pragma once

#include "core/Types.hpp"
#include "gpu/Device.hpp"
#include "gpu/GpuTypes.hpp"

#include <memory>
#include <span>

namespace quantiloom {
struct GpuDeviceHandles;  // renderer/GpuInterop.hpp
}

namespace quantiloom::gpu {

/// What the system device's default construction decides today. Nothing is chosen by
/// the caller yet; the struct exists so that a choice can be added without a new signature.
struct SystemDeviceOptions {};

/// A device of its own, which owns everything it created on the system: on Vulkan the
/// instance, the physical device it selected, the logical device and its allocator. Err
/// with the message the backend's setup failed with.
[[nodiscard]] Result<std::unique_ptr<Device>, String> CreateSystemDevice(
    Backend backend, const SystemDeviceOptions& options = {});

/// A device over a host's own objects, which the host keeps owning; dispatched on
/// handles.backend. The handles are taken as they are: validating them is the caller's job.
[[nodiscard]] Result<std::unique_ptr<Device>, String> CreateDeviceFromHandles(
    const GpuDeviceHandles& handles);

/// The backends this build can create a device on: {Vulkan} today.
[[nodiscard]] std::span<const Backend> CompiledBackends();

/// "Vulkan" or "Metal".
[[nodiscard]] StringView BackendName(Backend backend);

// ---- Backend entry points: what the neutral functions above dispatch to ------------------

namespace vulkan {
[[nodiscard]] bool IsCompiled();
[[nodiscard]] Result<std::unique_ptr<Device>, String> CreateSystemDevice(const SystemDeviceOptions& options);
[[nodiscard]] Result<std::unique_ptr<Device>, String> CreateDeviceFromHandles(const GpuDeviceHandles& handles);
}  // namespace vulkan

namespace metal {
[[nodiscard]] bool IsCompiled();
[[nodiscard]] Result<std::unique_ptr<Device>, String> CreateSystemDevice(const SystemDeviceOptions& options);
[[nodiscard]] Result<std::unique_ptr<Device>, String> CreateDeviceFromHandles(const GpuDeviceHandles& handles);
}  // namespace metal

}  // namespace quantiloom::gpu
