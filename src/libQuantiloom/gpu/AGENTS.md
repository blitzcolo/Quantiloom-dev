# src/libQuantiloom/gpu/

The internal compute RHI: what compute modules talk to instead of `vk*`. Internal only --
no `QL_API`, no header under `include/`, no `docs/abi/exports.golden` change.

- **Neutral headers.** No `gpu/*.hpp` includes a backend header; native handles travel as
  `u64`. `NeutralHeaders.cpp` (built without the PCH) fails the build if `vulkan.h` leaks in.
  `vulkan.h` lives only in `gpu/vulkan/`, a Metal backend would live only in `gpu/metal/`.
  Outside `gpu/vulkan/`, a comment names a Vulkan call without its opening parenthesis:
  `scripts/gpu-ab/vk_residual.sh` counts every `vk*(`, comments included.
- **Refs vs owners.** `Buffer`/`Texture`/`Sampler`/`ComputeKernel`/`CommandPool`/`Fence`/
  `TraceKernel`/`PipelineCache`/`RtAccelerationStructure`/... are owned through `unique_ptr`;
  `BufferRef`/`TextureRef`/`SamplerRef`/`EventRef`/`RtSceneRef`/`TimestampPoolRef`/`BindGroup`
  are non-owning PODs. Identity guards compare `TextureRef::Identity()`. A `CommandBuffer` dies
  before its `CommandPool`.
- **One recording call, one native command.** That is what makes the api_dump command-stream
  diff (`scripts/gpu-ab/apidump_diff.py`) meaningful. Do not batch or elide inside the backend.
- **Barriers** name `from` and `to` as `Access` bit sets; `VulkanMapping.cpp` is the only place
  that turns them into stages and masks (pinned by `GpuMappingTest`). Where a native barrier's
  stages are not the OR of its accesses, `BarrierBatch::srcStages`/`dstStages` say so; a host
  image's layouts travel verbatim in `TextureBarrier::hostLayoutBefore`/`After`. A storage
  image used as a copy or blit source is passed `WithLayout(TextureLayoutPolicy::Optimal)`.
  Metal ignores layouts, stage overrides and host layouts.
- **Bindings** mirror HLSL `register(<class><slot>, space<n>)` (`src/shaders/ql_bind.hlsli`):
  slot = register number = Vulkan binding, space = set, one slot number per space.
  `BindingSlot::visibility` and `KernelLayoutDesc::pushConstantVisibility` are Vulkan stage
  masks and default to Compute, so a compute table needs neither. The ray tracing set's 38
  slots are written nowhere in C++: `renderer/RtBindings.hpp` expands them from the shaders'
  manifest, `src/shaders/ql_rt_bindings.hlsli`, whose header also says how a Metal
  argument-buffer layout follows from the same rows.
- **Ray tracing.** `RtScene.hpp` holds an acceleration structure with its storage; a build or
  refit is one `CommandList::BuildRtAccelerationStructure`, and which barriers surround it stays
  with the caller. Instances are neutral `RtInstance` values that `Device::PackRtInstances`
  writes in the native layout (Vulkan names a bottom level by address, Metal 3 by index into
  `RtInstances::referencedBottomLevels`). `Trace.hpp`: a `TraceKernel` owns its shader binding
  table on Vulkan and no `gpu/` header mentions one; Metal has none, every ray being inline from
  raygen. The Vulkan ray tracing entry points are fetched once per `VulkanDevice` and copied into
  every list it hands out (`ExecuteImmediate`, `WrapRecording`, a `CommandPool`'s buffers); on
  a device without them an acceleration structure build throws.
- **No RGB32Float textures.** Metal has no 3-channel 2D format. Pad to RGBA32Float with alpha
  1: sampling a 3-component format already returns alpha 1, so the shader reads the same values.
- **Errors.** On Vulkan, the `Err` of a failed `gpu/Sync.hpp` call, `CreateSampler`, `CreateGpuEvent`,
  `CreateTimestampPool`, `CreateCommandPool`, `CreateFence`, `Submit` or `WaitQueueIdle` is the
  `VkResult` as a decimal, so a caller keeps a message that prints the code. `CreateTraceKernel`,
  `CreateRtAccelerationStructure`, `CreatePipelineCache` and `PipelineCache::Serialize` instead
  return the message `RayTracingPipeline` or `AccelerationStructure` threw or logged for the same
  failure, so a caller that throws or logs the `Err` keeps that text.
- **No member named after a `<windows.h>` macro.** `CreateEvent`, `CreateSemaphore`, `CreateMutex`,
  `GetObject`, `LoadImage`... are macros there; a translation unit that sees `<windows.h>` would
  declare a different member than the backend defines (and `VulkanDevice` is `final`, so calls
  devirtualize to a symbol that does not exist). Hence `CreateGpuEvent`.
- **Shaders** come from `core/ResourceLocator` through `ShaderLibrary`; there is no second search list.
- **`SharedLayoutChecks.cpp`** compiles the C++ half of `src/shaders/ql_shared_public.hlsli`:
  the layout locks of the public structs the shaders read. It defines nothing; the structs
  the host alone owns are single definitions in the other `ql_shared_*.hlsli`.
- **Backend files.** `gpu/vulkan/` is the whole Vulkan backend: the `gpu::` implementation
  (`VulkanDevice`, `VulkanDeviceFactory`, `VulkanCommandList`, `VulkanResources`, `VulkanSync`,
  `VulkanRt`, `VulkanTrace`, `VulkanMapping`), the context it runs on (`VulkanContext`, or
  `VulkanContextAdapter` over a host's handles, with `VulkanDeviceRequirements`), the VMA
  wrappers `GpuBuffer`/`GpuImage` and `VmaImpl.cpp`, `CommandHelper`'s immediate submissions,
  and the exported `vk::CreateDevice` (`VulkanInterop.cpp`). Library code outside it reaches
  the backend only through `gpu/DeviceFactory.hpp`; no file outside it calls a raw `vk*`
  function (`vk_residual.sh` prints 0).
- **No bridge.** The two conversions left between native objects and refs live beside their
  types in `VulkanResources.hpp`: `Ref(GpuImage)` and `AsGpuImage`. Only Vulkan-only test code
  uses them (`test_gpu_vulkan.cpp`, `tests/support/VulkanTestDevice.cpp`), and it reaches the
  context through `VulkanDevice::Context()`. Every other GPU test is written against `gpu::`
  (`tests/support/GpuTestDevice.hpp`).
- New compute code uses `gpu::`, not raw `vk*`.
