# src/libQuantiloom/gpu/

The internal compute RHI: what compute modules talk to instead of `vk*`. Internal only --
no `QL_API`, no header under `include/`, no `docs/abi/exports.golden` change.

- **Neutral headers.** No `gpu/*.hpp` includes a backend header; native handles travel as
  `u64`. `NeutralHeaders.cpp` (built without the PCH) fails the build if `vulkan.h` leaks in.
  `vulkan.h` lives only in `gpu/vulkan/`, a Metal backend would live only in `gpu/metal/`.
  Outside `gpu/vulkan/`, a comment names a Vulkan call without its opening parenthesis:
  `scripts/gpu-ab/vk_residual.sh` counts every `vk*(`, comments included.
- **Refs vs owners.** `Buffer`/`Texture`/`Sampler`/`ComputeKernel`/`CommandPool`/`Fence`/... are
  owned through `unique_ptr`; `BufferRef`/`TextureRef`/`SamplerRef`/`EventRef`/`RtSceneRef`/
  `TimestampPoolRef`/`BindGroup` are non-owning PODs. Identity guards compare
  `TextureRef::Identity()`. A `CommandBuffer` dies before its `CommandPool`.
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
  masks and default to Compute, so a compute table needs neither.
- **No RGB32Float textures.** Metal has no 3-channel 2D format. Pad to RGBA32Float with alpha
  1: sampling a 3-component format already returns alpha 1, so the shader reads the same values.
- **Errors.** On Vulkan, the `Err` of a failed `gpu/Sync.hpp` call, `CreateSampler`, `CreateGpuEvent`,
  `CreateTimestampPool`, `CreateCommandPool`, `CreateFence`, `Submit` or `WaitQueueIdle` is the
  `VkResult` as a decimal, so a caller keeps a message that prints the code.
- **No member named after a `<windows.h>` macro.** `CreateEvent`, `CreateSemaphore`, `CreateMutex`,
  `GetObject`, `LoadImage`... are macros there; a translation unit that sees `<windows.h>` would
  declare a different member than the backend defines (and `VulkanDevice` is `final`, so calls
  devirtualize to a symbol that does not exist). Hence `CreateGpuEvent`.
- **Shaders** come from `core/ResourceLocator` through `ShaderLibrary`; there is no second search list.
- **Transition.** `GpuBuffer`/`GpuImage`/`CommandHelper` are wrapped, not moved. `VulkanBridge.hpp`
  carries the transitional accessors and is deleted once ray tracing has migrated.
- New compute code uses `gpu::`, not raw `vk*`.
