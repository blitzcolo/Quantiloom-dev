# src/libQuantiloom/gpu/

The internal compute RHI: what compute modules talk to instead of `vk*`. Internal only --
no `QL_API`, no header under `include/`, no `docs/abi/exports.golden` change.

- **Neutral headers.** No `gpu/*.hpp` includes a backend header; native handles travel as
  `u64`. `NeutralHeaders.cpp` (built without the PCH) fails the build if `vulkan.h` leaks in.
  `vulkan.h` lives only in `gpu/vulkan/`, a Metal backend would live only in `gpu/metal/`.
- **Refs vs owners.** `Buffer`/`Texture`/`ComputeKernel`/... are owned through `unique_ptr`;
  `BufferRef`/`TextureRef`/`TimestampPoolRef`/`BindGroup` are non-owning PODs. Identity guards
  compare `TextureRef::Identity()`.
- **One recording call, one native command.** That is what makes the api_dump command-stream
  diff (`scripts/gpu-ab/apidump_diff.py`) meaningful. Do not batch or elide inside the backend.
- **Barriers** name `from` and `to` as `Access` bit sets; `VulkanMapping.cpp` is the only place
  that turns them into stages and masks (pinned by `GpuMappingTest`).
- **Bindings** mirror HLSL `register(<class><slot>, space<n>)` (`src/shaders/ql_bind.hlsli`):
  slot = register number = Vulkan binding, space = set, one slot number per space.
- **Shaders** come from `core/ResourceLocator` through `ShaderLibrary`; there is no second search list.
- **Transition.** `GpuBuffer`/`GpuImage`/`CommandHelper` are wrapped, not moved. `VulkanBridge.hpp`
  carries the transitional accessors and is deleted once ray tracing has migrated.
- New compute code uses `gpu::`, not raw `vk*`.
