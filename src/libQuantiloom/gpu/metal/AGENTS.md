# src/libQuantiloom/gpu/metal/

The Metal backend of the `gpu::` RHI, compiled on Apple only (`QUANTILOOM_BACKEND_METAL`,
on by default on a Mac, where Vulkan is off). Objective-C++ with ARC; `MetalBackend.hpp`
declares every class and is included from `.mm` files only.

How each Vulkan concept lands here:

- **Shaders.** A Metal build ships, per shader, the metallib Metal Shader Converter makes
  from DXIL and the reflection JSON it writes beside it (`<name>.reflection.json`); both
  install to `share/quantiloom/shaders/metallib` with the SHADERSET stamp
  (`src/shaders/CMakeLists.txt`). The reflection is the only source of a kernel's
  argument-buffer layout and threadgroup size: nothing here hard-codes a shader's bindings.
- **Bindings.** A `BindGroup` is a CPU record of its writes (`MetalBindGroup`). Every
  dispatch encodes a fresh top-level argument buffer into a per-command-list upload arena:
  MSC's linear layout (24-byte descriptors by register and space) for plain compute
  kernels, or `QL_ROOT_SIGNATURE`'s four tables plus the push constants as a root CBV for
  raygen and the ray-query kernels -- table 0 is t0..t37 then u0..u36, table 1 the
  samplers, tables 2 and 3 the bindless arrays (a slot whose count is above 1). Every
  resource named is declared with `useResources`, which is what makes it resident and
  hazard-tracked.
- **Barriers** end the open encoder. Resources are tracked, so encoder order is
  execution order for everything an argument buffer names. Layouts mean nothing here.
- **Ray tracing.** Instances are `MTLIndirectAccelerationStructureInstanceDescriptor`s
  naming their bottom level by `gpuResourceID` (`RtSceneRef::address`), so the renderer's
  `referencedBottomLevels` can stay empty; every acceleration structure sits in the
  device's residency set, which is how a TLAS's bottom levels are resident during a trace.
  A TLAS reaches a shader through an `IRRaytracingAccelerationStructureGPUHeader` buffer
  (`MetalRtAccelerationStructure::Header`).
- **Trace.** Raygen is converted as a compute kernel and dispatched over the launch grid
  with an `IRDispatchRaysArgument` at `kIRRayDispatchArgumentsBindPoint`; every ray is
  inline, so there is no shader binding table. The two specialisation constants are Metal
  function constants (`QL_METAL_FUNCTION_CONSTANT_SPACE`, `ql_bind.hlsli`), set when the
  pipeline is created; a constant the caller does not specialise is an error, not a
  default.
- **Blits.** A scaling or converting `BlitTexture` is a draw with a built-in fullscreen
  pipeline per destination format, so destinations need render-target usage (a host's
  texture too); an unscaled same-format one is a blit copy.
- **Events and fences** are `MTLSharedEvent`s, which only count up: a reset moves the
  value waited for, not the event.
- **Timestamps** are counter samples at the start of an empty blit pass, Apple GPUs
  sampling at encoder boundaries only.
- **Memory.** Buffers are shared (one memory on Apple silicon); DeviceLocal textures are
  private.
