# tests/

GoogleTest, compiled into one binary `libquantiloom_tests.exe`. Directories mirror
`src/libQuantiloom/` module names (`test_core/` ↔ `core/`, and so on).

## Adding a test file

Sources are listed explicitly — there is no glob. Add the `.cpp` to the matching
`set(TEST_<MODULE>_SOURCES ...)` block in `CMakeLists.txt` (lines 207-263). A file
that is not in that list is never compiled and never runs, with no warning.

## Tests that need a GPU

The suite links `quantiloom_core`, not the DLL, so anything in the library is
reachable — being internal is not a reason a thing cannot be tested.

For the render path, derive from `GpuDeviceTest` (`support/GpuTestDevice.hpp`)
rather than building a device yourself, and write the case against `gpu::`:

```cpp
class BlasTest : public quantiloom::testing::GpuDeviceTest {};
QL_INSTANTIATE_GPU_SUITE(BlasTest);   // once, in the file that defines the fixture

TEST_P(BlasTest, BuildsABottomLevelStructure) {
    auto buffer = Gpu().CreateBuffer(...);
}
```

The fixture is value-parameterized over the backends this build compiles
(`gpu::CompiledBackends()`), so the case is `Gpu/BlasTest.BuildsABottomLevelStructure/Vulkan`
and gains a `/Metal` twin when that backend is compiled in. Cases on `GpuDeviceTest`
itself are `TEST_P(GpuDeviceTest, ...)` with no instantiation of their own: it is
instantiated once, in `GpuTestDevice.cpp`. Filter with a leading `*`
(`--gtest_filter='*BlasTest.*'`); `BlasTest.*` matches nothing.

One headless device per backend is shared by every such case and created on first
use, so a filtered run that touches none of them pays nothing; the whole suite pays
~250 ms once. A machine without a ray-tracing GPU **skips** these cases with the
reason attached — that is not a broken build, and a new GPU test must keep it that
way. The suites are instantiated over the compiled backends, not the available ones,
so the test totals do not depend on the machine.

Shared plumbing, all on `gpu::` and recording what the hand-written Vulkan it
replaced recorded:

- `support/GpuInteropTestUtils.hpp` — the host PODs `ExternalRenderContext` takes:
  `GpuHandles(device)`, `GpuRecording(list)`, `GpuTarget(backend, texture, ...)`.
- `support/GpuTestHelpers.hpp` — staging uploads, the uint readback, and the two
  native inspections (mip count, packed instance transform). Float readbacks are
  `rendercore::ReadbackTexture`; allocations are `CreateTextureOrThrow` /
  `CreateBufferOrThrow` (`renderer/GpuAlloc.hpp`).
- `support/FacadeScene.hpp` — an `ExternalRenderContext` over the Cornell box.
- `support/vulkan/QueueBlocker.hpp` — holds a submitted frame in the queue until the
  host releases it; a case that needs it skips on a backend without one.

Only tests about Vulkan itself stay on `VulkanDeviceTest`
(`support/VulkanTestDevice.hpp`, the same shared device seen as a `VulkanContext`):
`test_vulkan_device.cpp` (VMA round trips, SBT alignment). Nothing else in `tests/`
calls a raw `vk*` function or names `VulkanContext`, apart from `tests/support/vulkan/`
and the Vulkan side of the support files.

Assert structural invariants, not pixels: instance counts, buffer offsets, a
non-null pipeline, a generated map whose size matches the render size. Image-level
checking belongs to the `render-verify` skill, which drives the CLI.

## Running

```bash
# from the repo root, not from tests/
./build/tests/Release/libquantiloom_tests.exe --gtest_filter='MaterialTest.*'
```

The binary reruns in ~3 s with no rebuild. Only rebuild after changing C++ under
`src/`: `cmake.exe --build build --target libquantiloom_tests --config Release -j`
(~76 s).

## Interpreting failures

A red test is not automatically a code bug. No CI runs this suite, so tests can be
stale relative to deliberate physics corrections — check `git log` on both the test
and the implementation before "fixing" code. See the `run-tests` skill.

## Commits

**No Claude Code session link in a commit message.** No `Claude-Session:` trailer,
no `https://claude.ai/code/...` URL, in the subject, the body or a trailer. Same for
PR descriptions.
