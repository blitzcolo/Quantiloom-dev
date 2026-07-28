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

For the render path, derive from `VulkanDeviceTest` (`support/VulkanTestDevice.hpp`)
rather than building a context yourself:

```cpp
TEST_F(quantiloom::testing::VulkanDeviceTest, BuildsABottomLevelStructure) {
    GpuBuffer buffer(Device().GetAllocator(), ...);
}
```

One headless device is shared by every such case and created on first use, so a
filtered run that touches none of them pays nothing; the whole suite pays ~250 ms
once. A machine without a ray-tracing GPU **skips** these cases with the reason
attached — that is not a broken build, and a new GPU test must keep it that way.

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
