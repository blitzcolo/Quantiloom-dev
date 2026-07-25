# tests/

GoogleTest, compiled into one binary `libquantiloom_tests.exe`. Directories mirror
`src/libQuantiloom/` module names (`test_core/` ↔ `core/`, and so on).

## Adding a test file

Sources are listed explicitly — there is no glob. Add the `.cpp` to the matching
`set(TEST_<MODULE>_SOURCES ...)` block in `CMakeLists.txt` (lines 193-249). A file
that is not in that list is never compiled and never runs, with no warning.

## Running

```bash
# from the repo root, not from tests/
./build/tests/Release/libquantiloom_tests.exe --gtest_filter='MaterialTest.*'
```

The binary reruns in ~3 s with no rebuild. Only rebuild after changing C++ under
`src/`: `cmake.exe --build build --target libquantiloom_tests --config Release -j`
(~76 s).

## The header comment in CMakeLists.txt is stale

It recommends `mkdir build && cd build && cmake ..` and `ctest --output-on-failure`.
Neither applies: the real configure uses the Visual Studio generator via
`build_wsl.sh`, and `ctest` registers only one aggregate test.

## Interpreting failures

A red test is not automatically a code bug. No CI runs this suite, so tests can be
stale relative to deliberate physics corrections — check `git log` on both the test
and the implementation before "fixing" code. See the `run-tests` skill.
