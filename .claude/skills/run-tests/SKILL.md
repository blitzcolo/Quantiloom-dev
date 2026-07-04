---
name: run-tests
description: Configure, build, and run the Quantiloom GoogleTest unit test suite (libquantiloom_tests) from WSL2 via the Windows toolchain. Use this whenever the user asks to run tests, verify a change didn't break anything, check test coverage, or run a specific test case — and after any nontrivial change to libQuantiloom (core/io/scene/renderer/postprocess/hs_core) before claiming the work is done.
---

# Run Tests (Quantiloom core)

Tests live in `tests/` (GoogleTest, ~37 test files across test_core / test_scene / test_io / test_renderer / test_postprocess / test_hs_core). They build into a single Windows executable and run fine from WSL via interop.

## Build and run

Tests build in the main `build/` tree (`QUANTILOOM_BUILD_TESTS` defaults ON) and
`./build_wsl.sh` already runs the full suite as a gate between build and SDK
install — so a successful script run means the suite was green. To iterate on
tests directly without the full script:

```bash
cd /mnt/d/Quantiloom-dev
cmake.exe --build build --target libquantiloom_tests --config Release -j
./build/tests/Release/libquantiloom_tests.exe
```

Note: `build/` has `QUANTILOOM_USE_BC7ENC=OFF`, so BC7 tests report SKIPPED there.
For coverage runs or a BC7-enabled suite, use the dedicated `build-test/` tree:

```bash
cmake.exe -B build-test -DBUILD_TESTING=ON -DENABLE_COVERAGE=ON \
    -DQUANTILOOM_USE_BC7ENC=ON -DQUANTILOOM_USE_OPENUSD=ON -DUSD_ROOT=C:/openusd
cmake.exe --build build-test --target libquantiloom_tests --config Release -j
./build-test/tests/Release/libquantiloom_tests.exe
```

Use the Windows `cmake.exe`, never Linux cmake (Visual Studio generator + Windows SDK paths — see the build-and-install skill). Build takes minutes on a cold tree; use a long Bash timeout.

## Running a subset

Full suite output is long. When iterating on one area, filter:

```bash
./build/tests/Release/libquantiloom_tests.exe --gtest_list_tests | head -50
./build/tests/Release/libquantiloom_tests.exe --gtest_filter='SpectralCurve*'
./build/tests/Release/libquantiloom_tests.exe --gtest_filter='*Blackbody*:*Planck*'
```

## Interpreting results

- Exit code 0 + `[  PASSED  ]` summary = green. Report the actual pass/fail counts, never "tests pass" without having seen the summary.
- Some renderer tests need a working Vulkan device (NVIDIA RTX on the Windows side). If GPU-dependent tests fail with device/extension errors on a machine without RTX, say so explicitly instead of treating it as a code regression.
- Physics tests (`test_blackbody_physics`, spectral data) encode CODATA constants and closed-form laws — a failure there is a physics regression, not a flaky test. Cross-check against `scripts/physics-audit/harness.py` (see the physics-audit skill) before touching tolerances.
- **A failing test is not automatically a code bug.** CI never runs this suite, so it can go months without compiling; meanwhile physics-audit fixes deliberately change behavior. Before "fixing" code, check `git log` on both the test and the implementation, and check `scripts/physics-audit/evidence/*.json` — if the implementation was corrected on purpose (e.g. ClearDay `mie_beta_550nm` 2.0e-6 → 1.70e-4 per ATM-01, IR reflectance fallback → neutral 0.1 per commit a8509a3), the test is stale: update the test to pin the corrected behavior and record it in the evidence JSON.

## When adding code

New public behavior in libQuantiloom gets a test in the matching `tests/test_<module>/` directory; register new files in `tests/CMakeLists.txt`. Note per SRS NFR-MAINT-03: CI currently builds but does not run tests, so running this suite locally is the only gate that actually executes.
