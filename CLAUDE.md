# Quantiloom

Spectral path tracer on Vulkan ray tracing. Renders VIS/SWIR/MWIR/LWIR bands and
hyperspectral cubes from TOML scene configs. C++20 + HLSL, one CMake project.

## Build: WSL2 shell, Windows toolchain

Development happens in WSL2; the product is a Windows binary. Only `.exe` tools are
available — there is no Linux `cmake`/`ctest` on this machine, and Linux cmake cannot
drive the Visual Studio generator. Run everything from the repo root.

| Change | Command |
|---|---|
| Full build + test gate + ABI gate + SDK install | `./build_wsl.sh` |
| C++ only | `cmake.exe --build build --config Release -j` (~76 s) |
| HLSL only | `cmd.exe /c "src\shaders\compile_shaders.bat" </dev/null` (~1 s), then rebuild |
| CMakeLists / new file | `./build_wsl.sh` (re-runs configure) |

`./build_wsl.sh` wipes and reinstalls `D:/Quantiloom-SDK/windows_amd64`, which the
Quantiloom-Qt frontend links against. It takes minutes — use a 600000 ms timeout and
never interrupt it mid-install.

Two gates run before the install, so neither a red suite nor an unreviewed ABI
change can reach the SDK: the test suite, then `scripts/check_exports.sh` against
`docs/abi/*.golden`. The export gate prints what to do when it trips; `src/libQuantiloom/CLAUDE.md`
has the rule it enforces.

## Tests

874 tests run in ~3 s, and the binary reruns without rebuilding. They link the
objects, not the DLL, so internal code is testable without being exported.

```bash
./build/tests/Release/libquantiloom_tests.exe --gtest_brief=1          # all (122 lines, not 2086)
./build/tests/Release/libquantiloom_tests.exe --gtest_filter='Foo*'    # one suite (~0.04 s)
```

`ctest` registers a single aggregate test here, so `ctest -R` cannot select a case —
always use `--gtest_filter`. 10 SKIPPED is the normal baseline: 8 BC7 (deliberately
off, see `build_wsl.sh`) and 2 EXR multipart (unimplemented). A test needing an asset
must build its path from `QUANTILOOM_SOURCE_ROOT`, never a relative or absolute one —
those resolve against the caller's cwd and skip on miss, so a wrong path reads as
"no test data" rather than as a failure.

## Repo map

| Path | What |
|---|---|
| `include/quantiloom/` | **Public** headers — the only ones the SDK installs, 7 modules |
| `src/libQuantiloom/` | Core library sources and **internal** headers |
| `src/shaders/` | HLSL → SPIR-V, ray tracing + compute |
| `src/app/` | CLI: `Quantiloom.exe <config.toml>` |
| `src/libSpectraForge/` | IR material generation library |
| `src/tools/` | `fusion_tool`, `QLTrans` (MODTRAN wrapper) |
| `tests/` | GoogleTest; directories mirror libQuantiloom module names |
| `docs/abi/` | Reviewed export baselines — the ABI gate's reference |
| `assets/configs/` | TOML scene configs — the CLI's only input |
| `scripts/` | Python/PowerShell tooling (spectral baking, LUT gen, physics audit) |

The core is compiled once into `quantiloom_core` (an OBJECT library) and consumed
two ways. **A new target links one or the other, never both** — two copies of the
library's global state (the spdlog logger, static caches) in one process is a
correctness bug:

| Link | Who | Sees |
|---|---|---|
| `quantiloom_core` | `libquantiloom_tests`, `fusion_tool` | everything, `QL_API` irrelevant |
| `libQuantiloom` | CLI, `libSpectraForge`, Quantiloom-Qt | only `QL_API`, only `include/quantiloom/` |

## Conventions

- Commits: Conventional Commits — `feat:`, `fix(shaders):`, `chore:`.
- **No Claude Code session link in a commit message.** No `Claude-Session:` trailer,
  no `https://claude.ai/code/...` URL, in the subject, the body or a trailer. Same for
  PR descriptions.
- clang-tidy runs through clangd via `.clang-tidy` — warnings surface as you edit,
  there is no separate lint command. No formatter is configured; match nearby style.
