# Quantiloom

Spectral path tracer on Vulkan ray tracing. Renders VIS/SWIR/MWIR/LWIR bands and
hyperspectral cubes from TOML scene configs. C++20 + HLSL, one CMake project.

## Build: WSL2 shell, Windows toolchain

Development happens in WSL2; the product is a Windows binary. Only `.exe` tools are
available — there is no Linux `cmake`/`ctest` on this machine, and Linux cmake cannot
drive the Visual Studio generator. Run everything from the repo root.

| Change | Command |
|---|---|
| Full build + test gate + SDK install | `./build_wsl.sh` |
| C++ only | `cmake.exe --build build --config Release -j` (~76 s) |
| HLSL only | `cmd.exe /c "src\shaders\compile_shaders.bat" </dev/null` (~1 s), then rebuild |
| CMakeLists / new file | `./build_wsl.sh` (re-runs configure) |

`./build_wsl.sh` wipes and reinstalls `D:/Quantiloom-SDK/windows_amd64`, which the
Quantiloom-Qt frontend links against. It takes minutes — use a 600000 ms timeout and
never interrupt it mid-install.

## Tests

863 tests run in ~3 s, and the binary reruns without rebuilding.

```bash
./build/tests/Release/libquantiloom_tests.exe --gtest_brief=1          # all (122 lines, not 2086)
./build/tests/Release/libquantiloom_tests.exe --gtest_filter='Foo*'    # one suite (~0.04 s)
```

`ctest` registers a single aggregate test here, so `ctest -R` cannot select a case —
always use `--gtest_filter`. 24 SKIPPED is the normal baseline (BC7 and glTF sample
assets disabled in `build/`).

## Repo map

| Path | What |
|---|---|
| `src/libQuantiloom/` | Core shared library (`Quantiloom.dll`) — 7 modules |
| `src/shaders/` | HLSL → SPIR-V, ray tracing + compute |
| `src/app/` | CLI: `Quantiloom.exe <config.toml>` |
| `src/libSpectraForge/` | IR material generation library |
| `src/tools/` | `fusion_tool`, `QLTrans` (MODTRAN wrapper) |
| `tests/` | GoogleTest; directories mirror libQuantiloom module names |
| `assets/configs/` | TOML scene configs — the CLI's only input |
| `scripts/` | Python/PowerShell tooling (spectral baking, LUT gen, physics audit) |

## Conventions

- Commits: Conventional Commits — `feat:`, `fix(shaders):`, `chore:`.
- clang-tidy runs through clangd via `.clang-tidy` — warnings surface as you edit,
  there is no separate lint command. No formatter is configured; match nearby style.
