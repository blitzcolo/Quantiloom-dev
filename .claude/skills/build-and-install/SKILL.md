---
name: build-and-install
description: Build the Quantiloom core SDK (libQuantiloom + libSpectraForge + CLI + tools) from WSL2 using the Windows MSVC toolchain, compile HLSL shaders with DXC, and install the SDK to D:/Quantiloom-SDK. Use this whenever the user asks to build, rebuild, compile, install, or "make" this project, after any C++ or shader source change that needs compiling, or when the downstream Quantiloom-Qt GUI needs a fresh SDK. Also covers build failure diagnosis and the Linux portability rules that every code change must respect.
---

# Build and Install (Quantiloom core)

## The one command

```bash
cd /mnt/d/Quantiloom-dev && ./build_wsl.sh
```

This is the canonical path. It runs, in order:
1. Shader compile: `cmd.exe /c "src\shaders\compile_shaders.bat"` (Windows DXC, HLSL → SPIR-V)
2. Configure: `cmake.exe -B build -G "Visual Studio 18 2026" -A x64` with `QUANTILOOM_USE_OPENUSD=ON -DUSD_ROOT=C:/openusd`
3. Build: `cmake.exe --build build --config Release -j` (includes the test suite)
4. **Test gate**: runs `libquantiloom_tests.exe` — a red suite aborts the script
   before install, so broken code never lands in the SDK. Don't "fix" a red gate
   by skipping it — fix the tests (see run-tests skill for adjudicating stale vs
   real failures).
5. **ABI gate**: `scripts/check_exports.sh` diffs the built DLLs' export tables
   against `docs/abi/*.golden` — see "Common failures" below.
6. Install: wipes and repopulates `D:/Quantiloom-SDK/windows_amd64`

There is no CI; these two gates are the only automated ones.

`set -euo pipefail` guarantees install never runs on a failed build or red tests. A full run takes minutes; use a generous Bash timeout (600000 ms) and never kill it mid-install — a half-written SDK breaks the Qt frontend.

## Why everything is `.exe`

Development happens in WSL2 but the product runs on Windows. All builds go through WSL interop to the **Windows** toolchain: `cmake.exe` (Windows CMake, VS 2026 generator), Windows DXC from the Windows Vulkan SDK, MSVC. Never use the Linux `cmake` for this tree — it cannot resolve the Visual Studio generator or Windows Qt/Vulkan/USD paths. Binaries produced are Windows EXE/DLLs, runnable directly from WSL via interop.

## Partial rebuilds

Don't re-run the full script when only part changed:

| Changed | Command |
|---|---|
| C++ only | `cmake.exe --build build --config Release -j` |
| HLSL shaders (`src/shaders/*.hlsl*`) | `cmd.exe /c "src\shaders\compile_shaders.bat" </dev/null` from repo root, then rebuild (build copies `.spv` into place) |
| CMakeLists / new files | re-run full `./build_wsl.sh` (configure step included) |

After any change to public headers, exported symbols, or shaders, finish with the **install** step (or full script) — the Qt frontend links the installed SDK at `D:/Quantiloom-SDK/windows_amd64`, not this repo's build tree. A stale SDK is the #1 source of "my core fix doesn't show up in the GUI" confusion. After installing, rebuild Quantiloom-Qt (see its `build-and-run` skill).

Key outputs:
- CLI: `build/src/app/Release/Quantiloom.exe`
- Tools: `build/src/tools/Release/{fusion_tool,QLTrans}.exe`
- SDK: `D:/Quantiloom-SDK/windows_amd64/{bin,include,lib}` (bin contains `Quantiloom.exe` + compiled `.spv` shaders)

## Common failures

- **`cmake.exe: command not found`** — Windows CMake not on the Windows PATH visible to WSL interop. Check `which cmake.exe`; fix the Windows PATH, don't substitute Linux cmake.
- **`ERROR: DXC not found in PATH`** — Windows Vulkan SDK missing from Windows PATH. The `.bat` prints this and would `pause`; the script's `</dev/null` prevents hanging.
- **USD errors at configure** — `USD_ROOT=C:/openusd` must exist (OpenUSD prebuilt). To build without USD support, pass `-DQUANTILOOM_USE_OPENUSD=OFF` instead.
- **Install step "device or resource busy" / access denied** — a running `QuantiloomQt.exe` or `Quantiloom.exe` holds the DLLs. Close them first.
- **C2065 "undeclared identifier" for a variable declared on the PREVIOUS line, often with garbled line numbers** — MSVC is parsing a UTF-8-no-BOM source file in the system codepage (GBK): multibyte characters in a comment (`²`, `×`, `λ`, Chinese text) swallow the newline and eat the next code line. The root CMakeLists sets `/utf-8` to prevent this (fixed 2026-07-04; do not remove it). If it reappears, check that new targets inherit the flag rather than "fixing" the source lines.
- **Script fails after "Build" with GoogleTest output** — that's the test gate doing its job, not a build error. Read the `[  FAILED  ]` lines and follow the run-tests skill.
- **`libQuantiloom export surface changed`** — the ABI gate. It prints the added (`+`) and removed (`-`) symbols; decide which case you are in before touching anything:
  - **Intended** (you deliberately promoted something to public API): `./scripts/check_exports.sh --update`, then commit `docs/abi/exports.golden` **with** the change. The baseline diff is what gets reviewed, and it doubles as the SemVer trigger — a removed line is a major, an added line a minor.
  - **Unintended** (`+` lines you did not mean to publish): almost always a new class that picked up `QL_API` out of habit. New code in `src/libQuantiloom/` is internal by default; the tests link the objects and see it without any export. See `src/libQuantiloom/CLAUDE.md`.
  - Never silence the gate by editing `build_wsl.sh`. An unreviewed export is a promise to Quantiloom-Qt that nobody made on purpose.

## Linux portability rules (apply to every code change)

Windows is the primary target, but Linux support (headless CI/batch, `D:/Quantiloom-SDK/linux_amd64` exists) must stay cheap to revive. When writing or reviewing C++ in this repo:

- No MSVC-only constructs (`__declspec` outside the existing `QL_API` macro machinery, `#pragma` MSVC extensions, `_s` CRT functions). Compiler must remain MSVC / Clang / GCC.
- GCC < 14 has a known ICE on this codebase — workaround is `-O1` on affected TUs; don't introduce code that deepens that dependency.
- Paths: use `std::filesystem`, never hard-coded `\\` separators or drive letters in library code. Drive letters belong only in build scripts.
- Case-sensitive includes: `#include` file names must match on-disk casing exactly (Windows tolerates mismatch, Linux does not).
- Keep CMake logic generator-agnostic; anything Windows-specific goes behind `if(WIN32)`.
