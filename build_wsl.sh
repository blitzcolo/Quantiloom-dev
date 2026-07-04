#!/usr/bin/env bash
# Build Quantiloom from WSL2 using the Windows toolchain via WSL interop.
# Produces Windows binaries (EXE/DLL), identical to running
# build_windows.ps1 + install_windows.ps1 on the Windows side.
#
# set -e guarantees the install step is never reached if any build step fails.
set -euo pipefail
cd "$(dirname "$0")"

# --- Build -----------------------------------------------------------------

# Compile shaders with the Windows DXC (Vulkan SDK must be on Windows PATH).
# </dev/null keeps the bat's `pause` from blocking a non-interactive run;
# the exit code still propagates through cmd.exe.
cmd.exe /c "src\\shaders\\compile_shaders.bat" </dev/null

cmake.exe -B build -G "Visual Studio 18 2026" -A x64 \
    -DQUANTILOOM_BUILD_TESTS=OFF \
    -DQUANTILOOM_USE_BC7ENC=OFF \
    -DQUANTILOOM_USE_OPENUSD=ON \
    -DUSD_ROOT=C:/openusd

cmake.exe --build build --config Release -j

# --- Install (only reached on successful build) ------------------------------

rm -rf /mnt/d/Quantiloom-SDK/windows_amd64
cmake.exe --install build --prefix D:/Quantiloom-SDK/windows_amd64 --config Release

echo "Build and install OK: D:\\Quantiloom-SDK\\windows_amd64"
