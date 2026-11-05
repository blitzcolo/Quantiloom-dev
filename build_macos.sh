#!/usr/bin/env bash
# Build Quantiloom on macOS (Apple silicon) with the Metal backend, run the
# same gates build_wsl.sh runs, and install the SDK to
# ../Quantiloom-SDK/macos_arm64 -- where Quantiloom-Qt's CMake looks on a Mac.
#
# The gates are the Windows paths' gates, unchanged: the test suite, the ABI
# gate (against docs/abi/exports.macos.golden), the furnace and illumination
# suites, and the camera checkers. A red gate stops the script before the
# install, so nothing unchecked reaches the SDK.
#
# Prerequisites (the build-and-install skill has the details):
#   - Xcode and its Metal toolchain; CMake and Ninja (Homebrew)
#   - DXC: the LunarG Vulkan SDK's macOS dxc (VULKAN_SDK, or tools/dxc/bin)
#   - Metal Shader Converter (/usr/local/bin/metal-shaderconverter)
#   - Python with numpy and OpenEXR for the render gates: PYTHON=...
#   - OpenUSD (optional): USD_ROOT=... (scripts/build_openusd.sh)
#
# set -e guarantees the install step is never reached if any build step fails.
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR=build-mac

if [ -z "${VULKAN_SDK:-}" ] && [ -d "$HOME/VulkanSDK" ]; then
    # The newest LunarG SDK, for its macOS dxc; nothing else of it is used.
    latest=$(ls -d "$HOME"/VulkanSDK/*/macOS 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$latest" ] && export VULKAN_SDK="$latest"
fi

usd_args=(-DQUANTILOOM_USE_OPENUSD=OFF)
if [ -n "${USD_ROOT:-}" ]; then
    usd_args=(-DQUANTILOOM_USE_OPENUSD=ON "-DUSD_ROOT=${USD_ROOT}")
fi

# --- Stale build directory guard -------------------------------------------
# CMakeCache.txt records the absolute source path; a tree moved elsewhere
# would keep building against the old one.
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    cached=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)
    if [ -n "$cached" ] && [ "$cached" != "$PWD" ]; then
        echo "Stale build cache in $BUILD_DIR/ ($cached != $PWD) -- removing." >&2
        rm -rf "$BUILD_DIR"
    fi
fi

# --- Build -----------------------------------------------------------------

cmake -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DQUANTILOOM_BUILD_TESTS=ON \
    -DQUANTILOOM_USE_BC7ENC=OFF \
    -DQUANTILOOM_BACKEND_VULKAN=OFF \
    -DQUANTILOOM_BACKEND_METAL=ON \
    "${usd_args[@]}"

cmake --build "$BUILD_DIR" -j "$(sysctl -n hw.ncpu)"

# --- Test gate ---------------------------------------------------------------

"./$BUILD_DIR/tests/libquantiloom_tests" --gtest_brief=1

# --- ABI gate ----------------------------------------------------------------

# The scripts are stored without an executable bit, so they are run through bash.
BUILD_DIR="$BUILD_DIR" bash ./scripts/check_exports.sh

# --- Physics, illumination and camera gates ------------------------------------
# Exit 3 from a suite means no usable GPU: a warning, as on Windows.

export CLI="./$BUILD_DIR/src/app/Quantiloom"
export TESTS="./$BUILD_DIR/tests/libquantiloom_tests"
export PYTHON="${PYTHON:-python3}"

run_gate() {
    local name="$1"; shift
    set +e
    "$@"
    local status=$?
    set -e
    case "$status" in
        0) ;;
        3) echo "WARNING: $name skipped, no GPU on this machine" >&2 ;;
        *) echo "$name failed -- not installing." >&2; exit "$status" ;;
    esac
}

run_gate "physics gate" bash ./scripts/render-tests/run_furnace_suite.sh
run_gate "illumination gate" bash ./scripts/render-tests/run_illumination_suite.sh
for checker in check_camera_dynamic check_camera_isp check_camera_history check_camera_ae \
               check_dataset_prerequisites check_fusion_ready; do
    run_gate "camera gate ($checker)" "$PYTHON" "scripts/render-tests/$checker.py"
done

# --- Install (only reached on successful build + green gates) ------------------

SDK_ROOT="${QUANTILOOM_SDK_ROOT:-$(cd .. && pwd)/Quantiloom-SDK}"
SDK_DIR="${SDK_ROOT}/macos_arm64"

rm -rf "${SDK_DIR}"
cmake --install "$BUILD_DIR" --prefix "${SDK_DIR}" >/dev/null

echo "Build and install OK: ${SDK_DIR}"
