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
#   - Python for the render gates: found or provisioned automatically (uv)
#   - OpenUSD (optional): ~/openusd or /opt/openusd is picked up, or USD_ROOT=...
#     (scripts/build_openusd.sh --install-dir ~/openusd needs no sudo)
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

# OpenUSD: USD_ROOT when set, else the first install scripts/build_openusd.sh
# would have made. Without one the build goes on with USD loading disabled.
if [ -z "${USD_ROOT:-}" ]; then
    for candidate in "$HOME/openusd" /opt/openusd; do
        if [ -f "$candidate/pxrConfig.cmake" ]; then USD_ROOT="$candidate"; break; fi
    done
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
# The render gates measure with numpy and OpenEXR. Use PYTHON when it has them,
# else python3 when it does, else a venv of our own made with uv (created once,
# kept in the build tree), so a fresh machine needs no setup step for this.
gate_python_ok() { "$1" -c 'import numpy, OpenEXR' >/dev/null 2>&1; }
if [ -n "${PYTHON:-}" ] && gate_python_ok "$PYTHON"; then
    :
elif gate_python_ok python3; then
    PYTHON=python3
else
    venv="$BUILD_DIR/gates-venv"
    if ! gate_python_ok "$venv/bin/python"; then
        command -v uv >/dev/null 2>&1 || {
            echo "The render gates need Python with numpy and OpenEXR: install uv" \
                 "(brew install uv) or set PYTHON to such an interpreter." >&2
            exit 1
        }
        uv venv --quiet "$venv"
        uv pip install --quiet --python "$venv/bin/python" numpy OpenEXR
    fi
    PYTHON="$venv/bin/python"
fi
export PYTHON

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
