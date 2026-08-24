#!/usr/bin/env bash
# Build Quantiloom from WSL2 using the Windows toolchain via WSL interop.
# Produces Windows binaries (EXE/DLL), identical to running
# build_windows.ps1 + install_windows.ps1 on the Windows side.
#
# That last sentence is an invariant to maintain, not an observation: both paths
# install to the same prefix, and Quantiloom-Qt cannot tell which one produced
# the SDK it links. It stopped being true once -- build_windows.ps1 configured
# tests OFF and ran no gate, so the Windows pair published whatever it built --
# and the same four gates now run on both sides, from the same scripts. A gate
# added here belongs there too.
#
# set -e guarantees the install step is never reached if any build step fails.
set -euo pipefail
cd "$(dirname "$0")"

# --- Build -----------------------------------------------------------------

# Shaders are compiled by CMake now, as part of the build below, so there is no
# separate step here. src/shaders/compile_shaders.bat still works and is kept as
# a manual fallback, but calling it from here would compile every shader twice
# per build -- and with a different DXC than CMake finds, since the bat takes
# whatever is on PATH. Which of the two won then depended on file timestamps.

# BC7 stays OFF deliberately, not for lack of trying: it measurably slowed
# rendering and saved little VRAM, so it is a net loss here. vendor/bc7enc_rdo/
# is also gitignored, so ON would not build from a fresh clone anyway. This
# leaves the 8 BC7CompressionActualTest cases skipped, which is expected.
# QUANTILOOM_BUILD_TESTS is passed explicitly rather than left to its default,
# because CMake caches it and build_windows.ps1 used to force it OFF: a tree
# configured by that script kept tests off, and the gate below then failed for
# want of a binary rather than for want of a passing test.
cmake.exe -B build -G "Visual Studio 18 2026" -A x64 \
    -DQUANTILOOM_BUILD_TESTS=ON \
    -DQUANTILOOM_USE_BC7ENC=OFF \
    -DQUANTILOOM_USE_OPENUSD=ON \
    -DUSD_ROOT=C:/openusd

# MSBuild quiet verbosity: warnings and errors still print in full; per-file
# compile lines and POST_BUILD copy chatter are suppressed.
cmake.exe --build build --config Release -j -- /v:q /nologo

# --- Test gate ---------------------------------------------------------------
# No CI runs this suite; this is the only automated gate. A red suite must
# never reach the install step, or broken code lands in the SDK consumed by
# Quantiloom-Qt. (~2 s for the full suite, cheap insurance.)
# --gtest_brief prints only failures and the final summary.

./build/tests/Release/libquantiloom_tests.exe --gtest_brief=1

# --- ABI gate ---------------------------------------------------------------
# The export table is the SDK's contract with Quantiloom-Qt, and it drifts
# quietly: a new class picks up QL_API by habit and is public forever. Compare
# against docs/abi/*.golden and stop before the install if it moved. Intended
# changes are accepted with ./scripts/check_exports.sh --update.

./scripts/check_exports.sh

# --- Physics gate -----------------------------------------------------------
# Eight isothermal furnace cavities -- five LWIR, three MWIR -- each of which
# must return the Planck
# radiance of its own temperature whatever its emissivity. Kirchhoff fixes the
# answer with no free parameters, so unlike every other render in the repo
# these can fail on their own rather than by comparison with a previous run.
# Four shader bugs in one week were visible here and in no unit test: the unit
# suite links the core and checks structure, and none of it puts a photon on a
# surface.
#
# Exit 3 means no usable GPU. That is a warning, not a failure -- the same call
# the unit suite makes for its GPU cases -- but it does mean this build was
# never checked against anything that renders.

set +e
./scripts/render-tests/run_furnace_suite.sh
furnace_status=$?
set -e
case "$furnace_status" in
    0) ;;
    3) echo "WARNING: physics gate skipped, no GPU on this machine" >&2 ;;
    *) echo "Physics gate failed -- not installing." >&2; exit "$furnace_status" ;;
esac

# The cavities have no sun, no sky and no scene outside their own walls, so
# nothing above can see a bug in how light REACHES a surface. This is the
# complement: sun occlusion per band, an open sky where the traced bounce must
# contribute exactly zero, and a Cornell box lit only by emissive geometry
# where every lit pixel is bounced light by construction.

set +e
./scripts/render-tests/run_illumination_suite.sh
illum_status=$?
set -e
case "$illum_status" in
    0) ;;
    3) echo "WARNING: illumination gate skipped, no GPU on this machine" >&2 ;;
    *) echo "Illumination gate failed -- not installing." >&2; exit "$illum_status" ;;
esac

# --- Install (only reached on successful build + green tests + stable ABI) ---

# The SDK install prefix is derived from where this repo sits, not hard-coded to
# a drive. Quantiloom-Qt resolves the SDK as ../Quantiloom-SDK relative to its own
# checkout, so a literal D:/ here would keep installing to the old drive after the
# trees are moved -- Studio would then link a stale SDK, or SdkGuard would abort on
# the hash mismatch. Override with QUANTILOOM_SDK_ROOT if the SDK lives elsewhere.

SDK_ROOT_WSL="${QUANTILOOM_SDK_ROOT:-$(cd .. && pwd)/Quantiloom-SDK}"
SDK_DIR_WSL="${SDK_ROOT_WSL}/windows_amd64"
SDK_DIR_WIN="$(wslpath -w "${SDK_ROOT_WSL}" 2>/dev/null || printf '%s' "${SDK_ROOT_WSL}")"
SDK_DIR_WIN="${SDK_DIR_WIN//\\//}/windows_amd64"

rm -rf "${SDK_DIR_WSL}"
# Per-file "Installing:" lines go to stdout (dropped); errors go to stderr (kept).
cmake.exe --install build --prefix "${SDK_DIR_WIN}" --config Release >/dev/null

echo "Build and install OK: ${SDK_DIR_WIN}"
