#!/bin/bash
# ============================================================================
# OpenUSD Build Script for Quantiloom (Linux)
# ============================================================================
#
# This script builds a minimal OpenUSD installation without Python support,
# imaging (Hydra), or tools. Only core USD libraries are built.
#
# Usage:
#   ./build_openusd.sh [--install-dir DIR] [--version TAG] [--build-type TYPE]
#
# Examples:
#   ./build_openusd.sh
#   ./build_openusd.sh --install-dir /opt/openusd --version v25.11
#
# ============================================================================

set -e

# Default values
INSTALL_DIR="/opt/openusd"
USD_VERSION="v25.08"
BUILD_TYPE="Release"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --version)
            USD_VERSION="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--install-dir DIR] [--version TAG] [--build-type TYPE]"
            echo ""
            echo "Options:"
            echo "  --install-dir DIR    Installation directory (default: /opt/openusd)"
            echo "  --version TAG        OpenUSD version tag (default: v25.08)"
            echo "  --build-type TYPE    CMake build type: Release, Debug (default: Release)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================================"
echo " OpenUSD Build Script for Quantiloom (Linux)"
echo "============================================================================"
echo ""
echo "  Version:     $USD_VERSION"
echo "  Install Dir: $INSTALL_DIR"
echo "  Build Type:  $BUILD_TYPE"
echo ""

# [1/5] Check prerequisites
echo "[1/5] Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.9+"
    exit 1
fi
echo "  Python: $(python3 --version)"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake not found. Please install CMake 3.26+"
    exit 1
fi
echo "  CMake: $(cmake --version | head -n1)"

# Check GCC
if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found. Please install build-essential"
    exit 1
fi
echo "  GCC: $(g++ --version | head -n1)"

# Check TBB
if ! pkg-config --exists tbb 2>/dev/null; then
    echo "WARNING: TBB not found via pkg-config. OpenUSD will build TBB from source."
fi

# [2/5] Prepare source
echo ""
echo "[2/5] Preparing OpenUSD source..."

USD_SOURCE_DIR="/tmp/OpenUSD-$USD_VERSION"

if [ -d "$USD_SOURCE_DIR" ]; then
    echo "  Using existing source at: $USD_SOURCE_DIR"
else
    echo "  Cloning OpenUSD $USD_VERSION..."
    git clone --depth 1 --branch "$USD_VERSION" https://github.com/PixarAnimationStudios/OpenUSD.git "$USD_SOURCE_DIR"
fi

# [3/5] Build OpenUSD
echo ""
echo "[3/5] Building OpenUSD (this may take 30-60 minutes)..."
echo ""
echo "  Build options:"
echo "    --no-python       (disable Python bindings)"
echo "    --no-imaging      (disable Hydra/usdview)"
echo "    --no-usdview      (disable usdview tool)"
echo "    --no-examples     (skip examples)"
echo "    --no-tutorials    (skip tutorials)"
echo "    --no-tools        (skip command-line tools)"
echo "    --no-docs         (skip documentation)"
echo ""

# Create install directory
sudo mkdir -p "$INSTALL_DIR" 2>/dev/null || mkdir -p "$INSTALL_DIR"

# Determine number of parallel jobs
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "  Using $NPROC parallel jobs"
echo ""

# Run build script
python3 "$USD_SOURCE_DIR/build_scripts/build_usd.py" \
    --no-python \
    --no-imaging \
    --no-usdview \
    --no-examples \
    --no-tutorials \
    --no-tools \
    --no-docs \
    --build-variant "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" \
    -j "$NPROC" \
    "$INSTALL_DIR"

# [4/5] Verify installation
echo ""
echo "[4/5] Verifying installation..."

REQUIRED_LIBS=(
    "libusd.so"
    "libusdGeom.so"
    "libusdShade.so"
    "libsdf.so"
    "libtf.so"
    "libgf.so"
    "libar.so"
    "libplug.so"
    "libvt.so"
    "libwork.so"
    "libarch.so"
)

LIB_DIR="$INSTALL_DIR/lib"
MISSING_LIBS=()

for lib in "${REQUIRED_LIBS[@]}"; do
    if [ -f "$LIB_DIR/$lib" ]; then
        echo "  [OK] $lib"
    else
        echo "  [MISSING] $lib"
        MISSING_LIBS+=("$lib")
    fi
done

if [ ${#MISSING_LIBS[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: Some required libraries are missing. Build may have failed partially."
fi

# [5/5] Print summary
echo ""
echo "[5/5] Build complete!"
echo ""
echo "============================================================================"
echo " OpenUSD installed to: $INSTALL_DIR"
echo "============================================================================"
echo ""
echo "To use in Quantiloom CMake:"
echo ""
echo "  cmake -B build -DUSD_ROOT=\"$INSTALL_DIR\" ..."
echo ""
echo "Or set environment variable:"
echo ""
echo "  export USD_ROOT=\"$INSTALL_DIR\""
echo ""
echo "You may also need to add the lib directory to LD_LIBRARY_PATH:"
echo ""
echo "  export LD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\$LD_LIBRARY_PATH\""
echo ""
