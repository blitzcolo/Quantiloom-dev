#!/bin/sh
# ============================================================================
# DXC Shader Compiler Setup Script (Linux/macOS)
# ============================================================================
# Downloads and extracts DirectXShaderCompiler (DXC) from GitHub releases
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"
DXC_DIR="$TOOLS_DIR/dxc"

DXC_VERSION="v1.8.2505.1"  # Latest stable release as of Nov. 2025
DXC_ARCHIVE_NAME="2025_07_14"
PLATFORM="$(uname -s)"

echo "[DXC Setup] Detecting platform: $PLATFORM"

# Determine download URL based on platform
case "$PLATFORM" in
    Linux*)
        DXC_ARCHIVE="linux-dxc-$DXC_ARCHIVE_NAME.x86_64.tar.gz"
        DXC_URL="https://github.com/microsoft/DirectXShaderCompiler/releases/download/$DXC_VERSION/$DXC_ARCHIVE"
        DXC_BIN_PATH="$DXC_DIR/bin/dxc"
        ;;
    Darwin*)
        DXC_ARCHIVE="macos-dxc-$DXC_ARCHIVE_NAME.tar.gz"
        DXC_URL="https://github.com/microsoft/DirectXShaderCompiler/releases/download/$DXC_VERSION/$DXC_ARCHIVE"
        DXC_BIN_PATH="$DXC_DIR/bin/dxc"
        ;;
    *)
        echo "[ERROR] Unsupported platform: $PLATFORM"
        echo "        Use setup_dxc.ps1 on Windows"
        exit 1
        ;;
esac

# Check if DXC is already installed
if [ -f "$DXC_BIN_PATH" ]; then
    echo "[DXC Setup] DXC already installed at: $DXC_BIN_PATH"
    "$DXC_BIN_PATH" --version || true
    echo "[DXC Setup] To reinstall, delete: $DXC_DIR"
    exit 0
fi

echo "[DXC Setup] DXC not found, downloading..."
echo "[DXC Setup] Version: $DXC_VERSION"
echo "[DXC Setup] URL: $DXC_URL"

# Create DXC directory
mkdir -p "$DXC_DIR"
cd "$DXC_DIR"

# Download DXC archive
echo "[DXC Setup] Downloading DXC..."
if command -v curl > /dev/null 2>&1; then
    curl -L -o "$DXC_ARCHIVE" "$DXC_URL"
elif command -v wget > /dev/null 2>&1; then
    wget -O "$DXC_ARCHIVE" "$DXC_URL"
else
    echo "[ERROR] Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Extract archive
echo "[DXC Setup] Extracting DXC..."
tar -xzf "$DXC_ARCHIVE"
rm "$DXC_ARCHIVE"

# Verify installation
if [ -f "$DXC_BIN_PATH" ]; then
    chmod +x "$DXC_BIN_PATH"
    echo "[DXC Setup] Installation successful!"
    echo "[DXC Setup] DXC binary: $DXC_BIN_PATH"
    "$DXC_BIN_PATH" --version || true
else
    echo "[ERROR] DXC binary not found after extraction"
    echo "        Expected: $DXC_BIN_PATH"
    exit 1
fi

echo "[DXC Setup] Done. You can now build shaders with CMake."
