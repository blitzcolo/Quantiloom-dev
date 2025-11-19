#!/bin/bash
# ============================================================================
# Quantiloom M1 - Shader Compilation Script
# ============================================================================
# Compiles all HLSL ray tracing shaders to SPIR-V using DXC
# ============================================================================

set -e  # Exit on error

echo "========================================="
echo "  Quantiloom Shader Compilation"
echo "========================================="
echo "You need to run it from the root folder of the project (where the main CMakeLists.txt is located)"

# Check if DXC is available
if ! command -v dxc &> /dev/null; then
    echo "ERROR: DXC not found in PATH"
    echo ""
    echo "Please install DXC:"
    echo "  - Windows: Vulkan SDK (https://vulkan.lunarg.com/)"
    echo "  - Linux:   sudo apt install dxc"
    echo "  - macOS:   brew install dxc"
    echo ""
    exit 1
fi

echo "DXC found: $(command -v dxc)"
echo ""

# Compilation flags
FLAGS="-spirv -T lib_6_3 -fspv-target-env=vulkan1.3"

# Compile raygen shader
echo "[1/3] Compiling raygen.rgen..."
dxc $FLAGS -Fo src/shaders/raygen.spv src/shaders/raygen.rgen
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/raygen.spv created"
else
    echo "      ✗ Failed to compile src/shaders/raygen.rgen"
    exit 1
fi

# Compile closesthit shader
echo "[2/3] Compiling closesthit.rchit..."
dxc $FLAGS -Fo src/shaders/closesthit.spv src/shaders/closesthit.rchit
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/closesthit.spv created"
else
    echo "      ✗ Failed to compile src/shaders/closesthit.rchit"
    exit 1
fi

# Compile miss shader
echo "[3/3] Compiling miss.rmiss..."
dxc $FLAGS -Fo src/shaders/miss.spv src/shaders/miss.rmiss
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/miss.spv created"
else
    echo "      ✗ Failed to compile src/shaders/miss.rmiss"
    exit 1
fi

echo ""
echo "========================================="
echo "  All shaders compiled successfully!"
echo "========================================="
echo ""
echo "You can now run the M1 test application."
