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
FLAGS="-spirv -T lib_6_3 -fspv-target-env=vulkan1.2"
COMP_FLAGS="-spirv -T cs_6_0 -fspv-target-env=vulkan1.2"

# Compile raygen shader
echo "[1/7] Compiling raygen.rgen..."
dxc $FLAGS -Fo src/shaders/raygen.spv src/shaders/raygen.rgen
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/raygen.spv created"
else
    echo "      ✗ Failed to compile src/shaders/raygen.rgen"
    exit 1
fi

# Compile closesthit shader
echo "[2/7] Compiling closesthit.rchit..."
dxc $FLAGS -Fo src/shaders/closesthit.spv src/shaders/closesthit.rchit
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/closesthit.spv created"
else
    echo "      ✗ Failed to compile src/shaders/closesthit.rchit"
    exit 1
fi

# Compile miss shader
echo "[3/7] Compiling miss.rmiss..."
dxc $FLAGS -Fo src/shaders/miss.spv src/shaders/miss.rmiss
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/miss.spv created"
else
    echo "      ✗ Failed to compile src/shaders/miss.rmiss"
    exit 1
fi

# Compile shadow_miss shader
echo "[4/7] Compiling shadow_miss.rmiss..."
dxc $FLAGS -Fo src/shaders/shadow_miss.spv src/shaders/shadow_miss.rmiss
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/shadow_miss.spv created"
else
    echo "      ✗ Failed to compile src/shaders/shadow_miss.rmiss"
    exit 1
fi

# Compile CLAHE compute shaders (3 passes)
echo "[5/7] Compiling clahe_histogram.comp..."
dxc $COMP_FLAGS -E main -D CLAHE_PASS_HISTOGRAM -Fo src/shaders/clahe_histogram.spv src/shaders/clahe.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/clahe_histogram.spv created"
else
    echo "      ✗ Failed to compile clahe_histogram"
    exit 1
fi

echo "[6/7] Compiling clahe_cdf.comp..."
dxc $COMP_FLAGS -E main -D CLAHE_PASS_CDF -Fo src/shaders/clahe_cdf.spv src/shaders/clahe.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/clahe_cdf.spv created"
else
    echo "      ✗ Failed to compile clahe_cdf"
    exit 1
fi

echo "[7/7] Compiling clahe_apply.comp..."
dxc $COMP_FLAGS -E main -D CLAHE_PASS_APPLY -Fo src/shaders/clahe_apply.spv src/shaders/clahe.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/clahe_apply.spv created"
else
    echo "      ✗ Failed to compile clahe_apply"
    exit 1
fi

echo ""
echo "========================================="
echo "  All shaders compiled successfully!"
echo "========================================="
echo ""

