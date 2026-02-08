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
echo "[1/12] Compiling raygen.rgen..."
dxc $FLAGS -Fo src/shaders/raygen.spv src/shaders/raygen.rgen
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/raygen.spv created"
else
    echo "      ✗ Failed to compile src/shaders/raygen.rgen"
    exit 1
fi

# Compile closesthit shader
echo "[2/12] Compiling closesthit.rchit..."
dxc $FLAGS -Fo src/shaders/closesthit.spv src/shaders/closesthit.rchit
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/closesthit.spv created"
else
    echo "      ✗ Failed to compile src/shaders/closesthit.rchit"
    exit 1
fi

# Compile miss shader
echo "[3/12] Compiling miss.rmiss..."
dxc $FLAGS -Fo src/shaders/miss.spv src/shaders/miss.rmiss
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/miss.spv created"
else
    echo "      ✗ Failed to compile src/shaders/miss.rmiss"
    exit 1
fi

# Compile shadow_miss shader
echo "[4/12] Compiling shadow_miss.rmiss..."
dxc $FLAGS -Fo src/shaders/shadow_miss.spv src/shaders/shadow_miss.rmiss
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/shadow_miss.spv created"
else
    echo "      ✗ Failed to compile src/shaders/shadow_miss.rmiss"
    exit 1
fi

# Compile CLAHE compute shaders (3 passes)
echo "[5/12] Compiling clahe_histogram.comp..."
dxc $COMP_FLAGS -E main -D CLAHE_PASS_HISTOGRAM -Fo src/shaders/clahe_histogram.spv src/shaders/clahe.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/clahe_histogram.spv created"
else
    echo "      ✗ Failed to compile clahe_histogram"
    exit 1
fi

echo "[6/12] Compiling clahe_cdf.comp..."
dxc $COMP_FLAGS -E main -D CLAHE_PASS_CDF -Fo src/shaders/clahe_cdf.spv src/shaders/clahe.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/clahe_cdf.spv created"
else
    echo "      ✗ Failed to compile clahe_cdf"
    exit 1
fi

echo "[7/12] Compiling clahe_apply.comp..."
dxc $COMP_FLAGS -E main -D CLAHE_PASS_APPLY -Fo src/shaders/clahe_apply.spv src/shaders/clahe.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/clahe_apply.spv created"
else
    echo "      ✗ Failed to compile clahe_apply"
    exit 1
fi

# Compile GPU Sensor compute shaders (5 passes)
echo "[8/12] Compiling sensor_radiance_to_electrons.comp..."
dxc $COMP_FLAGS -E main -Fo src/shaders/sensor_radiance_to_electrons.spv src/shaders/sensor_radiance_to_electrons.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/sensor_radiance_to_electrons.spv created"
else
    echo "      ✗ Failed to compile sensor_radiance_to_electrons"
    exit 1
fi

echo "[9/12] Compiling sensor_poisson_noise.comp..."
dxc $COMP_FLAGS -E main -Fo src/shaders/sensor_poisson_noise.spv src/shaders/sensor_poisson_noise.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/sensor_poisson_noise.spv created"
else
    echo "      ✗ Failed to compile sensor_poisson_noise"
    exit 1
fi

echo "[10/12] Compiling sensor_psf_blur_horizontal.comp..."
dxc $COMP_FLAGS -E main -D SENSOR_PSF_HORIZONTAL -Fo src/shaders/sensor_psf_blur_horizontal.spv src/shaders/sensor_psf_blur.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/sensor_psf_blur_horizontal.spv created"
else
    echo "      ✗ Failed to compile sensor_psf_blur_horizontal"
    exit 1
fi

echo "[11/12] Compiling sensor_psf_blur_vertical.comp..."
dxc $COMP_FLAGS -E main -D SENSOR_PSF_VERTICAL -Fo src/shaders/sensor_psf_blur_vertical.spv src/shaders/sensor_psf_blur.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/sensor_psf_blur_vertical.spv created"
else
    echo "      ✗ Failed to compile sensor_psf_blur_vertical"
    exit 1
fi

echo "[12/12] Compiling sensor_quantize_to_radiance.comp..."
dxc $COMP_FLAGS -E main -Fo src/shaders/sensor_quantize_to_radiance.spv src/shaders/sensor_quantize_to_radiance.comp.hlsl
if [ $? -eq 0 ]; then
    echo "      ✓ src/shaders/sensor_quantize_to_radiance.spv created"
else
    echo "      ✗ Failed to compile sensor_quantize_to_radiance"
    exit 1
fi

echo ""
echo "========================================="
echo "  All shaders compiled successfully!"
echo "========================================="
echo ""

