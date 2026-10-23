#!/bin/bash
# ============================================================================
# Quantiloom M1 - Shader Compilation Script
# ============================================================================
# Compiles all HLSL and GLSL shaders to SPIR-V using DXC and glslc
# ============================================================================
# MANUAL FALLBACK. src/shaders/CMakeLists.txt is the authoritative list; this
# script mirrors it (27 invocations: 25 HLSL via DXC, 2 GLSL via glslc). Keep
# the flags and the list in step with DXC_RT_FLAGS / DXC_COMP_FLAGS /
# DXC_RQ_FLAGS and the quantiloom_compile_hlsl / COMPUTE_SHADER_SOURCES
# blocks there.
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
    echo "Please install Vulkan SDK from https://vulkan.lunarg.com/"
    echo ""
    exit 1
fi
echo "DXC found: $(command -v dxc)"

# Check if glslc is available (GLSL compute shaders)
if ! command -v glslc &> /dev/null; then
    echo "ERROR: glslc not found in PATH"
    echo ""
    echo "Please install Vulkan SDK which includes glslc"
    echo ""
    exit 1
fi
echo "glslc found: $(command -v glslc)"
echo ""

# Compilation flags
FLAGS="-spirv -T lib_6_3 -fspv-target-env=vulkan1.2"
COMP_FLAGS="-spirv -T cs_6_0 -fspv-target-env=vulkan1.2"
RQ_FLAGS="-spirv -T cs_6_5 -fspv-target-env=vulkan1.2"

compile() {
    local label="$1"; shift
    echo "$label"
    if "$@"; then
        return 0
    else
        echo "      FAILED: $*"
        return 1
    fi
}

compile "[1/27] Compiling fusion_geometry.rayq..." \
    dxc $RQ_FLAGS -E main -Fo src/shaders/fusion_geometry.spv src/shaders/fusion_geometry.rayq.hlsl || exit 1
echo "      OK src/shaders/fusion_geometry.spv created"

compile "[2/27] Compiling raygen.rgen..." \
    dxc $FLAGS -Fo src/shaders/raygen.spv src/shaders/raygen.rgen || exit 1
echo "      OK src/shaders/raygen.spv created"

compile "[3/27] Compiling closesthit.rchit..." \
    dxc $FLAGS -Fo src/shaders/closesthit.spv src/shaders/closesthit.rchit || exit 1
echo "      OK src/shaders/closesthit.spv created"

compile "[4/27] Compiling anyhit.rahit..." \
    dxc $FLAGS -Fo src/shaders/anyhit.spv src/shaders/anyhit.rahit || exit 1
echo "      OK src/shaders/anyhit.spv created"

compile "[5/27] Compiling miss.rmiss..." \
    dxc $FLAGS -Fo src/shaders/miss.spv src/shaders/miss.rmiss || exit 1
echo "      OK src/shaders/miss.spv created"

compile "[6/27] Compiling shadow_miss.rmiss..." \
    dxc $FLAGS -Fo src/shaders/shadow_miss.spv src/shaders/shadow_miss.rmiss || exit 1
echo "      OK src/shaders/shadow_miss.spv created"

compile "[7/27] Compiling clahe_histogram.comp..." \
    dxc $COMP_FLAGS -E main -D CLAHE_PASS_HISTOGRAM -Fo src/shaders/clahe_histogram.spv src/shaders/clahe.comp.hlsl || exit 1
echo "      OK src/shaders/clahe_histogram.spv created"

compile "[8/27] Compiling clahe_cdf.comp..." \
    dxc $COMP_FLAGS -E main -D CLAHE_PASS_CDF -Fo src/shaders/clahe_cdf.spv src/shaders/clahe.comp.hlsl || exit 1
echo "      OK src/shaders/clahe_cdf.spv created"

compile "[9/27] Compiling clahe_apply.comp..." \
    dxc $COMP_FLAGS -E main -D CLAHE_PASS_APPLY -Fo src/shaders/clahe_apply.spv src/shaders/clahe.comp.hlsl || exit 1
echo "      OK src/shaders/clahe_apply.spv created"

compile "[10/27] Compiling display_range_extents.comp..." \
    dxc $COMP_FLAGS -E main -D DISPLAY_RANGE_PASS_EXTENTS -Fo src/shaders/display_range_extents.comp.spv src/shaders/display_range.comp.hlsl || exit 1
echo "      OK src/shaders/display_range_extents.comp.spv created"

compile "[11/27] Compiling display_range_histogram.comp..." \
    dxc $COMP_FLAGS -E main -D DISPLAY_RANGE_PASS_HISTOGRAM -Fo src/shaders/display_range_histogram.comp.spv src/shaders/display_range.comp.hlsl || exit 1
echo "      OK src/shaders/display_range_histogram.comp.spv created"

compile "[12/27] Compiling camera_psf.comp..." \
    dxc $COMP_FLAGS -E main -Fo src/shaders/camera_psf.comp.spv src/shaders/camera_psf.comp.hlsl || exit 1
echo "      OK src/shaders/camera_psf.comp.spv created"

compile "[13/27] Compiling camera_readout.comp..." \
    dxc $COMP_FLAGS -E main -Fo src/shaders/camera_readout.comp.spv src/shaders/camera_readout.comp.hlsl || exit 1
echo "      OK src/shaders/camera_readout.comp.spv created"

compile "[14/27] Compiling camera_stats_tiles.comp..." \
    dxc $COMP_FLAGS -E main -D CAMERA_STATS_PHASE_TILES -Fo src/shaders/camera_stats_tiles.comp.spv src/shaders/camera_stats.comp.hlsl || exit 1
echo "      OK src/shaders/camera_stats_tiles.comp.spv created"

compile "[15/27] Compiling camera_stats_reduce.comp..." \
    dxc $COMP_FLAGS -E main -D CAMERA_STATS_PHASE_REDUCE -Fo src/shaders/camera_stats_reduce.comp.spv src/shaders/camera_stats.comp.hlsl || exit 1
echo "      OK src/shaders/camera_stats_reduce.comp.spv created"

compile "[16/27] Compiling camera_stats_hist.comp..." \
    dxc $COMP_FLAGS -E main -D CAMERA_STATS_PHASE_HISTOGRAM -Fo src/shaders/camera_stats_hist.comp.spv src/shaders/camera_stats.comp.hlsl || exit 1
echo "      OK src/shaders/camera_stats_hist.comp.spv created"

compile "[17/27] Compiling camera_cdf.comp..." \
    dxc $COMP_FLAGS -E main -Fo src/shaders/camera_cdf.comp.spv src/shaders/camera_cdf.comp.hlsl || exit 1
echo "      OK src/shaders/camera_cdf.comp.spv created"

compile "[18/27] Compiling camera_demosaic.comp..." \
    dxc $COMP_FLAGS -E main -Fo src/shaders/camera_demosaic.comp.spv src/shaders/camera_demosaic.comp.hlsl || exit 1
echo "      OK src/shaders/camera_demosaic.comp.spv created"

compile "[19/27] Compiling camera_rng_vectors.comp..." \
    dxc $COMP_FLAGS -E main -Fo src/shaders/camera_rng_vectors.comp.spv src/shaders/camera_rng_vectors.comp.hlsl || exit 1
echo "      OK src/shaders/camera_rng_vectors.comp.spv created"

compile "[20/27] Compiling camera_fast_rgb.comp..." \
    dxc $COMP_FLAGS -E main -Fo src/shaders/camera_fast_rgb.comp.spv src/shaders/camera_fast_rgb.comp.hlsl || exit 1
echo "      OK src/shaders/camera_fast_rgb.comp.spv created"

compile "[21/27] Compiling camera_dynamic.comp..." \
    dxc $COMP_FLAGS -E main -Fo src/shaders/camera_dynamic.comp.spv src/shaders/camera_dynamic.comp.hlsl || exit 1
echo "      OK src/shaders/camera_dynamic.comp.spv created"

compile "[22/27] Compiling camera_hsv.comp..." \
    dxc $COMP_FLAGS -E main -Fo src/shaders/camera_hsv.comp.spv src/shaders/camera_hsv.comp.hlsl || exit 1
echo "      OK src/shaders/camera_hsv.comp.spv created"

compile "[23/27] Compiling pick.rayq..." \
    dxc $RQ_FLAGS -E main -Fo src/shaders/pick.spv src/shaders/pick.rayq.hlsl || exit 1
echo "      OK src/shaders/pick.spv created"

compile "[24/27] Compiling thermal_exchange.rayq..." \
    dxc $RQ_FLAGS -E main -Fo src/shaders/thermal_exchange.spv src/shaders/thermal_exchange.rayq.hlsl || exit 1
echo "      OK src/shaders/thermal_exchange.spv created"

compile "[25/27] Compiling thermal_step.comp..." \
    dxc $COMP_FLAGS -E main -Fo src/shaders/thermal_step.spv src/shaders/thermal_step.comp.hlsl || exit 1
echo "      OK src/shaders/thermal_step.spv created"

compile "[26/27] Compiling ibl_brdf_lut.comp..." \
    glslc -fshader-stage=compute -o src/shaders/ibl_brdf_lut.spv src/shaders/ibl_brdf_lut.comp || exit 1
echo "      OK src/shaders/ibl_brdf_lut.spv created"

compile "[27/27] Compiling spectral_reconstruct.comp..." \
    glslc -fshader-stage=compute -o src/shaders/spectral_reconstruct.spv src/shaders/spectral_reconstruct.comp || exit 1
echo "      OK src/shaders/spectral_reconstruct.spv created"

echo ""
echo "========================================="
echo "  All shaders compiled successfully!"
echo "========================================="
echo ""
