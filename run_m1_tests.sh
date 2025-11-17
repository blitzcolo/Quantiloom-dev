#!/bin/bash
# ============================================================================
# Quantiloom M1 - Test Runner Script
# ============================================================================
# Runs M1 tests with all 3 scene presets and validates output
# ============================================================================

set -e  # Exit on error

echo "========================================="
echo "  Quantiloom M1 Test Suite"
echo "========================================="
echo ""

# Determine build directory
if [ -d "build" ]; then
    BUILD_DIR="build"
elif [ -d "out" ]; then
    BUILD_DIR="out"
else
    echo "ERROR: No build directory found (build/ or out/)"
    echo "Please run CMake and build the project first:"
    echo "  mkdir build && cd build"
    echo "  cmake .."
    echo "  cmake --build ."
    exit 1
fi

# Check if executable exists
EXECUTABLE="$BUILD_DIR/src/app/quantiloom_m1_test"
if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: M1 test executable not found at $EXECUTABLE"
    echo "Please build the project first:"
    echo "  cd $BUILD_DIR && cmake --build ."
    exit 1
fi

# Check if shaders are compiled
if [ ! -f "src/shaders/raygen.spv" ] || [ ! -f "src/shaders/closesthit.spv" ] || [ ! -f "src/shaders/miss.spv" ]; then
    echo "ERROR: Compiled shaders not found in src/shaders/"
    echo "Please compile shaders first:"
    echo "  cd src/shaders"
    echo "  ./compile_shaders.sh"
    exit 1
fi

# Copy shaders to working directory (where executable runs)
echo "[Setup] Copying shaders to working directory..."
cp src/shaders/*.spv .
echo "        ✓ Shaders copied"
echo ""

# Test 1: Cornell Box
echo "========================================="
echo "  Test 1: Cornell Box Scene"
echo "========================================="
echo "Scene:    Minimal (ground + single cube)"
echo "Expected: Gray cube centered, lit from upper-left"
echo ""

# Note: This script assumes you'll manually change SCENE_PRESET in main_m1_test.cpp
# For automated testing, you'd need to add command-line arguments or config file
echo "[INFO] Running M1 test with CornellBox preset..."
echo "[INFO] (Ensure SCENE_PRESET = CornellBox in main_m1_test.cpp)"
echo ""

$EXECUTABLE

if [ -f "m1_output.exr" ]; then
    mv m1_output.exr m1_cornell_box.exr
    echo ""
    echo "✓ Test 1 completed: m1_cornell_box.exr"
else
    echo "✗ Test 1 failed: No output file generated"
    exit 1
fi

echo ""
echo "========================================="
echo "  All Tests Completed"
echo "========================================="
echo ""
echo "Output files:"
ls -lh m1_*.exr
echo ""
echo "Please review the images and verify:"
echo "  1. Lighting direction is correct (not all faces same brightness)"
echo "  2. Geometric normals are working (faces facing away from sun are darker)"
echo "  3. No artifacts or black pixels (except shadows)"
echo ""
echo "Next steps:"
echo "  1. Manually change SCENE_PRESET in main_m1_test.cpp to MultiObject"
echo "  2. Recompile and run this script again"
echo "  3. Repeat for LightingTest preset"
echo ""
echo "Or better: Add command-line argument support to main_m1_test.cpp"
echo "to automate testing all 3 presets in one run."
