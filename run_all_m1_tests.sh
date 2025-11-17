#!/bin/bash
# ============================================================================
# Quantiloom M1 - Automated Test Suite Runner
# ============================================================================
# Runs all 3 scene presets and generates comparison images
# ============================================================================

set -e  # Exit on error

echo "========================================="
echo "  Quantiloom M1 Automated Test Suite"
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
    echo "  ./compile_shaders.sh  (Linux/macOS)"
    echo "  compile_shaders.bat   (Windows)"
    exit 1
fi

# Copy shaders to working directory (where executable runs)
echo "[Setup] Copying shaders to working directory..."
cp src/shaders/*.spv .
echo "        ✓ Shaders copied"
echo ""

# Create output directory for test results
mkdir -p m1_test_results
echo "[Setup] Output directory: m1_test_results/"
echo ""

# ============================================================================
# Test 1: Cornell Box Scene
# ============================================================================
echo "========================================="
echo "  Test 1/3: Cornell Box Scene"
echo "========================================="
echo "Scene:    Minimal (ground + single cube)"
echo "Expected: Gray cube centered, lit from upper-left"
echo ""

$EXECUTABLE --scene cornell --output m1_test_results/cornell_box.exr

if [ -f "m1_test_results/cornell_box.exr" ]; then
    echo ""
    echo "✓ Test 1 PASSED: cornell_box.exr generated"
else
    echo "✗ Test 1 FAILED: No output file"
    exit 1
fi

# ============================================================================
# Test 2: Multi-Object Scene
# ============================================================================
echo ""
echo "========================================="
echo "  Test 2/3: Multi-Object Scene"
echo "========================================="
echo "Scene:    Ground + tall box + cube + sphere"
echo "Expected: Multiple objects with correct shading"
echo ""

$EXECUTABLE --scene multiobject --output m1_test_results/multi_object.exr

if [ -f "m1_test_results/multi_object.exr" ]; then
    echo ""
    echo "✓ Test 2 PASSED: multi_object.exr generated"
else
    echo "✗ Test 2 FAILED: No output file"
    exit 1
fi

# ============================================================================
# Test 3: Lighting Test Scene
# ============================================================================
echo ""
echo "========================================="
echo "  Test 3/3: Lighting Test Scene"
echo "========================================="
echo "Scene:    Ground + 5 cubes in a row"
echo "Expected: Left cubes brighter than right (sun from left)"
echo ""

$EXECUTABLE --scene lighting --output m1_test_results/lighting_test.exr

if [ -f "m1_test_results/lighting_test.exr" ]; then
    echo ""
    echo "✓ Test 3 PASSED: lighting_test.exr generated"
else
    echo "✗ Test 3 FAILED: No output file"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================="
echo "  All Tests Completed Successfully!"
echo "========================================="
echo ""
echo "Output files (in m1_test_results/):"
ls -lh m1_test_results/*.exr
echo ""
echo "========================================="
echo "  Validation Checklist"
echo "========================================="
echo ""
echo "For cornell_box.exr:"
echo "  [  ] Cube is visible and centered"
echo "  [  ] Top face brighter than side faces (sun from upper-left)"
echo "  [  ] Ground plane visible around cube"
echo "  [  ] Blue sky background"
echo ""
echo "For multi_object.exr:"
echo "  [  ] All 3 objects visible (tall box, cube, sphere)"
echo "  [  ] Each object has different brightness on different faces"
echo "  [  ] No all-same-brightness artifacts"
echo ""
echo "For lighting_test.exr:"
echo "  [  ] 5 cubes in a row are visible"
echo "  [  ] Leftmost cube BRIGHTEST (faces sun)"
echo "  [  ] Rightmost cube DARKEST (faces away)"
echo "  [  ] Clear lighting gradient from left to right"
echo ""
echo "If any validation fails, the geometric normal calculation"
echo "may still have issues. Expected behavior:"
echo "  - Faces toward sun: bright"
echo "  - Faces away from sun: dark (only sky radiance)"
echo "  - Lighting should vary across cube faces"
echo ""
echo "M1 is ready for M2 if all validations pass!"
