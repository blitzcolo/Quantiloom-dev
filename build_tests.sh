#!/bin/bash
set -e  # Exit on error

# Clean and recreate build directory
#rm -rf out-test
mkdir -p out-test
cd out-test

echo "========================================"
echo "Configuring Quantiloom Tests..."
echo "========================================"

# Configure with tests enabled
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DENABLE_COVERAGE=ON \
    -DQUANTILOOM_BUILD_TESTS=ON \
    -DQUANTILOOM_USE_BC7ENC=OFF \
    -DQUANTILOOM_USE_OPENUSD=OFF

echo ""
echo "========================================"
echo "Building Tests..."
echo "========================================"

# Build tests
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    cmake --build . --target libquantiloom_tests -j$(nproc)

echo ""
echo "========================================"
echo "Running Tests..."
echo "========================================"

# Run tests
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    ./tests/libquantiloom_tests

cd ..
