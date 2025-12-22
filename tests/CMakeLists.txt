# ============================================================================
# Quantiloom - Unit Tests CMakeLists.txt
# ============================================================================
# Builds comprehensive unit tests using Google Test framework
#
# Test organization:
# - test_core/      : Tests for libQuantiloom/core (Config, Log, Types, Image, etc.)
# - test_scene/     : Tests for libQuantiloom/scene (Material, Camera, Scene, etc.)
# - test_io/        : Tests for libQuantiloom/io (ImageIO, SpectralIO, GltfLoader, etc.)
# - test_renderer/  : Tests for libQuantiloom/renderer (Vulkan components - optional)
# - test_hs_core/   : Tests for HS-core algorithms (MIS, delta-tracking, etc.)
# - test_postprocess/: Tests for sensor chain and noise models
#
# RECOMMENDED: Build from project root (tests will be built automatically)
#   cd /path/to/Quantiloom-dev
#   mkdir build && cd build
#   cmake ..
#   cmake --build . --target libquantiloom_tests
#   ctest --output-on-failure
#
# ALTERNATIVE: Standalone build (if libQuantiloom is already built)
#   cd tests
#   mkdir build && cd build
#   cmake ..
#   cmake --build .
# ============================================================================

cmake_minimum_required(VERSION 3.20)

# ============================================================================
# Detect Build Context
# ============================================================================
# Check if this is a standalone build or part of the main project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(QUANTILOOM_STANDALONE_TESTS ON)
    message(STATUS "Building tests in STANDALONE mode")

    # Define project only for standalone builds
    project(libQuantiloomTests
        VERSION 0.0.1
        DESCRIPTION "libQuantiloom Unit Tests"
        LANGUAGES CXX
    )

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
    set(QUANTILOOM_STANDALONE_TESTS OFF)
    message(STATUS "Building tests as SUBPROJECT of main libQuantiloom project")
endif()

# ============================================================================
# Enable Testing
# ============================================================================

enable_testing()

# ============================================================================
# Find or Fetch Google Test
# ============================================================================
# Multi-platform strategy:
# 1. Try to find installed GTest (vcpkg, system package, etc.)
# 2. If not found, automatically fetch from GitHub (recommended for CI/CD)
# ============================================================================

option(QUANTILOOM_FETCH_GTEST "Automatically fetch GTest if not found" ON)

# Try to find GTest first
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    if(QUANTILOOM_FETCH_GTEST)
        message(STATUS "GTest not found locally. Fetching from GitHub...")

        include(FetchContent)

        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.17.0  # Latest stable release
            GIT_SHALLOW    TRUE
        )

        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        # Disable installation of GTest
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

        # Fetch and make available
        FetchContent_MakeAvailable(googletest)

        # Create aliases to match find_package interface
        if(NOT TARGET GTest::GTest)
            add_library(GTest::GTest ALIAS gtest)
        endif()

        if(NOT TARGET GTest::Main)
            add_library(GTest::Main ALIAS gtest_main)
        endif()

        message(STATUS "GTest fetched successfully (v1.17.0)")
    else()
        message(FATAL_ERROR
            "GTest not found! Please either:\n"
            "  1. Install GTest via vcpkg: vcpkg install gtest\n"
            "  2. Enable auto-fetch: cmake .. -DQUANTILOOM_FETCH_GTEST=ON\n"
            "  3. Specify GTest path: cmake .. -DGTest_DIR=/path/to/gtest"
        )
    endif()
else()
    message(STATUS "GTest found: ${GTEST_LIBRARIES}")

    # Ensure GTest::GTest and GTest::Main targets exist
    if(NOT TARGET GTest::GTest AND TARGET GTest::gtest)
        add_library(GTest::GTest ALIAS GTest::gtest)
    endif()

    if(NOT TARGET GTest::Main AND TARGET GTest::gtest_main)
        add_library(GTest::Main ALIAS GTest::gtest_main)
    endif()
endif()

# ============================================================================
# Include Directories
# ============================================================================

if(QUANTILOOM_STANDALONE_TESTS)
    # Standalone mode: point to parent directory structure
    set(QUANTILOOM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
    message(STATUS "Quantiloom root directory: ${QUANTILOOM_ROOT_DIR}")

    include_directories(
        ${QUANTILOOM_ROOT_DIR}/src/libQuantiloom
        ${QUANTILOOM_ROOT_DIR}/vendor
    )

    # Add vendor dependencies manually for standalone build
    # GLM
    if(EXISTS "${QUANTILOOM_ROOT_DIR}/vendor/glm")
        include_directories(${QUANTILOOM_ROOT_DIR}/vendor/glm)
    endif()

    # toml++
    if(EXISTS "${QUANTILOOM_ROOT_DIR}/vendor/tomlplusplus/include")
        include_directories(${QUANTILOOM_ROOT_DIR}/vendor/tomlplusplus/include)
    endif()

    # spdlog
    if(EXISTS "${QUANTILOOM_ROOT_DIR}/vendor/spdlog/include")
        include_directories(${QUANTILOOM_ROOT_DIR}/vendor/spdlog/include)
    endif()
else()
    # Subproject mode: use main project's include directories
    include_directories(
        ${CMAKE_SOURCE_DIR}/src/libQuantiloom
        ${CMAKE_SOURCE_DIR}/vendor
    )
endif()

# ============================================================================
# Compiler Settings
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# Test Source Files
# ============================================================================

# Core tests
set(TEST_CORE_SOURCES
    test_core/test_types.cpp
    test_core/test_image.cpp
    test_core/test_spectral_data.cpp
    test_core/test_config.cpp
    test_core/test_spectral_cube.cpp
)

# Scene tests
set(TEST_SCENE_SOURCES
    test_scene/test_material.cpp
    test_scene/test_camera.cpp
    test_scene/test_mesh.cpp
    test_scene/test_scene.cpp
)

# IO tests
set(TEST_IO_SOURCES
    test_io/test_image_io.cpp
    test_io/test_spectral_io.cpp
    test_io/test_lut_loader.cpp
    test_io/test_gltf_loader.cpp
)

# Renderer tests (non-Vulkan components only)
set(TEST_RENDERER_SOURCES
    test_renderer/test_brdf_lut.cpp
    test_renderer/test_lighting_params.cpp
    test_renderer/test_acceleration_structure.cpp
)

# Postprocess tests (sensor simulation, multiband fusion)
set(TEST_POSTPROCESS_SOURCES
    test_postprocess/test_generic_sensor.cpp
    test_postprocess/test_multiband_fusion.cpp
)

# Test main (custom main function with logging initialization)
set(TEST_MAIN_SOURCE
    test_main.cpp
)

# Combine all test sources
set(ALL_TEST_SOURCES
    ${TEST_MAIN_SOURCE}
    ${TEST_CORE_SOURCES}
    ${TEST_SCENE_SOURCES}
    ${TEST_IO_SOURCES}
    ${TEST_RENDERER_SOURCES}
    ${TEST_POSTPROCESS_SOURCES}
)

# ============================================================================
# Create Test Executable
# ============================================================================

add_executable(libquantiloom_tests
    ${ALL_TEST_SOURCES}
)

# ============================================================================
# Link Libraries
# ============================================================================

if(QUANTILOOM_STANDALONE_TESTS)
    # Standalone mode: tests are header-only verification
    # NOTE: This will only compile tests, not link to libQuantiloom
    # For full functionality, build from project root!
    target_link_libraries(libquantiloom_tests
        PRIVATE
            GTest::GTest
            # Note: Using custom main in test_main.cpp instead of GTest::Main
    )

    message(WARNING
        "================================ IMPORTANT ================================\n"
        "Tests built in STANDALONE mode (header-only).\n"
        "Most tests will FAIL at link time because libQuantiloom is not built.\n"
        "\n"
        "RECOMMENDED: Build from project root instead:\n"
        "  cd ${QUANTILOOM_ROOT_DIR}\n"
        "  mkdir build && cd build\n"
        "  cmake ..\n"
        "  cmake --build . --target libquantiloom_tests\n"
        "===========================================================================\n"
    )
else()
    # Subproject mode: link against the main library
    target_link_libraries(libquantiloom_tests
        PRIVATE
            GTest::GTest
            # Note: Using custom main in test_main.cpp instead of GTest::Main
            libQuantiloom  # Link against the main Quantiloom library
    )
endif()

# ============================================================================
# Register Tests with CTest
# ============================================================================

include(GoogleTest)
gtest_discover_tests(libquantiloom_tests)

# Alternative: Add as a single test
# add_test(NAME libQuantiloomAllTests COMMAND libquantiloom_tests)

# ============================================================================
# Custom Targets for Convenience
# ============================================================================

# Run tests with verbose output
add_custom_target(test_verbose
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    DEPENDS libquantiloom_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Run only core tests
add_custom_target(test_core
    COMMAND $<TARGET_FILE:libquantiloom_tests> --gtest_filter=*Test.*
    DEPENDS libquantiloom_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Run only scene tests
add_custom_target(test_scene
    COMMAND $<TARGET_FILE:libquantiloom_tests> --gtest_filter=MaterialTest.*:CameraTest.*
    DEPENDS libquantiloom_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Run only IO tests
add_custom_target(test_io
    COMMAND $<TARGET_FILE:libquantiloom_tests> --gtest_filter=ImageIOTest.*
    DEPENDS libquantiloom_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Run only renderer tests (non-Vulkan)
add_custom_target(test_renderer
    COMMAND $<TARGET_FILE:libquantiloom_tests> --gtest_filter=BRDFLutGeneratorTest.*:LightingParamsTest.*:AccelerationStructureTest.*
    DEPENDS libquantiloom_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Run only postprocess tests (sensor simulation, fusion)
add_custom_target(test_postprocess
    COMMAND $<TARGET_FILE:libquantiloom_tests> --gtest_filter=GenericSensorTest.*:MultibandFusionTest.*
    DEPENDS libquantiloom_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ============================================================================
# Optional: Code Coverage (Linux/GCC only)
# ============================================================================

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    option(ENABLE_COVERAGE "Enable code coverage" OFF)

    if(ENABLE_COVERAGE)
        target_compile_options(libquantiloom_tests PRIVATE --coverage)
        target_link_options(libquantiloom_tests PRIVATE --coverage)

        message(STATUS "Code coverage enabled (use gcov/lcov to generate reports)")
    endif()
endif()

# ============================================================================
# Installation (Optional)
# ============================================================================

# Install test executable (optional, for CI/CD systems)
install(TARGETS libquantiloom_tests
    RUNTIME DESTINATION bin/tests
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "==================================================")
message(STATUS "libQuantiloom Unit Tests Configuration")
message(STATUS "==================================================")
message(STATUS "  Test executable:   libquantiloom_tests")
message(STATUS "  Core tests:        ${TEST_CORE_SOURCES}")
message(STATUS "  Scene tests:       ${TEST_SCENE_SOURCES}")
message(STATUS "  IO tests:          ${TEST_IO_SOURCES}")
message(STATUS "  Renderer tests:    ${TEST_RENDERER_SOURCES}")
message(STATUS "  Postprocess tests: ${TEST_POSTPROCESS_SOURCES}")
message(STATUS "  Total tests:       ${ALL_TEST_SOURCES}")
message(STATUS "==================================================")
message(STATUS "Build and run:")
message(STATUS "  cmake --build . --target libquantiloom_tests")
message(STATUS "  ctest --output-on-failure")
message(STATUS "==================================================")
message(STATUS "Targeted test runs:")
message(STATUS "  cmake --build . --target test_core")
message(STATUS "  cmake --build . --target test_scene")
message(STATUS "  cmake --build . --target test_io")
message(STATUS "  cmake --build . --target test_renderer")
message(STATUS "  cmake --build . --target test_postprocess")
message(STATUS "==================================================")
