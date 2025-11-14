cmake_minimum_required(VERSION 3.21)

# ============================================================================
# Project Definition
# ============================================================================
project(Quantiloom
    VERSION 0.0.1
    DESCRIPTION "Unified Spectral Path Tracing System on Vulkan Ray Tracing"
    LANGUAGES CXX
)

# ============================================================================
# C++20 Standard & Compiler Requirements
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clangd/IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Build Type Configuration
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ============================================================================
# CPM.cmake Integration (Dependency Management)
# ============================================================================
set(CPM_DOWNLOAD_VERSION 0.42.0)
set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cpm_cache" CACHE PATH "CPM source cache directory")

if(NOT EXISTS "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
    message(STATUS "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION}...")
    file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        "${CMAKE_BINARY_DIR}/cmake/CPM.cmake"
        EXPECTED_HASH SHA256=2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a
    )
endif()

include("${CMAKE_BINARY_DIR}/cmake/CPM.cmake")

# ============================================================================
# Platform Detection & Cross-Platform Macros
# ============================================================================
if(WIN32)
    set(QUANTILOOM_PLATFORM_WINDOWS 1)
    add_compile_definitions(QUANTILOOM_PLATFORM_WINDOWS)
elseif(UNIX AND NOT APPLE)
    set(QUANTILOOM_PLATFORM_LINUX 1)
    add_compile_definitions(QUANTILOOM_PLATFORM_LINUX)
elseif(APPLE)
    set(QUANTILOOM_PLATFORM_MACOS 1)
    add_compile_definitions(QUANTILOOM_PLATFORM_MACOS)
endif()

# ============================================================================
# Compiler-Specific Flags
# ============================================================================
if(MSVC)
    # MSVC: Enable UTF-8, modern conformance, and high warning level
    add_compile_options(
        /W4
        /permissive-
        /Zc:__cplusplus
        /source-charset:utf-8
        /execution-charset:gbk
    )
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    # GCC/Clang: Enable comprehensive warnings
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
    )
endif()

# ============================================================================
# Project-Wide Options
# ============================================================================
option(QUANTILOOM_BUILD_TESTS "Build unit tests" ON)
option(QUANTILOOM_BUILD_EXAMPLES "Build example applications" ON)
option(QUANTILOOM_ENABLE_VALIDATION "Enable Vulkan validation layers" ON)

# ============================================================================
# Third-Party Dependencies (via CPM.cmake)
# ============================================================================

# spdlog: Fast C++ logging library (uses bundled fmt)
CPMAddPackage(
    NAME spdlog
    VERSION 1.14.1
    GITHUB_REPOSITORY gabime/spdlog
    OPTIONS
        "SPDLOG_BUILD_SHARED OFF"
        "SPDLOG_BUILD_EXAMPLE OFF"
        "SPDLOG_BUILD_TESTS OFF"
        "SPDLOG_FMT_EXTERNAL OFF"  # Use bundled fmt
        "CMAKE_POSITION_INDEPENDENT_CODE ON"  # Enable -fPIC for static library
)

# Ensure spdlog is built with PIC
if(TARGET spdlog)
    set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# tomlplusplus: TOML config parser (header-only)
CPMAddPackage(
    NAME tomlplusplus
    VERSION 3.4.0
    GITHUB_REPOSITORY marzer/tomlplusplus
    OPTIONS
        "BUILD_SHARED_LIBS OFF"
        "BUILD_TESTS OFF"
        "BUILD_EXAMPLES OFF"
)

# Imath: Dependency for OpenEXR
CPMAddPackage(
    NAME Imath
    VERSION 3.2.2
    GITHUB_REPOSITORY AcademySoftwareFoundation/Imath
    OPTIONS
        "BUILD_SHARED_LIBS OFF"
        "CMAKE_POSITION_INDEPENDENT_CODE ON"
)

# OpenEXR: High dynamic range image I/O (for multi-channel EXR)
CPMAddPackage(
    NAME openexr
    VERSION 3.4.3
    GITHUB_REPOSITORY AcademySoftwareFoundation/openexr
    OPTIONS
        "BUILD_SHARED_LIBS OFF"
        "BUILD_TESTING OFF"
        "OPENEXR_BUILD_TOOLS OFF"
        "OPENEXR_INSTALL_EXAMPLES OFF"
        "OPENEXR_INSTALL_PKG_CONFIG OFF"
)

# --- WORKAROUND for OpenEXR include paths ---
# Forcefully add OpenEXR include directories to the global path
# This is needed because the library's CMake target does not propagate them correctly.
include_directories(SYSTEM "${openexr_SOURCE_DIR}/src/lib")
include_directories(SYSTEM "${openexr_BINARY_DIR}/src/lib/OpenEXRCore")
# --- END WORKAROUND ---

# HDF5: Hierarchical Data Format for hyperspectral cubes
# Try CPM first, fall back to find_package if available
CPMAddPackage(
    NAME hdf5
    GIT_TAG hdf5-1.14.5
    GITHUB_REPOSITORY HDFGroup/hdf5
    OPTIONS
        "BUILD_SHARED_LIBS OFF"
        "HDF5_USE_STATIC_LIBRARIES ON"
        "BUILD_TESTING OFF"
        "HDF5_BUILD_EXAMPLES OFF"
        "HDF5_BUILD_TOOLS OFF"
        "HDF5_ENABLE_Z_LIB_SUPPORT ON"
        "HDF5_BUILD_CPP_LIB ON"
        "HDF5_BUILD_HL_LIB OFF"
)

# If CPM failed to fetch HDF5, try system package
if(NOT hdf5_ADDED)
    message(STATUS "HDF5 not found via CPM, trying find_package...")
    find_package(HDF5 REQUIRED COMPONENTS CXX)
endif()

message(STATUS "Dependencies resolved via CPM.cmake")

# ============================================================================
# Subdirectories
# ============================================================================

# Core library
add_subdirectory(src/libQuantiloom)

# Main application
add_subdirectory(src/app)

# Examples (optional)
if(QUANTILOOM_BUILD_EXAMPLES)
    # add_subdirectory(examples)  # TODO: M2+
endif()

# Tests (optional)
if(QUANTILOOM_BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)  # TODO: M1.5+
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "========================================")
message(STATUS "  Quantiloom Configuration Summary")
message(STATUS "========================================")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform:       ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler:       ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build Tests:    ${QUANTILOOM_BUILD_TESTS}")
message(STATUS "  Build Examples: ${QUANTILOOM_BUILD_EXAMPLES}")
message(STATUS "========================================")
