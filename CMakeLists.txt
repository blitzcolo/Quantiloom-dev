cmake_minimum_required(VERSION 3.21)

# ============================================================================
# Project Definition
# ============================================================================
project(Quantiloom
    VERSION 0.1.3
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
# GCC 13 ICE Workaround
# ============================================================================
# GCC 13.x has internal compiler errors (segfaults in RTL passes: reload,
# cse_local, dse1, ree) when compiling C++20 code at -O2/-O3.
# Use -O1 as workaround. Recommend upgrading to GCC 14+.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14")
    string(REPLACE "-O3" "-O1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-O2" "-O1" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    message(WARNING "GCC <14 detected: using -O1 to work around ICE bugs. "
                    "Recommend upgrading to GCC 14+ for full optimization.")
endif()

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
    add_compile_definitions(NOMINMAX)
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
    # MSVC: Enable UTF-8, modern conformance, exception handling, and high warning level
    # /wd4251: Disable DLL interface warnings for STL/header-only types (safe when using same runtime)
    add_compile_options(
        /W4
        /EHsc
        /wd4251
        /permissive-
        /Zc:__cplusplus
        # Sources are UTF-8 without BOM. Without this flag MSVC parses them in
        # the system codepage (GBK), where multibyte comment characters can
        # swallow the newline and eat the next line of code (C2065 errors).
        # /utf-8 sets BOTH source and execution charset, so narrow string
        # literals keep their UTF-8 bytes -- identical runtime behavior to the
        # old GBK byte-passthrough parse.
        /utf-8
    )
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    # Debug symbols in Release. The SDK ships the PDBs, so a crash in
    # Quantiloom-Qt resolves to a core function and line instead of stopping at
    # the DLL boundary. /Zi does not change code generation -- optimisation is
    # unaffected -- but /DEBUG turns OFF the linker's default folding, so
    # /OPT:REF,ICF have to be asked for back or the binaries grow.
    add_compile_options($<$<CONFIG:Release>:/Zi>)
    add_link_options(
        $<$<CONFIG:Release>:/DEBUG>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
else()
    # GCC/Clang: Enable comprehensive warnings
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -w
    )
endif()

# ============================================================================
# Project-Wide Options
# ============================================================================
option(QUANTILOOM_BUILD_TESTS "Build unit tests" ON)

# Support standard CMake BUILD_TESTING option
if(BUILD_TESTING)
    set(QUANTILOOM_BUILD_TESTS ON CACHE BOOL "Build unit tests" FORCE)
endif()
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
        "OPENEXR_BUILD_EXAMPLES OFF"
        "OPENEXR_INSTALL_EXAMPLES OFF"
        "OPENEXR_INSTALL_PKG_CONFIG OFF"
)

# --- WORKAROUND for OpenEXR include paths ---
# Forcefully add OpenEXR include directories to the global path
# This is needed because the library's CMake target does not propagate them correctly.
include_directories(SYSTEM "${openexr_SOURCE_DIR}/src/lib")
include_directories(SYSTEM "${openexr_BINARY_DIR}/src/lib/OpenEXRCore")
# --- END WORKAROUND ---

# NOTE: HDF5 dependency removed - hyperspectral cube I/O not needed for current use cases
# If needed in future, re-add with proper static linking configuration

# ============================================================================
# Vulkan SDK
# ============================================================================
find_package(Vulkan REQUIRED)

if(Vulkan_FOUND)
    message(STATUS "Vulkan SDK found: ${Vulkan_VERSION}")
    message(STATUS "Vulkan Include: ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "Vulkan Library: ${Vulkan_LIBRARIES}")
else()
    message(FATAL_ERROR "Vulkan SDK not found. Please install Vulkan SDK from https://vulkan.lunarg.com/")
endif()

# VMA (Vulkan Memory Allocator): Single-header memory management library
CPMAddPackage(
    NAME VulkanMemoryAllocator
    VERSION 3.4.0
    GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    DOWNLOAD_ONLY YES
)

# VMA is header-only, create interface library
if(VulkanMemoryAllocator_ADDED)
    add_library(VMA INTERFACE)
    target_include_directories(VMA INTERFACE
        ${VulkanMemoryAllocator_SOURCE_DIR}/include
    )
    target_link_libraries(VMA INTERFACE Vulkan::Vulkan)
    message(STATUS "VMA (Vulkan Memory Allocator) v3.4.0 added")
endif()

# tinygltf: glTF 2.0 loader (header-only)
CPMAddPackage(
    NAME tinygltf
    VERSION 3.0.0
    GITHUB_REPOSITORY syoyo/tinygltf
    DOWNLOAD_ONLY YES
)

# tinygltf is header-only, create interface library
if(tinygltf_ADDED)
    add_library(tinygltf INTERFACE)
    target_include_directories(tinygltf INTERFACE
        ${tinygltf_SOURCE_DIR}
    )
    # tinygltf uses bundled stb_image for texture loading (PNG/JPEG decoding)
    # No additional compile definitions needed - defaults are correct:
    #   - External images (URI) will be loaded automatically
    #   - Embedded images (DataURI) will be decoded
    #   - Binary buffers will be loaded from .bin files
    message(STATUS "tinygltf v3.0.0 added (header-only glTF 2.0 loader)")
endif()

# nlohmann/json: JSON parser (header-only), used by the safetensors reader
CPMAddPackage(
    NAME nlohmann_json
    VERSION 3.11.3
    GITHUB_REPOSITORY nlohmann/json
    DOWNLOAD_ONLY YES
)

if(nlohmann_json_ADDED)
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE
        ${nlohmann_json_SOURCE_DIR}/include
    )
    message(STATUS "nlohmann/json v3.11.3 added (header-only JSON parser)")
endif()

# ============================================================================
# OpenUSD (Pixar USD) - Official USD Library
# ============================================================================
# Replaces TinyUSDZ for full USD composition support (sublayers, references, payloads)
# Build OpenUSD with: scripts/build_openusd.ps1 (Windows) or scripts/build_openusd.sh (Linux)
#
# Required build flags for OpenUSD:
#   --no-python --no-imaging --no-usdview (minimal core libraries)
# ============================================================================

set(USD_ROOT "" CACHE PATH "Path to OpenUSD installation (e.g., C:/openusd or /opt/openusd)")

# Allow environment variable override
if(NOT USD_ROOT AND DEFINED ENV{USD_ROOT})
    set(USD_ROOT $ENV{USD_ROOT})
endif()

if(USD_ROOT)
    message(STATUS "OpenUSD root: ${USD_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "${USD_ROOT}")

    # Find OpenUSD package
    find_package(pxr CONFIG REQUIRED)

    # Define USD libraries to link (core only, no Python/Imaging)
    # Note: OpenUSD 24.x uses usd_ prefix for library names
    set(USD_LIBRARIES
        usd_usd
        usd_usdGeom
        usd_usdShade
        usd_sdf
        usd_tf
        usd_gf
        usd_ar
        usd_plug
        usd_vt
        usd_work
        usd_arch
        usd_pcp
        usd_sdr
        usd_kind
        usd_js
        usd_trace
    )

    # Find TBB (required by OpenUSD)
    find_package(TBB CONFIG QUIET)
    if(NOT TBB_FOUND AND NOT TARGET TBB::tbb)
        # Try to find TBB from OpenUSD's bundled version
        if(EXISTS "${USD_ROOT}/lib/cmake/TBB")
            list(APPEND CMAKE_PREFIX_PATH "${USD_ROOT}/lib/cmake/TBB")
            find_package(TBB CONFIG REQUIRED)
        elseif(EXISTS "${USD_ROOT}/include/tbb/tbb.h")
            # TBB 2020.x doesn't have CMake config - create target manually
            message(STATUS "Creating TBB target from USD_ROOT bundled TBB (legacy)")
            add_library(TBB::tbb SHARED IMPORTED)
            set_target_properties(TBB::tbb PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${USD_ROOT}/include"
            )
            if(WIN32)
                set_target_properties(TBB::tbb PROPERTIES
                    IMPORTED_LOCATION "${USD_ROOT}/bin/tbb.dll"
                    IMPORTED_IMPLIB "${USD_ROOT}/lib/tbb.lib"
                )
            else()
                set_target_properties(TBB::tbb PROPERTIES
                    IMPORTED_LOCATION "${USD_ROOT}/lib/libtbb.so"
                )
            endif()
            set(TBB_FOUND TRUE)
            set(TBB_VERSION "2020.3 (bundled)")
        else()
            message(FATAL_ERROR "TBB not found. Please install oneTBB or ensure USD_ROOT contains bundled TBB.")
        endif()
    elseif(TARGET TBB::tbb)
        set(TBB_FOUND TRUE)
        set(TBB_VERSION "from OpenUSD")
    endif()

    math(EXPR USD_BITS "${CMAKE_SIZEOF_VOID_P} * 8")
    message(STATUS "OpenUSD found with ${USD_BITS}-bit libraries")
    message(STATUS "  TBB: ${TBB_VERSION}")

    set(QUANTILOOM_USE_OPENUSD ON)

    # Helper function to copy USD DLLs to target output directory (Windows only)
    if(WIN32)
        function(copy_usd_dlls target_name)
            # Copy USD library DLLs
            file(GLOB USD_DLLS "${USD_ROOT}/lib/usd_*.dll")
            # Copy TBB DLLs
            file(GLOB TBB_DLLS "${USD_ROOT}/bin/tbb*.dll")
            # Copy MaterialX DLLs (if present)
            file(GLOB MATERIALX_DLLS "${USD_ROOT}/bin/MaterialX*.dll")

            foreach(dll_file ${USD_DLLS} ${TBB_DLLS} ${MATERIALX_DLLS})
                add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll_file}"
                        "$<TARGET_FILE_DIR:${target_name}>"
                    COMMENT "Copying ${dll_file} to output directory"
                )
            endforeach()

            # Also copy the USD plugInfo resources if they exist
            if(EXISTS "${USD_ROOT}/lib/usd")
                add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        "${USD_ROOT}/lib/usd"
                        "$<TARGET_FILE_DIR:${target_name}>/usd"
                    COMMENT "Copying USD plugins to output directory"
                )
            endif()
        endfunction()
    endif()
else()
    message(WARNING "USD_ROOT not set. OpenUSD support will be disabled.")
    message(WARNING "Set USD_ROOT to OpenUSD installation path, or run scripts/build_openusd.ps1")
    set(QUANTILOOM_USE_OPENUSD OFF)
endif()

# GLM: Mathematics library for graphics (header-only)
CPMAddPackage(
    NAME glm
    VERSION 1.0.2
    GITHUB_REPOSITORY g-truc/glm
    GIT_TAG 1.0.2
)

message(STATUS "Dependencies resolved via CPM.cmake")

# ============================================================================
# Suppress Warnings for Third-Party Libraries (MSVC)
# ============================================================================
if(MSVC)
    # Helper function to suppress warnings for a target
    function(suppress_warnings_for_target target_name)
        if(TARGET ${target_name})
            target_compile_options(${target_name} PRIVATE /W0)
        endif()
    endfunction()

    # OpenEXR libraries
    suppress_warnings_for_target(OpenEXR)
    suppress_warnings_for_target(OpenEXRCore)
    suppress_warnings_for_target(OpenEXRUtil)
    suppress_warnings_for_target(Iex)
    suppress_warnings_for_target(IlmThread)

    # Silence C++20 deprecated shared_ptr atomic warnings in IlmThread
    if(TARGET IlmThread)
        target_compile_definitions(IlmThread PRIVATE
            _SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING)
    endif()

    # Imath
    suppress_warnings_for_target(Imath)

    # OpenJPH (JPEG 2000 HT codec, dependency of OpenEXR)
    suppress_warnings_for_target(openjph)

    # spdlog
    suppress_warnings_for_target(spdlog)
endif()

# ============================================================================
# Subdirectories
# ============================================================================

# Shaders (compile before libraries/apps)
add_subdirectory(src/shaders)

# Core library
add_subdirectory(src/libQuantiloom)

# IR material generator (depends on libQuantiloom)
add_subdirectory(src/libSpectraForge)

# Main application
add_subdirectory(src/app)

# Tools (fusion, etc.)
add_subdirectory(src/tools)

# Examples (optional)
if(QUANTILOOM_BUILD_EXAMPLES)
    # add_subdirectory(examples)  # TODO: M2+
endif()

# Tests (optional)
if(QUANTILOOM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Unit tests enabled - target: libquantiloom_tests")
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
