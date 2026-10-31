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
# Adding a test file:
#   Sources below are listed explicitly -- there is no glob. A .cpp that is not
#   in the matching set(TEST_<MODULE>_SOURCES ...) block is never compiled and
#   never runs, and the suite still reports green.
#
# Building: from the repo root, never from tests/.
#   ./build_wsl.sh                                         # full: configure, build, test, install SDK
#   cmake.exe --build build --config Release -j            # incremental, ~76 s
#
#   The tree is configured by build_wsl.sh as
#     cmake.exe -B build -G "Visual Studio 18 2026" -A x64 ...
#   That is a multi-config generator, so --config is required on every build and
#   the binary lands in build/tests/Release/. A plain `mkdir build && cmake ..`
#   does not reproduce this configuration.
#
# Running:
#   ./build/tests/Release/libquantiloom_tests.exe --gtest_brief=1
#   ./build/tests/Release/libquantiloom_tests.exe --gtest_filter='MaterialTest.*'
#
#   The binary reruns in ~3 s without rebuilding. Use --gtest_filter to select
#   cases: add_test below registers the whole suite as ONE aggregate test
#   (libQuantiloomAllTests), so `ctest -R <case>` cannot pick out an individual
#   test. `ctest` runs everything or nothing.
#
#   24 SKIPPED is the expected baseline here (BC7 is off and the glTF sample
#   assets are unavailable in build/).
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
        VERSION 0.8.3
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
# Include Directories
# ============================================================================

if(QUANTILOOM_STANDALONE_TESTS)
    # Standalone mode: point to parent directory structure
    set(QUANTILOOM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
    message(STATUS "Quantiloom root directory: ${QUANTILOOM_ROOT_DIR}")

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}   # so tests can include "support/..."
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
        ${CMAKE_CURRENT_SOURCE_DIR}   # so tests can include "support/..."
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
    add_compile_options(/W4 /wd4251 /wd4819)
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
    test_core/test_spectral_colour.cpp
    test_core/test_rgb_to_spectrum.cpp
    test_core/test_sha256.cpp
    test_core/test_endmember_unmix.cpp
    test_core/test_spectral_data.cpp
    test_core/test_blackbody_physics.cpp
    test_core/test_blackbody.cpp
    test_core/test_sky_thermal.cpp
    test_core/test_shortwave_gains.cpp
    test_core/test_thermal_conduction.cpp
    test_core/test_thermal_convection_forcing.cpp
    test_core/test_thermal_lateral.cpp
    test_core/test_thermal_shell.cpp
    test_core/test_thermal_parameter_sensitivity.cpp
    test_core/test_thermal_element_dump.cpp
    test_core/test_thermal_sun_sensitivity.cpp
    test_core/test_thermal_timeline.cpp
    test_core/test_thermal_epochs.cpp
    test_core/test_thermal_solve_cache.cpp
    test_core/test_ir_energy_conservation.cpp
    test_core/test_config.cpp
    test_core/test_spectral_cube.cpp
    test_core/test_resource_locator.cpp
)

# Scene tests
set(TEST_SCENE_SOURCES
    test_scene/test_material.cpp
    test_scene/test_tangent_generator.cpp
    test_scene/test_camera.cpp
    test_scene/test_camera_projection.cpp
    test_io/test_rig_config.cpp
    test_renderer/test_fusion_transport.cpp
    test_renderer/test_fusion_publication.cpp
    test_scene/test_mesh.cpp
    test_scene/test_mesh_optimizer.cpp
    test_scene/test_scene.cpp
    test_scene/test_motion.cpp
    test_scene/test_scene_merge.cpp
)

# IO tests
set(TEST_IO_SOURCES
    test_io/test_export_session.cpp
    test_io/test_product_geometry.cpp
    test_io/test_image_io.cpp
    test_io/test_spectral_io.cpp
    test_io/test_spectral_basis_loader.cpp
    test_io/test_atmos_net.cpp
    test_io/test_private_executable.cpp
    test_io/test_lut_loader.cpp
    test_io/test_atmosphere_transmittance_lut_loader.cpp
    test_io/test_gltf_loader.cpp
    test_io/test_usd_loader.cpp
    test_io/test_usd_texture_bank.cpp
    test_io/test_cross_format_parity.cpp
)

# Renderer tests (non-Vulkan components only)
set(TEST_RENDERER_SOURCES
    test_renderer/test_brdf_lut.cpp
    test_renderer/test_lighting_params.cpp
    test_renderer/test_sky_dome.cpp
    test_renderer/test_acceleration_structure.cpp
    test_renderer/test_texture_manager.cpp
    test_renderer/test_render_core.cpp
    test_renderer/test_config_resolve.cpp
    test_renderer/test_apply_config.cpp
    test_renderer/test_vulkan_device.cpp
    test_renderer/test_deferred_buffer_uploads.cpp
    test_renderer/test_brdf_lut_gpu.cpp
    test_renderer/test_environment_cubemap.cpp
    test_renderer/test_scene_geometry.cpp
    test_renderer/test_material_conversion.cpp
    test_renderer/test_sheen_brdf.cpp
    test_renderer/test_extension_brdf.cpp
    test_renderer/test_spectral_unmixer.cpp
    test_renderer/test_temperature_texture.cpp
    test_renderer/test_thermal_exchange_gpu.cpp
    test_renderer/test_thermal_step_gpu.cpp
    test_renderer/test_thermal_preview.cpp
    test_renderer/test_depth_aov.cpp
    test_renderer/test_pick.cpp
    test_renderer/test_config_duplicates.cpp
    test_renderer/test_scene_edit.cpp
    test_renderer/test_tlas_refit.cpp
    test_renderer/test_timeline_context.cpp
    test_renderer/test_interactive_bench.cpp
    test_renderer/test_camera_gpu.cpp
    test_renderer/test_camera_response_gpu.cpp
    test_renderer/test_offline_batch_scheduler.cpp
    test_renderer/test_display_range_gpu.cpp
    test_renderer/test_compute_migration_ab.cpp
)

# Shared test infrastructure (headless Vulkan device, created on first use)
set(TEST_SUPPORT_SOURCES
    support/VulkanTestDevice.cpp
)

# Internal compute RHI (gpu/): CPU mapping tables and a GPU smoke suite
set(TEST_GPU_SOURCES
    test_gpu/test_gpu_mapping.cpp
    test_gpu/test_gpu_vulkan.cpp
)

# Postprocess tests (sensor simulation, multiband fusion)
set(TEST_POSTPROCESS_SOURCES
    test_postprocess/test_camera_pipeline.cpp
    test_postprocess/test_camera_cpu.cpp
    test_postprocess/test_camera_history.cpp
    test_postprocess/test_camera_isp.cpp
    test_postprocess/test_camera_auto.cpp
    test_postprocess/test_camera_hsv.cpp
    test_postprocess/test_isp_coefficients.cpp
    test_postprocess/test_camera_config.cpp
    test_postprocess/test_camera_presets.cpp
    test_postprocess/test_generic_sensor.cpp
    test_postprocess/test_multiband_fusion.cpp
    test_postprocess/test_gpu_sensor_math.cpp
    test_postprocess/test_sensor_config.cpp
    test_postprocess/test_sensor_chain_order.cpp
    test_postprocess/test_thermography.cpp
)

# Hyperspectral core tests
set(TEST_HS_CORE_SOURCES
    test_hs_core/test_hyperspectral_config.cpp
    test_hs_core/test_spectral_cube_io.cpp
    test_hs_core/test_spectral_analyzer.cpp
    test_hs_core/test_adaptive_grid_generator.cpp
    test_hs_core/test_spectral_reconstructor.cpp
    test_hs_core/test_batch_renderer.cpp
    test_hs_core/test_gpu_spectral_reconstructor.cpp
)

# MCP tests (protocol framing and the command queue; no GPU, no sockets)
set(TEST_MCP_SOURCES
    test_mcp/test_mcp_protocol.cpp
    test_mcp/test_mcp_registry.cpp
)

# Test main (custom main function with logging initialization)
set(TEST_MAIN_SOURCE
    test_main.cpp
)

# CLI tests. The manifest parser lives in src/app because deciding what a
# manifest line means is host business, not library business (see
# RenderJob.hpp) -- but it is pure text-to-Config with no device behind it, so
# it is compiled into the suite alongside its test rather than left untested
# for want of a target.
set(TEST_APP_SOURCES
    test_app/test_batch_manifest.cpp
    test_app/test_sequence_job.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/BatchManifest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/SequenceJob.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/RenderJob.cpp
)

# Compile SpectraForge against the same core objects as this executable; linking
# its DLL here would also load a second copy of Quantiloom's global state.
set(TEST_SPECTRAFORGE_SOURCES
    test_postprocess/test_spectraforge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/libSpectraForge/ColorCluster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/libSpectraForge/IRMaterialDatabase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/libSpectraForge/SpectraForge.cpp
)

# Combine all test sources
set(ALL_TEST_SOURCES
    ${TEST_MAIN_SOURCE}
    ${TEST_SUPPORT_SOURCES}
    ${TEST_GPU_SOURCES}
    ${TEST_CORE_SOURCES}
    ${TEST_SCENE_SOURCES}
    ${TEST_IO_SOURCES}
    ${TEST_RENDERER_SOURCES}
    ${TEST_POSTPROCESS_SOURCES}
    ${TEST_HS_CORE_SOURCES}
    ${TEST_MCP_SOURCES}
    ${TEST_APP_SOURCES}
    ${TEST_SPECTRAFORGE_SOURCES}
)

# ============================================================================
# Create Test Executable
# ============================================================================

add_executable(libquantiloom_tests
    ${ALL_TEST_SOURCES}
)

# Absolute path to the repository, baked in at configure time. Tests that need
# to reach a source file or an asset must use this rather than a relative path:
# a relative path silently resolves against the caller's working directory, and
# the usual fallback is to skip the test, so a wrong path looks like "no test
# data available" instead of a broken test.
if(QUANTILOOM_STANDALONE_TESTS)
    set(_ql_source_root "${CMAKE_CURRENT_SOURCE_DIR}/..")
else()
    set(_ql_source_root "${CMAKE_SOURCE_DIR}")
endif()
target_compile_definitions(libquantiloom_tests
    PRIVATE QUANTILOOM_SOURCE_ROOT="${_ql_source_root}")

# So test_app/ can include "BatchManifest.hpp" the way src/app does.
target_include_directories(libquantiloom_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/app)

# /MP: compile the test sources in parallel -- `cmake --build -j` alone only
# runs whole projects side by side. The precompiled header holds what nearly
# every test file includes: gtest, the standard library, Vulkan and GLM. Same
# rule as quantiloom_core's: stable third-party headers only, nothing that
# wants a macro defined before it.
if(MSVC)
    target_compile_options(libquantiloom_tests PRIVATE /MP)
endif()
target_precompile_headers(libquantiloom_tests PRIVATE
    <gtest/gtest.h>
    <algorithm> <array> <chrono> <cmath> <cstdint> <cstring> <filesystem>
    <format> <fstream> <functional> <limits> <memory> <numeric> <optional>
    <stdexcept> <string> <utility> <vector>
)
# Standalone mode links GTest alone, so Vulkan and GLM are not on its include
# path -- they come with quantiloom_core.
if(NOT QUANTILOOM_STANDALONE_TESTS)
    target_precompile_headers(libquantiloom_tests PRIVATE
        <vulkan/vulkan.h>
        <glm/glm.hpp>
    )
endif()

# Shaders come from the resource locator (the development fallback reads
# src/shaders from the working directory), so nothing is copied beside the
# test binary any more. The dependency stays: the suite needs them built.
if(NOT QUANTILOOM_STANDALONE_TESTS)
    add_dependencies(libquantiloom_tests CompileShaders)
endif()

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
    # Subproject mode: link the OBJECT library, not the DLL.
    #
    # Static linkage ignores __declspec(dllexport), so every symbol in the
    # library is reachable here -- including the parts that are deliberately
    # NOT part of the SDK's public contract (NormalGenerator, Upload, the
    # Vulkan wrappers). Testing something is no longer a reason to export it.
    #
    # Do not add libQuantiloom alongside this: linking both would put two
    # copies of the library's global state (spdlog logger, static caches) in
    # one process.
    target_link_libraries(libquantiloom_tests
        PRIVATE
            GTest::GTest
            # Note: Using custom main in test_main.cpp instead of GTest::Main
            quantiloom_core  # Objects, not the DLL -- see above
            # The core links this PRIVATE, which CMake records as $<LINK_ONLY:>
            # -- the objects come through, the include directory does not. The
            # MCP tests read JSON responses, so they ask for it by name.
            nlohmann_json
    )
endif()

# ============================================================================
# Runtime DLLs (Windows)
# ============================================================================
# Quantiloom.dll is deliberately absent: the core is linked statically above,
# so the test binary is self-contained apart from third-party runtimes.

if(WIN32 AND NOT QUANTILOOM_STANDALONE_TESTS)
    target_link_libraries(libquantiloom_tests PRIVATE bcrypt advapi32)
    # Copy OpenUSD DLLs if enabled
    if(QUANTILOOM_USE_OPENUSD AND COMMAND copy_usd_dlls)
        copy_usd_dlls(libquantiloom_tests)
    endif()
endif()

# ============================================================================
# Public header neutrality gate
# ============================================================================
# Every public header outside renderer/interop/ must compile with no Vulkan
# include directory at all: that is what keeps the SDK's neutral API neutral.
# A try_compile rather than a grep, so a transitive include through another
# public header fails too. The second try_compile is the control: the Vulkan
# interop header MUST fail under the same conditions, or the gate proves
# nothing.
if(NOT QUANTILOOM_STANDALONE_TESTS)
    set(_ql_neutrality_dir "${CMAKE_CURRENT_BINARY_DIR}/header_neutrality")
    set(_ql_public_inc "${CMAKE_SOURCE_DIR}/include/quantiloom")
    file(MAKE_DIRECTORY "${_ql_neutrality_dir}")
    file(GLOB_RECURSE _ql_public_headers RELATIVE "${_ql_public_inc}" "${_ql_public_inc}/*.hpp")
    set(_ql_neutral_tu "// Generated by tests/CMakeLists.txt: the public headers outside interop/.\n")
    foreach(_ql_header IN LISTS _ql_public_headers)
        if(NOT _ql_header MATCHES "^renderer/interop/")
            string(APPEND _ql_neutral_tu "#include \"${_ql_header}\"\n")
        endif()
    endforeach()
    file(WRITE "${_ql_neutrality_dir}/neutral.cpp" "${_ql_neutral_tu}")
    file(WRITE "${_ql_neutrality_dir}/control.cpp"
         "#include \"renderer/interop/VulkanInterop.hpp\"\n")

    # glm is the one third-party header the public API includes. Take its
    # directory from glm::glm when that carries one (CPM's source tree or a
    # find_package), with glm_SOURCE_DIR as the fallback, and insist that it
    # really holds glm/glm.hpp -- an empty entry would make the gate fail on
    # glm rather than on Vulkan, which proves nothing.
    set(_ql_glm_hints "${glm_SOURCE_DIR}")
    if(TARGET glm::glm)
        get_target_property(_ql_glm_dirs glm::glm INTERFACE_INCLUDE_DIRECTORIES)
        if(_ql_glm_dirs)
            string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]*)>" "\\1" _ql_glm_dirs "${_ql_glm_dirs}")
            string(REGEX REPLACE "\\$<INSTALL_INTERFACE:[^>]*>" "" _ql_glm_dirs "${_ql_glm_dirs}")
            list(APPEND _ql_glm_hints ${_ql_glm_dirs})
        endif()
    endif()
    find_path(QL_NEUTRALITY_GLM_INCLUDE glm/glm.hpp HINTS ${_ql_glm_hints} NO_DEFAULT_PATH)
    if(NOT QL_NEUTRALITY_GLM_INCLUDE)
        message(FATAL_ERROR
            "Header-neutrality gate: no glm/glm.hpp under '${_ql_glm_hints}'.")
    endif()

    # What a consumer of the base package compiles with: the public root, the
    # binary dir holding the generated core/LibVersion.hpp (both are
    # libQuantiloom's PUBLIC include directories), glm, the platform macro
    # core/Platform.hpp requires, and the dllimport side of QL_API. On MSVC
    # also /utf-8 and /permissive-, as the main build: several public headers
    # carry UTF-8 comments, and parsed in a GBK codepage those can swallow a
    # newline and fail for a reason that has nothing to do with Vulkan.
    set(_ql_neutrality_includes
        "${_ql_public_inc}"
        "${CMAKE_BINARY_DIR}/src/libQuantiloom"
        "${QL_NEUTRALITY_GLM_INCLUDE}"
    )
    if(WIN32)
        set(_ql_neutrality_defines -DQUANTILOOM_PLATFORM_WINDOWS -DQL_USE_SHARED)
    elseif(APPLE)
        set(_ql_neutrality_defines -DQUANTILOOM_PLATFORM_MACOS -DQL_USE_SHARED)
    else()
        set(_ql_neutrality_defines -DQUANTILOOM_PLATFORM_LINUX -DQL_USE_SHARED)
    endif()
    if(MSVC)
        list(APPEND _ql_neutrality_defines /utf-8 /permissive- /EHsc)
    endif()

    # The include list goes in ONE quoted argument, so its semicolons reach the
    # inner project as a list. Held in a list variable and expanded unquoted,
    # the separators split it and only the first directory survived. Both
    # try_compile calls below spell the same arguments from these variables.
    #
    # The gate is about compiling, not linking: a TU of #includes has no main,
    # so both calls build a static library rather than an executable. The
    # previous value is restored before anything else in this file runs.
    if(DEFINED CMAKE_TRY_COMPILE_TARGET_TYPE)
        set(_ql_saved_try_compile_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    else()
        unset(_ql_saved_try_compile_type)
    endif()
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

    try_compile(QL_PUBLIC_HEADERS_NEUTRAL
        "${_ql_neutrality_dir}/neutral_build"
        SOURCES "${_ql_neutrality_dir}/neutral.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${_ql_neutrality_includes}"
        COMPILE_DEFINITIONS ${_ql_neutrality_defines}
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        OUTPUT_VARIABLE _ql_neutral_output
    )

    try_compile(QL_INTEROP_CONTROL_COMPILES
        "${_ql_neutrality_dir}/control_build"
        SOURCES "${_ql_neutrality_dir}/control.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${_ql_neutrality_includes}"
        COMPILE_DEFINITIONS ${_ql_neutrality_defines}
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        OUTPUT_VARIABLE _ql_control_output
    )

    if(DEFINED _ql_saved_try_compile_type)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_ql_saved_try_compile_type}")
        unset(_ql_saved_try_compile_type)
    else()
        unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    endif()

    if(NOT QL_PUBLIC_HEADERS_NEUTRAL)
        message(FATAL_ERROR
            "Public headers outside renderer/interop/ must compile without any Vulkan "
            "include directory.\n${_ql_neutral_output}")
    endif()
    if(QL_INTEROP_CONTROL_COMPILES)
        message(FATAL_ERROR
            "The header-neutrality gate is vacuous: renderer/interop/VulkanInterop.hpp "
            "compiled without a Vulkan include directory.")
    endif()
    # The control must fail on vulkan.h and nothing else; failing on glm or on
    # a platform macro would make the neutral pass above meaningless too.
    #
    # Match only "[/\]vulkan.h", never the full "vulkan/vulkan.h". On a Chinese
    # Windows, MSVC prints C1083 with the file name inside UTF-8 curly quotes
    # (E2 80 9C), CMake decodes that output as code page 936, and the quote's
    # last byte pairs with the next one: "9C 76" becomes one CJK character, so
    # the first 'v' disappears and the text held here reads "...ulkan/vulkan.h".
    # The separator and the second segment survive in any UI language.
    if(NOT _ql_control_output MATCHES "[/\\\\]vulkan\\.h")
        message(FATAL_ERROR
            "The header-neutrality control failed for a reason other than the missing "
            "vulkan/vulkan.h:\n${_ql_control_output}")
    endif()
    message(STATUS
        "Public header neutrality: the non-interop public headers compile with no Vulkan "
        "include path (the interop control fails as required)")
endif()

# ============================================================================
# Register Tests with CTest
# ============================================================================

# Note: Using add_test instead of gtest_discover_tests to avoid DLL loading issues
# gtest_discover_tests runs at build time and may fail to find DLL
add_test(NAME libQuantiloomAllTests COMMAND libquantiloom_tests)

# Set test working directory to the repository root: the resource locator's
# development fallback (src/shaders, assets/) is working-directory relative,
# exactly as when build_wsl.sh runs the same binary from the repo root.
set_tests_properties(libQuantiloomAllTests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

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
    COMMAND $<TARGET_FILE:libquantiloom_tests> --gtest_filter=BRDFLutGeneratorTest.*:LightingParamsTest.*:AccelerationStructureTest.*:ComputeMigrationAB.*:GpuMappingTest.*:GpuShaderLibraryTest.*:GpuVulkanTest.*
    DEPENDS libquantiloom_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Run only postprocess tests (sensor simulation, fusion)
add_custom_target(test_postprocess
    COMMAND $<TARGET_FILE:libquantiloom_tests> --gtest_filter=CameraPhysicsTest.*:CpuCameraPipelineTest.*:CameraConfigIOTest.*:CameraGpuTest.*:CameraResponseGpuTest.*:GenericSensorTest.*:MultibandFusionTest.*:GPUSensorMathTest.*:SensorParamsTest.*:SensorConfigTest.*:SensorChainOrderTest.*
    DEPENDS libquantiloom_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Run only hyperspectral core tests
add_custom_target(test_hs_core
    COMMAND $<TARGET_FILE:libquantiloom_tests> --gtest_filter=HyperspectralConfigTest.*:SpectralCubeIOTest.*:SpectralFeatureTypeTest.*:SpectralFeatureTest.*:SpectralAnalysisParamsTest.*:SpectralAnalysisResultTest.*:SpectralAnalyzerTest.*:AdaptiveGridInfoTest.*:ReconstructionMappingTest.*:AdaptiveGridGeneratorTest.*:InterpolationMethodTest.*:SpectralReconstructorTest.*:BatchRenderStatusTest.*:BatchRenderProgressTest.*:BatchRenderParamsTest.*:ValidateWavelengthListTest.*:SortWavelengthsTest.*:EstimateBatchTimeTest.*:BatchRendererWorkflowTest.*:BatchRendererEdgeCaseTest.*:GpuReconstructorStatusTest.*:GpuReconstructorConfigTest.*:EstimateReconstructionMemoryTest.*:HyperspectralConfigGpuTest.*:CatmullRomReferenceTest.*:LinearInterpReferenceTest.*:WavelengthGridTest.*:SpectralCubeLayoutTest.*:ComputeShaderCompatTest.*:GpuReconstructorEdgeCaseTest.*
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
message(STATUS "  HS-Core tests:     ${TEST_HS_CORE_SOURCES}")
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
message(STATUS "  cmake --build . --target test_hs_core")
message(STATUS "==================================================")
