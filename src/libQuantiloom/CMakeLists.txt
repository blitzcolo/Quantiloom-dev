# ============================================================================
# libQuantiloom - Core Library
# ============================================================================
# Two targets, one compilation:
#
#   quantiloom_core (OBJECT)  -- every source, compiled once. In-repo consumers
#                                that legitimately need library internals (the
#                                unit tests, fusion_tool) link THIS. Static
#                                linkage ignores __declspec(dllexport)
#                                entirely, so they see every symbol without
#                                anything having to be marked QL_API.
#   libQuantiloom (SHARED)    -- the SDK artifact. Exports only what is marked
#                                QL_API. Quantiloom-Qt and the CLI link this.
#
# The split exists because export visibility and in-repo linkability used to be
# the same switch, which meant "cover it with a test" implied "write it into the
# SDK's public contract". Keep them separate: a target links EITHER the objects
# OR the DLL, never both -- two copies of the library's global state (the spdlog
# logger, static caches) in one process is a correctness bug, not a wart.
# ============================================================================

# Public headers live in include/quantiloom/, internal ones stay beside their
# .cpp here. That is the whole boundary: the install rule ships include/ and
# nothing else, and libQuantiloom puts only include/ on its consumers' search
# path, so a frontend physically cannot reach RayTracingPipeline.hpp. It is a
# compile error rather than a missing symbol at link time.
set(QL_PUBLIC_INC "${CMAKE_SOURCE_DIR}/include/quantiloom")

# Generate version header from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/core/LibVersion.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/core/LibVersion.hpp"
    @ONLY
)

add_library(quantiloom_core OBJECT
    # Core module
    core/Log.cpp
    ${QL_PUBLIC_INC}/core/Log.hpp
    core/Config.cpp
    ${QL_PUBLIC_INC}/core/Config.hpp
    core/SpectralColour.cpp
    ${QL_PUBLIC_INC}/core/Platform.hpp
    ${QL_PUBLIC_INC}/core/Types.hpp
    ${QL_PUBLIC_INC}/core/Image.hpp
    ${QL_PUBLIC_INC}/core/SpectralCube.hpp
    ${QL_PUBLIC_INC}/core/LUT.hpp
    core/AtmosphereTransmittanceLUT.hpp

    # IO module
    io/ImageIO.cpp
    ${QL_PUBLIC_INC}/io/ImageIO.hpp
    io/SpectralIO.cpp
    ${QL_PUBLIC_INC}/io/SpectralIO.hpp
    io/SpectralBasisLoader.cpp
    io/SpectralBasisLoader.hpp
    io/LUTLoader.cpp
    io/LUTLoader.hpp
    io/AtmosphereTransmittanceLUTLoader.cpp
    io/AtmosphereTransmittanceLUTLoader.hpp
    io/GltfLoader.cpp
    io/GltfLoader.hpp
    io/UsdLoader.cpp
    io/UsdLoader.hpp
    io/SpectralCubeIO.cpp
    io/SpectralCubeIO.hpp

    # Atmosphere NN module (MODTRAN surrogate inference + LUT baking)
    atmos/Safetensors.cpp
    atmos/Safetensors.hpp
    atmos/ResMLP.cpp
    atmos/ResMLP.hpp
    atmos/AtmosNet.cpp
    atmos/AtmosNet.hpp
    atmos/AtmosModelPack.cpp
    atmos/AtmosModelPack.hpp
    atmos/AtmosphereNNConfig.cpp
    ${QL_PUBLIC_INC}/atmos/AtmosphereNNConfig.hpp
    atmos/AtmosphereBaker.cpp
    atmos/AtmosphereBaker.hpp
    atmos/AtmosNNGpuTypes.hpp

    # Renderer module (Vulkan + VMA wrappers)
    renderer/VmaImpl.cpp
    renderer/VulkanContext.cpp
    renderer/VulkanContext.hpp
    renderer/GpuBuffer.cpp
    renderer/GpuBuffer.hpp
    renderer/GpuImage.cpp
    renderer/GpuImage.hpp
    renderer/Upload.cpp
    renderer/Upload.hpp
    renderer/TextureManager.cpp
    renderer/TextureManager.hpp
    renderer/RayTracingPipeline.cpp
    renderer/RayTracingPipeline.hpp
    renderer/AccelerationStructure.cpp
    renderer/AccelerationStructure.hpp
    renderer/CommandHelper.cpp
    renderer/CommandHelper.hpp
    renderer/BRDFLutGenerator.cpp
    renderer/BRDFLutGenerator.hpp
    renderer/PerformanceLogger.cpp
    renderer/PerformanceLogger.hpp
    renderer/TextureCompressor.cpp
    renderer/TextureCompressor.hpp
    ${QL_PUBLIC_INC}/renderer/LightingParams.hpp
    renderer/VulkanContextAdapter.cpp
    renderer/VulkanContextAdapter.hpp
    renderer/RenderCore.cpp
    renderer/RenderCore.hpp
    renderer/ExternalRenderContext.cpp
    ${QL_PUBLIC_INC}/renderer/ExternalRenderContext.hpp

    # Scene module
    scene/Camera.cpp
    ${QL_PUBLIC_INC}/scene/Camera.hpp
    ${QL_PUBLIC_INC}/scene/Mesh.hpp
    scene/MeshOptimizer.cpp
    scene/MeshOptimizer.hpp
    scene/NormalGenerator.cpp
    scene/NormalGenerator.hpp
    ${QL_PUBLIC_INC}/scene/BRDFModels.hpp
    scene/Material.cpp
    ${QL_PUBLIC_INC}/scene/Material.hpp
    ${QL_PUBLIC_INC}/scene/Texture.hpp
    scene/Scene.cpp
    ${QL_PUBLIC_INC}/scene/Scene.hpp

    # Postprocess module
    ${QL_PUBLIC_INC}/postprocess/SensorModel.hpp
    postprocess/GenericSensor.cpp
    ${QL_PUBLIC_INC}/postprocess/GenericSensor.hpp
    postprocess/MultibandFusion.cpp
    ${QL_PUBLIC_INC}/postprocess/MultibandFusion.hpp
    ${QL_PUBLIC_INC}/postprocess/PostprocessConfig.hpp

    # Hyperspectral core module
    hs_core/HyperspectralConfig.hpp
    hs_core/HyperspectralRenderer.cpp
    hs_core/HyperspectralRenderer.hpp
    hs_core/BatchRenderer.cpp
    hs_core/BatchRenderer.hpp
    hs_core/SpectralAnalyzer.cpp
    hs_core/SpectralAnalyzer.hpp
    hs_core/AdaptiveGridGenerator.cpp
    hs_core/AdaptiveGridGenerator.hpp
    hs_core/SpectralReconstructor.cpp
    hs_core/SpectralReconstructor.hpp
    hs_core/GpuSpectralReconstructor.cpp
    hs_core/GpuSpectralReconstructor.hpp

    # Generated files
    ${CMAKE_CURRENT_BINARY_DIR}/core/LibVersion.hpp
)

target_compile_definitions(quantiloom_core
    PUBLIC
        # PUBLIC, not PRIVATE: targets that link the objects statically compile
        # the same headers, and their declarations must carry the same
        # dllexport the definitions in those objects do. Consumers of the DLL
        # never see this -- libQuantiloom links quantiloom_core PRIVATE and
        # hands them QL_USE_SHARED (dllimport) instead.
        QL_BUILD_SHARED
    PRIVATE
        QUANTILOOM_LIB_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        QUANTILOOM_LIB_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        QUANTILOOM_LIB_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        QUANTILOOM_LIB_VERSION_STRING=${PROJECT_VERSION}
)

# Enable validation layers in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(quantiloom_core PRIVATE QUANTILOOM_ENABLE_VALIDATION)
endif()

set_target_properties(quantiloom_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Both roots, because the library's own sources include from either. Targets
# that link the objects (the tests, fusion_tool) inherit both by design -- they
# are entitled to internals. Targets that link the DLL get only the public root.
target_include_directories(quantiloom_core
    PUBLIC
        $<BUILD_INTERFACE:${QL_PUBLIC_INC}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/quantiloom>
)

# GLM headers are needed by consumers (public API uses glm types)
# Consumers must find_package(glm) to get these directories
target_include_directories(quantiloom_core
    PUBLIC
        $<BUILD_INTERFACE:${glm_SOURCE_DIR}>
)

# Link dependencies (from CPM.cmake)
# PUBLIC: Exposed in public headers (Vulkan types)
# PRIVATE: Implementation details hidden via PIMPL (spdlog, toml++, VMA)
# Note: GLM is header-only - exposed via include directories, not link
#       Consumers must find_package(glm) - see QuantiloomConfig.cmake.in
# PRIVATE deps still reach targets that link quantiloom_core: CMake records them
# in INTERFACE_LINK_LIBRARIES wrapped in $<LINK_ONLY:>, so the tests link spdlog
# and friends without inheriting their include directories.
target_link_libraries(quantiloom_core
    PUBLIC
        Vulkan::Vulkan
    PRIVATE
        spdlog::spdlog
        tomlplusplus::tomlplusplus
        nlohmann_json
        VMA
        OpenEXR::OpenEXR
        Imath::Imath
        tinygltf
        glm::glm  # Link privately for build, consumers use find_package(glm)
)

# OpenUSD linking (conditional)
if(QUANTILOOM_USE_OPENUSD)
    target_link_libraries(quantiloom_core
        PRIVATE
            ${USD_LIBRARIES}
            TBB::tbb
    )
    target_include_directories(quantiloom_core
        PRIVATE
            ${USD_ROOT}/include
    )
    target_compile_definitions(quantiloom_core PRIVATE QUANTILOOM_USE_OPENUSD=1)
    message(STATUS "libQuantiloom: OpenUSD support enabled")
else()
    message(WARNING "libQuantiloom: OpenUSD support disabled (USD_ROOT not set)")
    target_compile_definitions(quantiloom_core PRIVATE QUANTILOOM_USE_OPENUSD=0)
endif()

# BC7 texture compression (optional, requires bc7enc_rdo)
option(QUANTILOOM_USE_BC7ENC "Enable BC7 texture compression (requires bc7enc_rdo)" OFF)
if(QUANTILOOM_USE_BC7ENC)
    # Look for bc7enc_rdo in vendor directory
    set(BC7ENC_DIR "${CMAKE_SOURCE_DIR}/vendor/bc7enc_rdo")
    if(EXISTS "${BC7ENC_DIR}/bc7enc.h")
        target_include_directories(quantiloom_core PRIVATE ${BC7ENC_DIR})
        target_compile_definitions(quantiloom_core PRIVATE QUANTILOOM_USE_BC7ENC=1)

        # Add bc7enc_rdo source files (not header-only!)
        target_sources(quantiloom_core PRIVATE
            ${BC7ENC_DIR}/bc7enc.cpp
            ${BC7ENC_DIR}/rgbcx.cpp
        )

        message(STATUS "libQuantiloom: BC7 compression enabled (bc7enc_rdo found)")
    else()
        message(WARNING "libQuantiloom: BC7 compression requested but bc7enc_rdo not found at ${BC7ENC_DIR}")
        message(WARNING "  Download from: https://github.com/richgel999/bc7enc_rdo")
        message(WARNING "  Place files in: vendor/bc7enc_rdo/")
        target_compile_definitions(quantiloom_core PRIVATE QUANTILOOM_USE_BC7ENC=0)
    endif()
else()
    target_compile_definitions(quantiloom_core PRIVATE QUANTILOOM_USE_BC7ENC=0)
    message(STATUS "libQuantiloom: BC7 compression disabled (use -DQUANTILOOM_USE_BC7ENC=ON to enable)")
endif()

# Compiler-specific flags
# /wd4275: a QL_API class whose base is deliberately NOT exported. That is the
#          intended shape now -- e.g. GenericSensor stays public while its
#          SensorModel base is internal -- so the warning reports the design,
#          not a defect.
if(MSVC)
    target_compile_options(quantiloom_core PRIVATE /W4 /EHsc /wd4251 /wd4275 /wd4819)
else()
    # -fvisibility=hidden: Only export symbols explicitly marked with QL_API.
    # Static consumers are unaffected: visibility governs the dynamic symbol
    # table, not link-time resolution against the object files.
    target_compile_options(quantiloom_core PRIVATE -Wall -Wextra -Wpedantic -fvisibility=hidden)
endif()

# ============================================================================
# libQuantiloom -- the SDK artifact
# ============================================================================
# Carries only the .rc (version resource); all code comes from quantiloom_core.
# The link is PRIVATE so the object library's include directories and its
# QL_BUILD_SHARED define stop here: consumers get QL_USE_SHARED (dllimport) and
# only the headers this target declares PUBLIC.
# ============================================================================

add_library(libQuantiloom SHARED libQuantiloom.rc)

target_link_libraries(libQuantiloom PRIVATE quantiloom_core)

target_compile_definitions(libQuantiloom
    INTERFACE
        QL_USE_SHARED  # Consumers linking against this library will define QL_USE_SHARED
)

# The public root only -- src/libQuantiloom is deliberately absent, so nothing
# that links this target can include an internal header. The binary dir is here
# for the generated core/LibVersion.hpp, which is public.
target_include_directories(libQuantiloom
    PUBLIC
        $<BUILD_INTERFACE:${QL_PUBLIC_INC}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${glm_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/quantiloom>
)

target_link_libraries(libQuantiloom PUBLIC Vulkan::Vulkan)

# Set library output name (remove "lib" prefix for cleaner names)
set_target_properties(libQuantiloom PROPERTIES
    OUTPUT_NAME "Quantiloom"
    LINKER_LANGUAGE CXX  # the only listed source is a .rc
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON

    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

message(STATUS "libQuantiloom configured successfully")

# ============================================================================
# Install Targets for find_package(Quantiloom) Support
# ============================================================================
# Now enabled: spdlog, toml++, VMA are PRIVATE dependencies hidden via PIMPL.
# Only glm and Vulkan are PUBLIC - consumers need find_package(glm) and Vulkan SDK.
# ============================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install library binary
install(TARGETS libQuantiloom
    EXPORT QuantiloomTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Ship the PDB next to the DLL. Without it a crash in Quantiloom-Qt yields a
# stack that stops at the DLL boundary; with it the frame resolves to a core
# function and line. OPTIONAL because non-MSVC toolchains produce no PDB.
if(MSVC)
    install(FILES $<TARGET_PDB_FILE:libQuantiloom>
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        OPTIONAL
    )
endif()

# Install public headers (preserve directory structure)
#
# include/quantiloom/ only. This used to glob src/libQuantiloom/, which shipped
# all 61 headers regardless of whether anything outside the library could use
# them -- including EnvironmentPrefilter.hpp, which has no .cpp anywhere.
install(DIRECTORY ${QL_PUBLIC_INC}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quantiloom
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)

# Install generated headers
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/core/LibVersion.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quantiloom/core
)

# ============================================================================
# Bundle GLM headers into SDK (so consumers don't need separate glm install)
# ============================================================================
# Install glm headers (fetched via CPM in root CMakeLists.txt)
install(DIRECTORY ${glm_SOURCE_DIR}/glm
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
        PATTERN "*.inl"
)

# Create a minimal glmConfig.cmake for find_package(glm)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/glmConfig.cmake"
"# glmConfig.cmake - Bundled with Quantiloom SDK
# This allows find_package(glm) to work with bundled headers

if(NOT TARGET glm::glm)
    add_library(glm::glm INTERFACE IMPORTED)
    get_filename_component(_GLM_PREFIX \"\${CMAKE_CURRENT_LIST_DIR}/../../..\" ABSOLUTE)
    set_target_properties(glm::glm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES \"\${_GLM_PREFIX}/include\"
    )
    unset(_GLM_PREFIX)
endif()

set(glm_FOUND TRUE)
set(glm_VERSION \"1.0.1\")
")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/glmConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/glm
)

# ============================================================================
# Install compiled shaders (required at runtime)
# ============================================================================
# Shaders are compiled in src/shaders during build
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/shaders/
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILES_MATCHING PATTERN "*.spv"
)

# ============================================================================
# Install NN atmosphere model pack (required at runtime when atmosphere enabled)
# ============================================================================
install(DIRECTORY ${CMAKE_SOURCE_DIR}/assets/atmos_models
    DESTINATION assets
    FILES_MATCHING PATTERN "*.safetensors"
)

# ============================================================================
# Install OpenUSD Runtime Dependencies (required for USD file loading)
# ============================================================================
if(QUANTILOOM_USE_OPENUSD AND WIN32)
    # Install USD DLLs
    file(GLOB USD_DLLS "${USD_ROOT}/lib/usd_*.dll")
    if(USD_DLLS)
        install(FILES ${USD_DLLS}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    # Install TBB DLLs
    file(GLOB TBB_DLLS "${USD_ROOT}/bin/tbb*.dll")
    if(TBB_DLLS)
        install(FILES ${TBB_DLLS}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    # Install MaterialX DLLs (optional)
    file(GLOB MATERIALX_DLLS "${USD_ROOT}/bin/MaterialX*.dll")
    if(MATERIALX_DLLS)
        install(FILES ${MATERIALX_DLLS}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    # Install USD plugin directory
    if(EXISTS "${USD_ROOT}/lib/usd")
        install(DIRECTORY "${USD_ROOT}/lib/usd"
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()

# Export targets for find_package
install(EXPORT QuantiloomTargets
    FILE QuantiloomTargets.cmake
    NAMESPACE Quantiloom::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Quantiloom
)

# Generate QuantiloomConfig.cmake
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/QuantiloomConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/QuantiloomConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Quantiloom
)

# Generate QuantiloomConfigVersion.cmake
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/QuantiloomConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/QuantiloomConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/QuantiloomConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Quantiloom
)
