# ============================================================================
# libQuantiloom - Core Library
# ============================================================================

# Generate version header from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/core/LibVersion.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/core/LibVersion.hpp"
    @ONLY
)

add_library(libQuantiloom SHARED
    # Core module
    core/Log.cpp
    core/Log.hpp
    core/Config.cpp
    core/Config.hpp
    core/Platform.hpp
    core/Types.hpp
    core/Image.hpp
    core/SpectralCube.hpp
    core/LUT.hpp
    core/AtmosphereTransmittanceLUT.hpp
    libQuantiloom.rc

    # IO module
    io/ImageIO.cpp
    io/ImageIO.hpp
    io/SpectralIO.cpp
    io/SpectralIO.hpp
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
    renderer/AtmosphericConfig.cpp
    renderer/AtmosphericConfig.hpp
    renderer/LightingParams.hpp
    renderer/VulkanContextAdapter.cpp
    renderer/VulkanContextAdapter.hpp
    renderer/ExternalRenderContext.cpp
    renderer/ExternalRenderContext.hpp

    # Scene module
    scene/Camera.cpp
    scene/Camera.hpp
    scene/Mesh.hpp
    scene/Material.cpp
    scene/Material.hpp
    scene/Texture.hpp
    scene/Scene.cpp
    scene/Scene.hpp

    # Postprocess module
    postprocess/SensorModel.hpp
    postprocess/GenericSensor.cpp
    postprocess/GenericSensor.hpp
    postprocess/MultibandFusion.cpp
    postprocess/MultibandFusion.hpp
    postprocess/PostprocessConfig.hpp

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

target_compile_definitions(libQuantiloom
    PRIVATE
        QL_BUILD_SHARED
        QUANTILOOM_LIB_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        QUANTILOOM_LIB_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        QUANTILOOM_LIB_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        QUANTILOOM_LIB_VERSION_STRING=${PROJECT_VERSION}
    INTERFACE
        QL_USE_SHARED  # Consumers linking against this library will define QL_USE_SHARED
)

# Enable validation layers in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(libQuantiloom PRIVATE QUANTILOOM_ENABLE_VALIDATION)
endif()

# Set library output name (remove "lib" prefix for cleaner names)
set_target_properties(libQuantiloom PROPERTIES
    OUTPUT_NAME "Quantiloom"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON

    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories (use generator expressions for proper install/export)
target_include_directories(libQuantiloom
    PUBLIC
        # Build tree: use source and binary dirs
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        # Install tree: use installed include dir
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/quantiloom>
)

# GLM headers are needed by consumers (public API uses glm types)
# Consumers must find_package(glm) to get these directories
target_include_directories(libQuantiloom
    PUBLIC
        $<BUILD_INTERFACE:${glm_SOURCE_DIR}>
)

# Link dependencies (from CPM.cmake)
# PUBLIC: Exposed in public headers (Vulkan types)
# PRIVATE: Implementation details hidden via PIMPL (spdlog, toml++, VMA)
# Note: GLM is header-only - exposed via include directories, not link
#       Consumers must find_package(glm) - see QuantiloomConfig.cmake.in
target_link_libraries(libQuantiloom
    PUBLIC
        Vulkan::Vulkan
    PRIVATE
        spdlog::spdlog
        tomlplusplus::tomlplusplus
        VMA
        OpenEXR::OpenEXR
        Imath::Imath
        tinygltf
        glm::glm  # Link privately for build, consumers use find_package(glm)
)

# OpenUSD linking (conditional)
if(QUANTILOOM_USE_OPENUSD)
    target_link_libraries(libQuantiloom
        PRIVATE
            ${USD_LIBRARIES}
            TBB::tbb
    )
    target_include_directories(libQuantiloom
        PRIVATE
            ${USD_ROOT}/include
    )
    target_compile_definitions(libQuantiloom PRIVATE QUANTILOOM_USE_OPENUSD=1)
    message(STATUS "libQuantiloom: OpenUSD support enabled")
else()
    message(WARNING "libQuantiloom: OpenUSD support disabled (USD_ROOT not set)")
    target_compile_definitions(libQuantiloom PRIVATE QUANTILOOM_USE_OPENUSD=0)
endif()

# Compiler-specific flags
if(MSVC)
    target_compile_options(libQuantiloom PRIVATE /W4 /EHsc /wd4251 /wd4819)
else()
    # -fvisibility=hidden: Only export symbols explicitly marked with QL_API
    target_compile_options(libQuantiloom PRIVATE -Wall -Wextra -Wpedantic -fvisibility=hidden)
endif()

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

# Install public headers (preserve directory structure)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quantiloom
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
    PATTERN "*.cpp" EXCLUDE
    PATTERN "*.rc" EXCLUDE
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
