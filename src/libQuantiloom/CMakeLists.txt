# ============================================================================
# libQuantiloom - Core Library
# ============================================================================

# Generate version header from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/core/LibVersion.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/core/LibVersion.hpp"
    @ONLY
)

add_library(libQuantiloom STATIC
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
    libQuantiloom.rc

    # IO module
    io/ImageIO.cpp
    io/ImageIO.hpp
    io/SpectralIO.cpp
    io/SpectralIO.hpp
    io/LUTLoader.cpp
    io/LUTLoader.hpp
    io/GltfLoader.cpp
    io/GltfLoader.hpp

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

    # Scene module
    scene/Camera.cpp
    scene/Camera.hpp
    scene/Mesh.hpp
    scene/Material.hpp
    scene/Texture.hpp
    scene/Scene.cpp
    scene/Scene.hpp

    # Generated files
    ${CMAKE_CURRENT_BINARY_DIR}/core/LibVersion.hpp
)

target_compile_definitions(libQuantiloom
    PRIVATE
        QL_BUILD_STATIC
        QUANTILOOM_LIB_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        QUANTILOOM_LIB_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        QUANTILOOM_LIB_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        QUANTILOOM_LIB_VERSION_STRING=${PROJECT_VERSION}
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

# Include directories
target_include_directories(libQuantiloom
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # Allows #include "core/Log.hpp"
        ${CMAKE_CURRENT_BINARY_DIR}  # Allows #include "core/LibVersion.hpp"
)

# Link dependencies (from CPM.cmake)
target_link_libraries(libQuantiloom
    PUBLIC
        spdlog::spdlog
        tomlplusplus::tomlplusplus
        OpenEXR::OpenEXR
        Imath::Imath
        hdf5-static
        hdf5_cpp-static
        Vulkan::Vulkan
        VMA
        glm::glm
        tinygltf
)

# Add HDF5 include directories if using find_package
if(HDF5_FOUND AND NOT hdf5_ADDED)
    target_include_directories(libQuantiloom PUBLIC ${HDF5_INCLUDE_DIRS})
endif()

# Compiler-specific flags
if(MSVC)
    target_compile_options(libQuantiloom PRIVATE /W4)
else()
    target_compile_options(libQuantiloom PRIVATE -Wall -Wextra -Wpedantic)
endif()

message(STATUS "libQuantiloom configured successfully")
