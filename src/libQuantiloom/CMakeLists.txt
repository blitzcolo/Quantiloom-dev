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
    libQuantiloom.rc

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
)

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
)

# Compiler-specific flags
if(MSVC)
    target_compile_options(libQuantiloom PRIVATE /W4)
else()
    target_compile_options(libQuantiloom PRIVATE -Wall -Wextra -Wpedantic)
endif()

message(STATUS "libQuantiloom configured successfully")
