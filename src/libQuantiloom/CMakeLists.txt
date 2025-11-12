# ============================================================================
# libQuantiloom - Core Library
# ============================================================================

add_library(libQuantiloom STATIC
    # Core module
    core/Log.cpp
    core/Log.hpp
    core/Config.cpp
    core/Config.hpp
    core/Platform.hpp
    core/Types.hpp
)

# Set library output name (remove "lib" prefix for cleaner names)
set_target_properties(libQuantiloom PROPERTIES
    OUTPUT_NAME "Quantiloom"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(libQuantiloom
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # Allows #include "core/Log.hpp"
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
