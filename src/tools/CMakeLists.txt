# ============================================================================
# Quantiloom Tools
# ============================================================================
# Utility tools for postprocessing and analysis
# ============================================================================

# ============================================================================
# Multiband Fusion Tool
# ============================================================================
add_executable(fusion_tool
    fusion_tool.cpp
)

target_link_libraries(fusion_tool
    PRIVATE
        libQuantiloom
)

set_target_properties(fusion_tool PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Compiler warnings
if(MSVC)
    target_compile_options(fusion_tool PRIVATE /W4)
else()
    target_compile_options(fusion_tool PRIVATE -Wall -Wextra -Wpedantic)
endif()

message(STATUS "Fusion tool configured successfully")
