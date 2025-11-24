# ============================================================================
# Quantiloom Shader Compilation
# ============================================================================
# Automatically compiles HLSL shaders to SPIR-V using DXC
# ============================================================================

# Find DXC compiler
set(DXC_SEARCH_PATHS
    "${CMAKE_SOURCE_DIR}/tools/dxc/bin"           # Local DXC from setup script
    "${CMAKE_SOURCE_DIR}/tools/dxc/bin/x64"       # Windows local DXC
    "$ENV{VULKAN_SDK}/Bin"                        # Vulkan SDK on Windows
    "$ENV{VULKAN_SDK}/bin"                        # Vulkan SDK on Linux/macOS
)

find_program(DXC_EXECUTABLE
    NAMES dxc dxc.exe
    PATHS ${DXC_SEARCH_PATHS}
    DOC "DirectX Shader Compiler (DXC)"
)

if(NOT DXC_EXECUTABLE)
    message(WARNING
        "DXC not found! Shaders will NOT be compiled automatically.\n"
        "To install DXC, run:\n"
        "  Linux/macOS: ./tools/setup_dxc.sh\n"
        "  Windows:     powershell -ExecutionPolicy Bypass -File tools/setup_dxc.ps1\n"
        "\n"
        "Alternatively, install Vulkan SDK which includes DXC."
    )
    return()
endif()

message(STATUS "Found DXC: ${DXC_EXECUTABLE}")

# Shader source files
set(SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/raygen.rgen
    ${CMAKE_CURRENT_SOURCE_DIR}/closesthit.rchit
    ${CMAKE_CURRENT_SOURCE_DIR}/miss.rmiss
)

# Compiled shader output directory
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# DXC compilation flags
set(DXC_FLAGS
    -spirv                          # Generate SPIR-V
    -T lib_6_3                      # Shader model 6.3 (required for ray tracing)
    -fspv-target-env=vulkan1.3      # Target Vulkan 1.3
    -WX                             # Treat warnings as errors
    -Qembed_debug                   # Embed debug info
    -Zi                             # Generate PDB file, -Qembed_debug requires it
)

# Create custom targets for each shader
set(COMPILED_SHADERS)

foreach(SHADER_SOURCE ${SHADER_SOURCES})
    # Get shader filename without extension
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)

    # Output SPIR-V file
    set(SHADER_OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

    # Add custom command to compile shader
    #add_custom_command(
    #    OUTPUT ${SHADER_OUTPUT}
    #    COMMAND ${DXC_EXECUTABLE} ${DXC_FLAGS} -Fo ${SHADER_OUTPUT} ${SHADER_SOURCE}
    #    DEPENDS ${SHADER_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/common.hlsli
    #    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    #    COMMENT "Compiling shader: ${SHADER_NAME}"
    #    VERBATIM
    #)

    list(APPEND COMPILED_SHADERS ${SHADER_OUTPUT})
endforeach()

# Create custom target that depends on all compiled shaders
add_custom_target(CompileShaders ALL
    DEPENDS ${COMPILED_SHADERS}
    COMMENT "Compiling all shaders"
)

# Print summary
message(STATUS "Shader compilation configured:")
message(STATUS "  Input directory:  ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Output directory: ${SHADER_OUTPUT_DIR}")
message(STATUS "  Shaders to compile:")
foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_FILE ${SHADER_SOURCE} NAME)
    message(STATUS "    - ${SHADER_FILE}")
endforeach()
