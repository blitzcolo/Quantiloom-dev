# ============================================================================
# Quantiloom Shader Compilation
# ============================================================================
# Automatically compiles HLSL shaders to SPIR-V using DXC
# and GLSL compute shaders using glslc
# ============================================================================

# Find DXC compiler (for HLSL ray tracing shaders)
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

# Find glslc compiler (for GLSL compute shaders)
find_program(GLSLC_EXECUTABLE
    NAMES glslc glslc.exe
    PATHS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
        "/usr/bin"
    DOC "GLSL Compiler (glslc)"
)

if(NOT DXC_EXECUTABLE)
    message(WARNING
        "DXC not found! HLSL shaders will NOT be compiled automatically.\n"
        "To install DXC, run:\n"
        "  Linux/macOS: ./tools/setup_dxc.sh\n"
        "  Windows:     powershell -ExecutionPolicy Bypass -File tools/setup_dxc.ps1\n"
        "\n"
        "Alternatively, install Vulkan SDK which includes DXC."
    )
else()
    message(STATUS "Found DXC: ${DXC_EXECUTABLE}")
endif()

if(NOT GLSLC_EXECUTABLE)
    message(WARNING
        "glslc not found! GLSL compute shaders will NOT be compiled automatically.\n"
        "Please install Vulkan SDK which includes glslc."
    )
else()
    message(STATUS "Found glslc: ${GLSLC_EXECUTABLE}")
endif()

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

# ============================================================================
# GLSL Compute Shaders (using glslc)
# ============================================================================

set(COMPUTE_SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ibl_brdf_lut.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/spectral_reconstruct.comp
)

set(COMPILED_COMPUTE_SHADERS)

if(GLSLC_EXECUTABLE)
    foreach(COMP_SHADER ${COMPUTE_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${COMP_SHADER} NAME_WE)
        set(SHADER_OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

        add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${GLSLC_EXECUTABLE}
                -fshader-stage=compute
                -o ${SHADER_OUTPUT}
                ${COMP_SHADER}
            DEPENDS ${COMP_SHADER}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Compiling compute shader: ${SHADER_NAME}"
            VERBATIM
        )

        list(APPEND COMPILED_COMPUTE_SHADERS ${SHADER_OUTPUT})
    endforeach()

    add_custom_target(CompileComputeShaders ALL
        DEPENDS ${COMPILED_COMPUTE_SHADERS}
        COMMENT "Compiling all compute shaders"
    )

    message(STATUS "  Compute shaders to compile:")
    foreach(COMP_SHADER ${COMPUTE_SHADER_SOURCES})
        get_filename_component(SHADER_FILE ${COMP_SHADER} NAME)
        message(STATUS "    - ${SHADER_FILE}")
    endforeach()
endif()

# Print summary
message(STATUS "Shader compilation configured:")
message(STATUS "  Input directory:  ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Output directory: ${SHADER_OUTPUT_DIR}")
message(STATUS "  HLSL shaders to compile:")
foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_FILE ${SHADER_SOURCE} NAME)
    message(STATUS "    - ${SHADER_FILE}")
endforeach()
