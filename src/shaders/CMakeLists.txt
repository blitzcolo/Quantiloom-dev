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

# Compiled shader output directory. The .spv sit next to their sources and are
# gitignored -- they are build output, not artefacts to commit.
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Every .hlsli is a dependency of every shader. Listing only common.hlsli, as
# this did before, meant an edit to pbr.hlsli or blackbody.hlsli left the .spv
# stale. Globbed at configure time: adding a new .hlsli needs a re-configure.
file(GLOB SHADER_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hlsli")

# Flags. These are exactly what compile_shaders.bat uses, and the .spv they
# produce are byte-identical to the ones that shipped before this rule existed
# -- which is how the rule was checked. -Zi/-Qembed_debug are deliberately
# absent: they inflate closesthit.spv from 118 KB to 438 KB, and the SDK ships
# these files.
set(DXC_RT_FLAGS   -spirv -T lib_6_3 -fspv-target-env=vulkan1.2)
set(DXC_COMP_FLAGS -spirv -T cs_6_0  -fspv-target-env=vulkan1.2 -E main)
# Inline ray query needs SM 6.5; everything else stays on the byte-stable 6.0
set(DXC_RQ_FLAGS   -spirv -T cs_6_5  -fspv-target-env=vulkan1.2 -E main)

set(COMPILED_SHADERS)

# Compile one HLSL source to one .spv. Several compute sources produce more
# than one output, differing only by a -D, so the output name is a parameter
# rather than derived from the source.
function(quantiloom_compile_hlsl OUT_NAME SRC_NAME)
    set(SHADER_OUTPUT "${SHADER_OUTPUT_DIR}/${OUT_NAME}.spv")
    set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_NAME}")
    if(${SRC_NAME} MATCHES "\\.rayq\\.hlsl$")
        set(FLAGS ${DXC_RQ_FLAGS})
    elseif(${SRC_NAME} MATCHES "\\.comp\\.hlsl$")
        set(FLAGS ${DXC_COMP_FLAGS})
    else()
        set(FLAGS ${DXC_RT_FLAGS})
    endif()
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${DXC_EXECUTABLE} ${FLAGS} ${ARGN} -Fo ${SHADER_OUTPUT} ${SHADER_SOURCE}
        DEPENDS ${SHADER_SOURCE} ${SHADER_HEADERS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling shader: ${OUT_NAME}"
        VERBATIM
    )
    set(COMPILED_SHADERS ${COMPILED_SHADERS} ${SHADER_OUTPUT} PARENT_SCOPE)
endfunction()

if(DXC_EXECUTABLE)
    # Ray tracing shaders
    quantiloom_compile_hlsl(raygen       raygen.rgen)
    quantiloom_compile_hlsl(closesthit   closesthit.rchit)
    quantiloom_compile_hlsl(anyhit       anyhit.rahit)
    quantiloom_compile_hlsl(miss         miss.rmiss)
    quantiloom_compile_hlsl(shadow_miss  shadow_miss.rmiss)

    # CLAHE display enhancement: three passes from one source
    quantiloom_compile_hlsl(clahe_histogram clahe.comp.hlsl -D CLAHE_PASS_HISTOGRAM)
    quantiloom_compile_hlsl(clahe_cdf       clahe.comp.hlsl -D CLAHE_PASS_CDF)
    quantiloom_compile_hlsl(clahe_apply     clahe.comp.hlsl -D CLAHE_PASS_APPLY)

    # GPU sensor chain
    quantiloom_compile_hlsl(sensor_radiance_to_electrons sensor_radiance_to_electrons.comp.hlsl)
    quantiloom_compile_hlsl(sensor_poisson_noise         sensor_poisson_noise.comp.hlsl)
    quantiloom_compile_hlsl(sensor_psf_blur_horizontal   sensor_psf_blur.comp.hlsl -D SENSOR_PSF_HORIZONTAL)
    quantiloom_compile_hlsl(sensor_psf_blur_vertical     sensor_psf_blur.comp.hlsl -D SENSOR_PSF_VERTICAL)
    quantiloom_compile_hlsl(sensor_quantize_to_radiance  sensor_quantize_to_radiance.comp.hlsl)
    quantiloom_compile_hlsl(sensor_fpn                   sensor_fpn.comp.hlsl)

    # Viewport pick (inline ray query, cs_6_5)
    quantiloom_compile_hlsl(pick pick.rayq.hlsl)
    quantiloom_compile_hlsl(thermal_exchange thermal_exchange.rayq.hlsl)
    quantiloom_compile_hlsl(thermal_step    thermal_step.comp.hlsl)
endif()

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
list(LENGTH COMPILED_SHADERS SHADER_COUNT)
message(STATUS "  HLSL shaders to compile: ${SHADER_COUNT}")
foreach(SHADER_OUTPUT ${COMPILED_SHADERS})
    get_filename_component(SHADER_FILE ${SHADER_OUTPUT} NAME)
    message(STATUS "    - ${SHADER_FILE}")
endforeach()
