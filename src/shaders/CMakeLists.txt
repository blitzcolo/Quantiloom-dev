# ============================================================================
# Quantiloom Shader Compilation
# ============================================================================
# Compiles every shader -- all HLSL -- to SPIR-V with DXC, and, as a compile
# check for the Metal backend, to DXIL and (when Apple's Metal Shader Converter
# is installed) on to a metallib.
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

# ----------------------------------------------------------------------------
# Portability check. Every shader is also compiled to DXIL, and from there
# through Apple's Metal Shader Converter (MSC) to a metallib when MSC is
# installed -- MSC has a Windows build, and DXIL is the only input it takes.
# Nothing installs these outputs. They exist so that a construct the Metal
# backend cannot take fails a Windows build today rather than a Mac build later.
#
# MSC 4.0 turns a NaN/Inf optimisation on by default. The shaders guard
# radiance with isnan/isinf/isfinite in some fifty places, and an optimiser
# allowed to assume no NaN may fold those guards away, so every conversion
# passes --disableNanInfOptimization. Ray generation is compiled as a compute
# kernel, the mode Apple recommends since MSC 3.0. Bindings come from the root
# signature in ql_bind.hlsli (see there for which stages carry one).
#
# MSC 4.0.3 for Windows exits 5 with no message when handed any
# --root-signature JSON, including the example in its own manual; a root
# signature embedded in the DXIL is read correctly, which is why it lives in
# the shader source rather than in a JSON file here.
# ----------------------------------------------------------------------------
option(QUANTILOOM_SHADER_PORTABILITY_CHECK
       "Also compile every shader to DXIL, and to a metallib when Metal Shader Converter is found" ON)

find_program(QUANTILOOM_MSC_EXECUTABLE
    NAMES metal-shaderconverter metal-shaderconverter.exe
    PATHS
        "$ENV{METAL_SHADER_CONVERTER_DIR}"
        "$ENV{METAL_SHADER_CONVERTER_DIR}/bin"
        "C:/Program Files/Metal Shader Converter/bin"
        "/usr/local/bin"
    DOC "Apple Metal Shader Converter (DXIL -> metallib)"
)
set(QUANTILOOM_MSC_FLAGS "" CACHE STRING "Extra metal-shaderconverter flags")
separate_arguments(QUANTILOOM_MSC_FLAG_LIST NATIVE_COMMAND "${QUANTILOOM_MSC_FLAGS}")

set(PORTABILITY_DIR "${CMAKE_CURRENT_BINARY_DIR}/portability")
set(PORTABILITY_OUTPUTS)
set(PORTABILITY_NAMES)
if(QUANTILOOM_SHADER_PORTABILITY_CHECK)
    file(MAKE_DIRECTORY "${PORTABILITY_DIR}")
    if(QUANTILOOM_MSC_EXECUTABLE)
        message(STATUS "Found Metal Shader Converter: ${QUANTILOOM_MSC_EXECUTABLE}")
    else()
        message(STATUS "Metal Shader Converter not found: the shader portability "
                       "check stops at DXIL (set METAL_SHADER_CONVERTER_DIR)")
    endif()
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

# The A/B behind the first-bounce stratification measurement of the paper's
# Section VIII-B. ON replaces the six padded Owen-scrambled Sobol' slots with
# the same PCG stream the deeper bounces already use, changing nothing else, so
# the two convergence curves differ only in the sampler. It is not a supported
# rendering mode -- it exists so the reported speed-up can be reproduced rather
# than taken on trust, and it must be OFF for anything that ships.
option(QUANTILOOM_UNSTRATIFIED_FIRST_BOUNCE
       "Draw first-bounce samples from PCG instead of Sobol' (measurement only)" OFF)
if(QUANTILOOM_UNSTRATIFIED_FIRST_BOUNCE)
    message(WARNING "First-bounce stratification is DISABLED. This build is for "
                    "sampling measurements only.")
    list(APPEND DXC_RT_FLAGS -D QUANTILOOM_UNSTRATIFIED_FIRST_BOUNCE=1)
    list(APPEND DXC_COMP_FLAGS -D QUANTILOOM_UNSTRATIFIED_FIRST_BOUNCE=1)
endif()
# Inline ray query needs SM 6.5; everything else stays on the byte-stable 6.0
set(DXC_RQ_FLAGS   -spirv -T cs_6_5  -fspv-target-env=vulkan1.2 -E main)

# A .spv depends on the flags it was compiled with as much as on its source, and
# add_custom_command cannot see a flag change: flipping the option above leaves
# every previously built .spv newer than its inputs, so nothing recompiles and
# the next measurement silently runs the old shader. That is not hypothetical --
# it made both arms of the Section VIII-B A/B return bit-identical numbers.
# A stamp whose *content* is the flag set gives the dependency something to
# notice; file(CONFIGURE) rewrites it only when the content actually differs, so
# an unchanged configure still costs nothing.
set(SHADER_FLAGS_STAMP "${CMAKE_CURRENT_BINARY_DIR}/shader_flags.stamp")
file(CONFIGURE OUTPUT "${SHADER_FLAGS_STAMP}"
     CONTENT "RT=${DXC_RT_FLAGS}\nCOMP=${DXC_COMP_FLAGS}\nRQ=${DXC_RQ_FLAGS}\nMSC=${QUANTILOOM_MSC_FLAGS}\n")

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
        DEPENDS ${SHADER_SOURCE} ${SHADER_HEADERS} ${SHADER_FLAGS_STAMP}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling shader: ${OUT_NAME}"
        VERBATIM
    )
    set(COMPILED_SHADERS ${COMPILED_SHADERS} ${SHADER_OUTPUT} PARENT_SCOPE)

    if(QUANTILOOM_SHADER_PORTABILITY_CHECK)
        # Same source, same target profile and defines, minus the SPIR-V switches.
        set(DXIL_FLAGS ${FLAGS})
        list(REMOVE_ITEM DXIL_FLAGS -spirv -fspv-target-env=vulkan1.2)
        if(${SRC_NAME} MATCHES "\\.rayq\\.hlsl$")
            list(APPEND DXIL_FLAGS -rootsig-define QL_ROOT_SIGNATURE)
        endif()
        set(DXIL_OUTPUT "${PORTABILITY_DIR}/${OUT_NAME}.dxil")
        add_custom_command(
            OUTPUT ${DXIL_OUTPUT}
            COMMAND ${DXC_EXECUTABLE} ${DXIL_FLAGS} ${ARGN} -Fo ${DXIL_OUTPUT} ${SHADER_SOURCE}
            DEPENDS ${SHADER_SOURCE} ${SHADER_HEADERS} ${SHADER_FLAGS_STAMP}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Shader portability (DXIL): ${OUT_NAME}"
            VERBATIM
        )
        set(OUTPUTS ${PORTABILITY_OUTPUTS} ${DXIL_OUTPUT})
        if(QUANTILOOM_MSC_EXECUTABLE)
            set(METALLIB_OUTPUT "${PORTABILITY_DIR}/${OUT_NAME}.metallib")
            set(MSC_STAGE_FLAGS)
            if(${SRC_NAME} MATCHES "\\.rgen$")
                set(MSC_STAGE_FLAGS --rt-ray-generation-compilation=kernel)
            endif()
            add_custom_command(
                OUTPUT ${METALLIB_OUTPUT}
                COMMAND ${QUANTILOOM_MSC_EXECUTABLE} --disableNanInfOptimization
                        ${MSC_STAGE_FLAGS} ${QUANTILOOM_MSC_FLAG_LIST}
                        ${DXIL_OUTPUT} -o=${METALLIB_OUTPUT}
                DEPENDS ${DXIL_OUTPUT} ${SHADER_FLAGS_STAMP}
                COMMENT "Shader portability (metallib): ${OUT_NAME}"
                VERBATIM
            )
            list(APPEND OUTPUTS ${METALLIB_OUTPUT})
        endif()
        set(PORTABILITY_OUTPUTS ${OUTPUTS} PARENT_SCOPE)
        set(PORTABILITY_NAMES ${PORTABILITY_NAMES} ${OUT_NAME} PARENT_SCOPE)
    endif()
endfunction()

if(DXC_EXECUTABLE)
    quantiloom_compile_hlsl(fusion_geometry fusion_geometry.rayq.hlsl)
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
    quantiloom_compile_hlsl(display_range_extents.comp display_range.comp.hlsl -D DISPLAY_RANGE_PASS_EXTENTS)
    quantiloom_compile_hlsl(display_range_histogram.comp display_range.comp.hlsl -D DISPLAY_RANGE_PASS_HISTOGRAM)

    # Unified camera device preview (rate PSF, detector/RAW, ISP, RNG audit).
    quantiloom_compile_hlsl(camera_psf.comp         camera_psf.comp.hlsl)
    quantiloom_compile_hlsl(camera_readout.comp     camera_readout.comp.hlsl)
    quantiloom_compile_hlsl(camera_stats_tiles.comp camera_stats.comp.hlsl -D CAMERA_STATS_PHASE_TILES)
    quantiloom_compile_hlsl(camera_stats_reduce.comp camera_stats.comp.hlsl -D CAMERA_STATS_PHASE_REDUCE)
    quantiloom_compile_hlsl(camera_stats_hist.comp  camera_stats.comp.hlsl -D CAMERA_STATS_PHASE_HISTOGRAM)
    quantiloom_compile_hlsl(camera_cdf.comp         camera_cdf.comp.hlsl)
    quantiloom_compile_hlsl(camera_demosaic.comp    camera_demosaic.comp.hlsl)
    quantiloom_compile_hlsl(camera_rng_vectors.comp camera_rng_vectors.comp.hlsl)
    quantiloom_compile_hlsl(camera_fast_rgb.comp    camera_fast_rgb.comp.hlsl)
    quantiloom_compile_hlsl(camera_dynamic.comp     camera_dynamic.comp.hlsl)
    quantiloom_compile_hlsl(camera_hsv.comp         camera_hsv.comp.hlsl)

    # Viewport pick (inline ray query, cs_6_5)
    quantiloom_compile_hlsl(pick pick.rayq.hlsl)
    quantiloom_compile_hlsl(thermal_exchange thermal_exchange.rayq.hlsl)
    quantiloom_compile_hlsl(thermal_step    thermal_step.comp.hlsl)

    # Hyperspectral cube reconstruction (hs_core/GpuSpectralReconstructor)
    quantiloom_compile_hlsl(spectral_reconstruct spectral_reconstruct.comp.hlsl)
endif()

# ----------------------------------------------------------------------------
# SHADERSET: one digest over the exact .spv set, installed beside the shaders
# and compiled into the library. A resource copy whose stamp does not match is
# rejected at runtime instead of being silently rendered with (f.7).
# ----------------------------------------------------------------------------
set(QL_SHADER_SET_NAMES)
foreach(SHADER_OUTPUT ${COMPILED_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER_OUTPUT} NAME)
    list(APPEND QL_SHADER_SET_NAMES ${SHADER_NAME})
endforeach()
list(SORT QL_SHADER_SET_NAMES)
set(QL_SHADER_SET_STAMP "${CMAKE_CURRENT_BINARY_DIR}/SHADERSET")
set(QL_SHADER_SET_DIGEST_HEADER "${CMAKE_CURRENT_BINARY_DIR}/core/ShaderSetDigest.hpp")
# A stub so the first configure always has a header to compile against; the
# custom command below replaces it before the library is compiled.
if(NOT EXISTS "${QL_SHADER_SET_DIGEST_HEADER}")
    file(WRITE "${QL_SHADER_SET_DIGEST_HEADER}"
         "#pragma once\nnamespace quantiloom::core { inline constexpr const char* kShaderSetDigest = \"\"; }\n")
endif()
list(JOIN QL_SHADER_SET_NAMES ";" QL_SHADER_SET_NAME_LIST)
add_custom_command(
    OUTPUT ${QL_SHADER_SET_STAMP} ${QL_SHADER_SET_DIGEST_HEADER}
    COMMAND ${CMAKE_COMMAND}
        -DSHADER_DIR=${SHADER_OUTPUT_DIR}
        "-DNAMES=${QL_SHADER_SET_NAME_LIST}"
        -DSTAMP=${QL_SHADER_SET_STAMP}
        -DHEADER=${QL_SHADER_SET_DIGEST_HEADER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/ShaderSetStamp.cmake
    DEPENDS ${COMPILED_SHADERS} ${CMAKE_CURRENT_SOURCE_DIR}/ShaderSetStamp.cmake
    COMMENT "Stamping the shader set (SHADERSET)"
    VERBATIM
)

# One target owns every rule above: the .spv compiles and the stamp over them.
# The Visual Studio generator copies a custom command into every target of
# this directory that depends on its output, following DEPENDS chains, so a
# second target reaching the .spv -- even only through the stamp -- got its own
# copy of all the DXC rules, and under -j two projects ran the same rule at
# once (MSB8066 on ShaderSetStamp.vcxproj). Inside one target the stamp still
# runs after the shaders it hashes, and again whenever any of them changes.
add_custom_target(CompileShaders ALL
    DEPENDS ${COMPILED_SHADERS} ${QL_SHADER_SET_STAMP} ${QL_SHADER_SET_DIGEST_HEADER}
    COMMENT "Compiling all shaders"
)
# The name the library orders itself after. It carries no rule of its own:
# depending on the stamp file here would copy the stamp's rule, and through it
# every .spv rule, back into this target.
add_custom_target(ShaderSetStamp)
add_dependencies(ShaderSetStamp CompileShaders)
set(QL_SHADER_SET_STAMP "${QL_SHADER_SET_STAMP}" PARENT_SCOPE)
set(QL_SHADER_SET_DIGEST_HEADER "${QL_SHADER_SET_DIGEST_HEADER}" PARENT_SCOPE)

# Portability check target, plus a manifest the build scripts read to report
# it: which shaders were checked, and whether the metallib half ran. A failed
# DXC or MSC step already fails the build; the manifest exists so that a
# machine without MSC says so in the gate output instead of passing quietly.
if(QUANTILOOM_SHADER_PORTABILITY_CHECK AND PORTABILITY_OUTPUTS)
    add_custom_target(ShaderPortabilityCheck ALL
        DEPENDS ${PORTABILITY_OUTPUTS}
        COMMENT "Shader portability check (DXIL / metallib)"
    )
    if(QUANTILOOM_MSC_EXECUTABLE)
        set(PORTABILITY_MSC "1")
    else()
        set(PORTABILITY_MSC "0")
    endif()
    list(JOIN PORTABILITY_NAMES "\n" PORTABILITY_NAME_LINES)
    file(CONFIGURE OUTPUT "${PORTABILITY_DIR}/manifest.txt"
         CONTENT "msc=${PORTABILITY_MSC}\n${PORTABILITY_NAME_LINES}\n"
         NEWLINE_STYLE UNIX)
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
