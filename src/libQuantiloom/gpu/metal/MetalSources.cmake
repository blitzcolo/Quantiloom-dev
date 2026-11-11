# The Metal backend's sources, included by src/libQuantiloom/CMakeLists.txt when
# QUANTILOOM_BACKEND_METAL is on. Objective-C++ with ARC: the backend holds
# Metal objects in C++ classes and lets ARC own them.

find_path(QL_MSC_RUNTIME_INCLUDE metal_irconverter_runtime/metal_irconverter_runtime.h
    HINTS
        "$ENV{METAL_SHADER_CONVERTER_DIR}/include"
        "$ENV{METAL_SHADER_CONVERTER_DIR}/../include"
        /usr/local/include
        /opt/metal-shaderconverter/include
    DOC "Metal Shader Converter's runtime companion header (metal_irconverter_runtime.h)"
)
if(NOT QL_MSC_RUNTIME_INCLUDE)
    message(FATAL_ERROR "The Metal backend needs Metal Shader Converter's runtime header "
                        "metal_irconverter_runtime/metal_irconverter_runtime.h (installed with MSC).")
endif()

set(QL_METAL_SOURCES
    gpu/metal/MetalBackend.hpp
    gpu/metal/MetalCommon.mm
    gpu/metal/MetalResources.mm
    gpu/metal/MetalDevice.mm
    gpu/metal/MetalCommandList.mm
    gpu/metal/MetalRt.mm
    gpu/metal/MetalSync.mm
    gpu/metal/MetalDeviceFactory.mm
    gpu/metal/MetalIRRuntime.mm
)
target_sources(quantiloom_core PRIVATE ${QL_METAL_SOURCES})
set_source_files_properties(${QL_METAL_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-fobjc-arc"
    SKIP_PRECOMPILE_HEADERS ON
)
target_include_directories(quantiloom_core PRIVATE ${QL_MSC_RUNTIME_INCLUDE})
target_link_libraries(quantiloom_core PUBLIC "-framework Metal" "-framework Foundation")
