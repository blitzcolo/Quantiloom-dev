# ============================================================================
# Quantiloom Application (Main Executable)
# ============================================================================

# UTC build timestamp (set once at configure time)
string(TIMESTAMP QUANTILOOM_BUILD_TIMESTAMP_UTC "%Y-%m-%d %H:%M:%S UTC" UTC)

# Generate version header from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Version.hpp"
    @ONLY
)

# The Windows version resource, from the same project(VERSION) as the header
# above. It used to be a checked-in Quantiloom.rc with the version typed into
# four lines of it, which meant a bump touched five files across this repo and
# nothing failed when one was missed -- a stale FILEVERSION is only visible in
# Explorer's Details tab. @ONLY because a .rc is full of ${...} belonging to
# the resource compiler.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Quantiloom.rc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Quantiloom.rc"
    @ONLY
)

# Enable warnings
if(MSVC)
    add_compile_options(/W4 /wd4251 /wd4819)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_executable(Quantiloom
    main.cpp
    BatchJob.cpp
    BatchJob.hpp
    BatchManifest.cpp
    BatchManifest.hpp
    RenderJob.cpp
    RenderJob.hpp
    McpServe.cpp
    McpServe.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/Quantiloom.rc
    ${CMAKE_CURRENT_BINARY_DIR}/Version.hpp
)

# Include directories for generated files
target_include_directories(Quantiloom
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}  # Allows #include "Version.hpp"
)

# The DLL, and nothing but its public headers.
#
# This target is the working proof that the public API is sufficient: it renders
# every scene the product renders, through the same include/ and the same QL_API
# exports the SDK ships. src/libQuantiloom is not on its search path, so reaching
# for a library internal here fails at compile time rather than at review time --
# which makes "the header is missing" a signal to widen the SDK on purpose.
#
# It used to link quantiloom_core (the objects) with the internal source root on
# its include path, because main.cpp drove the Vulkan layer itself: GpuBuffer,
# CommandHelper, BLAS/TLAS, RayTracingPipeline, a second render orchestrator
# running in parallel with ExternalRenderContext's private steps. That is
# OfflineRenderer now.
#
# libSpectraForge is deliberately absent: the CLI imports nothing from it --
# verified against the import table.
#
# nlohmann_json is a third-party header-only library, not a Quantiloom internal:
# serve mode reads tool arguments and writes tool results, and the constraint
# above is about reaching into src/libQuantiloom, not about doing without JSON.
target_link_libraries(Quantiloom
    PRIVATE
        libQuantiloom
        nlohmann_json
)

# C++20 standard and RPATH set to executable directory
set_target_properties(Quantiloom PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "$ORIGIN"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Copy the library to the executable directory
if(WIN32)
    add_custom_command(
        TARGET Quantiloom POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:libQuantiloom>"
            "$<TARGET_FILE_DIR:Quantiloom>"
        COMMENT "Copying Quantiloom.dll to executable directory"
    )

    # Copy OpenUSD DLLs if enabled
    if(QUANTILOOM_USE_OPENUSD AND COMMAND copy_usd_dlls)
        copy_usd_dlls(Quantiloom)
    endif()
endif()

# Copy compiled shaders to the executable directory
add_custom_command(
    TARGET Quantiloom POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/shaders/raygen.spv"
        "${CMAKE_SOURCE_DIR}/src/shaders/closesthit.spv"
        "${CMAKE_SOURCE_DIR}/src/shaders/anyhit.spv"
        "${CMAKE_SOURCE_DIR}/src/shaders/miss.spv"
        "${CMAKE_SOURCE_DIR}/src/shaders/shadow_miss.spv"
        "${CMAKE_SOURCE_DIR}/src/shaders/ibl_brdf_lut.spv"
        "${CMAKE_SOURCE_DIR}/src/shaders/spectral_reconstruct.spv"
        "${CMAKE_SOURCE_DIR}/src/shaders/clahe_histogram.spv"
        "${CMAKE_SOURCE_DIR}/src/shaders/clahe_cdf.spv"
        "${CMAKE_SOURCE_DIR}/src/shaders/clahe_apply.spv"
        "$<TARGET_FILE_DIR:Quantiloom>"
    COMMENT "Copying shaders to executable directory"
)

# CMake ignores DEPENDS in the TARGET form of add_custom_command, so the
# `DEPENDS CompileShaders` that used to sit in the block above did nothing.
# It went unnoticed because CompileShaders is an ALL target and an ordinary
# full build reaches it anyway -- but `--build build --target Quantiloom`
# copied whatever .spv happened to be lying in the source tree.
add_dependencies(Quantiloom CompileShaders)

# Install target (optional, for packaging)
install(TARGETS Quantiloom
    RUNTIME DESTINATION bin
)

# PDB alongside the executable — see the note in src/libQuantiloom/CMakeLists.txt.
# Renamed because this target and libQuantiloom both output the base name
# "Quantiloom", so both PDBs default to Quantiloom.pdb and the second one
# installed silently overwrites the first in bin/. The linker records this name
# in the executable, so the debugger still finds it.
if(MSVC)
    set_target_properties(Quantiloom PROPERTIES PDB_NAME "QuantiloomCLI")
    install(FILES $<TARGET_PDB_FILE:Quantiloom>
        DESTINATION bin
        OPTIONAL
    )
endif()

message(STATUS "Quantiloom executables configured successfully")
