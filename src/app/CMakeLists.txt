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

# Enable warnings
if(MSVC)
    add_compile_options(/W4 /wd4251 /wd4819)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_executable(Quantiloom
    main.cpp
    Quantiloom.rc
    ${CMAKE_CURRENT_BINARY_DIR}/Version.hpp
)

# Include directories for generated files
target_include_directories(Quantiloom
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}  # Allows #include "Version.hpp"

        # TRANSITIONAL, paired with the object-library link below. main.cpp
        # drives the Vulkan layer directly -- GpuBuffer, CommandHelper,
        # BLAS/TLAS, RayTracingPipeline -- a second renderer running in
        # parallel with ExternalRenderContext's private steps.
        #
        # Delete both once the CLI drives a core-owned render facade. Together
        # they are the marker for that debt.
        ${CMAKE_SOURCE_DIR}/src/libQuantiloom
)

# The objects, not the DLL -- TRANSITIONAL, see the include note above.
#
# Render stages are moving into the core's internal RenderCore one at a time, and
# internal means not exported: the CLI can compile against those headers but cannot
# link the symbols through Quantiloom.dll. Linking the objects is what lets the
# migration proceed a stage at a time without exporting each stage on the way past.
# Switch back to libQuantiloom once the offline facade is complete and the CLI needs
# nothing but its public API -- that is also what restores the CLI to being the
# working proof that the public API is sufficient.
#
# libSpectraForge is deliberately absent, and now load-bearing rather than tidiness:
# it is a DLL that links Quantiloom.dll, so linking it here while the core is linked
# statically would put two copies of the library's global state in one process. The
# CLI imported nothing from it anyway -- verified against the import table.
target_link_libraries(Quantiloom
    PRIVATE
        quantiloom_core
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
