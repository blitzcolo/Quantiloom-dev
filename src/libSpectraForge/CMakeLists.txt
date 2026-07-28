# ============================================================================
# libSpectraForge - IR Material Generator from Visible Textures
# ============================================================================

# Generate version header from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/LibVersion.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/LibVersion.hpp"
    @ONLY
)

add_library(libSpectraForge SHARED
    ColorCluster.cpp
    ColorCluster.hpp
    IRMaterialDatabase.cpp
    IRMaterialDatabase.hpp
    SpectraForge.cpp
    SpectraForge.hpp
    Platform.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/LibVersion.hpp
)

# Windows resource file
if(WIN32)
    target_sources(libSpectraForge PRIVATE libSpectraForge.rc)
endif()

target_compile_definitions(libSpectraForge
    PRIVATE
        SF_BUILD_SHARED
    INTERFACE
        SF_USE_SHARED
)

set_target_properties(libSpectraForge PROPERTIES
    OUTPUT_NAME "SpectraForge"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_include_directories(libSpectraForge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# Link against libQuantiloom (gives us Types.hpp, Scene, Material, Log, etc.)
target_link_libraries(libSpectraForge
    PUBLIC
        libQuantiloom
)

# Compiler flags
if(MSVC)
    target_compile_options(libSpectraForge PRIVATE /W4 /EHsc /wd4251 /wd4819)
else()
    target_compile_options(libSpectraForge PRIVATE -Wall -Wextra -Wpedantic -fvisibility=hidden)
endif()

# ============================================================================
# Install Targets (included in SDK via cmake --install)
# ============================================================================

include(GNUInstallDirs)

# Install library binary (.dll/.so + .lib import stub)
install(TARGETS libSpectraForge
    EXPORT QuantiloomTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# PDB alongside the DLL — see the note in src/libQuantiloom/CMakeLists.txt.
if(MSVC)
    install(FILES $<TARGET_PDB_FILE:libSpectraForge>
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        OPTIONAL
    )
endif()

# Install public headers (flat — no subdirectories in libSpectraForge)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spectraforge
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
    PATTERN "*.cpp" EXCLUDE
    PATTERN "*.rc" EXCLUDE
    PATTERN "*.in" EXCLUDE
)

# Install generated version header
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LibVersion.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spectraforge
)

message(STATUS "libSpectraForge configured successfully")
