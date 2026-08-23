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

# Links the objects rather than the DLL: fusion_tool drives MultibandFusion,
# which is library-internal and no longer exported. See src/libQuantiloom/CMakeLists.txt.
target_link_libraries(fusion_tool
    PRIVATE
        quantiloom_core
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

# ============================================================================
# Measurement tools
# ============================================================================
# Two harnesses that drive library internals directly, for experiments whose
# quantity is not an image:
#
#   thermal_column_tool  one 1-D column through the surface energy balance,
#                        with no scene around it -- a site validation, and the
#                        pointwise reference a mesh-resolution study measures
#                        against
#   sensor_lab           one sensor instance over many frames, which is the
#                        only arrangement that separates temporal noise from
#                        the fixed pattern
#
# Both link quantiloom_core rather than the DLL, for the same reason
# fusion_tool does: what they drive is internal and is not going to be
# exported to give a measurement script a way in.
foreach(tool thermal_column_tool sensor_lab)
    add_executable(${tool} ${tool}.cpp)
    target_link_libraries(${tool} PRIVATE quantiloom_core)
    set_target_properties(${tool} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    if(MSVC)
        target_compile_options(${tool} PRIVATE /W4)
    else()
        target_compile_options(${tool} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

# quantiloom_core carries the USD loader, so anything linking it needs the USD
# runtime beside the binary or it will not start -- the same POST_BUILD copy the
# CLI and the test binary already do. fusion_tool is on this list because it was
# missing it too: it links the same objects and has been unable to launch from
# its own build directory since USD came in.
if(QUANTILOOM_USE_OPENUSD AND COMMAND copy_usd_dlls)
    # One copy for all three, not one each. The tools share an output
    # directory, so three POST_BUILD passes writing the same sixty DLLs into it
    # race for the file handles and the build fails with MSB3073 on whichever
    # link finished second. fusion_tool carries the copy; the other two depend
    # on it, which both serialises the link order and guarantees the runtime is
    # in place before anything is asked to start.
    copy_usd_dlls(fusion_tool)
    add_dependencies(thermal_column_tool fusion_tool)
    add_dependencies(sensor_lab fusion_tool)
endif()

message(STATUS "Measurement tools configured successfully")

# ============================================================================
# QLTrans — MODTRAN atmospheric LUT generator (Windows only)
# ============================================================================
if(WIN32)
    add_executable(QLTrans
        QLTrans/main.cpp
        QLTrans/QLTrans.rc
    )
    target_include_directories(QLTrans PRIVATE QLTrans)
    target_link_libraries(QLTrans PRIVATE tomlplusplus::tomlplusplus)
    set_target_properties(QLTrans PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    if(MSVC)
        target_compile_options(QLTrans PRIVATE /W4 /wd4996)
    endif()
    message(STATUS "QLTrans configured successfully")
endif()
