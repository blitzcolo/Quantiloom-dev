# ============================================================================
# Quantiloom Application (Main Executable)
# ============================================================================

# Generate version header from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Version.hpp"
    @ONLY
)

add_executable(Quantiloom
    main.cpp
    Quantiloom.rc
    ${CMAKE_CURRENT_BINARY_DIR}/Version.hpp
)

# Include directories for generated files
target_include_directories(Quantiloom
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}  # Allows #include "Version.hpp"
)

# Link against the core library
target_link_libraries(Quantiloom
    PRIVATE
        libQuantiloom
)

target_compile_definitions(Quantiloom
    PRIVATE
        QL_USE_STATIC
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
endif()

# Install target (optional, for packaging)
install(TARGETS Quantiloom
    RUNTIME DESTINATION bin
)

# ============================================================================
# Quantiloom M1 Test (Standalone Ray Tracing Test)
# ============================================================================

add_executable(QuantiloomM1Test
    main_m1_test.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/Version.hpp
)

target_include_directories(QuantiloomM1Test
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(QuantiloomM1Test
    PRIVATE
        libQuantiloom
)

target_compile_definitions(QuantiloomM1Test
    PRIVATE
        QL_USE_STATIC
)

set_target_properties(QuantiloomM1Test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

message(STATUS "Quantiloom executables configured successfully")
