# ============================================================================
# Quantiloom Application (Main Executable)
# ============================================================================

add_executable(Quantiloom
    main.cpp
    Quantiloom.rc
)

# Link against the core library
target_link_libraries(Quantiloom
    PRIVATE
        libQuantiloom
)

# C++20 standard
set_target_properties(Quantiloom PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Copy the library to the executable directory
add_custom_command(
    TARGET Quantiloom POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:libQuantiloom>" # DLL/shared library file
        "$<TARGET_FILE_DIR:Quantiloom>"  # Executable output directory
    COMMENT "Copying Quantiloom.dll to executable directory"
)

# Install target (optional, for packaging)
install(TARGETS Quantiloom
    RUNTIME DESTINATION bin
)

message(STATUS "Quantiloom executable configured successfully")
