# ============================================================================
# Quantiloom Application (Main Executable)
# ============================================================================

add_executable(Quantiloom
    main.cpp
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

# Install target (optional, for packaging)
install(TARGETS Quantiloom
    RUNTIME DESTINATION bin
)

message(STATUS "Quantiloom executable configured successfully")
