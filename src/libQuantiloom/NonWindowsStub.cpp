/**
 * @file NonWindowsStub.cpp
 * @brief Sole source of the non-Windows libQuantiloom target
 *
 * The Windows build of libQuantiloom lists only the generated version
 * resource (libQuantiloom.rc); every symbol comes from quantiloom_core
 * through the PRIVATE link. The resource compiler is Windows-only and CMake
 * requires at least one compilable source for a SHARED target, so other
 * platforms build this (deliberately empty) unit instead. It exports
 * nothing: the SDK surface is exactly what quantiloom_core marks QL_API.
 */
