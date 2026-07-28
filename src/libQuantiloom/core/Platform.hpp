/**
 * @file Platform.hpp
 * @brief Cross-platform abstraction layer and compiler detection macros
 *
 * Provides platform/compiler-specific macros for:
 * - Platform identification (QL_WINDOWS, QL_LINUX, QL_MACOS)
 * - Compiler detection (QL_COMPILER_MSVC, QL_COMPILER_GCC, QL_COMPILER_CLANG)
 * - Build configuration (QL_DEBUG, QL_RELEASE)
 * - DLL export/import (QL_API)
 * - Debug utilities (QL_DEBUGBREAK, QL_ASSERT)
 * - Warning suppression (QL_DISABLE_WARNINGS_PUSH/POP)
 *
 * Platform macros are defined by CMake at configure time based on target system.
 * This header ensures Quantiloom compiles correctly on Windows/Linux with MSVC/GCC/Clang.
 *
 * @author blitzcolo
 */

#pragma once

// ============================================================================
// Platform Detection & Abstraction Layer
// Cross-platform compatibility macros
// ============================================================================

// Platform identification macros (defined by CMake)
#if defined(QUANTILOOM_PLATFORM_WINDOWS)
    #define QL_WINDOWS 1
#elif defined(QUANTILOOM_PLATFORM_LINUX)
    #define QL_LINUX 1
#elif defined(QUANTILOOM_PLATFORM_MACOS)
    #define QL_MACOS 1
#else
    #error "Unsupported platform! Quantiloom requires Windows, Linux, or macOS."
#endif

// Compiler detection
#if defined(_MSC_VER)
    #define QL_COMPILER_MSVC 1
#elif defined(__clang__)
    #define QL_COMPILER_CLANG 1
#elif defined(__GNUC__)
    #define QL_COMPILER_GCC 1
#elif defined(__INTEL_COMPILER)
    #define QL_COMPILER_INTEL 1
#else
    #error "Unsupported compiler! C++20 support required."
#endif

// Build configuration
#if defined(NDEBUG)
    #define QL_RELEASE 1
#else
    #define QL_DEBUG 1
#endif

// DLL export/import macros (for shared library support)
//
// QL_API is the SDK's public contract, not a convenience marker. A symbol that
// carries it is exported from Quantiloom.dll, visible to Quantiloom-Qt, and
// covered by SemVer -- removing it later is a breaking change.
//
// New code defaults to NOT having it. Two rules keep it that way:
//
//   * Needing a unit test is not a reason to export. The tests link
//     quantiloom_core (the OBJECT library) and see every symbol regardless.
//   * A new GUI capability extends the ExternalRenderContext facade with a
//     method plus a POD parameter struct. It does not export the class that
//     implements the capability -- the frontend needs to set a parameter, not
//     to own an object.
//
// Adding QL_API is a deliberate act: it also means moving the header into the
// public set and updating docs/abi/exports.golden, which is reviewed.
#if defined(QL_WINDOWS)
    #if defined(QL_BUILD_SHARED)
        #define QL_API __declspec(dllexport)
    #elif defined(QL_USE_SHARED)
        #define QL_API __declspec(dllimport)
    #else
        #define QL_API
    #endif
#else
    // Linux/macOS: Use visibility attribute when building shared library
    // Requires -fvisibility=hidden compiler flag to hide non-exported symbols
    #if defined(QL_BUILD_SHARED)
        #define QL_API __attribute__((visibility("default")))
    #else
        #define QL_API
    #endif
#endif

// Debug break macro
#if defined(QL_DEBUG)
    #if defined(QL_WINDOWS)
        #define QL_DEBUGBREAK() __debugbreak()
    #elif defined(QL_LINUX) || defined(QL_MACOS)
        #include <signal.h>
        #define QL_DEBUGBREAK() raise(SIGTRAP)
    #endif
#else
    #define QL_DEBUGBREAK()
#endif

// Function signature macro (for logging)
#if defined(QL_COMPILER_MSVC)
    #define QL_FUNC_SIG __FUNCSIG__
#else
    #define QL_FUNC_SIG __PRETTY_FUNCTION__
#endif

// Disable specific warnings for third-party headers
#if defined(QL_COMPILER_MSVC)
    #define QL_DISABLE_WARNINGS_PUSH __pragma(warning(push, 0))
    #define QL_DISABLE_WARNINGS_POP  __pragma(warning(pop))
#elif defined(QL_COMPILER_CLANG) || defined(QL_COMPILER_GCC)
    #define QL_DISABLE_WARNINGS_PUSH \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wall\"") \
        _Pragma("GCC diagnostic ignored \"-Wextra\"")
    #define QL_DISABLE_WARNINGS_POP _Pragma("GCC diagnostic pop")
#elif defined(QL_COMPILER_INTEL)
    #define QL_DISABLE_WARNINGS_PUSH __pragma(warning(push, 0))
    #define QL_DISABLE_WARNINGS_POP  __pragma(warning(pop))
#endif

// Assert macro (active in debug builds)
#if defined(QL_DEBUG)
    #include <cassert>
    #define QL_ASSERT(condition, message) \
        do { \
            if (!(condition)) { \
                ::quantiloom::Log::Critical("Assertion failed: {} at {}:{}", \
                    message, __FILE__, __LINE__); \
                QL_DEBUGBREAK(); \
                assert(condition); \
            } \
        } while (false)
#elif defined(QL_RELEASE)
    #define QL_ASSERT(condition, message) ((void)0)
#elif defined(QL_RELWITHDEBINFO)
    #define QL_ASSERT(condition, message) ((void)0)
#elif defined(QL_MINSIZEREL)
    #define QL_ASSERT(condition, message) ((void)0)
#else
    #define QL_ASSERT(condition, message) ((void)0)
#endif

namespace quantiloom {

/// Returns a human-readable platform string
constexpr const char* GetPlatformName() {
    #if defined(QL_WINDOWS)
        return "Windows";
    #elif defined(QL_LINUX)
        return "Linux";
    #elif defined(QL_MACOS)
        return "macOS";
    #else
        return "Unknown";
    #endif
}

/// Returns a human-readable compiler string
constexpr const char* GetCompilerName() {
    #if defined(QL_COMPILER_MSVC)
        return "MSVC";
    #elif defined(QL_COMPILER_CLANG)
        return "Clang";
    #elif defined(QL_COMPILER_GCC)
        return "GCC";
    #elif defined(QL_COMPILER_INTEL)
        return "Intel";
    #endif
}

/// Returns build configuration string
constexpr const char* GetBuildConfig() {
    #if defined(QL_DEBUG)
        return "Debug";
    #elif defined(QL_RELEASE)
        return "Release";
    #elif defined(QL_RELWITHDEBINFO)
        return "RelWithDebInfo";
    #elif defined(QL_MINSIZEREL)
        return "MinSizeRel";
    #else
        return "Unknown";
    #endif
}

} // namespace quantiloom
