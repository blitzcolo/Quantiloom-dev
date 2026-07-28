/**
 * @file Log.hpp
 * @brief Logging system facade - public API without third-party dependencies
 *
 * This header exposes only standard C++ types. The underlying spdlog
 * implementation is hidden in Log.cpp.
 *
 * @author blitzcolo
 */

#pragma once

#include "Platform.hpp"
#include "Types.hpp"

#include <string_view>
#include <format>

// ============================================================================
// Logging System Facade
// Quantiloom M0 - PIMPL-based logging with hidden spdlog backend
// ============================================================================

namespace quantiloom {

/**
 * @class Log
 * @brief Centralized logging system with hidden spdlog backend
 *
 * Public API uses only standard C++ types. The spdlog implementation
 * is completely hidden, allowing DLL distribution without spdlog headers.
 */
class QL_API Log {
public:
    /// Log severity levels
    enum class Level {
        Trace,    ///< Verbose debugging info
        Debug,    ///< Development-time diagnostic
        Info,     ///< General informational messages
        Warn,     ///< Warnings (non-critical issues)
        Error,    ///< Errors (recoverable failures)
        Critical, ///< Critical errors (program-terminating)
        Off       ///< Disable logging
    };

    /// Initialize the logging system with console and file output
    /// @param logFilePath Optional path to log file (nullptr = console only)
    /// @param level Minimum severity level to display
    static void Init(const char* logFilePath = "quantiloom.log", Level level = Level::Info);

    /// Shutdown the logging system (flushes buffers)
    static void Shutdown();

    /// Set the global log level at runtime
    static void SetLevel(Level level);

    /// Retrieve the current log level
    static Level GetLevel();

    /// Flush all log buffers immediately
    static void Flush();

    // ========================================================================
    // Core Logging Interface (string_view based)
    // ========================================================================

    /// Log a message at the specified level
    static void LogMessage(Level level, std::string_view message);

    /// Log at Trace level
    static void Trace(std::string_view message) { LogMessage(Level::Trace, message); }

    /// Log at Debug level
    static void Debug(std::string_view message) { LogMessage(Level::Debug, message); }

    /// Log at Info level
    static void Info(std::string_view message) { LogMessage(Level::Info, message); }

    /// Log at Warn level
    static void Warn(std::string_view message) { LogMessage(Level::Warn, message); }

    /// Log at Error level
    static void Error(std::string_view message) { LogMessage(Level::Error, message); }

    /// Log at Critical level
    static void Critical(std::string_view message) { LogMessage(Level::Critical, message); }

    // ========================================================================
    // Formatted Logging Interface (C++20 std::format)
    // ========================================================================

    /// Log formatted message at Trace level
    template<typename... Args>
    static void Trace(std::format_string<Args...> fmt, Args&&... args) {
        LogMessage(Level::Trace, std::format(fmt, std::forward<Args>(args)...));
    }

    /// Log formatted message at Debug level
    template<typename... Args>
    static void Debug(std::format_string<Args...> fmt, Args&&... args) {
        LogMessage(Level::Debug, std::format(fmt, std::forward<Args>(args)...));
    }

    /// Log formatted message at Info level
    template<typename... Args>
    static void Info(std::format_string<Args...> fmt, Args&&... args) {
        LogMessage(Level::Info, std::format(fmt, std::forward<Args>(args)...));
    }

    /// Log formatted message at Warn level
    template<typename... Args>
    static void Warn(std::format_string<Args...> fmt, Args&&... args) {
        LogMessage(Level::Warn, std::format(fmt, std::forward<Args>(args)...));
    }

    /// Log formatted message at Error level
    template<typename... Args>
    static void Error(std::format_string<Args...> fmt, Args&&... args) {
        LogMessage(Level::Error, std::format(fmt, std::forward<Args>(args)...));
    }

    /// Log formatted message at Critical level
    template<typename... Args>
    static void Critical(std::format_string<Args...> fmt, Args&&... args) {
        LogMessage(Level::Critical, std::format(fmt, std::forward<Args>(args)...));
    }

private:
    static Level s_CurrentLevel;
};

} // namespace quantiloom

// ============================================================================
// Convenience Macros
// ============================================================================

#define QL_LOG_TRACE(...)    ::quantiloom::Log::Trace(__VA_ARGS__)
#define QL_LOG_DEBUG(...)    ::quantiloom::Log::Debug(__VA_ARGS__)
#define QL_LOG_INFO(...)     ::quantiloom::Log::Info(__VA_ARGS__)
#define QL_LOG_WARN(...)     ::quantiloom::Log::Warn(__VA_ARGS__)
#define QL_LOG_ERROR(...)    ::quantiloom::Log::Error(__VA_ARGS__)
#define QL_LOG_CRITICAL(...) ::quantiloom::Log::Critical(__VA_ARGS__)

// Short-form aliases
#define LOG_TRACE(...)    ::quantiloom::Log::Trace(__VA_ARGS__)
#define LOG_DEBUG(...)    ::quantiloom::Log::Debug(__VA_ARGS__)
#define LOG_INFO(...)     ::quantiloom::Log::Info(__VA_ARGS__)
#define LOG_WARN(...)     ::quantiloom::Log::Warn(__VA_ARGS__)
#define LOG_ERROR(...)    ::quantiloom::Log::Error(__VA_ARGS__)
#define LOG_CRITICAL(...) ::quantiloom::Log::Critical(__VA_ARGS__)
