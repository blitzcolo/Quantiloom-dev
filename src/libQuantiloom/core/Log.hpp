#pragma once

#include "Platform.hpp"
#include "Types.hpp"

QL_DISABLE_WARNINGS_PUSH
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
QL_DISABLE_WARNINGS_POP

#include <memory>

// ============================================================================
// Logging System Facade
// Quantiloom M0 - spdlog wrapper with multiple severity levels
// ============================================================================

namespace quantiloom {

/// Centralized logging system using spdlog backend
class QL_API Log {
public:
    /// Log severity levels
    enum class Level {
        Trace,    // Verbose debugging info
        Debug,    // Development-time diagnostic
        Info,     // General informational messages
        Warn,     // Warnings (non-critical issues)
        Error,    // Errors (recoverable failures)
        Critical, // Critical errors (program-terminating)
        Off       // Disable logging
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

    // ========================================================================
    // Templated Logging Interface (supports fmt-style formatting)
    // ========================================================================

    template<typename... Args>
    static void Trace(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        s_Logger->trace(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Debug(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        s_Logger->debug(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Info(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        s_Logger->info(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Warn(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        s_Logger->warn(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Error(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        s_Logger->error(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Critical(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        s_Logger->critical(fmt, std::forward<Args>(args)...);
    }

    /// Flush all log buffers immediately
    static void Flush();

private:
    static std::shared_ptr<spdlog::logger> s_Logger;
};

} // namespace quantiloom

// ============================================================================
// Convenience Macros (optional - can be disabled if conflicts exist)
// ============================================================================

#define QL_LOG_TRACE(...)    ::quantiloom::Log::Trace(__VA_ARGS__)
#define QL_LOG_DEBUG(...)    ::quantiloom::Log::Debug(__VA_ARGS__)
#define QL_LOG_INFO(...)     ::quantiloom::Log::Info(__VA_ARGS__)
#define QL_LOG_WARN(...)     ::quantiloom::Log::Warn(__VA_ARGS__)
#define QL_LOG_ERROR(...)    ::quantiloom::Log::Error(__VA_ARGS__)
#define QL_LOG_CRITICAL(...) ::quantiloom::Log::Critical(__VA_ARGS__)
