/**
 * @file Log.cpp
 * @brief Logging system implementation with spdlog backend
 *
 * spdlog is only included here, not in the public header.
 *
 * @author blitzcolo
 */

#include "core/Log.hpp"

QL_DISABLE_WARNINGS_PUSH
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "core/LogTap.hpp"

#include <mutex>
#include <spdlog/sinks/basic_file_sink.h>
QL_DISABLE_WARNINGS_POP

#include <vector>
#include <memory>

namespace quantiloom {

// Static member definitions
Log::Level Log::s_CurrentLevel = Log::Level::Info;

// File-local spdlog logger (hidden from public API)
static std::shared_ptr<spdlog::logger> s_Logger;

// The tap is read on every message and written almost never, but texture
// decoding logs from worker threads, so the read is under the same lock.
static std::mutex s_TapMutex;
static logtap::Fn s_Tap;

void logtap::Set(Fn fn) {
    std::lock_guard<std::mutex> lock(s_TapMutex);
    s_Tap = std::move(fn);
}

void Log::Init(const char* logFilePath, const Level level) {
    // Create multi-sink logger (console + file)
    std::vector<spdlog::sink_ptr> sinks;

    // Console sink (with color support)
    const auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    consoleSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    sinks.push_back(consoleSink);

    // File sink (if path provided)
    if (logFilePath && logFilePath[0] != '\0') {
        const auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, true);
        fileSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
        sinks.push_back(fileSink);
    }

    // Create logger
    s_Logger = std::make_shared<spdlog::logger>("Quantiloom", sinks.begin(), sinks.end());
    s_Logger->set_level(spdlog::level::trace); // Capture all levels, filter below
    s_Logger->flush_on(spdlog::level::err);    // Auto-flush on errors

    // Register as default logger
    spdlog::register_logger(s_Logger);
    spdlog::set_default_logger(s_Logger);

    // Set user-specified level
    SetLevel(level);

    Info("Quantiloom Logger initialized");
    Info("Platform: {}, Compiler: {}, Config: {}",
         GetPlatformName(), GetCompilerName(), GetBuildConfig());
}

void Log::Shutdown() {
    if (s_Logger) {
        Info("Shutting down logger...");
        s_Logger->flush();
        spdlog::shutdown();
        s_Logger.reset();
    }
}

void Log::SetLevel(const Level level) {
    s_CurrentLevel = level;

    if (!s_Logger) return;

    switch (level) {
        case Level::Trace:    s_Logger->set_level(spdlog::level::trace); break;
        case Level::Debug:    s_Logger->set_level(spdlog::level::debug); break;
        case Level::Info:     s_Logger->set_level(spdlog::level::info); break;
        case Level::Warn:     s_Logger->set_level(spdlog::level::warn); break;
        case Level::Error:    s_Logger->set_level(spdlog::level::err); break;
        case Level::Critical: s_Logger->set_level(spdlog::level::critical); break;
        case Level::Off:      s_Logger->set_level(spdlog::level::off); break;
    }
}

Log::Level Log::GetLevel() {
    return s_CurrentLevel;
}

void Log::Flush() {
    if (s_Logger) {
        s_Logger->flush();
    }
}

void Log::LogMessage(Level level, std::string_view message) {
    {
        std::lock_guard<std::mutex> lock(s_TapMutex);
        if (s_Tap) {
            s_Tap(level, message);
        }
    }
    if (!s_Logger) return;

    // Convert string_view to string for spdlog
    std::string msg(message);

    switch (level) {
        case Level::Trace:    s_Logger->trace("{}", msg); break;
        case Level::Debug:    s_Logger->debug("{}", msg); break;
        case Level::Info:     s_Logger->info("{}", msg); break;
        case Level::Warn:     s_Logger->warn("{}", msg); break;
        case Level::Error:    s_Logger->error("{}", msg); break;
        case Level::Critical: s_Logger->critical("{}", msg); break;
        case Level::Off:      break;
    }
}

} // namespace quantiloom
