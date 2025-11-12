#include "Log.hpp"

QL_DISABLE_WARNINGS_PUSH
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
QL_DISABLE_WARNINGS_POP

#include <vector>

namespace quantiloom {

// Static member definition
std::shared_ptr<spdlog::logger> Log::s_Logger;

void Log::Init(const char* logFilePath, Level level) {
    // Create multi-sink logger (console + file)
    std::vector<spdlog::sink_ptr> sinks;

    // Console sink (with color support)
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    consoleSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    sinks.push_back(consoleSink);

    // File sink (if path provided)
    if (logFilePath && logFilePath[0] != '\0') {
        auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, true);
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

void Log::SetLevel(Level level) {
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
    if (!s_Logger) return Level::Off;

    switch (s_Logger->level()) {
        case spdlog::level::trace:    return Level::Trace;
        case spdlog::level::debug:    return Level::Debug;
        case spdlog::level::info:     return Level::Info;
        case spdlog::level::warn:     return Level::Warn;
        case spdlog::level::err:      return Level::Error;
        case spdlog::level::critical: return Level::Critical;
        default:                      return Level::Off;
    }
}

void Log::Flush() {
    if (s_Logger) {
        s_Logger->flush();
    }
}

} // namespace quantiloom
