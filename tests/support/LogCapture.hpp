/**
 * @file LogCapture.hpp
 * @brief Capture what the library logs for the length of one test
 *
 * A loader's warnings are part of its contract -- "this input is authored but
 * not applied, and here is why" -- and this is how a test pins one. The tap
 * runs before the level filter, so an INFO line is captured while the suite
 * runs at WARN.
 */

#pragma once

#include "core/Log.hpp"
#include "core/LogTap.hpp"

#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace quantiloom::support {

class ScopedLogCapture {
public:
    struct Line {
        Log::Level level;
        std::string text;
    };

    ScopedLogCapture() {
        logtap::Set([this](Log::Level level, std::string_view message) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_lines.push_back({level, std::string(message)});
        });
    }
    ~ScopedLogCapture() { logtap::Set(nullptr); }

    ScopedLogCapture(const ScopedLogCapture&) = delete;
    ScopedLogCapture& operator=(const ScopedLogCapture&) = delete;

    /// Whether any line at exactly `level` contains `needle`.
    [[nodiscard]] bool Has(Log::Level level, std::string_view needle) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const Line& line : m_lines) {
            if (line.level == level && line.text.find(needle) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool HasWarning(std::string_view needle) const {
        return Has(Log::Level::Warn, needle);
    }

    [[nodiscard]] bool HasInfo(std::string_view needle) const {
        return Has(Log::Level::Info, needle);
    }

    /// How many lines at `level` contain `needle` -- for "once per file" claims.
    [[nodiscard]] int Count(Log::Level level, std::string_view needle) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        int count = 0;
        for (const Line& line : m_lines) {
            if (line.level == level && line.text.find(needle) != std::string::npos) {
                ++count;
            }
        }
        return count;
    }

    /// Everything captured, joined -- for a failure message.
    [[nodiscard]] std::string Dump() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::string out;
        for (const Line& line : m_lines) {
            out += line.text;
            out += '\n';
        }
        return out;
    }

private:
    mutable std::mutex m_mutex;
    std::vector<Line> m_lines;
};

}  // namespace quantiloom::support
