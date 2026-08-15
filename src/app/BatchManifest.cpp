#include "BatchManifest.hpp"

#include "core/Log.hpp"

#include <fstream>
#include <sstream>

namespace quantiloom::app {

namespace fs = std::filesystem;

namespace {

String Trim(const String& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == String::npos) return "";
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

/// Split the override half of a line into tokens, honouring double quotes.
///
/// Quotes are kept rather than stripped: a quoted value is quoted because TOML
/// wants it that way, and the token goes into a TOML document verbatim. The
/// only thing quoting does here is stop a space inside a value from ending the
/// token.
std::vector<String> Tokenise(const String& text) {
    std::vector<String> tokens;
    String current;
    bool inQuotes = false;

    for (usize i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '"') {
            inQuotes = !inQuotes;
            current += c;
        } else if (c == '\\' && inQuotes && i + 1 < text.size()) {
            // A backslash escape belongs to TOML, not to this tokeniser; pass
            // both characters through so \" does not end the quoted run.
            current += c;
            current += text[++i];
        } else if (!inQuotes && (c == ' ' || c == '\t')) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

}  // namespace

Result<std::vector<ManifestEntry>, String> ParseManifest(const fs::path& manifestPath) {
    using ParseResult = Result<std::vector<ManifestEntry>, String>;

    std::ifstream in(manifestPath);
    if (!in) {
        return ParseResult::Err("Cannot open manifest: " + manifestPath.string());
    }

    const fs::path manifestDir = fs::absolute(manifestPath).parent_path();

    std::vector<ManifestEntry> entries;
    String line;
    usize lineNumber = 0;

    while (std::getline(in, line)) {
        ++lineNumber;
        const String trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        ManifestEntry entry;
        entry.lineNumber = lineNumber;

        const auto bar = trimmed.find('|');
        const String pathText = Trim(bar == String::npos ? trimmed : trimmed.substr(0, bar));
        if (pathText.empty()) {
            return ParseResult::Err("Manifest line " + std::to_string(lineNumber) +
                                    " names overrides but no configuration");
        }

        fs::path configPath(pathText);
        if (configPath.is_relative()) configPath = manifestDir / configPath;
        entry.configPath = configPath.lexically_normal();

        if (bar != String::npos) {
            String document;
            for (const String& token : Tokenise(trimmed.substr(bar + 1))) {
                if (token.front() == '@') {
                    const String sidecarText = token.substr(1);
                    if (sidecarText.empty()) {
                        return ParseResult::Err("Manifest line " + std::to_string(lineNumber) +
                                                ": '@' names no file");
                    }
                    fs::path sidecar(sidecarText);
                    if (sidecar.is_relative()) sidecar = manifestDir / sidecar;
                    entry.sidecars.push_back(sidecar.lexically_normal());
                    continue;
                }

                const auto equals = token.find('=');
                if (equals == String::npos || equals == 0) {
                    return ParseResult::Err(
                        "Manifest line " + std::to_string(lineNumber) + ": '" + token +
                        "' is neither key=value nor @sidecar.toml");
                }
                // TOML's own grammar decides what the value is, so a typo is
                // rejected here rather than rendering the unmodified scene.
                document += token.substr(0, equals) + " = " + token.substr(equals + 1) + "\n";
            }

            if (!document.empty()) {
                if (auto parsed = Config::Parse(document); !parsed.has_value()) {
                    return ParseResult::Err("Manifest line " + std::to_string(lineNumber) +
                                            ": " + parsed.error());
                }
                entry.inlineOverrides = std::move(document);
            }
        }

        entries.push_back(std::move(entry));
    }

    if (entries.empty()) {
        return ParseResult::Err("Manifest names no configurations: " + manifestPath.string());
    }
    return ParseResult(std::move(entries));
}

Result<Config, String> ApplyEntryOverrides(const Config& config, const ManifestEntry& entry) {
    using ApplyResult = Result<Config, String>;

    Config merged = config;

    for (const fs::path& sidecar : entry.sidecars) {
        auto loaded = Config::Load(sidecar);
        if (!loaded.has_value()) {
            return ApplyResult::Err("cannot load sidecar " + sidecar.string() + ": " +
                                    loaded.error());
        }
        merged = merged.MergedWith(loaded.value());
    }

    if (!entry.inlineOverrides.empty()) {
        // Already validated by ParseManifest; a failure here would be a bug in
        // one of the two, so it is reported rather than ignored.
        auto parsed = Config::Parse(entry.inlineOverrides);
        if (!parsed.has_value()) {
            return ApplyResult::Err("inline overrides did not parse: " + parsed.error());
        }
        merged = merged.MergedWith(parsed.value());
    }

    return ApplyResult(std::move(merged));
}

String DescribeEntryOverrides(const ManifestEntry& entry) {
    if (entry.sidecars.empty() && entry.inlineOverrides.empty()) {
        return "";
    }

    std::ostringstream summary;
    for (const fs::path& sidecar : entry.sidecars) {
        summary << "@" << sidecar.filename().string() << " ";
    }
    // The document is one key per line; the summary is one line, so the
    // newlines become spaces.
    std::istringstream lines(entry.inlineOverrides);
    String assignment;
    while (std::getline(lines, assignment)) {
        if (!assignment.empty()) summary << assignment << "  ";
    }
    return Trim(summary.str());
}

}  // namespace quantiloom::app
