#include "Config.hpp"
#include "Log.hpp"

#include <fstream>
#include <sstream>

namespace quantiloom {

Config::Config(toml::table&& root) : m_Root(std::move(root)) {}

Result<Config, String> Config::Load(const std::filesystem::path& filePath) {
    if (!std::filesystem::exists(filePath)) {
        return Result<Config, String>::Err("Config file not found: " + filePath.string());
    }

    try {
        toml::table table = toml::parse_file(filePath.string());
        Log::Info("Loaded configuration from: {}", filePath.string());
        return Config(std::move(table));
    }
    catch (const toml::parse_error& err) {
        std::ostringstream oss;
        oss << "TOML parse error: " << err.description()
            << " at line " << err.source().begin.line
            << ", column " << err.source().begin.column;
        return Result<Config, String>::Err(oss.str());
    }
    catch (const std::exception& ex) {
        return Result<Config, String>::Err(String("Failed to load config: ") + ex.what());
    }
}

bool Config::Has(StringView key) const {
    return Navigate(key) != nullptr;
}

Result<Config, String> Config::GetTable(StringView key) const {
    const toml::node* node = Navigate(key);
    if (!node) {
        return Result<Config, String>::Err("Table not found: " + String(key));
    }

    if (!node->is_table()) {
        return Result<Config, String>::Err("Key is not a table: " + String(key));
    }

    // Clone the table (toml++ doesn't provide a non-const access pattern here)
    toml::table clonedTable = *node->as_table();
    return Config(std::move(clonedTable));
}

void Config::Print() const {
    std::ostringstream oss;
    oss << m_Root;
    Log::Info("Configuration:\n{}", oss.str());
}

const toml::node* Config::Navigate(StringView key) const {
    // Split key by '.' and traverse the hierarchy
    const toml::node* current = &m_Root;
    usize start = 0;

    while (start < key.size()) {
        usize end = key.find('.', start);
        if (end == StringView::npos) {
            end = key.size();
        }

        StringView segment = key.substr(start, end - start);
        String segmentStr(segment);

        if (current->is_table()) {
            const toml::table* table = current->as_table();
            auto it = table->find(segmentStr);
            if (it == table->end()) {
                return nullptr;
            }
            current = &it->second;
        } else {
            return nullptr;
        }

        start = end + 1;
    }

    return current;
}

} // namespace quantiloom
