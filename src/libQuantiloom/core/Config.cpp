/**
 * @file Config.cpp
 * @brief Configuration loader implementation with toml++ backend
 *
 * toml++ is only included here, not in the public header (PIMPL pattern).
 *
 * @author wtflmao
 */

#include "Config.hpp"
#include "Log.hpp"

QL_DISABLE_WARNINGS_PUSH
#include <toml++/toml.h>
QL_DISABLE_WARNINGS_POP

#include <fstream>
#include <sstream>

namespace quantiloom {

// ============================================================================
// PIMPL Implementation
// ============================================================================

struct Config::Impl {
    toml::table root;

    Impl() = default;
    explicit Impl(toml::table&& t) : root(std::move(t)) {}
    Impl(const Impl& other) : root(other.root) {}

    // Navigate to a nested node by dot-separated path
    const toml::node* Navigate(StringView key) const {
        const toml::node* current = &root;
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
};

// ============================================================================
// Constructors / Destructor / Assignment
// ============================================================================

Config::Config() : m_impl(std::make_unique<Impl>()) {}

Config::~Config() = default;

Config::Config(const Config& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}

Config::Config(Config&& other) noexcept = default;

Config& Config::operator=(const Config& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}

Config& Config::operator=(Config&& other) noexcept = default;

Config::Config(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) {}

// ============================================================================
// Static Factory
// ============================================================================

Result<Config, String> Config::Load(const std::filesystem::path& filePath) {
    if (!std::filesystem::exists(filePath)) {
        return Result<Config, String>::Err("Config file not found: " + filePath.string());
    }

    try {
        toml::table table = toml::parse_file(filePath.string());
        Log::Info("Loaded configuration from: {}", filePath.string());

        auto impl = std::make_unique<Impl>(std::move(table));
        return Config(std::move(impl));
    }
    catch (const toml::parse_error& err) {
        std::ostringstream oss;
        oss << "TOML parse error: " << err.description()
            << " at line " << err.source().begin.line
            << ", column " << err.source().begin.column;
        return Result<Config, String>::Err(oss.str());
    }
}

// ============================================================================
// Key Existence
// ============================================================================

bool Config::Has(const StringView key) const {
    return m_impl->Navigate(key) != nullptr;
}

bool Config::HasSection(const StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    return node != nullptr && node->is_table();
}

// ============================================================================
// Type-safe Getters
// ============================================================================

String Config::GetString(StringView key, const String& defaultValue) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return defaultValue;
    if (const auto val = node->value<std::string>()) {
        return *val;
    }
    return defaultValue;
}

i32 Config::GetInt(StringView key, i32 defaultValue) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return defaultValue;
    if (const auto val = node->value<int64_t>()) {
        return static_cast<i32>(*val);
    }
    return defaultValue;
}

i64 Config::GetInt64(StringView key, i64 defaultValue) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return defaultValue;
    if (const auto val = node->value<int64_t>()) {
        return *val;
    }
    return defaultValue;
}

u32 Config::GetUInt(StringView key, u32 defaultValue) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return defaultValue;
    if (const auto val = node->value<int64_t>()) {
        return static_cast<u32>(*val);
    }
    return defaultValue;
}

u64 Config::GetUInt64(StringView key, u64 defaultValue) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return defaultValue;
    if (const auto val = node->value<int64_t>()) {
        return static_cast<u64>(*val);
    }
    return defaultValue;
}

f32 Config::GetFloat(StringView key, f32 defaultValue) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return defaultValue;
    if (const auto val = node->value<double>()) {
        return static_cast<f32>(*val);
    }
    return defaultValue;
}

f64 Config::GetDouble(StringView key, f64 defaultValue) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return defaultValue;
    if (const auto val = node->value<double>()) {
        return *val;
    }
    return defaultValue;
}

bool Config::GetBool(StringView key, bool defaultValue) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return defaultValue;
    if (const auto val = node->value<bool>()) {
        return *val;
    }
    return defaultValue;
}

// ============================================================================
// Required Getters
// ============================================================================

Result<String, String> Config::GetRequiredString(StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) {
        return Result<String, String>::Err("Missing required key: " + String(key));
    }
    if (const auto val = node->value<std::string>()) {
        return *val;
    }
    return Result<String, String>::Err("Type mismatch for key: " + String(key));
}

Result<i32, String> Config::GetRequiredInt(StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) {
        return Result<i32, String>::Err("Missing required key: " + String(key));
    }
    if (const auto val = node->value<int64_t>()) {
        return static_cast<i32>(*val);
    }
    return Result<i32, String>::Err("Type mismatch for key: " + String(key));
}

Result<f32, String> Config::GetRequiredFloat(StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) {
        return Result<f32, String>::Err("Missing required key: " + String(key));
    }
    if (const auto val = node->value<double>()) {
        return static_cast<f32>(*val);
    }
    return Result<f32, String>::Err("Type mismatch for key: " + String(key));
}

Result<bool, String> Config::GetRequiredBool(StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) {
        return Result<bool, String>::Err("Missing required key: " + String(key));
    }
    if (const auto val = node->value<bool>()) {
        return *val;
    }
    return Result<bool, String>::Err("Type mismatch for key: " + String(key));
}

// ============================================================================
// Array Getters
// ============================================================================

Vector<String> Config::GetStringArray(StringView key) const {
    Vector<String> result;
    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_array()) return result;

    const toml::array* arr = node->as_array();
    result.reserve(arr->size());
    for (const auto& elem : *arr) {
        if (auto val = elem.value<std::string>()) {
            result.push_back(*val);
        }
    }
    return result;
}

Vector<i32> Config::GetIntArray(StringView key) const {
    Vector<i32> result;
    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_array()) return result;

    const toml::array* arr = node->as_array();
    result.reserve(arr->size());
    for (const auto& elem : *arr) {
        if (auto val = elem.value<int64_t>()) {
            result.push_back(static_cast<i32>(*val));
        }
    }
    return result;
}

Vector<f32> Config::GetFloatArray(StringView key) const {
    Vector<f32> result;
    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_array()) return result;

    const toml::array* arr = node->as_array();
    result.reserve(arr->size());
    for (const auto& elem : *arr) {
        if (auto val = elem.value<double>()) {
            result.push_back(static_cast<f32>(*val));
        }
    }
    return result;
}

Vector<f64> Config::GetDoubleArray(StringView key) const {
    Vector<f64> result;
    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_array()) return result;

    const toml::array* arr = node->as_array();
    result.reserve(arr->size());
    for (const auto& elem : *arr) {
        if (auto val = elem.value<double>()) {
            result.push_back(*val);
        }
    }
    return result;
}

// ============================================================================
// Section/Table Access
// ============================================================================

Result<Config, String> Config::GetTable(const StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) {
        return Result<Config, String>::Err("Table not found: " + String(key));
    }
    if (!node->is_table()) {
        return Result<Config, String>::Err("Key is not a table: " + String(key));
    }

    toml::table clonedTable = *node->as_table();
    auto impl = std::make_unique<Impl>(std::move(clonedTable));
    return Config(std::move(impl));
}

Vector<Config> Config::GetTableArray(const StringView key) const {
    Vector<Config> result;

    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_array()) {
        return result;
    }

    const toml::array* arr = node->as_array();
    for (const auto& elem : *arr) {
        if (elem.is_table()) {
            toml::table clonedTable = *elem.as_table();
            auto impl = std::make_unique<Impl>(std::move(clonedTable));
            result.push_back(Config(std::move(impl)));
        }
    }

    return result;
}

std::unordered_map<String, String> Config::GetSection(const StringView key) const {
    std::unordered_map<String, String> result;

    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_table()) {
        return result;
    }

    const toml::table* table = node->as_table();
    for (const auto& [k, v] : *table) {
        if (v.is_string()) {
            result[String(k)] = *v.value<std::string>();
        }
    }

    return result;
}

void Config::Print() const {
    std::ostringstream oss;
    oss << m_impl->root;
    Log::Info("Configuration:\n{}", oss.str());
}

// ============================================================================
// Template Specializations
// ============================================================================

template<>
String Config::Get<String>(StringView key, const String& defaultValue) const {
    return GetString(key, defaultValue);
}

template<>
i32 Config::Get<i32>(StringView key, const i32& defaultValue) const {
    return GetInt(key, defaultValue);
}

template<>
i64 Config::Get<i64>(StringView key, const i64& defaultValue) const {
    return GetInt64(key, defaultValue);
}

template<>
u32 Config::Get<u32>(StringView key, const u32& defaultValue) const {
    return GetUInt(key, defaultValue);
}

template<>
u64 Config::Get<u64>(StringView key, const u64& defaultValue) const {
    return GetUInt64(key, defaultValue);
}

template<>
f32 Config::Get<f32>(StringView key, const f32& defaultValue) const {
    return GetFloat(key, defaultValue);
}

template<>
f64 Config::Get<f64>(StringView key, const f64& defaultValue) const {
    return GetDouble(key, defaultValue);
}

template<>
bool Config::Get<bool>(StringView key, const bool& defaultValue) const {
    return GetBool(key, defaultValue);
}

// GetRequired specializations
template<>
Result<String, String> Config::GetRequired<String>(StringView key) const {
    return GetRequiredString(key);
}

template<>
Result<i32, String> Config::GetRequired<i32>(StringView key) const {
    return GetRequiredInt(key);
}

template<>
Result<i64, String> Config::GetRequired<i64>(StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return Result<i64, String>::Err("Missing required key: " + String(key));
    if (const auto val = node->value<int64_t>()) return *val;
    return Result<i64, String>::Err("Type mismatch for key: " + String(key));
}

template<>
Result<u32, String> Config::GetRequired<u32>(StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return Result<u32, String>::Err("Missing required key: " + String(key));
    if (const auto val = node->value<int64_t>()) return static_cast<u32>(*val);
    return Result<u32, String>::Err("Type mismatch for key: " + String(key));
}

template<>
Result<u64, String> Config::GetRequired<u64>(StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return Result<u64, String>::Err("Missing required key: " + String(key));
    if (const auto val = node->value<int64_t>()) return static_cast<u64>(*val);
    return Result<u64, String>::Err("Type mismatch for key: " + String(key));
}

template<>
Result<f32, String> Config::GetRequired<f32>(StringView key) const {
    return GetRequiredFloat(key);
}

template<>
Result<f64, String> Config::GetRequired<f64>(StringView key) const {
    const toml::node* node = m_impl->Navigate(key);
    if (!node) return Result<f64, String>::Err("Missing required key: " + String(key));
    if (const auto val = node->value<double>()) return *val;
    return Result<f64, String>::Err("Type mismatch for key: " + String(key));
}

template<>
Result<bool, String> Config::GetRequired<bool>(StringView key) const {
    return GetRequiredBool(key);
}

// GetArray specializations
template<>
Vector<String> Config::GetArray<String>(StringView key) const {
    return GetStringArray(key);
}

template<>
Vector<i32> Config::GetArray<i32>(StringView key) const {
    return GetIntArray(key);
}

template<>
Vector<i64> Config::GetArray<i64>(StringView key) const {
    Vector<i64> result;
    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_array()) return result;
    const toml::array* arr = node->as_array();
    result.reserve(arr->size());
    for (const auto& elem : *arr) {
        if (auto val = elem.value<int64_t>()) result.push_back(*val);
    }
    return result;
}

template<>
Vector<u32> Config::GetArray<u32>(StringView key) const {
    Vector<u32> result;
    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_array()) return result;
    const toml::array* arr = node->as_array();
    result.reserve(arr->size());
    for (const auto& elem : *arr) {
        if (auto val = elem.value<int64_t>()) result.push_back(static_cast<u32>(*val));
    }
    return result;
}

template<>
Vector<u64> Config::GetArray<u64>(StringView key) const {
    Vector<u64> result;
    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_array()) return result;
    const toml::array* arr = node->as_array();
    result.reserve(arr->size());
    for (const auto& elem : *arr) {
        if (auto val = elem.value<int64_t>()) result.push_back(static_cast<u64>(*val));
    }
    return result;
}

template<>
Vector<f32> Config::GetArray<f32>(StringView key) const {
    return GetFloatArray(key);
}

template<>
Vector<f64> Config::GetArray<f64>(StringView key) const {
    return GetDoubleArray(key);
}

template<>
Vector<bool> Config::GetArray<bool>(StringView key) const {
    Vector<bool> result;
    const toml::node* node = m_impl->Navigate(key);
    if (!node || !node->is_array()) return result;
    const toml::array* arr = node->as_array();
    result.reserve(arr->size());
    for (const auto& elem : *arr) {
        if (auto val = elem.value<bool>()) result.push_back(*val);
    }
    return result;
}

} // namespace quantiloom
