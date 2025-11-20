#pragma once

#include "Platform.hpp"
#include "Types.hpp"

QL_DISABLE_WARNINGS_PUSH
#include <toml++/toml.h>
QL_DISABLE_WARNINGS_POP

#include <filesystem>

// ============================================================================
// Configuration Loader (TOML)
// Parse and validate TOML configuration files
// ============================================================================

namespace quantiloom {

/// Configuration manager for Quantiloom (reads TOML files)
class QL_API Config {
public:
    /// Load a TOML configuration file
    /// @param filePath Path to the .toml file
    /// @return Result containing parsed config or error
    static Result<Config, String> Load(const std::filesystem::path& filePath);

    /// Create an empty configuration
    Config() = default;

    /// Check if a key exists in the configuration
    /// @param key Dot-separated key path (e.g., "renderer.resolution")
    bool Has(StringView key) const;

    /// Get a value from the configuration (with optional default)
    /// @tparam T Expected value type (i32, f32, String, bool, etc.)
    /// @param key Dot-separated key path
    /// @param defaultValue Fallback value if key is missing
    template<typename T>
    T Get(StringView key, const T& defaultValue = T{}) const;

    /// Get a value (throws if key is missing or wrong type)
    /// @tparam T Expected value type
    /// @param key Dot-separated key path
    template<typename T>
    Result<T, String> GetRequired(StringView key) const;

    /// Get a nested table as a Config object
    /// @param key Dot-separated key path
    Result<Config, String> GetTable(StringView key) const;

    /// Get an array of values
    /// @tparam T Element type
    /// @param key Dot-separated key path
    template<typename T>
    Vector<T> GetArray(StringView key) const;

    /// Access the underlying toml::table (for advanced usage)
    const toml::table& GetRoot() const { return m_Root; }

    /// Print the entire config to stdout (for debugging)
    void Print() const;

private:
    explicit Config(toml::table&& root);

    toml::table m_Root;

    /// Helper: Navigate to a nested node by dot-separated path
    const toml::node* Navigate(StringView key) const;
};

// ============================================================================
// Template Implementation
// ============================================================================

template<typename T>
T Config::Get(StringView key, const T& defaultValue) const {
    const toml::node* node = Navigate(key);
    if (!node) {
        return defaultValue;
    }

    if constexpr (std::is_same_v<T, String>) {
        if (auto val = node->value<std::string>()) {
            return *val;
        }
    } else if constexpr (std::is_same_v<T, i32>) {
        if (auto val = node->value<int64_t>()) {
            return static_cast<i32>(*val);
        }
    } else if constexpr (std::is_same_v<T, u32>) {
        if (auto val = node->value<int64_t>()) {
            return static_cast<u32>(*val);
        }
    } else if constexpr (std::is_same_v<T, f32>) {
        if (auto val = node->value<double>()) {
            return static_cast<f32>(*val);
        }
    } else if constexpr (std::is_same_v<T, f64>) {
        if (auto val = node->value<double>()) {
            return *val;
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (auto val = node->value<bool>()) {
            return *val;
        }
    }

    return defaultValue;
}

template<typename T>
Result<T, String> Config::GetRequired(StringView key) const {
    const toml::node* node = Navigate(key);
    if (!node) {
        return Result<T, String>::Err("Missing required key: " + String(key));
    }

    if constexpr (std::is_same_v<T, String>) {
        if (auto val = node->value<std::string>()) {
            return *val;
        }
    } else if constexpr (std::is_same_v<T, i32>) {
        if (auto val = node->value<int64_t>()) {
            return static_cast<i32>(*val);
        }
    } else if constexpr (std::is_same_v<T, u32>) {
        if (auto val = node->value<int64_t>()) {
            return static_cast<u32>(*val);
        }
    } else if constexpr (std::is_same_v<T, f32>) {
        if (auto val = node->value<double>()) {
            return static_cast<f32>(*val);
        }
    } else if constexpr (std::is_same_v<T, f64>) {
        if (auto val = node->value<double>()) {
            return *val;
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (auto val = node->value<bool>()) {
            return *val;
        }
    }

    return Result<T, String>::Err("Type mismatch for key: " + String(key));
}

template<typename T>
Vector<T> Config::GetArray(StringView key) const {
    const toml::node* node = Navigate(key);
    Vector<T> result;

    if (!node || !node->is_array()) {
        return result;
    }

    const toml::array* arr = node->as_array();
    result.reserve(arr->size());

    for (const auto& elem : *arr) {
        if constexpr (std::is_same_v<T, String>) {
            if (auto val = elem.value<std::string>()) {
                result.push_back(*val);
            }
        } else if constexpr (std::is_same_v<T, i8>) {
            if (auto val = elem.value<int64_t>()) {
                result.push_back(static_cast<i8>(*val));
            }
        } else if constexpr (std::is_same_v<T, i16>) {
            if (auto val = elem.value<int64_t>()) {
                result.push_back(static_cast<i16>(*val));
            }
        } else if constexpr (std::is_same_v<T, i32>) {
            if (auto val = elem.value<int64_t>()) {
                result.push_back(static_cast<i32>(*val));
            }
        } else if constexpr (std::is_same_v<T, i64>) {
            if (auto val = elem.value<int64_t>()) {
                result.push_back(static_cast<i64>(*val));
            }
        } else if constexpr (std::is_same_v<T, u8>) {
            if (auto val = elem.value<int64_t>()) {
                result.push_back(static_cast<u8>(*val));
            }
        } else if constexpr (std::is_same_v<T, u16>) {
            if (auto val = elem.value<int64_t>()) {
                result.push_back(static_cast<u16>(*val));
            }
        } else if constexpr (std::is_same_v<T, u32>) {
            if (auto val = elem.value<int64_t>()) {
                result.push_back(static_cast<u32>(*val));
            }
        } else if constexpr (std::is_same_v<T, u64>) {
            if (auto val = elem.value<int64_t>()) {
                result.push_back(static_cast<u64>(*val));
            }
        } else if constexpr (std::is_same_v<T, f32>) {
            if (auto val = elem.value<double>()) {
                result.push_back(static_cast<f32>(*val));
            }
        } else if constexpr (std::is_same_v<T, f64>) {
            if (auto val = elem.value<double>()) {
                result.push_back(*val);
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            if (auto val = elem.value<bool>()) {
                result.push_back(*val);
            }
        } else {
            return Result<Vector<T>, String>::Err("Unsupported type: " + std::string(typeid(T).name()));
        }
    }

    return result;
}

} // namespace quantiloom
