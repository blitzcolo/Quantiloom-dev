/**
 * @file Config.hpp
 * @brief Configuration loader - public API without third-party dependencies
 *
 * This header exposes only standard C++ types. The underlying toml++
 * implementation is hidden using PIMPL pattern.
 *
 * @author blitzcolo
 */

#pragma once

#include "Platform.hpp"
#include "Types.hpp"

#include <filesystem>
#include <unordered_map>
#include <memory>

// ============================================================================
// Configuration Loader (TOML)
// Parse and validate TOML configuration files
// ============================================================================

namespace quantiloom {

/**
 * @class Config
 * @brief Configuration manager for Quantiloom (reads TOML files)
 *
 * Public API uses only standard C++ types. The toml++ implementation
 * is completely hidden, allowing DLL distribution without toml++ headers.
 */
class QL_API Config {
public:
    /// Load a TOML configuration file
    /// @param filePath Path to the .toml file
    /// @return Result containing parsed config or error
    static Result<Config, String> Load(const std::filesystem::path& filePath);

    /// Parse a TOML document held in memory
    ///
    /// The same reading as Load(), for callers that have the text rather than a
    /// file: an agent sending a configuration over a socket, a test with its
    /// document inline. Relative paths inside the document still resolve
    /// against whatever base directory the consumer supplies -- parsing does not
    /// know where the text came from.
    ///
    /// @param document TOML source
    /// @return Result containing parsed config or the parse error
    static Result<Config, String> Parse(StringView document);

    /// Create an empty configuration
    Config();

    /// Destructor
    ~Config();

    /// Copy constructor
    Config(const Config& other);

    /// Move constructor
    Config(Config&& other) noexcept;

    /// Copy assignment
    Config& operator=(const Config& other);

    /// Move assignment
    Config& operator=(Config&& other) noexcept;

    /// Check if a key exists in the configuration
    /// @param key Dot-separated key path (e.g., "renderer.resolution")
    [[nodiscard]] bool Has(StringView key) const;

    /// Check if a section (table) exists
    /// @param key Section name (e.g., "spectral_curves")
    [[nodiscard]] bool HasSection(StringView key) const;

    // ========================================================================
    // Type-safe Getters (with default values)
    // ========================================================================

    /// Get string value
    [[nodiscard]] String GetString(StringView key, const String& defaultValue = "") const;

    /// Get 32-bit signed integer
    [[nodiscard]] i32 GetInt(StringView key, i32 defaultValue = 0) const;

    /// Get 64-bit signed integer
    [[nodiscard]] i64 GetInt64(StringView key, i64 defaultValue = 0) const;

    /// Get 32-bit unsigned integer
    [[nodiscard]] u32 GetUInt(StringView key, u32 defaultValue = 0) const;

    /// Get 64-bit unsigned integer
    [[nodiscard]] u64 GetUInt64(StringView key, u64 defaultValue = 0) const;

    /// Get 32-bit float
    [[nodiscard]] f32 GetFloat(StringView key, f32 defaultValue = 0.0f) const;

    /// Get 64-bit double
    [[nodiscard]] f64 GetDouble(StringView key, f64 defaultValue = 0.0) const;

    /// Get boolean
    [[nodiscard]] bool GetBool(StringView key, bool defaultValue = false) const;

    // ========================================================================
    // Required Getters (return Result, error if missing)
    // ========================================================================

    /// Get required string value
    [[nodiscard]] Result<String, String> GetRequiredString(StringView key) const;

    /// Get required 32-bit signed integer
    [[nodiscard]] Result<i32, String> GetRequiredInt(StringView key) const;

    /// Get required 32-bit float
    [[nodiscard]] Result<f32, String> GetRequiredFloat(StringView key) const;

    /// Get required boolean
    [[nodiscard]] Result<bool, String> GetRequiredBool(StringView key) const;

    // ========================================================================
    // Array Getters
    // ========================================================================

    /// Get array of strings
    [[nodiscard]] Vector<String> GetStringArray(StringView key) const;

    /// Get array of integers
    [[nodiscard]] Vector<i32> GetIntArray(StringView key) const;

    /// Get array of floats
    [[nodiscard]] Vector<f32> GetFloatArray(StringView key) const;

    /// Get array of doubles
    [[nodiscard]] Vector<f64> GetDoubleArray(StringView key) const;

    // ========================================================================
    // Section/Table Access
    // ========================================================================

    /// Get a nested table as a Config object
    /// @param key Dot-separated key path
    [[nodiscard]] Result<Config, String> GetTable(StringView key) const;

    /// Get an array of tables as a vector of Config objects
    /// @param key Dot-separated key path to array of tables
    /// @return Vector of Config objects, empty if not found or not array of tables
    [[nodiscard]] Vector<Config> GetTableArray(StringView key) const;

    /// Get all key-value pairs from a section as a map
    /// @param key Section name
    /// @return Map of string keys to string values
    [[nodiscard]] std::unordered_map<String, String> GetSection(StringView key) const;

    // ========================================================================
    // Merging
    // ========================================================================

    /// Layer another configuration over this one, key by key
    ///
    /// Tables merge recursively, so an override document naming only
    /// `[renderer] spp` leaves `renderer.resolution` as this configuration had
    /// it. Everything else -- scalars, and arrays including arrays of tables
    /// such as `[[materials]]` -- is replaced whole rather than combined:
    /// array elements carry no identity to pair them up by, so an override
    /// material list is *the* list, not an addition to this one.
    ///
    /// Neither input is modified.
    ///
    /// @param overrides Configuration whose keys win wherever both define one
    /// @return The merged configuration
    [[nodiscard]] Config MergedWith(const Config& overrides) const;

    // ========================================================================
    // Template Interface (for backward compatibility)
    // Supported types: String, i32, i64, u32, u64, f32, f64, bool
    // ========================================================================

    /// Generic getter with default value
    template<typename T>
    T Get(StringView key, const T& defaultValue = T{}) const;

    /// Generic required getter
    template<typename T>
    Result<T, String> GetRequired(StringView key) const;

    /// Generic array getter
    template<typename T>
    Vector<T> GetArray(StringView key) const;

    /// Print the entire config to stdout (for debugging)
    void Print() const;

private:
    // PIMPL - hide toml++ types
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    // Private constructor for internal use
    explicit Config(std::unique_ptr<Impl> impl);
};

// ============================================================================
// Template Specialization Declarations
// (Implementations in Config.cpp via explicit instantiation)
// ============================================================================

template<> QL_API String Config::Get<String>(StringView key, const String& defaultValue) const;
template<> QL_API i32 Config::Get<i32>(StringView key, const i32& defaultValue) const;
template<> QL_API i64 Config::Get<i64>(StringView key, const i64& defaultValue) const;
template<> QL_API u32 Config::Get<u32>(StringView key, const u32& defaultValue) const;
template<> QL_API u64 Config::Get<u64>(StringView key, const u64& defaultValue) const;
template<> QL_API f32 Config::Get<f32>(StringView key, const f32& defaultValue) const;
template<> QL_API f64 Config::Get<f64>(StringView key, const f64& defaultValue) const;
template<> QL_API bool Config::Get<bool>(StringView key, const bool& defaultValue) const;

template<> QL_API Result<String, String> Config::GetRequired<String>(StringView key) const;
template<> QL_API Result<i32, String> Config::GetRequired<i32>(StringView key) const;
template<> QL_API Result<i64, String> Config::GetRequired<i64>(StringView key) const;
template<> QL_API Result<u32, String> Config::GetRequired<u32>(StringView key) const;
template<> QL_API Result<u64, String> Config::GetRequired<u64>(StringView key) const;
template<> QL_API Result<f32, String> Config::GetRequired<f32>(StringView key) const;
template<> QL_API Result<f64, String> Config::GetRequired<f64>(StringView key) const;
template<> QL_API Result<bool, String> Config::GetRequired<bool>(StringView key) const;

template<> QL_API Vector<String> Config::GetArray<String>(StringView key) const;
template<> QL_API Vector<i32> Config::GetArray<i32>(StringView key) const;
template<> QL_API Vector<i64> Config::GetArray<i64>(StringView key) const;
template<> QL_API Vector<u32> Config::GetArray<u32>(StringView key) const;
template<> QL_API Vector<u64> Config::GetArray<u64>(StringView key) const;
template<> QL_API Vector<f32> Config::GetArray<f32>(StringView key) const;
template<> QL_API Vector<f64> Config::GetArray<f64>(StringView key) const;
template<> QL_API Vector<bool> Config::GetArray<bool>(StringView key) const;

} // namespace quantiloom
