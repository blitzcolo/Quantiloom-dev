#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <span>
#include <array>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <concepts>

// ============================================================================
// Fundamental Type Aliases & C++20 Utilities
// Quantiloom M0 - Modern C++20 type definitions
// ============================================================================

namespace quantiloom {

// ============================================================================
// Integer Types (explicit width)
// ============================================================================
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using usize = std::size_t;
using isize = std::ptrdiff_t;

// ============================================================================
// Floating-Point Types
// ============================================================================
using f32 = float;
using f64 = double;

// Spectral wavelength type (nanometers, typically in range [380, 2500])
using Wavelength = f32;

// ============================================================================
// String Types
// ============================================================================
using String = std::string;
using StringView = std::string_view;

// ============================================================================
// Container Type Aliases
// ============================================================================
template<typename T>
using Span = std::span<T>;

template<typename T, usize N>
using Array = std::array<T, N>;

template<typename T>
using Vector = std::vector<T>;

template<typename T>
using UniquePtr = std::unique_ptr<T>;

template<typename T>
using SharedPtr = std::shared_ptr<T>;

template<typename T>
using WeakPtr = std::weak_ptr<T>;

template<typename T>
using Optional = std::optional<T>;

// ============================================================================
// Error Handling (C++20-compatible Result type)
// ============================================================================

/// Simple Result type using std::variant (C++20 compatible)
/// Usage: Result<T, E> func() { return T{...}; } or { return Err<E>{msg}; }
template<typename T, typename E = String>
class Result {
public:
    // Construct from success value
    Result(T&& value) : m_data(std::move(value)) {}
    Result(const T& value) : m_data(value) {}

    // Construct from error
    struct Err {
        E error;
        explicit Err(E&& e) : error(std::move(e)) {}
        explicit Err(const E& e) : error(e) {}
    };

    Result(Err&& err) : m_data(std::move(err.error)) {}

    // Check if result holds a value
    bool has_value() const { return std::holds_alternative<T>(m_data); }
    explicit operator bool() const { return has_value(); }

    // Access value (throws if error)
    T& value() & { return std::get<T>(m_data); }
    const T& value() const & { return std::get<T>(m_data); }
    T&& value() && { return std::get<T>(std::move(m_data)); }

    T& operator*() & { return value(); }
    const T& operator*() const & { return value(); }
    T&& operator*() && { return std::move(value()); }

    // Access error (throws if value)
    const E& error() const & { return std::get<E>(m_data); }
    E& error() & { return std::get<E>(m_data); }
    E&& error() && { return std::get<E>(std::move(m_data)); }

private:
    std::variant<T, E> m_data;
};

// Helper function to create error result
template<typename E>
auto Err(E&& error) {
    return typename Result<int, E>::Err(std::forward<E>(error));
}

/// Error codes for Quantiloom operations
enum class ErrorCode : u32 {
    Success = 0,

    // File I/O errors (1-99)
    FileNotFound = 1,
    FileReadError = 2,
    FileWriteError = 3,

    // Configuration errors (100-199)
    ConfigParseError = 100,
    ConfigMissingKey = 101,
    ConfigInvalidValue = 102,

    // Vulkan errors (200-299)
    VulkanInitFailed = 200,
    VulkanDeviceNotFound = 201,

    // Scene errors (300-399)
    SceneLoadFailed = 300,
    MaterialInvalid = 301,

    // Spectral errors (400-499)
    WavelengthOutOfRange = 400,
    SpectralDataCorrupted = 401,

    Unknown = 9999
};

// ============================================================================
// C++20 Concepts for Generic Constraints
// ============================================================================

/// Concept: Arithmetic type (integral or floating-point)
template<typename T>
concept Arithmetic = std::integral<T> || std::floating_point<T>;

/// Concept: Numeric type supporting basic math operations
template<typename T>
concept Numeric = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
};

/// Concept: Spectral type (must have wavelength field or method)
template<typename T>
concept SpectralData = requires(T t) {
    { t.wavelength } -> std::convertible_to<Wavelength>;
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert ErrorCode to human-readable string
constexpr const char* ErrorCodeToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success:            return "Success";
        case ErrorCode::FileNotFound:       return "File not found";
        case ErrorCode::FileReadError:      return "File read error";
        case ErrorCode::FileWriteError:     return "File write error";
        case ErrorCode::ConfigParseError:   return "Config parse error";
        case ErrorCode::ConfigMissingKey:   return "Config missing key";
        case ErrorCode::ConfigInvalidValue: return "Config invalid value";
        case ErrorCode::VulkanInitFailed:   return "Vulkan initialization failed";
        case ErrorCode::VulkanDeviceNotFound: return "Vulkan device not found";
        case ErrorCode::SceneLoadFailed:    return "Scene load failed";
        case ErrorCode::MaterialInvalid:    return "Material invalid";
        case ErrorCode::WavelengthOutOfRange: return "Wavelength out of range";
        case ErrorCode::SpectralDataCorrupted: return "Spectral data corrupted";
        default:                            return "Unknown error";
    }
}

// ============================================================================
// Constants
// ============================================================================

namespace constants {
    // Physical constants
    inline constexpr f64 PI = 3.14159265358979323846;
    inline constexpr f64 TWO_PI = 2.0 * PI;
    inline constexpr f64 INV_PI = 1.0 / PI;

    // Spectral range (nanometers)
    inline constexpr Wavelength WAVELENGTH_MIN_VISIBLE = 380.0f;
    inline constexpr Wavelength WAVELENGTH_MAX_VISIBLE = 760.0f;
    inline constexpr Wavelength WAVELENGTH_MIN_IR = 760.0f;
    inline constexpr Wavelength WAVELENGTH_MAX_IR = 2500.0f;

    // Speed of light (m/s)
    inline constexpr f64 SPEED_OF_LIGHT = 299792458.0;

    // Planck constant (J⋅s)
    inline constexpr f64 PLANCK_CONSTANT = 6.62607015e-34;
}

} // namespace quantiloom
