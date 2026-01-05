/**
 * @file Types.hpp
 * @brief Fundamental type definitions and C++20 utilities for Quantiloom spectral renderer
 *
 * This header provides the core type system for Quantiloom, including:
 * - Fixed-width integer and floating-point type aliases (i8/u8, i32/u32, f32/f64)
 * - Spectral rendering mode enumeration (Single, RGB_Fused, MWIR_Fused, LWIR_Fused, etc.)
 * - Result<T,E> error handling type (variant-based, std::expected alternative)
 * - Error code definitions for all subsystems
 * - Container type aliases (Vector, Array, Span, Optional, etc.)
 * - C++20 concepts for generic programming (Arithmetic, Numeric, SpectralData)
 * - Physical constants (PI, speed of light, Planck constant, IR band ranges)
 *
 * All Quantiloom code uses these types for consistency and portability across Windows/Linux.
 *
 * @author wtflmao
 */

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
// Quantiloom - Modern C++20 type definitions
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
// Spectral Rendering Modes
// ============================================================================
/**
 * @enum SpectralMode
 * @brief Defines different spectral rendering pipelines supported by Quantiloom
 *
 * Controls how the renderer interprets wavelengths and produces output:
 * - Single: Monochromatic rendering at one wavelength (EXR grayscale output)
 * - RGB_Fused: Standard visible-light RGB rendering with CIE XYZ color matching
 * - IR bands: Thermal/near-IR fusion modes with wavelength-specific processing
 *
 * @note CRITICAL: Must match shader defines in common.hlsli exactly!
 * @note RGB_Fused outputs both EXR (HDR) and PNG (tone-mapped preview)
 * @note IR modes (MWIR/LWIR/SWIR/NIR) output thermal/reflectance imagery
 *
 * @see SpectralData.hpp for wavelength-dependent material properties
 */
enum class SpectralMode : u32 {
    Single       = 0,  // Single wavelength (grayscale output, EXR only)
    RGB_Fused    = 1,  // RGB fusion with CIE XYZ -> sRGB (outputs EXR + PNG)
    Multispectral = 2,  // Multiple wavelengths (hyperspectral cube) - TBD
    MWIR_Fused   = 3,  // Mid-wave IR fusion 3000-5000nm (outputs EXR + PNG)
    LWIR_Fused   = 4,  // Long-wave IR fusion 8000-12000nm (outputs EXR + PNG)
    SWIR_Fused   = 5,  // Short-wave IR fusion 1000-2500nm (outputs EXR + PNG)
    NIR_Fused    = 6   // Near IR fusion 780-1400nm (outputs EXR + PNG) - reflected solar
};

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

/**
 * @class Result
 * @brief C++20-compatible error handling type (variant-based, std::expected alternative)
 *
 * Represents either a successful value (T) or an error (E). Provides explicit error handling
 * without exceptions, forcing call sites to check for errors explicitly.
 *
 * @tparam T Success value type
 * @tparam E Error type (defaults to String)
 *
 * Usage example:
 * @code
 * Result<Image, String> LoadImage(const String& path) {
 *     if (!FileExists(path))
 *         return Result<Image, String>::Err("File not found");
 *     return Image{...};  // Implicit construction from T
 * }
 *
 * auto result = LoadImage("texture.png");
 * if (result.has_value()) {
 *     Image& img = result.value();
 *     // Process image...
 * } else {
 *     QL_LOG_ERROR("Failed to load: {}", result.error());
 * }
 * @endcode
 *
 * @note Prefer this over exceptions for expected failure modes (file I/O, parsing, etc.)
 * @note For unexpected failures (programming errors), use QL_ASSERT or throw exceptions
 * @see ErrorCode for common error categories
 */
template<typename T, typename E = String>
class Result {
public:
    // Construct from success value (implicit for convenient return)
    Result(T&& value) : m_data(std::move(value)) {}
    Result(const T& value) : m_data(value) {}

    // Construct from error
    struct Err {
        E error;
        explicit Err(E&& e) : error(std::move(e)) {}
        explicit Err(const E& e) : error(e) {}
    };

    Result(Err&& err) : m_data(std::move(err.error)) {}
    Result(const Err& err) : m_data(err.error) {}

    // Check if result holds a value
    [[nodiscard]] bool has_value() const { return std::holds_alternative<T>(m_data); }
    explicit operator bool() const { return has_value(); }

    // Access value (throws if error)
    T& value() & { return std::get<T>(m_data); }
    [[nodiscard]] const T& value() const & { return std::get<T>(m_data); }
    T&& value() && { return std::get<T>(std::move(m_data)); }

    T& operator*() & { return value(); }
    const T& operator*() const & { return value(); }
    T&& operator*() && { return std::move(value()); }

    // Access error (throws if value)
    [[nodiscard]] const E& error() const & { return std::get<E>(m_data); }
    E& error() & { return std::get<E>(m_data); }
    E&& error() && { return std::get<E>(std::move(m_data)); }

private:
    std::variant<T, E> m_data;
};

/**
 * @class Result<void, E>
 * @brief Specialization of Result for functions that return nothing on success
 *
 * Used for operations that can fail but don't return a value on success.
 * Uses std::monostate internally to represent the "success" case.
 *
 * Usage:
 * @code
 * Result<void, String> DoSomething() {
 *     if (failed) return Result<void, String>::Err("reason");
 *     return Result<void, String>::Ok();
 * }
 *
 * auto result = DoSomething();
 * if (!result.has_value()) {
 *     std::cerr << result.error() << std::endl;
 * }
 * @endcode
 */
template<typename E>
class Result<void, E> {
public:
    // Construct success (void value)
    Result() : m_data(std::monostate{}) {}

    // Static factory for success
    static Result Ok() { return Result(); }

    // Static factory for error
    static Result Err(E&& e) { return Result(std::move(e)); }
    static Result Err(const E& e) { return Result(e); }

    // Check if result is success
    [[nodiscard]] bool has_value() const { return std::holds_alternative<std::monostate>(m_data); }
    explicit operator bool() const { return has_value(); }

    // Access error (throws if success)
    [[nodiscard]] const E& error() const & { return std::get<E>(m_data); }
    E& error() & { return std::get<E>(m_data); }
    E&& error() && { return std::get<E>(std::move(m_data)); }

private:
    // Private constructor for error
    explicit Result(E&& e) : m_data(std::move(e)) {}
    explicit Result(const E& e) : m_data(e) {}

    std::variant<std::monostate, E> m_data;
};

// Helper function to create error result
template<typename E>
auto Err(E&& error) {
    return typename Result<int, E>::Err(std::forward<E>(error));
}

// ============================================================================
// Spectral Mode Parsing
// ============================================================================
// Convert string from config to SpectralMode
// Defined here (after String/Result) to avoid forward reference issues
// ============================================================================

inline Result<SpectralMode, String> ParseSpectralMode(const StringView mode_str) {
    if (mode_str == "single" || mode_str == "single_wavelength") {
        return Result(SpectralMode::Single);
    } else if (mode_str == "rgb_fused" || mode_str == "rgb" || mode_str == "RGB") {
        // Accept both new name (rgb_fused) and legacy name (rgb) for compatibility
        return Result(SpectralMode::RGB_Fused);
    } else if (mode_str == "multispectral") {
        return Result(SpectralMode::Multispectral);
    } else if (mode_str == "mwir_fused" || mode_str == "MWIR") {
        return Result(SpectralMode::MWIR_Fused);
    } else if (mode_str == "lwir_fused" || mode_str == "LWIR") {
        return Result(SpectralMode::LWIR_Fused);
    } else if (mode_str == "swir_fused" || mode_str == "SWIR") {
        return Result(SpectralMode::SWIR_Fused);
    } else if (mode_str == "nir_fused" || mode_str == "NIR") {
        return Result(SpectralMode::NIR_Fused);
    } else {
        return Result<SpectralMode>(Result<SpectralMode, String>::Err("Invalid spectral mode: " + String(mode_str)));
    }
}

/**
 * @enum ErrorCode
 * @brief Standardized error codes for all Quantiloom subsystems
 *
 * Categorized by subsystem for quick diagnosis:
 * - 1-99: File I/O errors
 * - 100-199: Configuration/parsing errors
 * - 200-299: Vulkan/GPU errors
 * - 300-399: Scene/geometry errors
 * - 400-499: Spectral data errors
 * - 9999: Unknown/uncategorized errors
 *
 * @note Use ErrorCodeToString() to convert to human-readable messages
 * @see Result<T, ErrorCode> for typed error returns
 */
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
    // Visible spectrum
    inline constexpr Wavelength WAVELENGTH_MIN_VISIBLE = 380.0f;
    inline constexpr Wavelength WAVELENGTH_MAX_VISIBLE = 780.0f;  // Standard CIE visible range ends at 780nm

    // Infrared bands (following ISO 20473 classification)
    // NIR: Near-Infrared (780-1400nm) - primarily reflected solar radiation
    inline constexpr Wavelength WAVELENGTH_MIN_NIR = 780.0f;
    inline constexpr Wavelength WAVELENGTH_MAX_NIR = 1400.0f;

    // SWIR: Short-Wave Infrared (1000-2500nm)
    inline constexpr Wavelength WAVELENGTH_MIN_SWIR = 1000.0f;
    inline constexpr Wavelength WAVELENGTH_MAX_SWIR = 2500.0f;

    // MWIR: Mid-Wave Infrared (3000-5000nm)
    inline constexpr Wavelength WAVELENGTH_MIN_MWIR = 3000.0f;
    inline constexpr Wavelength WAVELENGTH_MAX_MWIR = 5000.0f;

    // LWIR: Long-Wave Infrared (8000-12000nm)
    inline constexpr Wavelength WAVELENGTH_MIN_LWIR = 8000.0f;
    inline constexpr Wavelength WAVELENGTH_MAX_LWIR = 12000.0f;

    // Speed of light (m/s)
    inline constexpr f64 SPEED_OF_LIGHT = 299792458.0;

    // Planck constant (J⋅s)
    inline constexpr f64 PLANCK_CONSTANT = 6.62607015e-34;
}

} // namespace quantiloom
