/**
 * @file Types.hpp
 * @brief Fundamental type definitions and C++20 utilities for Quantiloom spectral renderer
 *
 * This header provides the core type system for Quantiloom, including:
 * - Fixed-width integer and floating-point type aliases (i8/u8, i32/u32, f32/f64)
 * - Spectral rendering mode enumeration (RGB, VIS_Fused, Single, MWIR_Fused, LWIR_Fused, etc.)
 * - Result<T,E> error handling type (variant-based, std::expected alternative)
 * - Error code definitions for all subsystems
 * - Container type aliases (Vector, Array, Span, Optional, etc.)
 * - C++20 concepts for generic programming (Arithmetic, Numeric, SpectralData)
 * - Physical constants (PI, speed of light, Planck constant, IR band ranges)
 *
 * All Quantiloom code uses these types for consistency and portability across Windows/Linux.
 *
 * @author blitzcolo
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
 * - RGB: Fast RGB-only pipeline (no spectral integration, best performance)
 * - VIS_Fused: 32-wavelength visible spectral integration with CIE XYZ color matching
 * - Single: Monochromatic rendering at one wavelength (EXR grayscale output)
 * - IR bands: Thermal/near-IR fusion modes with wavelength-specific processing
 *
 * @note CRITICAL: Must match shader defines in common.hlsli exactly!
 * @note RGB/VIS_Fused outputs both EXR (HDR) and PNG (sRGB-encoded preview)
 * @note IR modes (MWIR/LWIR/SWIR/NIR) output thermal/reflectance imagery
 *
 * @see SpectralData.hpp for wavelength-dependent material properties
 */
enum class SpectralMode : u32 {
    Single       = 0,  // Single wavelength (grayscale output, EXR only)
    VIS_Fused    = 1,  // Visible spectral integration 400-780nm, CIE XYZ -> sRGB (outputs EXR + PNG)
    Multispectral = 2,  // Multiple wavelengths (hyperspectral cube) - TBD
    MWIR_Fused   = 3,  // Mid-wave IR fusion 3000-5000nm (outputs EXR + PNG)
    LWIR_Fused   = 4,  // Long-wave IR fusion 8000-12000nm (outputs EXR + PNG)
    SWIR_Fused   = 5,  // Short-wave IR fusion 1400-2400nm (outputs EXR + PNG)
    NIR_Fused    = 6,  // Near IR fusion 930-1200nm (outputs EXR + PNG) - reflected solar
    RGB          = 7   // Fast RGB-only pipeline (no spectral integration, default)
};

// ============================================================================
// Debug Visualization Modes
// ============================================================================
/**
 * @enum DebugVisualizationMode
 * @brief Shader debug visualization modes for rendering pipeline debugging
 *
 * Enables visualization of intermediate rendering data for debugging:
 * - Geometry: normals, UVs, positions, material IDs
 * - Material: albedo, metallic, roughness, emissive
 * - Lighting: NdotL, NdotV, direct sun, diffuse
 * - BRDF: Fresnel F0, full BRDF evaluation
 * - IBL: prefiltered env, BRDF LUT, specular
 * - Spectral: XYZ tristimulus, pre-correction RGB
 * - IR: temperature, emissivity, emission/reflection
 *
 * Values are grouped by category (1-9: geometry, 10-19: material, etc.)
 * for easy navigation and future expansion.
 *
 * @note CRITICAL: Must match shader defines in common.hlsli exactly!
 * @note Set via CameraData.debug_mode, 0 = normal rendering
 */
enum class DebugVisualizationMode : u32 {
    None = 0,              // Normal rendering (no debug output)

    // Geometry (1-9)
    WorldPosition = 1,     // Hit point world coordinates (frac for visibility)
    GeometricNormal = 2,   // Raw geometric normal (before normal map)
    ShadedNormal = 3,      // Final shaded normal (with normal map applied)
    Tangent = 4,           // Tangent vector
    UV = 5,                // Texture coordinates (U, V, 0)
    MaterialID = 6,        // Material index (hashed to color)
    TriangleID = 7,        // Triangle index (hashed to color)
    Barycentric = 8,       // Barycentric coordinates (b0, b1, b2)

    // Material (10-19)
    BaseColor = 10,        // Albedo RGB (texture × factor)
    Metallic = 11,         // Metallic factor (grayscale)
    Roughness = 12,        // Roughness factor (grayscale)
    NormalMapDelta = 13,   // Normal map contribution (dx, dy, 0.5)
    Emissive = 14,         // Emissive RGB (HDR)
    Alpha = 15,            // Alpha channel (grayscale)

    // Lighting (20-29)
    NdotL = 20,            // Dot(Normal, LightDir) (grayscale)
    NdotV = 21,            // Dot(Normal, ViewDir) (grayscale)
    DirectSun = 22,        // Direct sun lighting contribution
    Diffuse = 23,          // Diffuse component (kD × albedo)
    AtmosphericTransmittance = 24, // Atmospheric transmittance (grayscale)

    // BRDF (30-39)
    FresnelF0 = 30,        // Fresnel at normal incidence (F0)
    Fresnel = 31,          // Fresnel at current angle
    BRDF_Full = 32,        // Full Cook-Torrance BRDF evaluation
    SpecularD = 33,        // GGX distribution term D
    SpecularG = 34,        // Geometry/masking term G

    // IBL (40-49)
    ReflectionDir = 40,    // Reflection direction vector
    PrefilteredEnv = 41,   // Prefiltered environment map sample
    BrdfLut = 42,          // BRDF LUT sample (scale, bias, 0)
    IblSpecular = 43,      // IBL specular contribution
    SkyAmbient = 44,       // Sky ambient/diffuse contribution

    // Spectral (50-59)
    XYZ_Tristimulus = 50,  // CIE XYZ tristimulus values (before matrix)
    BeforeChromaCorrection = 51, // RGB before chromaticity correction
    SpectralReflectance550 = 52, // Spectral reflectance at 550nm

    // IR (60-69)
    Temperature = 60,      // Surface temperature (colormap)
    IREmissivity = 61,     // IR emissivity (grayscale)
    IREmission = 62,       // Thermal emission component
    IRReflection = 63,     // IR reflection component

    // Geometry Diagnostics (70-79) - For debugging mesh/index corruption
    VertexPositions = 70,  // Hash of 3 vertex positions (R=v0, G=v1, B=v2)
    IndexValues = 71,      // Triangle indices as colors (normalized by 32)
    InstanceID = 72,       // Instance index (hashed to color)
    PrimitiveID = 73,      // PrimitiveIndex() value (R=id/12 for cube)
    IndexBufferPos = 74,   // Index buffer read position (debug offset calc)
    V0Position = 75,       // v0 vertex position directly (mapped to 0-1)
    RawIdx0 = 76,          // Raw idx0 value and read address
    V0Raw = 77,            // v0 position clamped (not frac)
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
 * @brief C++20-compatible error handling type (tagged union, std::expected alternative)
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
 * @note Uses tagged wrapper types to support Result<T,T> where T==E
 * @see ErrorCode for common error categories
 */
template<typename T, typename E = String>
class Result {
private:
    // Tagged wrapper types to disambiguate when T == E
    struct ValueTag { T data; };
    struct ErrorTag { E data; };

public:
    // Construct from success value (implicit for convenient return)
    Result(T&& value) : m_data(ValueTag{std::move(value)}) {}
    Result(const T& value) : m_data(ValueTag{value}) {}

    // Construct from error wrapper
    struct Err {
        E error;
        explicit Err(E&& e) : error(std::move(e)) {}
        explicit Err(const E& e) : error(e) {}
    };

    Result(Err&& err) : m_data(ErrorTag{std::move(err.error)}) {}
    Result(const Err& err) : m_data(ErrorTag{err.error}) {}

    // Check if result holds a value
    [[nodiscard]] bool has_value() const { return std::holds_alternative<ValueTag>(m_data); }
    explicit operator bool() const { return has_value(); }

    // Access value (throws if error)
    T& value() & { return std::get<ValueTag>(m_data).data; }
    [[nodiscard]] const T& value() const & { return std::get<ValueTag>(m_data).data; }
    T&& value() && { return std::move(std::get<ValueTag>(std::move(m_data)).data); }

    T& operator*() & { return value(); }
    const T& operator*() const & { return value(); }
    T&& operator*() && { return std::move(value()); }

    // Access error (throws if value)
    [[nodiscard]] const E& error() const & { return std::get<ErrorTag>(m_data).data; }
    E& error() & { return std::get<ErrorTag>(m_data).data; }
    E&& error() && { return std::move(std::get<ErrorTag>(std::move(m_data)).data); }

private:
    std::variant<ValueTag, ErrorTag> m_data;
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
    } else if (mode_str == "rgb" || mode_str == "RGB") {
        // Fast RGB-only mode (default, no spectral integration)
        return Result(SpectralMode::RGB);
    } else if (mode_str == "vis_fused" || mode_str == "VIS") {
        // Visible spectral integration mode (32-wavelength CIE XYZ)
        return Result(SpectralMode::VIS_Fused);
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

// ============================================================================
// Fused Band Info
// ============================================================================
// Single CPU-side source of truth for fused-band wavelength ranges.
// CRITICAL: Must match the shader constants in closesthit.rchit / miss.rmiss
// (VIS 400-780, NIR 930-1200, SWIR 1400-2400, MWIR 3000-5000, LWIR 8000-12000).
// The renderer stores per-nm AVERAGE spectral radiance for fused modes
// (band-integrated radiance divided by WidthNm); the sensor chain needs
// WidthNm to recover band-integrated radiance and CenterNm for photon energy.
// ============================================================================

struct SpectralBandInfo {
    f32 lambdaMinNm;
    f32 lambdaMaxNm;

    [[nodiscard]] constexpr f32 CenterNm() const { return 0.5f * (lambdaMinNm + lambdaMaxNm); }
    [[nodiscard]] constexpr f32 WidthNm() const { return lambdaMaxNm - lambdaMinNm; }
};

inline std::optional<SpectralBandInfo> GetFusedBandInfo(SpectralMode mode) {
    switch (mode) {
        case SpectralMode::VIS_Fused:  return SpectralBandInfo{400.0f, 780.0f};
        case SpectralMode::NIR_Fused:  return SpectralBandInfo{930.0f, 1200.0f};
        case SpectralMode::SWIR_Fused: return SpectralBandInfo{1400.0f, 2400.0f};
        case SpectralMode::MWIR_Fused: return SpectralBandInfo{3000.0f, 5000.0f};
        case SpectralMode::LWIR_Fused: return SpectralBandInfo{8000.0f, 12000.0f};
        default:                       return std::nullopt;
    }
}

// IR fused modes render scalar band radiance (grayscale) and need the
// IR-specific sensor unit handling; VIS_Fused outputs CIE-integrated RGB.
inline bool IsIRFusedMode(SpectralMode mode) {
    return mode == SpectralMode::NIR_Fused || mode == SpectralMode::SWIR_Fused ||
           mode == SpectralMode::MWIR_Fused || mode == SpectralMode::LWIR_Fused;
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
    //
    // NOTE: these are the ISO 20473 band *taxonomy* -- what "NIR" means as a
    // term. They are NOT the ranges the renderer integrates. Those come from
    // GetFusedBandInfo() above and differ on purpose:
    //
    //   band   taxonomy (here)   rendered (GetFusedBandInfo)
    //   NIR     780 - 1400        930 - 1200   narrowed to the NN atmosphere's coverage
    //   SWIR   1000 - 2500       1400 - 2400   avoids the NIR overlap below 1400
    //   VIS     380 -  780        400 -  780   CIE tail below 400 contributes ~nothing
    //
    // Quoting these constants to describe a render mode is the mistake that put
    // the wrong ranges into common.hlsli and the shader README. If you want the
    // range a *_Fused mode covers, call GetFusedBandInfo().

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
