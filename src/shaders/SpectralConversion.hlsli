// ============================================================================
// Quantiloom - Spectral Conversion Utilities
// ============================================================================
// Implements physically-based spectral ↔ RGB conversions for accurate
// color rendering in spectral path tracers.
//
// Key Components:
// 1. RGB → Spectrum Upsampling (Sigmoid-based, Jakob & Hanika method)
// 2. CIE 1931 Color Matching Functions (for Spectrum → XYZ)
// 3. XYZ ↔ RGB conversion matrices (sRGB D65 color space)
//
// References:
// - "Spectral and XYZ Color Functions" (PBRT v4, Chapter 4)
// - "A Low-Dimensional Function Space for Efficient Spectral Upsampling"
//   (Jakob & Hanika, 2019)
// - CIE 1931 Standard Observer (2-degree)
// ============================================================================

#ifndef QUANTILOOM_SPECTRAL_CONVERSION_HLSLI
#define QUANTILOOM_SPECTRAL_CONVERSION_HLSLI

// ============================================================================
// Constants
// ============================================================================

static const float LAMBDA_MIN = 380.0;  // Visible spectrum start (nm)
static const float LAMBDA_MAX = 780.0;  // Visible spectrum end (nm)
static const float LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;

// ============================================================================
// RGB → Spectrum Upsampling (Simplified Sigmoid Model)
// ============================================================================
// Converts linear RGB color to a smooth reflectance spectrum R(λ).
//
// This is a **simplified** version using a 3-component weighted sum of
// Gaussian-like basis functions. The full Jakob-Hanika method requires
// precomputed tables; this is a fast approximation for real-time use.
//
// IMPORTANT: Input RGB must be in LINEAR space (not sRGB)!
// ============================================================================

// Evaluate a smooth spectral basis function (approximate Gaussian)
// Center wavelength λ_center, width σ
float SpectralBasis(float lambda, float lambda_center, float sigma) {
    float x = (lambda - lambda_center) / sigma;
    return exp(-0.5 * x * x);
}

// Convert Linear RGB to reflectance spectrum at wavelength λ
// This approximation assumes RGB primaries map to ~450nm (B), ~550nm (G), ~650nm (R)
float ConvertLinearRGBToSpectrum(float3 rgb_linear, float lambda) {
    // Clamp RGB to [0, 1] (reflectance must be positive)
    rgb_linear = saturate(rgb_linear);

    // Define approximate wavelength centers for RGB primaries (sRGB D65)
    const float LAMBDA_RED   = 650.0;  // Red peak
    const float LAMBDA_GREEN = 550.0;  // Green peak
    const float LAMBDA_BLUE  = 450.0;  // Blue peak
    const float SIGMA = 60.0;          // Spectral width (tunable)

    // Weighted sum of basis functions
    float R_lambda =
        rgb_linear.r * SpectralBasis(lambda, LAMBDA_RED, SIGMA) +
        rgb_linear.g * SpectralBasis(lambda, LAMBDA_GREEN, SIGMA) +
        rgb_linear.b * SpectralBasis(lambda, LAMBDA_BLUE, SIGMA);

    // Normalize to ensure white (1,1,1) → uniform spectrum ≈ 1.0
    // The normalization factor is empirically tuned
    const float NORMALIZATION = 1.3;  // Adjust to match white point

    return R_lambda * NORMALIZATION;
}

// ============================================================================
// CIE 1931 Color Matching Functions (Analytical Approximation)
// ============================================================================
// These functions describe how the human eye responds to different wavelengths.
// We use analytical fits (Gaussian-like functions) for efficiency.
//
// Reference: Wyman et al., "Simple Analytic Approximations to the CIE XYZ
//            Color Matching Functions" (2013)
// ============================================================================

// CIE X color matching function (approximate)
float CIE_X(float lambda) {
    float t1 = (lambda - 442.0) * ((lambda < 442.0) ? 0.0624 : 0.0374);
    float t2 = (lambda - 599.8) * ((lambda < 599.8) ? 0.0264 : 0.0323);
    float t3 = (lambda - 501.1) * ((lambda < 501.1) ? 0.0490 : 0.0382);

    return 0.362 * exp(-0.5 * t1 * t1) +
           1.056 * exp(-0.5 * t2 * t2) -
           0.065 * exp(-0.5 * t3 * t3);
}

// CIE Y color matching function (luminosity, approximate)
float CIE_Y(float lambda) {
    float t = (lambda - 568.8) * ((lambda < 568.8) ? 0.0213 : 0.0247);
    return 0.821 * exp(-0.5 * t * t) + 0.286 * exp(-0.5 * pow((lambda - 530.9) / 84.0, 2.0));
}

// CIE Z color matching function (approximate)
float CIE_Z(float lambda) {
    float t = (lambda - 437.0) * ((lambda < 437.0) ? 0.0845 : 0.0278);
    return 1.217 * exp(-0.5 * t * t) + 0.681 * exp(-0.5 * pow((lambda - 459.0) / 50.0, 2.0));
}

// ============================================================================
// Spectrum → CIE XYZ Conversion (Monte Carlo Integration)
// ============================================================================
// Convert spectral radiance L(λ) at a single wavelength to XYZ contribution.
//
// For Monte Carlo integration:
//   XYZ = ∫ L(λ) * CMF(λ) dλ ≈ Σ [L(λ_i) * CMF(λ_i) / pdf(λ_i)]
//
// Parameters:
//   lambda: Sampled wavelength (nm)
//   L_lambda: Spectral radiance at this wavelength (W·sr⁻¹·m⁻²·nm⁻¹)
//   pdf_lambda: Probability density of sampling this wavelength (1/nm)
//
// Returns: XYZ contribution (accumulate over all samples)
// ============================================================================

float3 ConvertSpectrumToXYZ_MonteCarlo(float lambda, float L_lambda, float pdf_lambda) {
    // Evaluate CIE color matching functions at this wavelength
    float x_bar = CIE_X(lambda);
    float y_bar = CIE_Y(lambda);
    float z_bar = CIE_Z(lambda);

    // Monte Carlo estimator: L(λ) * CMF(λ) / pdf(λ)
    float weight = L_lambda / max(pdf_lambda, 1e-8);

    return float3(
        weight * x_bar,
        weight * y_bar,
        weight * z_bar
    );
}

// Simplified version for uniform wavelength sampling over visible spectrum
// Assumes pdf(λ) = 1 / (LAMBDA_MAX - LAMBDA_MIN)
float3 ConvertSpectrumToXYZ_Uniform(float lambda, float L_lambda) {
    float pdf = 1.0 / LAMBDA_RANGE;  // Uniform sampling PDF
    return ConvertSpectrumToXYZ_MonteCarlo(lambda, L_lambda, pdf);
}

// ============================================================================
// CIE XYZ → Linear RGB Conversion (sRGB D65 Color Space)
// ============================================================================
// Converts CIE XYZ tristimulus values to linear sRGB (D65 white point).
//
// Matrix from sRGB specification (IEC 61966-2-1:1999)
// This is the INVERSE of the RGB → XYZ transform.
//
// Output is in LINEAR RGB space (gamma correction applied later)
// ============================================================================

float3 ConvertXYZToLinearRGB(float3 XYZ) {
    // XYZ to Linear sRGB transformation matrix (D65)
    const float3x3 XYZ_TO_RGB = float3x3(
         3.2406, -1.5372, -0.4986,
        -0.9689,  1.8758,  0.0415,
         0.0557, -0.2040,  1.0570
    );

    float3 rgb_linear = mul(XYZ_TO_RGB, XYZ);

    // Clamp negative values (can occur due to out-of-gamut colors)
    // In production, you might want to use gamut mapping instead
    return max(rgb_linear, 0.0);
}

// ============================================================================
// Linear RGB → sRGB (Gamma Correction / OETF)
// ============================================================================
// Applies sRGB opto-electronic transfer function (OETF) for display encoding.
//
// This is NOT needed for intermediate HDR buffers - only apply at final output!
// ============================================================================

float LinearToSRGB_Component(float linear_value) {
    if (linear_value <= 0.0031308) {
        return 12.92 * linear_value;
    } else {
        return 1.055 * pow(linear_value, 1.0 / 2.4) - 0.055;
    }
}

float3 ConvertLinearRGBToSRGB(float3 rgb_linear) {
    return float3(
        LinearToSRGB_Component(rgb_linear.r),
        LinearToSRGB_Component(rgb_linear.g),
        LinearToSRGB_Component(rgb_linear.b)
    );
}

// ============================================================================
// sRGB → Linear RGB (Inverse OETF)
// ============================================================================
// Converts sRGB-encoded texture values to linear space for physically-based
// rendering. ALWAYS apply this to sRGB textures before shading!
// ============================================================================

float SRGBToLinear_Component(float srgb) {
    if (srgb <= 0.04045) {
        return srgb / 12.92;
    } else {
        return pow((srgb + 0.055) / 1.055, 2.4);
    }
}

float3 ConvertSRGBToLinearRGB(float3 srgb) {
    return float3(
        SRGBToLinear_Component(srgb.r),
        SRGBToLinear_Component(srgb.g),
        SRGBToLinear_Component(srgb.b)
    );
}

// ============================================================================
// Helper: Wavelength-dependent reflectance from RGB texture
// ============================================================================
// Convenience function that combines:
//   1. sRGB → Linear RGB conversion (if needed)
//   2. Linear RGB → Spectrum upsampling at target wavelength
//
// Use this when sampling a glTF texture to get spectral reflectance.
//
// Parameters:
//   rgb_srgb: Color sampled from texture (assumed sRGB-encoded)
//   lambda: Target wavelength (nm)
//   is_srgb: Whether the input is sRGB-encoded (true for most glTF textures)
//
// Returns: Reflectance at wavelength λ (0 to ~1, can exceed 1 for HDR)
// ============================================================================

float GetSpectralReflectanceFromRGBTexture(float3 rgb_srgb, float lambda, bool is_srgb = true) {
    // Convert to linear RGB if needed
    float3 rgb_linear = is_srgb ? ConvertSRGBToLinearRGB(rgb_srgb) : rgb_srgb;

    // Upsample to spectrum at target wavelength
    return ConvertLinearRGBToSpectrum(rgb_linear, lambda);
}

#endif // QUANTILOOM_SPECTRAL_CONVERSION_HLSLI
