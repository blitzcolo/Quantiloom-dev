#pragma once

#include "../core/Image.hpp"
#include "../core/Types.hpp"
#include "../core/Platform.hpp"

#include <cmath>
#include <numbers>

// ============================================================================
// Sensor Simulation - Abstract Interface
// ============================================================================
// Models the full sensor chain: Optics → Detector → ADC
// Converts radiance (W/m²/sr) to digital numbers (DN)
// ============================================================================

namespace quantiloom {

/// Sensor parameters (from TOML config)
struct SensorParams {
    // Optics
    f32 focalLength_mm = 50.0f;
    f32 fNumber = 2.8f;
    f32 pixelPitch_um = 5.0f;

    // PSF width override, in pixels. Negative (the default) derives the width
    // from the diffraction formula below; >= 0 replaces it outright, so 0 means
    // no blur at all -- which is why the sentinel is negative rather than zero.
    // Exists so a study sweeping blur can vary it without moving fNumber, which
    // would move the collection solid angle with it.
    f32 psfSigma_px = -1.0f;

    // Detector
    f32 quantumEfficiency = 0.8f;       // QE at peak wavelength [0.0-1.0]
    f32 wellCapacity_e = 50000.0f;      // Full well capacity (electrons)
    f32 readNoise_e_rms = 10.0f;        // Read noise (electrons RMS)
    f32 darkCurrent_e_s = 50.0f;        // Dark current (electrons/second)
    f32 integrationTime_s = 0.01f;      // Integration time (seconds)

    // ADC
    u32 bitDepth = 14;                  // ADC bit depth (12/14/16)
    f32 gain = 3.0f;                    // Gain (electrons/DN) - typical: 2-5

    // Noise flags
    bool enablePoissonNoise = true;     // Shot noise (photon counting)
    bool enableReadNoise = true;        // Readout noise
    bool enableDarkCurrent = true;      // Dark current noise
    bool enableFPN = false;             // Fixed pattern noise (PRNU/DSNU)

    // FPN parameters (Fixed Pattern Noise)
    f32 prnuSigma = 0.01f;              // PRNU standard deviation (typical: 0.005-0.02, i.e., 0.5-2%)
    f32 dsnuSigma_e = 5.0f;             // DSNU standard deviation (electrons, typical: 5-20)

    // Noise RNG seed. Fixed by default, so two renders of the same scene with
    // the same parameters produce bit-identical raw DN -- which a renderer
    // aimed at quantitative validation needs, and which was impossible while
    // the sensor seeded itself from std::random_device with no way to override.
    // Set 0 to draw a nondeterministic seed instead (frame-varying noise).
    u32 noiseSeed = 0x548CU;

    // NUC parameters (Non-Uniformity Correction)
    bool enableNUC = false;             // Apply NUC correction (leaves residual noise)
    f32 nucEfficiency = 0.98f;          // NUC efficiency (typical: 0.95-0.99, i.e., 95-99%)

    // Temperature (for IR sensors)
    f32 detectorTemperature_K = 77.0f;  // Detector temperature (K)

    // Wavelength (for QE calculation, optional)
    f32 wavelength_nm = 550.0f;         // Peak wavelength (nm)

    // Vignetting (optical falloff at image edges)
    bool enableVignetting = false;      // Enable cos^4 natural vignetting
    f32 fov_deg = 45.0f;                // Horizontal field of view (degrees)
    bool isTelecentric = false;         // Telecentric lens (no vignetting)
};

// ============================================================================
// Optics formulas
// ============================================================================
// Shared by the CPU chain (GenericSensor) and the GPU chain (the sensor_*
// compute shaders, dispatched from ExternalRenderContext). Both used to carry
// their own copy, and the tests carried a third; one definition is what keeps
// them from drifting.

/// Gaussian sigma matched to the FWHM of the diffraction-limited Airy core,
/// in units of lambda*fNumber.
///
/// The Airy intensity [2*J1(x)/x]^2 with x = pi*r/(lambda*N) falls to half at
/// x = 1.6163, so its FWHM is 2*1.6163/pi = 1.029 lambda*N, and a Gaussian of
/// equal FWHM has sigma = 1.029 / (2*sqrt(2*ln2)) = 0.437 lambda*N.
///
/// Before 2026-08-15 this was 1.22 -- the radius of the Airy pattern's first
/// zero, which is a correct Rayleigh radius but not a standard deviation. Every
/// PSF was therefore 2.79x too wide. Renders made before that date carry the old
/// width; see the commit body for the change.
constexpr f32 kAiryGaussianSigmaFactor = 0.437f;

/// Below this sigma the Gaussian is narrower than a pixel can express, and both
/// paths skip the blur entirely rather than convolving with a near-delta kernel.
constexpr f32 kMinPSFSigmaPixels = 0.1f;

/// Upper bound on the GPU kernel, which sizes its radius from sigma.
constexpr f32 kMaxPSFSigmaPixels = 10.0f;

/// PSF Gaussian sigma in pixels: the explicit override when given, otherwise
/// the diffraction-limited width. An override of 0 means no blur, which is why
/// the sentinel is negative.
inline auto PSFSigmaPixels(const SensorParams& p) -> f32 {
    if (p.psfSigma_px >= 0.0f) {
        return p.psfSigma_px;
    }
    const f32 wavelength_m = p.wavelength_nm * 1e-9f;
    const f32 pixelPitch_m = p.pixelPitch_um * 1e-6f;
    return kAiryGaussianSigmaFactor * wavelength_m * p.fNumber / pixelPitch_m;
}

/// Radius, in pixels, of the separable Gaussian kernel for a given sigma:
/// 3 sigma, which covers 99.7% of the Gaussian. Zero means no blur -- the CPU
/// chain returns the image untouched, and the GPU chain runs its two passes
/// with a single unit-weight tap, which copies.
inline auto PSFKernelRadiusPixels(f32 sigma_px) -> u32 {
    if (sigma_px < kMinPSFSigmaPixels) {
        return 0;
    }
    const f32 clamped = sigma_px < kMaxPSFSigmaPixels ? sigma_px : kMaxPSFSigmaPixels;
    return static_cast<u32>(std::ceil(3.0f * clamped));
}

/// Solid angle (sr) of the exit pupil seen from a pixel.
///
/// For a Lambertian scene through a cone of half-angle theta the irradiance is
/// E = pi*L*sin^2(theta), and tan(theta) = 1/(2N) gives sin^2 = 1/(1 + 4N^2).
/// Before 2026-08-15 this used the small-angle form pi/(4N^2), which overstates
/// collection by 1 + 1/(4N^2) -- 6.25% at f/2, 1.56% at f/4. Constant at a fixed
/// aperture, hence invisible to any exposure calibration, but a systematic bias
/// along the axis whenever fNumber is swept.
inline auto ApertureSolidAngleSr(f32 fNumber) -> f64 {
    const f64 n = static_cast<f64>(fNumber);
    return std::numbers::pi / (1.0 + 4.0 * n * n);
}

/// Sensor output containing both raw DN and enhanced preview
struct SensorOutput {
    Image rawDN;            // Raw sensor DN values [0, 2^bitDepth-1]
    Image enhancedPreview;  // Noisy radiance with PSF blur (same scale as input)
};

/// Abstract sensor model interface
class SensorModel {
public:
    virtual ~SensorModel() = default;

    /// Apply sensor effects to HDR radiance image
    /// @param hdr Input HDR image (radiance in W/m²/sr)
    /// @param params Sensor parameters
    /// @return SensorOutput containing raw DN and enhanced preview, or error
    virtual auto Apply(const Image& hdr, const SensorParams& params)
        -> Result<SensorOutput, String> = 0;
};

} // namespace quantiloom
