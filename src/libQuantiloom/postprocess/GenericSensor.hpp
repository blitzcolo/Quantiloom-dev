/**
 * @file GenericSensor.hpp
 * @brief Generic sensor model implementation for VIS/SWIR/MWIR/LWIR imaging simulation
 *
 * Provides GenericSensor class implementing complete sensor simulation chain:
 * 1. Optics: PSF (Point Spread Function) blur via Gaussian convolution
 * 2. Detector: Quantum efficiency, well capacity, photoelectron conversion
 * 3. Noise: Poisson (shot), read noise, dark current, FPN (PRNU/DSNU)
 * 4. ADC: Quantization to digital numbers (DN) with configurable bit depth
 *
 * Sensor chain:
 * @code
 * HDR Radiance (W·sr⁻¹·m⁻²)
 *   ↓ [Optics: PSF blur]
 * Blurred Radiance
 *   ↓ [Detector: QE × integration time × pixel area]
 * Photoelectrons (e⁻)
 *   ↓ [Noise: Poisson + Read + Dark + FPN]
 * Noisy Photoelectrons
 *   ↓ [ADC: quantize to [0, 2^bitDepth-1]]
 * Digital Numbers (DN)
 * @endcode
 *
 * Noise models:
 * - Poisson: sqrt(N) shot noise from photon statistics
 * - Read noise: Gaussian additive noise from readout circuitry
 * - Dark current: Temperature-dependent thermal electron generation
 * - FPN (Fixed Pattern Noise): PRNU (gain) + DSNU (dark) per-pixel maps
 *
 * Output products:
 * - rawDN: Quantized sensor output [0, 2^bitDepth-1] (realistic sensor data)
 * - enhancedPreview: Noisy radiance with PSF (for visualization)
 *
 * @note Implements SensorModel abstract interface
 * @note All noise sources can be toggled via SensorParams flags
 * @note FPN maps generated once and cached (deterministic per-run)
 *
 * @see SensorModel for abstract base class
 * @see SensorParams for configuration parameters
 * @see SensorOutput for output data structure
 *
 * @author blitzcolo
 */

#pragma once

#include "SensorModel.hpp"
#include "core/Platform.hpp"
#include <random>

namespace quantiloom {

/// Generic sensor implementation - covers VIS/SWIR/MWIR/LWIR
/// Full chain: Optics (PSF) → Detector (QE, noise) → ADC (quantization)
class QL_API GenericSensor final : public SensorModel {
public:
    GenericSensor();
    ~GenericSensor() override = default;

    /// Apply full sensor chain to HDR input
    auto Apply(const Image& hdr, const SensorParams& params)
        -> Result<SensorOutput, String> override;

private:
    // Seed the RNG from SensorParams on first use, or when the requested seed
    // changes. Invalidates the FPN maps, which belong to the previous stream.
    auto EnsureSeeded(u32 requestedSeed) -> void;

    // Step 1: Apply optical PSF (Gaussian approximation)
    static auto ApplyPSF(const Image& img, f32 sigma_pixels) -> Image;

    // Step 2: Convert radiance to photo-electrons
    static auto RadianceToElectrons(const Image& radiance, const SensorParams& p) -> Image;

    // Step 3: Add noise sources
    auto AddNoise(Image& electrons, const SensorParams& p) -> void;

    // Step 4: Quantize to digital numbers (DN)
    static auto QuantizeToDN(const Image& electrons, const SensorParams& p) -> Image;

    // Step 5: Convert noisy electrons back to radiance (for preview)
    static auto ElectronsToRadiance(const Image& electrons, const SensorParams& p) -> Image;

    // FPN: Generate fixed pattern noise maps (PRNU + DSNU)
    auto GenerateFPNMaps(u32 width, u32 height, const SensorParams& p) -> void;

    // FPN: Apply fixed pattern noise to electron signal
    auto ApplyFPN(Image& electrons, const SensorParams& p) -> void;

    // Helpers
    static auto MakeGaussianKernel(f32 sigma) -> Vector<f32>;

    static auto ConvolveX(const Image& img, const Vector<f32>& kernel) -> Image;
    static auto ConvolveY(const Image& img, const Vector<f32>& kernel) -> Image;

    // RNG for noise. Seeded from SensorParams::noiseSeed on first use rather
    // than at construction, so the seed travels with the parameters (and thus
    // with the scene TOML) instead of being fixed before they are known.
    // Re-seeded if a later Apply asks for a different seed.
    std::mt19937 m_Rng;
    bool m_Seeded = false;
    u32 m_SeededWith = 0;

    // FPN maps (generated once, reused for all frames)
    Image m_PRNUMap;  // Photo Response Non-Uniformity (multiplicative gain map)
    Image m_DSNUMap;  // Dark Signal Non-Uniformity (additive dark current map)
    bool m_FPNMapsGenerated = false;
};

} // namespace quantiloom
