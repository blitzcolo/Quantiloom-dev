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
#include <memory>

namespace quantiloom {

/// Generic sensor implementation - covers VIS/SWIR/MWIR/LWIR
/// Full chain: Optics (PSF) → Detector (QE, noise) → ADC (quantization)
///
/// The chain's state -- RNG, FPN maps, seeding -- lives behind a pimpl. It used
/// to sit in this header, which put sizeof(GenericSensor) into the ABI: a
/// frontend calling make_unique<GenericSensor>() baked the layout in, so adding
/// a member silently broke a Quantiloom-Qt built against the previous SDK. That
/// has happened (see SdkGuard in Quantiloom-Qt). Now only Apply() is contract.
class QL_API GenericSensor final : public SensorModel {
public:
    GenericSensor();
    ~GenericSensor() override;

    /// Apply full sensor chain to HDR input
    auto Apply(const Image& hdr, const SensorParams& params)
        -> Result<SensorOutput, String> override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace quantiloom
